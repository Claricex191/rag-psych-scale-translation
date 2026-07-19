import os

from certifi import contents

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import base64
import io
import json
import re
import logging
import uuid
from pathlib import Path
from typing import Any, List, Optional, Dict

import anthropic
from openai import AsyncOpenAI
from google import genai
import click
import mcp.types as types
import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from dotenv import load_dotenv
from mcp.server.lowlevel import Server
from pdf2image import convert_from_path
from qdrant_client import QdrantClient, models

from enum import Enum

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GUIDELINES_DIR = Path(os.getenv("GUIDELINES_DIR", "./guidelines-forward"))
GUIDELINES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------- RAG Server -------------------------------
class Rag_server:
    def __init__(self, use_memory: bool = False, anthropic_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None,
                 deepseek_api_key: Optional[str] = None):
        # Qdrant setup
        if use_memory:
            self.qdrant_client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant client")
        else:
            try:
                qdrant_path = os.getenv("QDRANT_PATH", "./qdrant-db-forward")
                self.qdrant_client = QdrantClient(path=qdrant_path)
                logger.info(f"Using embedded Qdrant at {qdrant_path}")
            except Exception as e:
                logger.error(f"Failed to connect to local Qdrant: {e}")
                raise
        self.qdrant_client.set_model("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")  # Supports multiple languages like English and Chinese

        self.colpali_model = None
        self.colpali_processor = None
        self.collection_name = "oise_mt"

        # Track two-stage flow
        self.session_state = {}

        # Claude Vision setup
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        # OpenAI Vision setup
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.deepseek_api_key = deepseek_api_key or os.getenv('DEEPSEEK_API_KEY')

        available_keys = [k for k in [self.anthropic_api_key, self.openai_api_key,
                                      self.gemini_api_key, self.deepseek_api_key] if k]
        if not available_keys:
            logger.error(
                "No API keys found! Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY")
            raise ValueError("No LLM API keys found")

        if self.anthropic_api_key:
            self.claude_client = anthropic.AsyncAnthropic(api_key=self.anthropic_api_key)
            logger.info("Claude client initialized successfully")
        else:
            self.claude_client = None
            logger.warning("No ANTHROPIC_API_KEY — Claude models unavailable")

        if self.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialized successfully")
        else:
            self.openai_client = None
            logger.warning("No OPENAI_API_KEY — OpenAI models unavailable")

        if self.gemini_api_key:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            logger.info("Gemini client initialized successfully")
        else:
            self.gemini_client = None
            logger.warning("No GEMINI_API_KEY — Gemini models unavailable")

        if self.deepseek_api_key:
            self.deepseek_client = AsyncOpenAI(api_key=self.deepseek_api_key, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
            logger.info("Qwen client initialized successfully")
        else:
            self.deepseek_client = None
            logger.warning("No DEEPSEEK_API_KEY — Qwen models unavailable")

        self.pdf_images = {}

        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.set_per_process_memory_fraction(0.95)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.warning("GPU not available, using CPU")

    def _build_prompt(self, scale_path: str) -> str:
        with open(scale_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # scale_name = data['scale_info']['name']
        # short_name = data['scale_info']['short_name']
        target_population = data['scale_info']['target_population']
        scale_text = '\n'.join([f"{item['number']}. {item['text']}" for item in data['items']])
        system_instruction = f"""
You are a bilingual professor in the fields of education and psychology, with expertise in psychological measurement, scale development, and cross-linguistic test adaptation. 
You are familiar with standardized self-report instruments and their use in research and applied settings.

Before performing the translation, please note the following information:
    1. This psychological scale is open use and does not have copyright restrictions that limit translation or adaptation for research purposes.
    2. The target population for this translation is: {target_population}.

You are now asked to perform a forward translation of a psychological self-report scale item from English into Chinese as part of a formal test adaptation process.
Please translate the item below in accordance with the established translation framework already provided to you.

**Source items:**
{scale_text}

**Expected output:**
Please provide:
    1. The forward translation of the item into Chinese.
    2. A brief decision log for each scale, identifying terms or phrases that required special consideration (e.g., ambiguity, meaning shift, interpretation uncertainty) during translation (if applicable).

**Output Format Requirement:**
    - If multiple items are provided (e.g., "1. ... 2. ... 3. ..."), translate ALL of them
    - Preserve item structure, numbering, and response scales
    - You must respond with valid JSON in this EXACT structure:
        {{
          "items": [
            {{"number": "1", "original": "original text here", "translation": "translated text here", "log": "decision log here"}},
            {{"number": "2", "original": "original text here", "translation": "translated text here", "log": "decision log here"}}
          ]
        }}
    Do not include any text outside the JSON object.
"""

        return f"""{system_instruction}
Please provide your response here: 
"""

    def _load_pdf_images(self, pdf_path: str, doc_id: str) -> Dict[int, Image.Image]:
        if doc_id in self.pdf_images:
            logger.debug(f"Images for {doc_id} already in memory cache")
            return self.pdf_images[doc_id]

        logger.info(f"Loading PDF: {pdf_path}")
        try:
            # Check if PDF file exists
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return {}

            # Convert PDF to images
            images = convert_from_path(pdf_path)

            # Store in memory cache
            self.pdf_images[doc_id] = {
                i + 1: img for i, img in enumerate(images)
            }

            logger.info(f"Successfully loaded {len(images)} images for doc_id: {doc_id}")
            return self.pdf_images[doc_id]

        except Exception as e:
            logger.error(f"Failed to load images from {pdf_path}: {e}")
            return {}

    async def initialize_colpali(self, model_name: str = "vidore/colpali-v1.3"):
        try:
            logger.info(f"Loading ColPali model: {model_name}")

            self.colpali_model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else "cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()

            self.colpali_processor = ColPaliProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            logger.info("ColPali model loaded successfully")

            await self._setup_collection()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to initialize ColPali: {e}")
            if "out of memory" in str(e).lower():
                logger.info("GPU out of memory, falling back to CPU")
                self.device = "cpu"
                self.colpali_model = ColPali.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).eval()
                self.colpali_processor = ColPaliProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                await self._setup_collection()
            else:
                raise

    async def _setup_collection(self):
        try:
            if self.qdrant_client.collection_exists(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                return

            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
            )
            logger.info(f"Created collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise

    def _embed_images_colpali(self, images: List[Image.Image]):
        """Generate ColPali embeddings for images"""
        batch_size = 1 if self.device == "cpu" else min(2, len(images))
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                batch_features = self.colpali_processor.process_images(batch_images)
                batch_features = {k: v.to(self.colpali_model.device) for k, v in batch_features.items()}

                embeddings = self.colpali_model(**batch_features)

                # Convert to numpy and keep multi-vector structure
                embeddings_np = embeddings.cpu().float().numpy()

                # Each image gets a multi-vector embedding
                for emb in embeddings_np:
                    all_embeddings.append(emb)  # Shape: (n_patches, 128)

                del batch_features, embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return all_embeddings

    def _embed_query_colpali(self, query: str):
        """Generate ColPali embedding for query"""
        with torch.no_grad():
            batch_queries = self.colpali_processor.process_queries([query])
            batch_queries = {k: v.to(self.colpali_model.device) for k, v in batch_queries.items()}
            query_embedding = self.colpali_model(**batch_queries)
            # Return as numpy array, keep multi-vector structure
            return query_embedding.cpu().float().numpy()[0]

    def _image_to_base64(self, image: Image.Image, max_size: tuple = (1024, 1024)) -> str:
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image = image.copy()
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')

    def _build_familiarity_check_prompt(self, scale_path: str) -> str:

        with open(scale_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        scale_name = data['scale_info']['name']
        short_name = data['scale_info']['short_name']

        prompt = f"""You are a bilingual professor in the fields of education and psychology, with expertise in psychological measurement, scale development, and cross-linguistic test adaptation. 
You are familiar with standardized self-report instruments and their use in research and applied settings.
The name of the scale is: {scale_name} ({short_name}).
Before performing the translation, Please provide a best-effort familiarity check for this scale. The familiarity check should include:
    1. Whether the scale seems recognizable and your familiarity level (high/moderate/low/none)
    2. What the scale measures and how it is typically used
    3. Whether you know of any publicly available target-language translations or sample translated items
"""
        return prompt

    async def _check_with_openai(self, scale_path: str, model: str="gpt-5.2-2025-12-11") -> Dict:
        prompt_text = self._build_familiarity_check_prompt(scale_path)
        user_input = [{"role": "user", "content": prompt_text}]
        response = await self.openai_client.responses.create(
            model=model,
            input=user_input,
            max_output_tokens=8000,
        )
        summary = (getattr(response, "output_text", None)
                   or response.output[0].content[0].text).strip()
        return {"success": True, "summary": summary}

    async def _check_with_claude(self, scale_path: str, model: str="claude-opus-4-5-20251101") -> Dict:
        prompt_text = self._build_familiarity_check_prompt(scale_path)
        content_parts = []
        content_parts.append({"type": "text","text": prompt_text})
        response = await self.claude_client.messages.create(
            model=model,
            max_tokens=8000,
            messages=[{"role": "user", "content": content_parts}],
        )
        summary = response.content[0].text.strip()
        return {"success": True, "summary": summary}

    async def _check_with_gemini(self, scale_path: str, model: str="gemini-3-pro-preview") -> Dict:
        prompt_text = self._build_familiarity_check_prompt(scale_path)
        content_parts = [prompt_text]
        response = await asyncio.to_thread(
            self.gemini_client.models.generate_content,
            model=model,
            contents=content_parts,
            config={
                "max_output_tokens": 8000
            }
        )
        summary = response.text.strip()
        return {"success": True, "summary": summary}

    async def _check_with_qwen(self, scale_path: str, model: str="qwen3.5-plus") -> Dict:
        prompt_text = self._build_familiarity_check_prompt(scale_path)
        content_parts = []
        content_parts.append({"type": "text","text": prompt_text})
        response = await self.deepseek_client.chat.completions.create(
            model=model,
            max_tokens=8000,
            messages=[{"role": "user", "content": content_parts}],
        )
        summary = response.choices[0].message.content.strip()
        return {"success": True, "summary": summary}

    async def _check_with_model(self, scale_path, model):
        if model.startswith("gpt") or model.startswith("o1"):
            return await self._check_with_openai(scale_path, model)
        elif model.startswith("claude"):
            return await self._check_with_claude(scale_path, model)
        elif model.startswith("gemini"):
            return await self._check_with_gemini(scale_path, model)
        elif model.startswith("qwen"):
            return await self._check_with_qwen(scale_path, model)
        else:
            raise ValueError(f"Unsupported model: {model}")

    async def _extract_with_openai(self, images: List[Image.Image], scale_path: str,
                                   model="gpt-5.2-2025-12-11", temperature=0.7) -> Dict:
        try:
            content_parts = []
            prompt_text = self._build_prompt(scale_path)

            # Query text
            content_parts.append({
                "type": "input_text",
                "text": prompt_text
            })

            # Add each page as input_image
            for img in images:
                b64 = self._image_to_base64(img)
                content_parts.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64}",
                })

            user_input = [{"role": "user", "content": content_parts}]

            response = await self.openai_client.responses.create(
                model=model,
                input=user_input,
                max_output_tokens=10000,
                temperature=temperature
            )

            summary = (getattr(response, "output_text", None)
                       or response.output[0].content[0].text).strip()

            return {"success": True, "summary": summary, "raw_response": summary}

        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            return {"success": False, "error": f"OpenAI extraction failed: {e}"}

    async def _extract_with_claude(self, images: List[Image.Image], scale_path: str,
                                   model="claude-opus-4-5-20251101", temperature=0.7) -> Dict:
        try:
            content_parts = []
            prompt_text = self._build_prompt(scale_path)

            # Query text
            content_parts.append({
                "type": "text",
                "text": prompt_text
            })

            # Add each page as input_image
            for img in images:
                b64 = self._image_to_base64(img)
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64
                    }
                })

            response = await self.claude_client.messages.create(
                model=model,
                max_tokens=10000,
                messages=[{"role": "user", "content": content_parts}],
                temperature=temperature
            )

            summary = response.content[0].text.strip()

            return {"success": True, "summary": summary, "raw_response": summary}

        except Exception as e:
            logger.error(f"Claude extraction failed: {e}")
            return {"success": False, "error": f"Claude extraction failed: {e}"}

    async def _extract_with_gemini(self, images: List[Image.Image], scale_path: str,
                                   model="gemini-3-pro-preview", temperature=0.7) -> Dict:
        try:
            prompt_text = self._build_prompt(scale_path)

            # Query text
            content_parts = [prompt_text]

            # Add each page as input_image
            for img in images:
                b64 = self._image_to_base64(img)
                content_parts.append(img)

            response = await asyncio.to_thread(
                self.gemini_client.models.generate_content,
                model=model,
                contents=content_parts,
                config={
                    "temperature": temperature,
                    "max_output_tokens": 10000
                }
            )

            summary = response.text.strip()

            return {"success": True, "summary": summary, "raw_response": summary}

        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            return {"success": False, "error": f"Gemini extraction failed: {e}"}

    async def _extract_with_qwen(self, images: List[Image.Image], scale_path: str,
                                 model="qwen3.5-plus", temperature=0.7) -> Dict:
        try:
            content_parts = []
            prompt_text = self._build_prompt(scale_path)

            # Query text
            content_parts.append({
                "type": "text",
                "text": prompt_text
            })

            # Add each page as input_image
            for img in images:
                b64 = self._image_to_base64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }
                })

            response = await self.deepseek_client.chat.completions.create(
                model=model,
                max_tokens=10000,
                messages=[{"role": "user", "content": content_parts}],
                temperature=temperature
            )

            summary = response.choices[0].message.content.strip()

            return {"success": True, "summary": summary, "raw_response": summary}

        except Exception as e:
            logger.error(f"Qwen extraction failed: {e}")
            return {"success": False, "error": f"Qwen extraction failed: {e}"}

    # ====================== Model Router ======================
    async def _extract_with_model(self, images, scale_path,
                              model="gpt-5.2-2025-12-11", temperature=0.7):
        if model.startswith("gpt") or model.startswith("o1"):
            return await self._extract_with_openai(images, scale_path, model, temperature)
        elif model.startswith("claude"):
            return await self._extract_with_claude(images, scale_path, model, temperature)
        elif model.startswith("gemini"):
            return await self._extract_with_gemini(images, scale_path, model, temperature)
        elif model.startswith("qwen"):
            return await self._extract_with_qwen(images, scale_path, model, temperature)
        else:
            raise ValueError(f"Unsupported model: {model}")

    async def index_pdf(self, pdf_path: str, doc_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        try:
            if doc_id is None:
                doc_id = Path(pdf_path).stem

            logger.info(f"Converting PDF to images: {pdf_path}")
            images = convert_from_path(pdf_path)

            self.pdf_images[doc_id] = {
                i + 1: img for i, img in enumerate(images)
            }

            logger.info("Generating ColPali embeddings...")
            embeddings = self._embed_images_colpali(images)

            points = []
            for i, (embedding, image) in enumerate(zip(embeddings, images)):
                payload = {
                    "pdf_path": pdf_path,
                    "doc_id": doc_id,
                    "page_number": i + 1,
                    "total_pages": len(images),
                    **(metadata or {})
                }

                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=payload
                ))

            logger.info("Uploading to Qdrant")
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Successfully indexed {len(images)} pages from {pdf_path}")
            return f"Indexed {len(images)} pages from {Path(pdf_path).name} (doc_id: {doc_id})"

        except Exception as e:
            logger.error(f"Failed to index PDF: {e}")
            raise

    async def check_familiarity(self, scale_path: str, model: str, session_id: str = None) -> Dict:
        if session_id is None:
            session_id = str(uuid.uuid4())
        try:
            familiarity_result = await self._check_with_model(scale_path, model)
            if familiarity_result.get("success") and "summary" in familiarity_result:
                summary = familiarity_result["summary"]
                raw_response = familiarity_result.get('summary', '').strip()
                logger.info("Summary extracted successfully")

                # Mark this session as having completed Stage 1 - Familiarity Check
                self.session_state[session_id] = {
                    "familiarity_checked": True,
                    "familiarity_result": familiarity_result,
                    "scale_path": scale_path,
                    "model": model,
                }
                return {
                    "success": True,
                    "session_id": session_id,
                    "familiarity_summary": familiarity_result["summary"],
                    "message": "Familiarity check complete."
                }
            else:
                return {
                    "success": False,
                    "session_id": session_id,
                    "error": familiarity_result.get("error", "Familiarity check returned no summary")
                }
        except Exception as e:
            logger.error(f"Familiarity check failed: {e}")
            return {"success": False, "session_id": session_id, "error": str(e)}

    async def search_file(self, scale_path: str, session_id: str = None, limit: int = 30, extract_from_top: int = 8,
                          temperature: float = 0.7, model: str = "gpt-5.2-2025-12-11") -> Dict:
        """Search for relevant pages using ColPali"""
        try:
            if session_id is None:
                session_id = str(uuid.uuid4())
            # Two-stage workflow: check whether the familiarity check has been done
            session = self.session_state.get(session_id)
            if not session or not session.get("familiarity_checked"):
                return {
                    "error": "Familiarity check not completed. Please check familiarity with LLM first.",
                    "session_id": session_id
                }
            logger.info(f"Stage 2 translation: Session: {session_id}")

            query = self._build_prompt(scale_path)
            query_embedding = self._embed_query_colpali(query)

            # Use the multi-vector query directly
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),  # Multi-vector query
                limit=limit,
                with_payload=True
            )

            if not search_result:
                return {"error": "No relevant pages found", "session_id": session_id}

            top_results = search_result[:extract_from_top]
            grouped_images = []
            page_info = []

            for result in top_results:
                doc_id = result.payload.get("doc_id")
                page_num = result.payload.get("page_number")
                pdf_path = result.payload.get("pdf_path")

                # Load images if not in cache
                if doc_id not in self.pdf_images:
                    logger.info(f"Images not in cache for {doc_id}, loading from {pdf_path}")
                    doc_images = self._load_pdf_images(pdf_path, doc_id)
                else:
                    doc_images = self.pdf_images[doc_id]

                # Get the specific page image
                if page_num in doc_images:
                    grouped_images.append(doc_images[page_num])
                    page_info.append({
                        "doc_id": doc_id,
                        "page_number": page_num,
                        "score": result.score,
                        "pdf_path": pdf_path
                    })
                else:
                    logger.warning(f"Page {page_num} not found for doc_id={doc_id}.")

            if not grouped_images:
                return {
                    "error": "Images not available for extraction",
                    "search_results": [
                        {
                            "score": r.score,
                            "page_number": r.payload.get("page_number"),
                            "doc_id": r.payload.get("doc_id"),
                            "pdf_path": r.payload.get("pdf_path")
                        } for r in search_result
                    ],
                    "session_id": session_id
                }

            logger.info(f"Extracting information with OpenAI from {len(grouped_images)} pages")

            openai_result = await self._extract_with_model(grouped_images, scale_path, model=model, temperature=temperature)
            if openai_result.get("success") and "summary" in openai_result:
                summary = openai_result["summary"]
                raw_response = openai_result.get('summary', '').strip()
                logger.info("Summary extracted successfully")

                # parse json
                llm_response = self._extract_json_translation(raw_response)

                return {
                    "summary": summary,
                    "translation": llm_response,
                    "search_results": page_info,
                    "session_id": session_id
                }
            else:
                logger.warning("Failed to extract summary")
                return {
                    "error": "Failed to extract summary",
                    "openai_success": openai_result.get("success", False),
                    "openai_error": openai_result.get("error", "Unknown error"),
                    "search_results": page_info,
                    "total_pages_searched": len(search_result),
                    "session_id": session_id
                }

        except Exception as e:
            logger.error(f"Failed in search and extract: {e}")
            return {"error": str(e), "session_id": session_id or str(uuid.uuid4())}

    def _extract_json_translation(self, text: str) -> str:
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag) and closing fence
            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)
            cleaned = cleaned.strip()

        # Try direct parse first
        try:
            parsed = json.loads(cleaned)
            if 'items' in parsed and isinstance(parsed['items'], list):
                return json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass

        # Fall back: find the outermost { ... } with balanced braces
        start = cleaned.find('{')
        if start == -1:
            return text

        depth = 0
        end = start
        for i in range(start, len(cleaned)):
            if cleaned[i] == '{':
                depth += 1
            elif cleaned[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        try:
            json_text = cleaned[start:end]
            parsed = json.loads(json_text)
            if 'items' in parsed and isinstance(parsed['items'], list):
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            elif 'translation' in parsed:
                return parsed['translation']
        except json.JSONDecodeError:
            pass

        # If no valid JSON found, return original text
        return text

    def clear_session(self, session_id: str) -> bool:
        """Clear session state (familiarity check tracking)."""
        if session_id in self.session_state:
            del self.session_state[session_id]
            return True
        return False


rag_server = None


# --------------------------------- Endpoint ---------------------------------
@click.command()
@click.option("--use-memory", is_flag=True, help="Use in-memory Qdrant")
@click.option("--port", default=8000)
@click.option("--host", default="127.0.0.1")
def main(use_memory: bool, port: int, host: str) -> int:
    global rag_server

    app = Server("rag")
    rag_server = Rag_server(use_memory=use_memory)

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="index_pdf",
                description="Index a PDF using ColPali",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pdf_path": {"type": "string", "description": "Path to the PDF file"},
                    },
                    "required": ["pdf_path"]
                }
            ),
            types.Tool(
                name="search_file",
                description="Use ColPali to find relevant pages",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "extract_page": {"type": "integer", "description": "Number of top results", "default": 3}
                    },
                    "required": ["query"]
                }
            )
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            if name == "index_pdf":
                if rag_server.colpali_model is None:
                    await rag_server.initialize_colpali()
                pdf_path = arguments["pdf_path"]
                result = await rag_server.index_pdf(pdf_path)
                return [types.TextContent(type="text", text=result)]

            elif name == "search_file":
                if rag_server.colpali_model is None:
                    await rag_server.initialize_colpali()
                query = arguments["query"]
                limit = arguments.get("limit", 30)
                extract_from_top = arguments.get("extract_from_top", 8)
                temperature = arguments.get("temperature", 0.7)
                model = arguments.get("model", "gpt-5.2-2025-12-11")
                result = await rag_server.search_file(query, limit, extract_from_top, temperature, model)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    from starlette.requests import Request
    from starlette.responses import JSONResponse

    async def health(request: Request):
        return JSONResponse({"ok": True})

    async def upload_endpoint(request: Request):
        # multipart/form-data with "file"
        form = await request.form()
        up = form.get("file")
        if up is None:
            return JSONResponse({"success": False, "error": "Missing file field 'file'."}, status_code=400)
        if not (up.filename or "").lower().endswith(".pdf"):
            return JSONResponse({"success": False, "error": "Only PDF files are supported."}, status_code=400)

        # Save to a stable path under GUIDELINES_DIR so it survives process restarts —
        # /translate re-reads this path from Qdrant payload via _load_pdf_images
        # whenever the in-memory pdf_images cache is cold (e.g. after a restart).
        safe_name = f"{uuid.uuid4()}_{Path(up.filename).name}"
        dest_path = str(GUIDELINES_DIR / safe_name)
        try:
            data = await up.read()
            with open(dest_path, "wb") as f:
                f.write(data)

            # Ensure ColPali is ready
            if rag_server.colpali_model is None:
                await rag_server.initialize_colpali()

            # Index with the existing pipeline
            result_msg = await rag_server.index_pdf(dest_path)
            return JSONResponse({"success": True, "filename": up.filename, "message": result_msg})
        except Exception as e:
            try:
                os.remove(dest_path)
            except Exception:
                pass
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    async def check_endpoint(request: Request):
        """Stage 1: Familiarity check. Returns LLM's familiarity with the scale."""
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"success": False, "error": "Invalid JSON body."}, status_code=400)

        scale_path = (payload.get("scale_path") or "").strip()
        if not scale_path:
            return JSONResponse({"success": False, "error": "Field 'scale_path' is required."}, status_code=400)

        model = payload.get("model", "gpt-5.2-2025-12-11")
        session_id = payload.get("session_id")  # optional, will be generated if not provided

        try:
            result = await rag_server.check_familiarity(scale_path, model, session_id=session_id)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    async def translate_endpoint(request: Request):
        """Stage 2: Translation with RAG. Requires a session_id that has passed Stage 1."""
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"success": False, "error": "Invalid JSON body."}, status_code=400)

        session_id = (payload.get("session_id") or "").strip()
        if not session_id:
            return JSONResponse({"success": False, "error": "Field 'session_id' is required. Complete /check first."},
                                status_code=400)

        scale_path = (payload.get("scale_path") or "").strip()
        if not scale_path:
            return JSONResponse({"success": False, "error": "Field 'scale_path' is required."}, status_code=400)

        limit = int(payload.get("limit", 30))
        extract_from_top = int(payload.get("extract_from_top", 8))
        model = payload.get("model", "gpt-5.2-2025-12-11")
        temperature = payload.get("temperature", 0.7)

        try:
            if rag_server.colpali_model is None:
                await rag_server.initialize_colpali()

            result = await rag_server.search_file(
                scale_path, session_id=session_id, limit=limit,
                extract_from_top=extract_from_top, temperature=temperature, model=model
            )
            if isinstance(result, dict):
                return JSONResponse({"success": True, **result})
            else:
                return JSONResponse({"success": True, "summary": str(result)})
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    async def ask_endpoint(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"success": False, "error": "Invalid JSON body."}, status_code=400)

        query = (payload.get("query") or "").strip()
        if not query:
            return JSONResponse({"success": False, "error": "Field 'query' is required."}, status_code=400)

        session_id = payload.get("session_id")
        limit = int(payload.get("limit", 30))
        extract_from_top = int(payload.get("extract_from_top", 8))
        model = payload.get("model", "gpt-5.2-2025-12-11")
        temperature = payload.get("temperature", 0.7)

        try:
            if rag_server.colpali_model is None:
                await rag_server.initialize_colpali()

            # Use your existing search method
            result = await rag_server.search_file(query, session_id=session_id, limit=limit,
                                                  extract_from_top=extract_from_top, temperature=temperature,
                                                  model=model)
            # result is already a dict in your code; make sure we wrap it consistently
            if isinstance(result, dict):
                return JSONResponse({"success": True, **result})
            else:
                return JSONResponse({"success": True, "summary": str(result)})
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    # Run server
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Mount, Route
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await app.run(streams[0], streams[1], app.create_initialization_options())
        return Response()

    starlette_app = Starlette(
        debug=True,
        routes=[
            # REST endpoints for GUI
            Route("/", endpoint=health, methods=["GET"]),
            Route("/health", endpoint=health, methods=["GET"]),
            Route("/upload", endpoint=upload_endpoint, methods=["POST"]),
            Route("/check", endpoint=check_endpoint, methods=["POST"]),  # Stage 1: familiarity check
            Route("/translate", endpoint=translate_endpoint, methods=["POST"]),  # Stage 2: RAG translation
            Route("/ask", endpoint=ask_endpoint, methods=["POST"]),  # Legacy combined endpoint

            # MCP endpoints (keep these)
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )
    uvicorn.run(starlette_app, host=host, port=port)
    return 0


if __name__ == "__main__":
    main()