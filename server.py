import os
from lib2to3.fixes.fix_input import context

from requests import session

# from qdrant_client.local.distances import DiscoveryQuery
# from streamlit import image
# from torchvision import message

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#!/usr/bin/env python3
import base64
import io
import json
import logging
import uuid
from pathlib import Path
from typing import Any, List, Optional, Dict

import anthropic
from openai import AsyncOpenAI
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

# Add long-term & short-term memory
import json
from collections import defaultdict
from datetime import datetime


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------- RAG Server -------------------------------
class Rag_server:
    def __init__(self, use_memory: bool = False, anthropic_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        # Qdrant setup
        if use_memory:
            self.qdrant_client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant client")
        else:
            try:
                self.qdrant_client = QdrantClient(path="/scratch/claricex/qdrant-db")
                logger.info("Using embedded Qdrant at /scratch/claricex/qdrant-db")
            except Exception as e:
                logger.error(f"Failed to connect to local Qdrant: {e}")
                raise
        self.qdrant_client.set_model("sentence-transformers/paraphrase-multilingual-mpnet-base-v2") # Supports multiple languages like English and Chinese

        self.colpali_model = None
        self.colpali_processor = None
        self.collection_name = "oise_mt"

        # Add memory dictionary
        self.conversation_history = defaultdict(list)
        self.max_memory_turns = 10 # At most 10 turns of dialogues

        # Claude Vision setup
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        # OpenAI Vision setup
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

        if not self.anthropic_api_key and not self.openai_api_key:
            logger.error("No Anthropic or OpenAI API key found!")
            raise ValueError("ANTHROPIC_API_KEY/OPENAI_API_KEY not found")

        self.claude_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        logger.info("Claude client initialized successfully")

        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        logger.info("OpenAI client initialized successfully")

        self.pdf_images = {}

        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.set_per_process_memory_fraction(0.7)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.warning("GPU not available, using CPU")

    # ============================ Memory Setup =============================
    def add_to_memory(self, session_id: str, query: str, response: str, search_results: List[Dict] = None):
        # Add conversation turn to memory
        turn = {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "search_results": search_results or []
        }

        self.conversation_history[session_id].append(turn)

        # Keep only recent turns
        if len(self.conversation_history[session_id]) > self.max_memory_turns:
            self.conversation_history[session_id] = self.conversation_history[session_id][-self.max_memory_turns:]

    def get_conversation_context(self, session_id: str, max_turns: int = 5) -> List[Dict]:
        # Get recent conversation turns for a session
        if session_id not in self.conversation_history:
            return []
        return self.conversation_history[session_id][-max_turns:]

    def _build_chat_history_prompt(self, current_query: str, chat_history: List[Dict]) -> str:
        """Build enhanced prompt with conversation context"""
        if not chat_history:
            return current_query

        history_text = "Previous conversation:\n"
        for turn in chat_history:  # Use last 3 turns
            history_text += f"Q: {turn['query']}\n"
            # If the response is too long then should truncate long responses to shorter.
            # Now we provide whole chat history to the LLM (which includes the user query and the llm response)
            response = turn['response']
            history_text += f"A: {response}\n\n"

        return f"""{history_text}Current question: {current_query}
                Based on the PDF pages shown and the conversation history above, please answer the current query. Consider the previous context when relevant.
                Provide a comprehensive summary that covers the key information relevant to the query.
                
"""


    # ============================ End of Memory Setup ============================
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
            # if self.qdrant_client.collection_exists(self.collection_name):
            #     logger.info(f"Collection {self.collection_name} already exists")
            #     return

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

    async def _extract_with_openai(self, images: List[Image.Image], query: str, chat_history: List[Dict] = None) -> Dict:
        try:
            content_parts = []
            prompt_text = self._build_chat_history_prompt(query, chat_history or [])

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
                model="gpt-4o-2024-11-20",
                input=user_input,
                max_output_tokens=2500,
            )

            summary = (getattr(response, "output_text", None)
                       or response.output[0].content[0].text).strip()

            return {"success": True, "summary": summary, "raw_response": summary}

        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            return {"success": False, "error": f"OpenAI extraction failed: {e}"}

    async def _extract_with_claude(self, images: List[Image.Image], query: str, chat_history: List[Dict] = None) -> Dict:
        try:
            image_contents = []
            for i, img in enumerate(images):
                base64_image = self._image_to_base64(img)
                image_contents.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image
                    }
                })

            prompt_text = self._build_chat_history_prompt(query, chat_history or [])
            messages = [
                {
                    "role": "user",
                    "content": image_contents + [
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                messages=messages
            )

            response_text = response.content[0].text.strip()

            return {
                "success": True,
                "summary": response_text,
                "raw_response": response_text
            }

        except Exception as e:
            logger.error(f"Claude extraction failed: {e}")
            return {
                "success": False,
                "error": f"Claude extraction failed: {str(e)}"
            }


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

    async def search_file(self, query: str, session_id: str = None, limit: int = 15, extract_from_top: int = 3) -> Dict:
        """Search for relevant pages using ColPali"""
        try:
            if session_id is None:
                session_id = str(uuid.uuid4())
            logger.info(f"Search with memory: Session: {session_id}, Query: {query}")

            chat_history = self.get_conversation_context(session_id, max_turns=5)
            logger.info(f"Searching for: {query}")
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

            openai_result = await self._extract_with_openai(grouped_images, query, chat_history)
            if openai_result.get("success") and "summary" in openai_result:
                summary = openai_result["summary"]
                logger.info("Summary extracted successfully")
                # Store this chat history to memory:
                self.add_to_memory(
                    session_id=session_id,
                    query=query,
                    response=summary,
                    search_results=page_info
                )

                return {
                    "summary": summary,
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

    def get_session_summary(self, session_id: str) -> Dict:
        if session_id not in self.conversation_history:
            return {
                "exists": False,
                "turn_count": 0,
                "message": "No conversation history found"
            }
        turns = self.conversation_history[session_id]
        return {
            "exists": True,
            "session_id": session_id,
            "turn_count": len(turns),
            "first_query": turns[0]["query"] if turns else None
        }
    def clear_session_memory(self, session_id: str) -> bool:
        if session_id in self.conversation_history[session_id]:
            del self.conversation_history[session_id]
            return True
        return False

    async def extract_items_from_pdf(self, pdf_path: str) -> str:
        try:
            logger.info(f"Extracting items from PDF: {pdf_path}")
            images = convert_from_path(pdf_path)
            extracted_text = await self._extract_text_with_vision(images)

            logger.info(f"Successfully extracted from {pdf_path}")
            return extracted_text
        except Exception as e:
            logger.error(f"Failed to extract items from PDF: {e}")
            raise

    async def _extract_text_with_vision(self, images: List[Image.Image]) -> str:
        """
        Use vision model to extract text from PDF images.
        Works well for both regular PDFs and scanned documents.
        """
        try:
            content_parts = []

            # Add instruction
            content_parts.append({
                "type": "input_text",
                "text": """Extract all text content from these PDF pages. 
                These contain psychological assessment items that need to be translated.

                Please:
                1. Extract all text exactly as shown
                2. Preserve numbering and formatting
                3. Keep item text separate and clear
                4. Include any response scales or options

                Output only the extracted text, maintaining the original structure."""
            })

            # Add images (limit to reasonable number for API)
            for img in images[:20]:  # Limit to 20 pages to avoid token limits
                b64 = self._image_to_base64(img)
                content_parts.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64}",
                })

            user_input = [{"role": "user", "content": content_parts}]

            response = await self.openai_client.responses.create(
                model="gpt-4o-2024-11-20",
                input=user_input,
                max_output_tokens=4000,
            )

            extracted_text = (getattr(response, "output_text", None)
                              or response.output[0].content[0].text).strip()

            if len(images) > 20:
                extracted_text += f"\n\n[Note: Only first 20 pages extracted. Total pages: {len(images)}]"

            return extracted_text

        except Exception as e:
            logger.error(f"Vision-based text extraction failed: {e}")
            # Fallback to basic text extraction if vision fails
            return ""



rag_server = None

# --------------------------- Memory Checkpointer ---------------------------


# --------------------------------- Endpoint ---------------------------------
@click.command()
@click.option("--use-memory", is_flag=True, help="Use in-memory Qdrant")
@click.option("--port", default=8000)
@click.option("--host", default="127.0.0.1")
@click.option("--anthropic-api-key", help="Anthropic API key")



def main(use_memory: bool, port: int, host: str, anthropic_api_key: str) -> int:
    global rag_server

    app = Server("rag")
    rag_server = Rag_server(use_memory=use_memory, anthropic_api_key=anthropic_api_key)

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
                limit = arguments.get("limit", 15)
                extract_from_top = arguments.get("extract_from_top", 3)
                result = await rag_server.search_file(query, limit, extract_from_top)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    from starlette.requests import Request
    from starlette.responses import JSONResponse
    import tempfile

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

        # Save to a temp path
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        try:
            data = await up.read()
            with open(tmp_path, "wb") as f:
                f.write(data)

            # Ensure ColPali is ready
            if rag_server.colpali_model is None:
                await rag_server.initialize_colpali()

            # Index with your existing pipeline
            result_msg = await rag_server.index_pdf(tmp_path)
            return JSONResponse({"success": True, "filename": up.filename, "message": result_msg})
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)
        finally:
            # Remove temp file
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    async def ask_endpoint(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"success": False, "error": "Invalid JSON body."}, status_code=400)

        query = (payload.get("query") or "").strip()
        if not query:
            return JSONResponse({"success": False, "error": "Field 'query' is required."}, status_code=400)

        session_id = payload.get("session_id")
        limit = int(payload.get("limit", 15))
        extract_from_top = int(payload.get("extract_from_top", 3))

        try:
            if rag_server.colpali_model is None:
                await rag_server.initialize_colpali()

            # Use your existing search method
            result = await rag_server.search_file(query, session_id=session_id, limit=limit, extract_from_top=extract_from_top)
            # result is already a dict in your code; make sure we wrap it consistently
            if isinstance(result, dict):
                return JSONResponse({"success": True, **result})
            else:
                return JSONResponse({"success": True, "summary": str(result)})
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    async def upload_items_endpoint(request: Request):
        """
        Upload a PDF containing psychological items to be translated.
        Returns extracted text from the PDF that can be used as query.
        """
        form = await request.form()
        up = form.get("file")

        if up is None:
            return JSONResponse({"success": False, "error": "Missing file field 'file'."}, status_code=400)
        if not (up.filename or "").lower().endswith(".pdf"):
            return JSONResponse({"success": False, "error": "Only PDF files are supported."}, status_code=400)

        # Save to temp path
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)

        try:
            data = await up.read()
            with open(tmp_path, "wb") as f:
                f.write(data)

            # Extract text from the items PDF
            extracted_items = await rag_server.extract_items_from_pdf(tmp_path)

            return JSONResponse({
                "success": True,
                "filename": up.filename,
                "extracted_items": extracted_items,
                "message": f"Extracted {len(extracted_items)} characters from {up.filename}"
            })

        except Exception as e:
            logger.error(f"Failed to extract items from PDF: {e}")
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)
        finally:
            # Remove temp file
            try:
                os.remove(tmp_path)
            except Exception:
                pass

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
            Route("/upload_items", endpoint=upload_items_endpoint, methods=["POST"]),
            Route("/ask", endpoint=ask_endpoint, methods=["POST"]),

            # MCP endpoints (keep these)
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )
    uvicorn.run(starlette_app, host=host, port=port)
    return 0




if __name__ == "__main__":
    main()