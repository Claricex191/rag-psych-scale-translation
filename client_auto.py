import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoTranslateClient:
    def __init__(self, server_url: str = "http://127.0.0.1:8000"):
        self.server_url = server_url.rstrip('/')
        self.session_id = None

    def check_server_health(self) -> bool:
        """Check if the server is running and healthy"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Server is healthy and ready")
                return True
            else:
                logger.error(f"Server returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to server at {self.server_url}: {e}")
            logger.error("Please ensure the server is running with: python server_v1208.py")
            return False

    def load_scales_from_json(self, input_path: str) -> List[Dict]:
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON formats
            if isinstance(data, dict) and 'items' in data:
                items = data['items']
            elif isinstance(data, list):
                items = data
            else:
                raise ValueError("JSON must contain 'items' array or be an array itself")

            # Validate items
            for i, item in enumerate(items):
                if 'original' not in item:
                    raise ValueError(f"Item {i} missing 'original' field")
                if 'number' not in item:
                    item['number'] = str(i + 1)  # Auto-assign number

            logger.info(f"✓ Loaded {len(items)} items from {input_path}")
            return items

        except FileNotFoundError:
            logger.error(f"File not found: {input_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {input_path}: {e}")
            raise

    def prepare_translation_query(self, items: List[Dict]) -> str:
        query_parts = []
        for item in items:
            query_parts.append(f"{item['number']}. {item['original']}")

        return "\n".join(query_parts)

    def translate_items(
            self,
            items: List[Dict],
            session_id: Optional[str] = None,
            batch_size: int = 10
    ) -> List[Dict]:
        all_translations = []
        total_items = len(items)

        # Process in batches
        for i in range(0, total_items, batch_size):
            batch = items[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_items + batch_size - 1) // batch_size

            logger.info(f"Translating batch {batch_num}/{total_batches} ({len(batch)} items)...")

            # Prepare query
            query = self.prepare_translation_query(batch)

            # Send to server
            payload = {
                "query": query,
                "limit": 30,
                "extract_from_top": 8
            }

            if session_id:
                payload["session_id"] = session_id

            try:
                response = requests.post(
                    f"{self.server_url}/ask",
                    json=payload,
                    timeout=120  # Allow up to 2 minutes for translation
                )
                response.raise_for_status()

                result = response.json()

                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Server error: {error_msg}")
                    raise RuntimeError(f"Translation failed: {error_msg}")

                # Extract translations from response
                summary = result.get("summary", "")

                # Parse JSON from summary
                translations = self._extract_translations(summary, batch)
                all_translations.extend(translations)

                logger.info(f"✓Batch {batch_num} completed ({len(translations)} items)")

                # Small delay between batches to avoid overwhelming the server
                if i + batch_size < total_items:
                    time.sleep(1)

            except requests.exceptions.Timeout:
                logger.error(f"Request timeout for batch {batch_num}")
                raise
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for batch {batch_num}: {e}")
                raise

        logger.info(f"Translation complete: {len(all_translations)} items translated")
        return all_translations

    def _extract_translations(self, summary: str, original_items: List[Dict]) -> List[Dict]:
        try:
            # Remove markdown code blocks if present
            json_text = summary
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()

            data = json.loads(json_text)

            if isinstance(data, dict) and 'items' in data:
                return data['items']
            elif isinstance(data, list):
                return data
            else:
                logger.warning("Unexpected JSON structure, using fallback")

        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from response, using text extraction")

        # Fallback: try to extract translations from text
        results = []
        for item in original_items:
            results.append({
                "number": item["number"],
                "original": item["original"],
                "translation": f"[Translation failed - check server response]"
            })

        return results

    def save_translations(self, translations: List[Dict], output_path: str):
        output_data = {
            "items": translations
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Saved {len(translations)} translated items to {output_path}")

    def process_file(
            self,
            input_path: str,
            output_path: Optional[str] = None,
            session_id: Optional[str] = None,
            batch_size: int = 10
    ) -> str:
        if not output_path:
            input_stem = Path(input_path).stem
            output_path = f"{input_stem}_translated.json"

        logger.info("=" * 60)
        logger.info("Starting automated translation workflow")
        logger.info("=" * 60)

        # Step 1: Check server
        if not self.check_server_health():
            raise RuntimeError("Server is not available")

        # Step 2: Load items
        items = self.load_scales_from_json(input_path)

        # Step 3: Translate
        translations = self.translate_items(items, session_id, batch_size)

        # Step 4: Save
        self.save_translations(translations, output_path)

        logger.info("=" * 60)
        logger.info(f"✓ Workflow complete! Output: {output_path}")
        logger.info("=" * 60)

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Automatically translate psychometric scales using RAG server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python auto_translate_client.py scales.json

  # Specify output file
  python auto_translate_client.py scales.json --output translated.json

  # Use custom server URL and batch size
  python auto_translate_client.py scales.json --server http://localhost:8000 --batch-size 5

  # Use session ID for conversation continuity
  python auto_translate_client.py scales.json --session-id my-session-001
        """
    )

    parser.add_argument(
        'input_file',
        help='Input JSON file containing scale items'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output JSON file path (default: {input}_translated.json)'
    )

    parser.add_argument(
        '--server',
        default='http://127.0.0.1:8000',
        help='RAG server URL (default: http://127.0.0.1:8000)'
    )

    parser.add_argument(
        '--session-id',
        help='Session ID for conversation continuity'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of items to translate per batch (default: 10)'
    )

    args = parser.parse_args()

    # Create client and process
    try:
        client = AutoTranslateClient(server_url=args.server)
        output_file = client.process_file(
            input_path=args.input_file,
            output_path=args.output,
            session_id=args.session_id,
            batch_size=args.batch_size
        )

        print(f"\n✓ Success! Translated file saved to: {output_file}")
        return 0

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())