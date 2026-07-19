"""Lightweight REST client for the forward/backward RAG servers.

Deliberately standalone (no import of client_bidirectional.py / embed_EGA.py,
which pull in rpy2 and other heavy deps not needed to talk to the servers).
"""

import requests


class RAGClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id = None

    def health(self, timeout: float = 5.0) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=timeout)
            return resp.ok
        except requests.RequestException:
            return False

    def upload_guideline(self, filename: str, file_bytes: bytes) -> dict:
        files = {"file": (filename, file_bytes, "application/pdf")}
        response = requests.post(f"{self.base_url}/upload", files=files, timeout=300)
        return response.json()

    def check_familiarity(self, scale_path: str, model: str) -> dict:
        payload = {"scale_path": scale_path, "model": model}
        response = requests.post(f"{self.base_url}/check", json=payload, timeout=180)
        result = response.json()
        if result.get("session_id"):
            self.session_id = result["session_id"]
        return result

    def translate(self, scale_path: str, model: str, temperature: float = 0.7,
                  limit: int = 30, extract_from_top: int = 5) -> dict:
        if not self.session_id:
            return {"success": False, "error": "No session_id. Run check_familiarity first."}

        payload = {
            "session_id": self.session_id,
            "scale_path": scale_path,
            "model": model,
            "temperature": temperature,
            "limit": limit,
            "extract_from_top": extract_from_top,
        }
        response = requests.post(f"{self.base_url}/translate", json=payload, timeout=300)
        result = response.json()
        if result.get("session_id"):
            self.session_id = result["session_id"]
        return result

    def back_translate(self, scale_path: str, model: str, temperature: float = 0.7,
                        limit: int = 30, extract_from_top: int = 5) -> dict:
        """Backward server has no /check gate — goes straight to /translate."""
        payload = {
            "scale_path": scale_path,
            "model": model,
            "temperature": temperature,
            "limit": limit,
            "extract_from_top": extract_from_top,
        }
        response = requests.post(f"{self.base_url}/translate", json=payload, timeout=300)
        return response.json()
