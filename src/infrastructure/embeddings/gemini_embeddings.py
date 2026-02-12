"""
Embedding Service — Gemini Embedding API.

Converts case summaries into vector embeddings for RAG.
Compatible with GCP Vertex AI in production.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GeminiEmbeddingService:
    """Generate embeddings using Google Gemini API."""

    # Gemini embedding model
    MODEL = "models/gemini-embedding-001"
    DIMENSION = 768  # default output dimension

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def embed_text(self, text: str) -> Optional[list[float]]:
        """Generate embedding for a single text."""
        try:
            client = self._get_client()
            result = client.models.embed_content(
                model=self.MODEL,
                contents=text,
            )
            if result and result.embeddings:
                vector = list(result.embeddings[0].values)
                logger.debug(f"Generated embedding: dim={len(vector)}")
                return vector
            return None
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Generate embeddings for multiple texts."""
        results = []
        for text in texts:
            vec = self.embed_text(text)
            results.append(vec)
        return results
