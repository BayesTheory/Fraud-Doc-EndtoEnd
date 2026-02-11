"""
Use Case: Search Similar Cases

Busca documentos/casos similares por embedding vetorial.
"""

from src.core.interfaces.embedding_service import IEmbeddingService, SearchResult
from src.core.interfaces.fraud_classifier import IFraudClassifier


class SearchSimilarUseCase:
    """
    Use Case: dado um case_id ou imagem, busca casos similares.
    """

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        fraud_classifier: IFraudClassifier,
    ):
        self._embeddings = embedding_service
        self._fraud = fraud_classifier

    def by_case_id(self, case_id: str, top_k: int = 5) -> SearchResult:
        """Busca similares a partir de um caso já processado."""
        # TODO: recuperar embedding do caso e buscar
        raise NotImplementedError

    def by_image(self, image_bytes: bytes, top_k: int = 5) -> SearchResult:
        """Busca similares a partir de uma nova imagem."""
        # TODO: gerar embedding e buscar
        raise NotImplementedError
