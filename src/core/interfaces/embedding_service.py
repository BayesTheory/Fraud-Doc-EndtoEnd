"""
Contract: Embedding Service

Armazena e busca embeddings vetoriais para encontrar
casos/documentos similares via nearest-neighbor search.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SimilarCase:
    """Um caso similar encontrado via busca vetorial."""
    case_id: str
    distance: float
    fraud_score: float
    fraud_label: str
    doc_type: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """Resultado de uma busca por similaridade."""
    query_case_id: str | None
    similar_cases: list[SimilarCase]
    search_time_ms: float = 0.0


class IEmbeddingService(ABC):
    """
    Port: Embedding Service (Vector Store)

    Armazena embeddings e permite busca por similaridade.
    Implementação pode ser pgvector, Qdrant, FAISS, etc.
    """

    @abstractmethod
    def store(self, case_id: str, embedding: list[float], metadata: dict) -> None:
        """
        Armazena um embedding com metadados.

        Args:
            case_id: ID do caso.
            embedding: Vetor numérico.
            metadata: Metadados extras (fraud_score, doc_type, etc.).
        """
        ...

    @abstractmethod
    def search(self, embedding: list[float], top_k: int = 5, filters: dict | None = None) -> SearchResult:
        """
        Busca os top_k casos mais similares.

        Args:
            embedding: Vetor de consulta.
            top_k: Quantidade de resultados.
            filters: Filtros opcionais (ex: doc_type, fraud_label).

        Returns:
            SearchResult com lista de casos similares.
        """
        ...
