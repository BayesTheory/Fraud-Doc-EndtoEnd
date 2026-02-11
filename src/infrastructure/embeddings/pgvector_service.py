"""
Adapter: pgvector Embedding Service

Implementação concreta do contrato IEmbeddingService
usando Postgres + extensão pgvector para busca vetorial.
"""

from src.core.interfaces.embedding_service import IEmbeddingService, SearchResult


class PgVectorService(IEmbeddingService):
    """
    Store e busca de embeddings usando pgvector no Postgres.

    Usa índice HNSW para busca aproximada eficiente.
    Metadados (fraud_score, doc_type, flags) armazenados
    como colunas para filtragem SQL nativa.
    """

    def __init__(self, connection_string: str, embedding_dim: int = 1280):
        self._connection_string = connection_string
        self._embedding_dim = embedding_dim

    def store(self, case_id: str, embedding: list[float], metadata: dict) -> None:
        # TODO: INSERT com pgvector
        raise NotImplementedError

    def search(self, embedding: list[float], top_k: int = 5, filters: dict | None = None) -> SearchResult:
        # TODO: KNN search com pgvector (<-> operator)
        raise NotImplementedError
