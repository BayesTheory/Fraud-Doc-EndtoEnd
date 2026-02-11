"""
Contract: Storage Service

Gerencia upload/download de artefatos (imagens, JSONs)
em object storage (MinIO/S3/local filesystem).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StorageRef:
    """Referência a um arquivo armazenado."""
    bucket: str
    key: str
    size_bytes: int
    sha256: str
    content_type: str


class IStorageService(ABC):
    """
    Port: Storage Service

    Gerencia persistência de artefatos binários.
    Implementação pode ser MinIO, S3, filesystem local, etc.
    """

    @abstractmethod
    def upload(self, data: bytes, key: str, content_type: str = "image/jpeg") -> StorageRef:
        """
        Faz upload de um arquivo.

        Args:
            data: Conteúdo em bytes.
            key: Caminho/chave no storage.
            content_type: MIME type.

        Returns:
            StorageRef com localização e hash.
        """
        ...

    @abstractmethod
    def download(self, key: str) -> bytes:
        """
        Baixa um arquivo do storage.

        Args:
            key: Caminho/chave no storage.

        Returns:
            Conteúdo em bytes.
        """
        ...

    @abstractmethod
    def get_url(self, key: str, expires_seconds: int = 3600) -> str:
        """
        Gera URL assinada para acesso temporário.

        Args:
            key: Caminho/chave no storage.
            expires_seconds: Tempo de expiração.

        Returns:
            URL assinada.
        """
        ...
