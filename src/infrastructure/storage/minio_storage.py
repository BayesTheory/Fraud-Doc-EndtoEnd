"""
Adapter: MinIO Storage Service

Implementação concreta do contrato IStorageService
usando MinIO (compatível com API S3).
"""

from src.core.interfaces.storage_service import IStorageService, StorageRef


class MinIOStorageService(IStorageService):
    """
    Storage de artefatos usando MinIO.

    Em produção, trocar por S3 real sem mudar nenhum
    outro código — só muda as credenciais de conexão.
    """

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket

    def upload(self, data: bytes, key: str, content_type: str = "image/jpeg") -> StorageRef:
        # TODO: Implementar upload MinIO
        raise NotImplementedError

    def download(self, key: str) -> bytes:
        # TODO: Implementar download MinIO
        raise NotImplementedError

    def get_url(self, key: str, expires_seconds: int = 3600) -> str:
        # TODO: Gerar presigned URL
        raise NotImplementedError
