"""
Application Settings.

Centraliza toda configuração via .env / variáveis de ambiente.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configurações carregadas de variáveis de ambiente."""

    # --- App ---
    env: str = "development"
    debug: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Quality Gate ---
    blur_threshold: float = 100.0
    brightness_min: int = 50
    brightness_max: int = 220
    min_resolution: int = 640
    min_doc_area_ratio: float = 0.70

    # --- OCR ---
    ocr_lang: str = "pt"
    ocr_use_gpu: bool = False
    ocr_min_confidence: float = 0.5

    # --- Fraud ---
    fraud_model_path: str = "models/weights/efficientnet_b0_fraud.pt"
    fraud_threshold: float = 0.5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Singleton de settings."""
    return Settings()
