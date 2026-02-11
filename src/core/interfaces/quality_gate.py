"""
Contract: Quality Gate

Avalia se a imagem do documento tem qualidade suficiente
para ser processada pelo pipeline. Qualquer implementação
(OpenCV, modelo ML, serviço externo) deve respeitar este contrato.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class QualityResult:
    """Resultado da avaliação de qualidade."""
    quality_ok: bool
    quality_score: float          # 0.0 (péssimo) a 1.0 (perfeito)
    reasons: list[str]            # ex: ["BLUR_HIGH", "CROP_PARTIAL"]
    recommendation: str           # "ACCEPT", "RECAPTURE", "REVIEW"
    details: dict | None = None   # métricas individuais (blur_score, brightness, etc.)


class IQualityGate(ABC):
    """
    Port: Quality Gate

    Responsável por decidir se uma imagem de documento
    está apta para processamento (OCR + fraud detection).
    """

    @abstractmethod
    def evaluate(self, image_bytes: bytes) -> QualityResult:
        """
        Avalia a qualidade da imagem.

        Args:
            image_bytes: Imagem em bytes (JPEG/PNG).

        Returns:
            QualityResult com score, flags e recomendação.
        """
        ...
