"""
Contract: Fraud Classifier

Classifica imagens de documentos como autênticas ou forjadas
usando modelos de deep learning.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FraudResult:
    """Resultado da classificação de fraude."""
    fraud_score: float            # 0.0 (bona fide) a 1.0 (forjado)
    fraud_label: str              # "BONA_FIDE", "FORGED"
    attack_type_predicted: str | None = None  # "crop_and_replace", "inpainting", etc.
    model_version: str = ""
    threshold_used: float = 0.5
    embedding: list[float] | None = None  # vetor do penúltimo layer (para busca vetorial)


class IFraudClassifier(ABC):
    """
    Port: Fraud Classifier

    Classifica a autenticidade de documentos usando visão computacional.
    A implementação pode ser EfficientNet, ResNet, ViT, etc.
    """

    @abstractmethod
    def classify(self, image_bytes: bytes) -> FraudResult:
        """
        Classifica se o documento é autêntico ou forjado.

        Args:
            image_bytes: Imagem em bytes.

        Returns:
            FraudResult com score, label e embedding opcional.
        """
        ...

    @abstractmethod
    def get_embedding(self, image_bytes: bytes) -> list[float]:
        """
        Gera o embedding visual do documento (sem classificar).

        Args:
            image_bytes: Imagem em bytes.

        Returns:
            Lista de floats (ex: 512-D ou 1280-D).
        """
        ...
