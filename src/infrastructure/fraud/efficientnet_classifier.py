"""
Adapter: EfficientNet Fraud Classifier

Implementação concreta do contrato IFraudClassifier
usando EfficientNet-B0 com transfer learning para
classificar documentos como BONA_FIDE vs FORGED.
"""

from src.core.interfaces.fraud_classifier import IFraudClassifier, FraudResult


class EfficientNetClassifier(IFraudClassifier):
    """
    Classificador de fraude baseado em EfficientNet-B0.

    Treinado via transfer learning no dataset SIDTD para
    detectar adulterações (crop & replace, inpainting).

    O penúltimo layer (1280-D) é usado como embedding
    para busca vetorial de casos similares.
    """

    def __init__(self, model_path: str, threshold: float = 0.5, device: str = "cpu"):
        self._model_path = model_path
        self._threshold = threshold
        self._device = device
        # TODO: Carregar modelo PyTorch

    def classify(self, image_bytes: bytes) -> FraudResult:
        # TODO: Implementar inferência
        raise NotImplementedError("EfficientNet classifier not yet implemented")

    def get_embedding(self, image_bytes: bytes) -> list[float]:
        # TODO: Extrair embedding do penúltimo layer
        raise NotImplementedError("Embedding extraction not yet implemented")
