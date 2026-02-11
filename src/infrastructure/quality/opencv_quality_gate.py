"""
Adapter: OpenCV Quality Gate

Implementação concreta do contrato IQualityGate
usando OpenCV para avaliar blur, iluminação, enquadramento e resolução.
"""

from src.core.interfaces.quality_gate import IQualityGate, QualityResult


class OpenCVQualityGate(IQualityGate):
    """
    Implementação do Quality Gate usando OpenCV.

    Avalia:
        - Blur (variância do Laplaciano)
        - Iluminação (média + desvio do histograma)
        - Enquadramento (detecção de bordas do documento)
        - Resolução mínima
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        brightness_min: int = 50,
        brightness_max: int = 220,
        min_resolution: int = 640,
        min_doc_area_ratio: float = 0.70,
    ):
        self._blur_threshold = blur_threshold
        self._brightness_min = brightness_min
        self._brightness_max = brightness_max
        self._min_resolution = min_resolution
        self._min_doc_area_ratio = min_doc_area_ratio

    def evaluate(self, image_bytes: bytes) -> QualityResult:
        # TODO: Implementar avaliação com OpenCV
        raise NotImplementedError("OpenCV quality gate not yet implemented")
