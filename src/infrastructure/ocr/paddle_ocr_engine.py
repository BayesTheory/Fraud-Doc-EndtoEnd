"""
Adapter: PaddleOCR Engine

Implementação concreta do contrato IOCREngine
usando PaddleOCR para extração de texto e campos.
"""

from src.core.interfaces.ocr_engine import IOCREngine, OCRResult


class PaddleOCREngine(IOCREngine):
    """
    Implementação do OCR usando PaddleOCR.

    Pipeline:
        1. Pré-processamento (binarização adaptativa, deskew)
        2. Detecção de texto (DB detector)
        3. Reconhecimento (CRNN recognizer)
        4. Pós-processamento (regex, normalização por campo)
    """

    def __init__(self, lang: str = "pt", use_gpu: bool = False):
        self._lang = lang
        self._use_gpu = use_gpu
        # TODO: Inicializar PaddleOCR engine

    def extract(self, image_bytes: bytes, doc_type_hint: str | None = None) -> OCRResult:
        # TODO: Implementar extração com PaddleOCR
        raise NotImplementedError("PaddleOCR engine not yet implemented")
