"""
Contract: OCR Engine

Extrai texto e campos estruturados de imagens de documentos.
Qualquer engine (PaddleOCR, EasyOCR, Tesseract, API externa)
deve implementar este contrato.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class OCRField:
    """Campo individual extraído pelo OCR."""
    name: str                     # ex: "cpf", "nome", "data_nascimento"
    value: str                    # valor extraído
    confidence: float             # 0.0 a 1.0
    bounding_box: list | None = None  # [x1, y1, x2, y2] opcional


@dataclass
class OCRResult:
    """Resultado completo da extração OCR."""
    raw_text: str                 # texto bruto completo
    fields: list[OCRField]       # campos estruturados
    avg_confidence: float         # confiança média
    doc_type_detected: str | None = None  # "RG", "CNH", "UNKNOWN"
    ocr_engine: str = ""          # identificação da engine usada
    details: dict = field(default_factory=dict)


class IOCREngine(ABC):
    """
    Port: OCR Engine

    Responsável por extrair texto e campos de documentos.
    A implementação cuida do pré-processamento (binarização,
    deskew) e pós-processamento (regex, normalização).
    """

    @abstractmethod
    def extract(self, image_bytes: bytes, doc_type_hint: str | None = None) -> OCRResult:
        """
        Extrai campos de um documento.

        Args:
            image_bytes: Imagem em bytes.
            doc_type_hint: Tipo esperado (opcional, ajuda parsing).

        Returns:
            OCRResult com texto bruto e campos estruturados.
        """
        ...
