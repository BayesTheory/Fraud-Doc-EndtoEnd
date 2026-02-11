"""
Adapter: PaddleOCR Engine — Implementação COMPLETA.

Extrai texto com PaddleOCR e pós-processa para identificar
campos de documentos brasileiros (nome, CPF, data, RG).
"""

import re
import logging
import numpy as np
from typing import Any

from src.core.interfaces.ocr_engine import IOCREngine, OCRResult, OCRField

logger = logging.getLogger(__name__)


# ─── Padrões regex para campos brasileiros ────────────────

PATTERNS = {
    "cpf": re.compile(r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}"),
    "rg": re.compile(r"\d{1,2}\.?\d{3}\.?\d{3}-?[\dXx]"),
    "data": re.compile(r"\d{2}[/\-\.]\d{2}[/\-\.]\d{4}"),
    "cnh_registro": re.compile(r"\d{11}"),
}

# Palavras-chave que indicam tipo de documento
DOC_TYPE_KEYWORDS = {
    "RG": ["republica", "identidade", "registro geral", "ssp", "detran", "instituto"],
    "CNH": ["habilitacao", "habilitação", "cnh", "permissao", "carteira nacional"],
    "CRLV": ["licenciamento", "veiculo", "veículo", "crlv", "renavam"],
}


class PaddleOCREngine(IOCREngine):
    """
    OCR usando PaddleOCR com pós-processamento para docs brasileiros.

    Pipeline:
        1. PaddleOCR extrai caixas + texto + confiança
        2. Pós-processamento identifica campos via regex
        3. Tipificação do documento por palavras-chave
    """

    def __init__(self, lang: str = "pt", use_gpu: bool = False):
        self._lang = lang
        self._use_gpu = use_gpu
        self._engine = None  # Lazy init (PaddleOCR é pesado)

    def _get_engine(self) -> Any:
        """Inicializa PaddleOCR sob demanda."""
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR

                self._engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=self._lang,
                    use_gpu=self._use_gpu,
                    show_log=False,
                )
                logger.info("PaddleOCR inicializado com sucesso")
            except ImportError:
                logger.warning("PaddleOCR não instalado — usando fallback simples")
                self._engine = "FALLBACK"
        return self._engine

    def extract(self, image_bytes: bytes, doc_type_hint: str | None = None) -> OCRResult:
        """Extrai texto e campos da imagem."""
        engine = self._get_engine()

        # Decodifica imagem
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        import cv2
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return OCRResult(
                raw_text="",
                fields=[],
                avg_confidence=0.0,
                doc_type_detected="UNKNOWN",
                ocr_engine="paddleocr",
                details={"error": "Imagem inválida"},
            )

        # --- Executar OCR ---
        if engine == "FALLBACK":
            return self._fallback_extract(img)

        result = engine.ocr(img, cls=True)

        if not result or not result[0]:
            return OCRResult(
                raw_text="",
                fields=[],
                avg_confidence=0.0,
                doc_type_detected="UNKNOWN",
                ocr_engine="paddleocr",
                details={"warning": "Nenhum texto detectado"},
            )

        # --- Processar resultados ---
        lines: list[dict] = []
        for line in result[0]:
            bbox = line[0]
            text = line[1][0]
            conf = float(line[1][1])
            lines.append({
                "text": text,
                "confidence": conf,
                "bbox": [
                    int(bbox[0][0]), int(bbox[0][1]),
                    int(bbox[2][0]), int(bbox[2][1]),
                ],
            })

        raw_text = " ".join(l["text"] for l in lines)
        avg_conf = sum(l["confidence"] for l in lines) / len(lines) if lines else 0.0

        # --- Pós-processamento: extrair campos ---
        fields = self._extract_fields(raw_text, lines)

        # --- Tipificar documento ---
        doc_type = doc_type_hint or self._detect_doc_type(raw_text)

        return OCRResult(
            raw_text=raw_text,
            fields=fields,
            avg_confidence=round(avg_conf, 3),
            doc_type_detected=doc_type,
            ocr_engine="paddleocr",
            details={
                "total_lines": len(lines),
                "lines": lines,
            },
        )

    def _extract_fields(self, raw_text: str, lines: list[dict]) -> list[OCRField]:
        """Extrai campos estruturados via regex."""
        fields: list[OCRField] = []
        text_upper = raw_text.upper()

        # CPF
        cpf_match = PATTERNS["cpf"].search(raw_text)
        if cpf_match:
            fields.append(OCRField(
                name="cpf",
                value=cpf_match.group(),
                confidence=self._get_confidence_for_region(cpf_match.group(), lines),
            ))

        # RG
        rg_match = PATTERNS["rg"].search(raw_text)
        if rg_match and (not cpf_match or rg_match.group() != cpf_match.group()):
            fields.append(OCRField(
                name="rg",
                value=rg_match.group(),
                confidence=self._get_confidence_for_region(rg_match.group(), lines),
            ))

        # Datas
        date_matches = PATTERNS["data"].findall(raw_text)
        for i, date_str in enumerate(date_matches):
            label = "data_nascimento" if i == 0 else f"data_{i}"
            fields.append(OCRField(
                name=label,
                value=date_str,
                confidence=self._get_confidence_for_region(date_str, lines),
            ))

        # Nome — heurística: texto mais longo que parece nome
        # (só letras, maiúsculo, >5 chars, sem números)
        for line_data in lines:
            text = line_data["text"].strip()
            if (
                len(text) > 8
                and re.match(r"^[A-ZÀ-Ü\s]+$", text.upper())
                and not any(c.isdigit() for c in text)
                and not self._is_keyword(text)
            ):
                fields.append(OCRField(
                    name="nome",
                    value=text.upper().strip(),
                    confidence=line_data["confidence"],
                    bounding_box=line_data.get("bbox"),
                ))
                break  # Pega só o primeiro candidato

        return fields

    def _detect_doc_type(self, raw_text: str) -> str:
        """Detecta tipo do documento por palavras-chave."""
        text_lower = raw_text.lower()
        scores = {}
        for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[doc_type] = score

        if scores:
            return max(scores, key=scores.get)
        return "UNKNOWN"

    def _get_confidence_for_region(self, text: str, lines: list[dict]) -> float:
        """Encontra a confiança da linha que contém o texto."""
        for line in lines:
            if text in line["text"]:
                return line["confidence"]
        return 0.5  # fallback

    def _is_keyword(self, text: str) -> bool:
        """Verifica se o texto é uma keyword de documento (não nome)."""
        lower = text.lower()
        all_keywords = []
        for kws in DOC_TYPE_KEYWORDS.values():
            all_keywords.extend(kws)
        return any(kw in lower for kw in all_keywords)

    def _fallback_extract(self, img: np.ndarray) -> OCRResult:
        """Fallback quando PaddleOCR não está instalado."""
        return OCRResult(
            raw_text="[PaddleOCR não disponível — instale com: pip install paddleocr paddlepaddle]",
            fields=[],
            avg_confidence=0.0,
            doc_type_detected="UNKNOWN",
            ocr_engine="fallback",
            details={"warning": "PaddleOCR não instalado"},
        )
