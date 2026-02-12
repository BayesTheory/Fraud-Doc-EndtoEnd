"""
Hybrid OCR Engine — PaddleOCR v5 (MRZ) + EasyOCR (VIZ).

PaddleOCR v5 is excellent at MRZ (monospace OCR-B font, 99%+ accuracy).
EasyOCR handles the variable-font VIZ (Visual Inspection Zone) text.
"""

import os
import re
import cv2
import numpy as np
import logging

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from src.core.interfaces.ocr_engine import IOCREngine, OCRResult, OCRField

logger = logging.getLogger(__name__)

MRZ_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")


def _clean_mrz_line(text: str) -> str:
    """Clean OCR output to valid MRZ characters."""
    text = text.upper().replace(" ", "").replace("\n", "")
    cleaned = []
    for ch in text:
        if ch in MRZ_CHARS:
            cleaned.append(ch)
    return "".join(cleaned)


class HybridOCREngine(IOCREngine):
    """
    Hybrid OCR: PaddleOCR v5 for MRZ + EasyOCR for VIZ text.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self.lang = lang
        self.use_gpu = use_gpu
        self._paddle = None
        self._easyocr = None

    def _get_paddle(self):
        if self._paddle is None:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(lang="en")
        return self._paddle

    def _get_easyocr(self):
        if self._easyocr is None:
            import easyocr
            self._easyocr = easyocr.Reader(
                [self.lang], gpu=self.use_gpu, verbose=False
            )
        return self._easyocr

    def extract(self, image_bytes: bytes, doc_type_hint: str | None = None) -> OCRResult:
        """Extract passport fields using hybrid OCR."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return OCRResult(
                raw_text="", fields=[], avg_confidence=0.0,
                doc_type_detected="UNKNOWN", ocr_engine="Hybrid"
            )

        # ── 1. Extract MRZ with PaddleOCR v5 ──
        mrz_upper, mrz_lower, mrz_conf = self._extract_mrz_paddle(image)

        # ── 2. Fallback to EasyOCR if PaddleOCR failed ──
        if not mrz_upper or not mrz_lower:
            logger.info("PaddleOCR MRZ incomplete, trying EasyOCR fallback...")
            easy_upper, easy_lower = self._extract_mrz_easyocr(image)
            if not mrz_upper and easy_upper:
                mrz_upper = easy_upper
            if not mrz_lower and easy_lower:
                mrz_lower = easy_lower

        # ── 3. Extract VIZ text with EasyOCR ──
        full_text = self._extract_viz(image)

        # ── 4. Build structured fields ──
        fields = []
        confs = []

        if mrz_upper:
            conf = mrz_conf
            fields.append(OCRField("mrz_upper_line", mrz_upper, conf))
            confs.append(conf)

            # Parse names from MRZ line 1
            if len(mrz_upper) >= 40:
                names = mrz_upper[5:].split("<<")
                if names:
                    primary = names[0].replace("<", " ").strip()
                    fields.append(OCRField("primary_identifier", primary, conf))
                    confs.append(conf)
                if len(names) >= 2:
                    secondary = names[1].replace("<", " ").strip()
                    if secondary:
                        fields.append(OCRField("secondary_identifier", secondary, conf))
                        confs.append(conf)

                # Issuing country from line 1 position 2:5
                issuing = mrz_upper[2:5]
                fields.append(OCRField("issuing_country", issuing, conf))
                confs.append(conf)

        if mrz_lower:
            fields.append(OCRField("mrz_lower_line", mrz_lower, mrz_conf))
            confs.append(mrz_conf)

            if len(mrz_lower) >= 28:
                doc_num = mrz_lower[0:9].replace("<", "")
                fields.append(OCRField("document_number", doc_num, mrz_conf))
                confs.append(mrz_conf)

                nationality = mrz_lower[10:13]
                fields.append(OCRField("nationality", nationality, mrz_conf))
                confs.append(mrz_conf)

                dob_raw = mrz_lower[13:19]
                dob = self._format_date(dob_raw)
                fields.append(OCRField("date_of_birth", dob, mrz_conf))
                confs.append(mrz_conf)

                sex = mrz_lower[20]
                fields.append(OCRField("sex", sex, mrz_conf))
                confs.append(mrz_conf)

                doe_raw = mrz_lower[21:27]
                doe = self._format_date(doe_raw)
                fields.append(OCRField("date_of_expiry", doe, mrz_conf))
                confs.append(mrz_conf)

        avg_conf = sum(confs) / len(confs) if confs else 0.0

        return OCRResult(
            raw_text=full_text,
            fields=fields,
            avg_confidence=round(avg_conf, 3),
            doc_type_detected="PASSPORT" if mrz_upper else "UNKNOWN",
            ocr_engine="Hybrid (PaddleOCR v5 + EasyOCR)",
            details={
                "mrz_upper": mrz_upper,
                "mrz_lower": mrz_lower,
                "mrz_lines_found": int(bool(mrz_upper)) + int(bool(mrz_lower)),
            },
        )

    def _extract_mrz_paddle(self, image: np.ndarray) -> tuple[str, str, float]:
        """
        Extract MRZ using PaddleOCR v5 predict() API.
        Returns (mrz_upper, mrz_lower, avg_confidence).
        """
        h, w = image.shape[:2]
        mrz_crop = image[int(h * 0.55):, :]  # bottom 45%

        try:
            paddle = self._get_paddle()
            results = paddle.predict(mrz_crop)

            # PaddleOCR v5 returns list of result objects
            mrz_candidates = []
            for res in results:
                if hasattr(res, 'rec_texts') and hasattr(res, 'rec_scores'):
                    texts = res.rec_texts
                    scores = res.rec_scores
                    polys = res.rec_polys if hasattr(res, 'rec_polys') else [None] * len(texts)

                    for text, score, poly in zip(texts, scores, polys):
                        clean = _clean_mrz_line(text)
                        y_pos = poly[0][1] if poly is not None and len(poly) > 0 else 0
                        if len(clean) >= 30:
                            # Pad to 44 if close
                            if 40 <= len(clean) <= 44:
                                clean = clean[:44].ljust(44, "<")
                            mrz_candidates.append((float(y_pos), clean, float(score)))

            # Sort by Y position (top to bottom)
            mrz_candidates.sort(key=lambda x: x[0])

            mrz_upper = ""
            mrz_lower = ""
            avg_conf = 0.95

            if len(mrz_candidates) >= 2:
                mrz_upper = mrz_candidates[-2][1]  # second to last (line 1)
                mrz_lower = mrz_candidates[-1][1]  # last (line 2)
                avg_conf = (mrz_candidates[-2][2] + mrz_candidates[-1][2]) / 2
            elif len(mrz_candidates) == 1:
                line = mrz_candidates[0][1]
                avg_conf = mrz_candidates[0][2]
                # Determine if it's line 1 or 2
                if line.startswith("P"):
                    mrz_upper = line
                else:
                    mrz_lower = line

            return mrz_upper, mrz_lower, round(avg_conf, 3)

        except Exception as e:
            logger.warning(f"PaddleOCR MRZ extraction failed: {e}")
            return "", "", 0.0

    def _extract_mrz_easyocr(self, image: np.ndarray) -> tuple[str, str]:
        """Fallback MRZ extraction using EasyOCR."""
        h, w = image.shape[:2]
        mrz_crop = image[int(h * 0.60):, :]

        try:
            reader = self._get_easyocr()
            results = reader.readtext(
                mrz_crop,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
                paragraph=False, width_ths=1.5,
            )
            candidates = []
            for r in sorted(results, key=lambda x: x[0][0][1]):
                text = _clean_mrz_line(r[1])
                if len(text) >= 30:
                    if 40 <= len(text) <= 44:
                        text = text.ljust(44, "<")
                    candidates.append(text)

            upper = candidates[0] if len(candidates) >= 1 else ""
            lower = candidates[1] if len(candidates) >= 2 else ""
            return upper, lower
        except Exception as e:
            logger.warning(f"EasyOCR MRZ fallback failed: {e}")
            return "", ""

    def _extract_viz(self, image: np.ndarray) -> str:
        """Extract full visible text using EasyOCR."""
        try:
            reader = self._get_easyocr()
            results = reader.readtext(image, paragraph=False)
            return " ".join([r[1] for r in results]) if results else ""
        except Exception as e:
            logger.warning(f"EasyOCR VIZ failed: {e}")
            return ""

    @staticmethod
    def _format_date(yymmdd: str) -> str:
        """Convert YYMMDD → DD.MM.YYYY."""
        if len(yymmdd) != 6 or not yymmdd.isdigit():
            return yymmdd
        yy, mm, dd = int(yymmdd[:2]), yymmdd[2:4], yymmdd[4:6]
        year = 2000 + yy if yy < 30 else 1900 + yy
        return f"{dd}.{mm}.{year}"
