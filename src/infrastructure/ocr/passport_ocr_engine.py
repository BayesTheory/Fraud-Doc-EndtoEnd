"""
Passport OCR Engine.

Uses COCO bounding box annotations to crop passport field regions,
then runs OCR on each region individually for precise field extraction.

Supports two modes:
1. **Annotated mode** (with COCO bboxes): Crops each field → OCR per field
2. **Blind mode** (no bboxes): Full-image OCR with MRZ detection heuristics

Falls back gracefully when PaddleOCR is not installed.
"""
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.interfaces.ocr_engine import IOCREngine, OCRResult, OCRField
from src.infrastructure.data.coco_loader import PassportSample, FieldRegion


# MRZ character set for filtering
MRZ_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")


def _clean_mrz_text(text: str) -> str:
    """Clean OCR'd MRZ text to valid MRZ characters."""
    text = text.upper().strip()
    # Common OCR misreads
    replacements = {
        "O": "0", " ": "", "\n": "", "I": "1", "L": "1",
        "S": "5", "B": "8", "G": "6", "Q": "0",
        "(": "<", ")": "<", "[": "<", "]": "<",
        "{": "<", "}": "<", "|": "<",
    }
    # Only apply replacements in numeric positions
    # For now, just clean invalid chars
    result = ""
    for c in text:
        if c in MRZ_CHARS:
            result += c
        elif c in replacements:
            result += replacements[c]
    return result


class PassportOCREngine(IOCREngine):
    """
    OCR engine specialized for passport documents.

    When COCO annotations (bounding boxes) are provided, it crops each
    annotated region and OCRs them individually for higher accuracy.
    When no annotations are available, falls back to full-image OCR.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self.lang = lang
        self.use_gpu = use_gpu
        self._ocr = None
        self._initialized = False

    def _init_ocr(self):
        """Lazy initialization of PaddleOCR."""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
            )
        except ImportError:
            print("[WARN] PaddleOCR not installed. Using stub OCR.")
            self._ocr = None

        self._initialized = True

    def extract(self, image: np.ndarray) -> OCRResult:
        """
        Extract text from a passport image (full image, no annotations).
        Falls back to basic OCR.
        """
        self._init_ocr()
        return self._ocr_full_image(image)

    def extract_with_regions(
        self,
        image: np.ndarray,
        sample: PassportSample,
    ) -> OCRResult:
        """
        Extract text using annotated bounding box regions.

        This is the preferred method — crops each field region and
        OCRs it individually for much better accuracy.
        """
        self._init_ocr()

        fields = []
        extracted = {}
        confidences = []

        for field_name, regions in sample.fields.items():
            for region in regions:
                x1, y1, x2, y2 = region.to_xyxy()

                # Add padding (5% of region size)
                pad_x = max(5, int((x2 - x1) * 0.05))
                pad_y = max(3, int((y2 - y1) * 0.1))

                # Clamp to image bounds
                h, w = image.shape[:2]
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)

                # Crop
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # OCR the cropped region
                text, confidence = self._ocr_region(crop, field_name)

                if text:
                    # Post-process based on field type
                    text = self._post_process_field(field_name, text)

                    ocr_field = OCRField(
                        name=field_name,
                        value=text,
                        confidence=confidence,
                        bbox=[x1, y1, x2, y2],
                    )
                    fields.append(ocr_field)

                    # Store in extracted dict (append if multiple regions)
                    if field_name in extracted:
                        extracted[field_name] += " " + text
                    else:
                        extracted[field_name] = text

                    confidences.append(confidence)

        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return OCRResult(
            full_text="\n".join(f"{f.name}: {f.value}" for f in fields),
            fields=fields,
            confidence=round(avg_confidence, 3),
            extracted_fields=extracted,
            document_type="passport",
        )

    def _ocr_region(
        self, crop: np.ndarray, field_name: str
    ) -> Tuple[str, float]:
        """OCR a single cropped region."""
        if self._ocr is None:
            return ("", 0.0)

        try:
            result = self._ocr.ocr(crop, cls=True)
            if not result or not result[0]:
                return ("", 0.0)

            # Combine all detected text in the region
            texts = []
            confs = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, tuple) and len(text_info) >= 2:
                        texts.append(str(text_info[0]))
                        confs.append(float(text_info[1]))

            combined_text = " ".join(texts)
            avg_conf = sum(confs) / len(confs) if confs else 0.0

            return (combined_text, avg_conf)

        except Exception as e:
            return ("", 0.0)

    def _ocr_full_image(self, image: np.ndarray) -> OCRResult:
        """Fallback: OCR the entire image."""
        if self._ocr is None:
            return OCRResult(
                full_text="[PaddleOCR not available]",
                fields=[],
                confidence=0.0,
                extracted_fields={},
                document_type="passport",
            )

        try:
            result = self._ocr.ocr(image, cls=True)
            if not result or not result[0]:
                return OCRResult(
                    full_text="",
                    fields=[],
                    confidence=0.0,
                    extracted_fields={},
                    document_type="passport",
                )

            all_texts = []
            all_confs = []
            fields = []

            for line in result[0]:
                if line and len(line) >= 2:
                    bbox_points = line[0]
                    text_info = line[1]
                    if isinstance(text_info, tuple):
                        text = str(text_info[0])
                        conf = float(text_info[1])

                        all_texts.append(text)
                        all_confs.append(conf)

                        # Try to auto-detect field type from position/content
                        field_name = self._guess_field_name(text)
                        fields.append(OCRField(
                            name=field_name,
                            value=text,
                            confidence=conf,
                            bbox=self._flatten_bbox(bbox_points),
                        ))

            full_text = "\n".join(all_texts)
            avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

            # Try to extract MRZ from full text
            extracted = self._extract_mrz_from_text(full_text)

            return OCRResult(
                full_text=full_text,
                fields=fields,
                confidence=round(avg_conf, 3),
                extracted_fields=extracted,
                document_type="passport",
            )

        except Exception:
            return OCRResult(
                full_text="",
                fields=[],
                confidence=0.0,
                extracted_fields={},
                document_type="passport",
            )

    def _post_process_field(self, field_name: str, text: str) -> str:
        """Clean up OCR'd text based on field type."""
        text = text.strip()

        if "mrz" in field_name:
            # MRZ should only contain A-Z, 0-9, <
            text = _clean_mrz_text(text)

        elif field_name in ("date_of_birth", "date_of_issue", "date_of_expiry"):
            # Keep only digits, dots, slashes, dashes
            text = re.sub(r"[^0-9./\-]", "", text)

        elif field_name == "sex":
            text = text.upper()
            if text and text[0] in ("M", "F"):
                text = text[0]

        elif field_name in ("document_number", "personal_number"):
            # Keep alphanumeric
            text = re.sub(r"[^A-Za-z0-9]", "", text)

        elif field_name in ("issuing_state_code", "nationality"):
            text = text.upper().strip()
            text = re.sub(r"[^A-Z]", "", text)[:3]

        return text

    def _guess_field_name(self, text: str) -> str:
        """Guess field name from content (for blind OCR mode)."""
        text_upper = text.upper().strip()

        # MRZ detection
        if len(text_upper) > 30 and text_upper.count("<") > 3:
            if text_upper.startswith("P"):
                return "mrz_upper_line"
            return "mrz_lower_line"

        # Date pattern
        if re.match(r"\d{2}[./\-]\d{2}[./\-]\d{2,4}", text):
            return "date_field"

        return "unknown"

    def _extract_mrz_from_text(self, full_text: str) -> Dict[str, str]:
        """Try to detect and extract MRZ lines from full OCR text."""
        extracted = {}
        lines = full_text.split("\n")

        mrz_lines = []
        for line in lines:
            clean = line.strip().upper()
            # MRZ lines are 44 chars and mostly contain valid MRZ chars
            if len(clean) >= 30:
                mrz_ratio = sum(1 for c in clean if c in MRZ_CHARS) / len(clean)
                if mrz_ratio > 0.8:
                    mrz_lines.append(_clean_mrz_text(clean))

        if len(mrz_lines) >= 2:
            # Last two MRZ-like lines should be the actual MRZ
            extracted["mrz_upper_line"] = mrz_lines[-2]
            extracted["mrz_lower_line"] = mrz_lines[-1]

        return extracted

    @staticmethod
    def _flatten_bbox(bbox_points) -> List[float]:
        """Convert PaddleOCR bbox format [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] to [x,y,x2,y2]."""
        if not bbox_points:
            return [0, 0, 0, 0]
        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        return [min(xs), min(ys), max(xs), max(ys)]
