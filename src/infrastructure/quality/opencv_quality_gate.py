"""
Adapter: OpenCV Quality Gate — Implementação COMPLETA.

Avalia qualidade de imagem usando 4 métricas de CV clássico:
  1. Blur     → variância do Laplaciano
  2. Brilho   → média + desvio do histograma
  3. Resolução → dimensão mínima
  4. Enquadramento → % da área ocupada pelo documento
"""

import cv2
import numpy as np

from src.core.interfaces.quality_gate import IQualityGate, QualityResult


class OpenCVQualityGate(IQualityGate):
    """
    Quality Gate usando OpenCV puro — rápido (~5ms), determinístico, auditável.
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        brightness_min: int = 50,
        brightness_max: int = 220,
        min_resolution: int = 640,
        min_doc_area_ratio: float = 0.05,
    ):
        self._blur_threshold = blur_threshold
        self._brightness_min = brightness_min
        self._brightness_max = brightness_max
        self._min_resolution = min_resolution
        self._min_doc_area_ratio = min_doc_area_ratio

    def evaluate(self, image_bytes: bytes) -> QualityResult:
        """Avalia a qualidade da imagem e retorna score + flags."""
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return QualityResult(
                quality_ok=False,
                quality_score=0.0,
                reasons=["INVALID_IMAGE"],
                recommendation="RECAPTURE",
                details={"error": "Não foi possível decodificar a imagem"},
            )

        reasons: list[str] = []
        scores: dict[str, float] = {}

        # --- 1. Blur (variância do Laplaciano) ---
        blur_score = self._check_blur(img)
        scores["blur_score"] = round(blur_score, 2)
        if blur_score < self._blur_threshold:
            reasons.append("BLUR_HIGH")

        # --- 2. Iluminação (brilho) ---
        brightness, brightness_std = self._check_brightness(img)
        scores["brightness_mean"] = round(brightness, 2)
        scores["brightness_std"] = round(brightness_std, 2)
        if brightness < self._brightness_min:
            reasons.append("TOO_DARK")
        elif brightness > self._brightness_max:
            reasons.append("TOO_BRIGHT")
        if brightness_std < 30:
            reasons.append("LOW_CONTRAST")

        # --- 3. Resolução ---
        h, w = img.shape[:2]
        min_side = min(h, w)
        scores["resolution_min_side"] = min_side
        scores["resolution"] = f"{w}x{h}"
        if min_side < self._min_resolution:
            reasons.append("LOW_RESOLUTION")

        # --- 4. Enquadramento (documento ocupa área suficiente?) ---
        doc_area_ratio = self._check_framing(img)
        scores["doc_area_ratio"] = round(doc_area_ratio, 3)
        if doc_area_ratio < self._min_doc_area_ratio:
            reasons.append("CROP_PARTIAL")

        # --- Score final (média ponderada normalizada) ---
        quality_score = self._compute_quality_score(
            blur_score, brightness, brightness_std, min_side, doc_area_ratio
        )

        quality_ok = len(reasons) == 0
        if quality_ok:
            recommendation = "ACCEPT"
        elif len(reasons) <= 1:
            recommendation = "REVIEW"
        else:
            recommendation = "RECAPTURE"

        return QualityResult(
            quality_ok=quality_ok,
            quality_score=round(quality_score, 3),
            reasons=reasons,
            recommendation=recommendation,
            details=scores,
        )

    # ─── Métodos internos ──────────────────────────────────

    def _check_blur(self, img: np.ndarray) -> float:
        """
        Variância do Laplaciano — quanto maior, mais nítido.
        Típico: >300 = nítido, <100 = borrado.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _check_brightness(self, img: np.ndarray) -> tuple[float, float]:
        """
        Média e desvio padrão do canal V (HSV).
        Retorna (mean, std).
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        return float(v_channel.mean()), float(v_channel.std())

    def _check_framing(self, img: np.ndarray) -> float:
        """
        Estimate document area ratio using multiple strategies.
        Returns ratio 0.0 to 1.0 — best result from 3 methods.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_area = img.shape[0] * img.shape[1]
        if image_area == 0:
            return 0.0

        ratios = []

        # Strategy 1: Adaptive threshold + largest contour
        try:
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 51, 10
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                ratios.append(cv2.contourArea(largest) / image_area)
        except Exception:
            pass

        # Strategy 2: Multi-scale Canny + contour approximation
        try:
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            for lo, hi in [(30, 100), (50, 150), (75, 200)]:
                edges = cv2.Canny(blurred, lo, hi)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
                dilated = cv2.dilate(edges, kernel, iterations=3)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    # Try to approximate as rectangle
                    peri = cv2.arcLength(largest, True)
                    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
                    if 4 <= len(approx) <= 8:
                        ratios.append(cv2.contourArea(approx) / image_area)
                    else:
                        ratios.append(cv2.contourArea(largest) / image_area)
        except Exception:
            pass

        # Strategy 3: Edge density — documents have text = high edge density
        try:
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / image_area
            # Document images typically have 3-15% edge density
            # Non-document (blank wall, sky) has <1%
            if edge_density > 0.02:
                ratios.append(min(edge_density * 8, 1.0))  # Scale to 0-1
        except Exception:
            pass

        return max(ratios) if ratios else 0.0

    def _compute_quality_score(
        self,
        blur: float,
        brightness: float,
        brightness_std: float,
        min_side: int,
        doc_ratio: float,
    ) -> float:
        """
        Score composto 0.0-1.0 (média ponderada das métricas normalizadas).
        """
        # Normaliza cada métrica para 0-1
        blur_norm = min(blur / 500.0, 1.0)

        # Brilho ideal ~128, penaliza extremos
        brightness_norm = 1.0 - abs(brightness - 128.0) / 128.0
        brightness_norm = max(brightness_norm, 0.0)

        contrast_norm = min(brightness_std / 70.0, 1.0)
        resolution_norm = min(min_side / 1280.0, 1.0)
        framing_norm = min(doc_ratio / 0.9, 1.0)

        # Pesos
        score = (
            blur_norm * 0.30
            + brightness_norm * 0.20
            + contrast_norm * 0.10
            + resolution_norm * 0.15
            + framing_norm * 0.25
        )
        return max(0.0, min(score, 1.0))
