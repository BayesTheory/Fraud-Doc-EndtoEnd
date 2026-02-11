"""
Entity: Analysis Result

Resultado consolidado de todo o pipeline de análise
(quality + OCR + rules + fraud). Agregação de domínio.
"""

from dataclasses import dataclass, field
from datetime import datetime

from src.core.interfaces.quality_gate import QualityResult
from src.core.interfaces.ocr_engine import OCRResult
from src.core.interfaces.rules_engine import RulesResult
from src.core.interfaces.fraud_classifier import FraudResult


@dataclass
class AnalysisResult:
    """Resultado consolidado de uma análise de documento."""
    case_id: str
    quality: QualityResult | None = None
    ocr: OCRResult | None = None
    rules: RulesResult | None = None
    fraud: FraudResult | None = None

    # Decisão final
    final_decision: str = "PENDING"    # "APPROVED", "REJECTED", "REVIEW"
    final_score: float = 0.0           # score agregado
    rejection_reasons: list[str] = field(default_factory=list)

    # Meta
    pipeline_version: str = ""
    total_latency_ms: float = 0.0
    stage_latencies: dict = field(default_factory=dict)  # {"quality": 5.2, "ocr": 120.3, ...}
    created_at: datetime = field(default_factory=datetime.utcnow)
