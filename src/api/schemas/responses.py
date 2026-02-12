"""
Pydantic schemas — Response models para a API.
"""

from pydantic import BaseModel


class QualityResponse(BaseModel):
    quality_ok: bool
    quality_score: float
    reasons: list[str]
    recommendation: str
    details: dict | None = None


class OCRFieldResponse(BaseModel):
    name: str
    value: str
    confidence: float
    bounding_box: list | None = None


class OCRResponse(BaseModel):
    raw_text: str
    fields: list[OCRFieldResponse]
    avg_confidence: float
    doc_type_detected: str | None = None
    ocr_engine: str = ""


class RuleViolationResponse(BaseModel):
    rule_id: str
    rule_name: str
    severity: str
    detail: str


class RulesResponse(BaseModel):
    rules_passed: int
    rules_failed: int
    rules_total: int
    violations: list[RuleViolationResponse]
    risk_score: float
    risk_level: str
    rules_version: str


class FraudResponse(BaseModel):
    fraud_score: float
    fraud_label: str
    attack_type_predicted: str | None = None
    model_version: str = ""
    threshold_used: float = 0.5


class AnalysisResponse(BaseModel):
    case_id: str
    final_decision: str
    final_score: float
    rejection_reasons: list[str]
    quality: QualityResponse | None = None
    ocr: OCRResponse | None = None
    rules: RulesResponse | None = None
    fraud: FraudResponse | None = None
    llm: dict | None = None
    pipeline_version: str = ""
    total_latency_ms: float = 0.0
    stage_latencies: dict = {}
