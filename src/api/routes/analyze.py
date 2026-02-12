"""
Route: POST /analyze — Upload and analyze a document.
"""

import uuid
import time

from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from src.api.schemas.responses import (
    AnalysisResponse,
    QualityResponse,
    OCRResponse,
    OCRFieldResponse,
    RulesResponse,
    RuleViolationResponse,
)
from src.infrastructure.quality.opencv_quality_gate import OpenCVQualityGate
from src.infrastructure.ocr.hybrid_ocr_engine import HybridOCREngine
from src.infrastructure.rules.passport_rules import PassportRulesEngine
from src.infrastructure.llm.llm_analyzer import LLMFraudAnalyzer
from src.core.use_cases.analyze_document import AnalyzeDocumentUseCase
from src.config.settings import get_settings

router = APIRouter()

# Lazy singletons
_use_case = None
_llm_analyzer = None


def _get_use_case() -> AnalyzeDocumentUseCase:
    """Factory — build use case with concrete adapters."""
    global _use_case
    if _use_case is None:
        settings = get_settings()
        _use_case = AnalyzeDocumentUseCase(
            quality_gate=OpenCVQualityGate(
                blur_threshold=settings.blur_threshold,
                brightness_min=settings.brightness_min,
                brightness_max=settings.brightness_max,
                min_resolution=settings.min_resolution,
                min_doc_area_ratio=settings.min_doc_area_ratio,
            ),
            ocr_engine=HybridOCREngine(
                lang=settings.ocr_lang,
                use_gpu=settings.ocr_use_gpu,
            ),
            rules_engine=PassportRulesEngine(),
        )
    return _use_case


def _get_llm() -> LLMFraudAnalyzer | None:
    """Get LLM analyzer if enabled."""
    global _llm_analyzer
    settings = get_settings()
    if not settings.llm_enabled or not settings.gemini_api_key:
        return None
    if _llm_analyzer is None:
        _llm_analyzer = LLMFraudAnalyzer(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
        )
    return _llm_analyzer


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(request: Request, file: UploadFile = File(...)):
    """
    Analyze a passport document.

    Upload an image (JPEG/PNG) and receive:
    - Quality Gate (blur, brightness, framing)
    - OCR (extracted fields)
    - Rules Engine (MRZ checksums, cross-checks)
    - LLM Analysis (AI fraud assessment)
    - Final Decision (APPROVED / REJECTED / REVIEW)
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    use_case = _get_use_case()
    result = use_case.execute(image_bytes)

    # ── Run LLM analysis ──
    llm_data = None
    llm_analyzer = _get_llm()
    if llm_analyzer and result.ocr:
        try:
            t0 = time.perf_counter()
            ocr_fields = {f.name: f.value for f in result.ocr.fields}
            violations = []
            if result.rules:
                violations = [
                    {"rule_id": v.rule_id, "rule_name": v.rule_name,
                     "severity": v.severity, "detail": v.detail}
                    for v in result.rules.violations
                ]
            llm_result = llm_analyzer.analyze(
                ocr_fields=ocr_fields,
                rules_violations=violations,
                risk_score=result.rules.risk_score if result.rules else 0,
                risk_level=result.rules.risk_level if result.rules else "LOW",
            )
            if not llm_result.error:
                llm_data = llm_result.to_dict()
            else:
                llm_data = {"error": llm_result.error, "latency_ms": llm_result.latency_ms}

            # Add LLM latency to stage latencies
            if result.stage_latencies:
                result.stage_latencies["llm_ms"] = round(llm_result.latency_ms, 2)
                result.total_latency_ms += llm_result.latency_ms
        except Exception as e:
            llm_data = {"error": str(e)}

    # ── Build response ──
    response = AnalysisResponse(
        case_id=result.case_id,
        final_decision=result.final_decision,
        final_score=result.final_score,
        rejection_reasons=result.rejection_reasons,
        pipeline_version=result.pipeline_version,
        total_latency_ms=result.total_latency_ms,
        stage_latencies=result.stage_latencies,
    )

    if result.quality:
        response.quality = QualityResponse(
            quality_ok=result.quality.quality_ok,
            quality_score=result.quality.quality_score,
            reasons=result.quality.reasons,
            recommendation=result.quality.recommendation,
            details=result.quality.details,
        )

    if result.ocr:
        response.ocr = OCRResponse(
            raw_text=result.ocr.raw_text,
            fields=[
                OCRFieldResponse(
                    name=f.name,
                    value=f.value,
                    confidence=f.confidence,
                    bounding_box=f.bounding_box,
                )
                for f in result.ocr.fields
            ],
            avg_confidence=result.ocr.avg_confidence,
            doc_type_detected=result.ocr.doc_type_detected,
            ocr_engine=result.ocr.ocr_engine,
        )

    if result.rules:
        response.rules = RulesResponse(
            rules_passed=result.rules.rules_passed,
            rules_failed=result.rules.rules_failed,
            rules_total=result.rules.rules_total,
            violations=[
                RuleViolationResponse(
                    rule_id=v.rule_id,
                    rule_name=v.rule_name,
                    severity=v.severity,
                    detail=v.detail,
                )
                for v in result.rules.violations
            ],
            risk_score=result.rules.risk_score,
            risk_level=result.rules.risk_level,
            rules_version=result.rules.rules_version,
        )

    # Add LLM to response
    if llm_data:
        response.llm = llm_data

    # ── Store result for dashboard/RAG ──
    try:
        store_fn = request.app.state.store_case
        store_fn(response.model_dump())
    except Exception:
        pass

    return response

