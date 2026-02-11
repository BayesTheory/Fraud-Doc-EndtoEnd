"""
Route: POST /analyze — Upload e análise de documento.
"""

import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from src.api.schemas.responses import (
    AnalysisResponse,
    QualityResponse,
    OCRResponse,
    OCRFieldResponse,
    RulesResponse,
    RuleViolationResponse,
)
from src.infrastructure.quality.opencv_quality_gate import OpenCVQualityGate
from src.infrastructure.ocr.paddle_ocr_engine import PaddleOCREngine
from src.infrastructure.rules.brazilian_doc_rules import BrazilianDocRulesEngine
from src.core.use_cases.analyze_document import AnalyzeDocumentUseCase
from src.config.settings import get_settings

router = APIRouter()


def _get_use_case() -> AnalyzeDocumentUseCase:
    """Factory — monta o use case com os adapters concretos."""
    settings = get_settings()
    return AnalyzeDocumentUseCase(
        quality_gate=OpenCVQualityGate(
            blur_threshold=settings.blur_threshold,
            brightness_min=settings.brightness_min,
            brightness_max=settings.brightness_max,
            min_resolution=settings.min_resolution,
            min_doc_area_ratio=settings.min_doc_area_ratio,
        ),
        ocr_engine=PaddleOCREngine(
            lang=settings.ocr_lang,
            use_gpu=settings.ocr_use_gpu,
        ),
        rules_engine=BrazilianDocRulesEngine(),
    )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(file: UploadFile = File(...)):
    """
    Analisa um documento de identidade.

    Upload a imagem (JPEG/PNG) e recebe:
    - Quality Gate (blur, iluminação, enquadramento)
    - OCR (campos extraídos)
    - Regras de Negócio (validações CPF, datas, etc.)
    - Decisão final (APPROVED / REJECTED / REVIEW)
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem (JPEG/PNG)")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Arquivo vazio")

    use_case = _get_use_case()
    result = use_case.execute(image_bytes)

    # Converte entities → response schemas
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

    return response
