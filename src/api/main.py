"""
FastAPI Application — Fraud-Doc Pipeline.

Architecture:
  - PostgreSQL + pgvector (prod) / SQLite (dev) for storage
  - Gemini Embeddings for vectorization
  - RAG-enhanced LLM chat
  - PaddleOCR v5 + EasyOCR hybrid engine
"""

import os
import time
import uuid
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.routes.analyze import router as analyze_router
from src.config.settings import get_settings
from src.infrastructure.db.database import init_db, get_db
from src.infrastructure.db.repository import CaseRepository
from src.infrastructure.db.models import CaseRecord

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud-Doc Pipeline",
    description="AI-powered document fraud detection with OCR, rules engine, LLM analysis, and RAG.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ──
case_repo = CaseRepository()
_rag_engine = None

API_PASSWORD = os.getenv("API_PASSWORD", "admin")
API_KEY = os.getenv("API_KEY", "bayes-fraud-doc-2024")


def get_rag_engine():
    """Lazy-init RAG engine."""
    global _rag_engine
    if _rag_engine is None:
        settings = get_settings()
        if settings.gemini_api_key:
            from src.infrastructure.rag.rag_engine import RAGChatEngine
            _rag_engine = RAGChatEngine(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
            )
    return _rag_engine


# ── Startup ──
@app.on_event("startup")
async def startup():
    """Initialize DB and load demo cases."""
    init_db()
    _load_demo_cases()
    # Embed demo cases in background
    _embed_existing_cases()
    logger.info("Fraud-Doc Pipeline started")


# Register analyze routes
app.include_router(analyze_router, prefix="/api/v1", tags=["Analysis"])


# ── Auth ──
class AuthRequest(BaseModel):
    password: str

class AuthResponse(BaseModel):
    success: bool
    token: str = ""
    message: str = ""

@app.post("/api/v1/auth", response_model=AuthResponse)
async def authenticate(req: AuthRequest):
    """Simple password auth. Default password: admin"""
    if req.password == API_PASSWORD:
        return AuthResponse(success=True, token=API_KEY, message="Authenticated successfully")
    return AuthResponse(success=False, message="Invalid password")


# ── Cases (from DB) ──
@app.get("/api/v1/cases")
async def list_cases(limit: int = 50, offset: int = 0, decision: str = None):
    """List all analysis runs from database."""
    return case_repo.list_cases(limit=limit, offset=offset, decision=decision)


@app.get("/api/v1/cases/{case_id}")
async def get_case(case_id: str):
    """Get a specific analysis run by case_id."""
    data = case_repo.get_by_id(case_id)
    if data:
        return data
    return JSONResponse(status_code=404, content={"detail": "Case not found"})


@app.get("/api/v1/stats")
async def get_stats():
    """Get aggregated statistics."""
    return case_repo.get_stats()


# ── Store helper (called from analyze route) ──
def store_case(result_dict: dict):
    """Store an analysis result in DB and generate embedding."""
    result_dict["timestamp"] = datetime.now().isoformat()
    result_dict["run_id"] = str(uuid.uuid4())[:8]

    # Save to DB
    record = case_repo.save(result_dict)

    # Generate embedding async-style (in-thread for now)
    try:
        rag = get_rag_engine()
        if rag:
            rag.embed_case(record.case_id)
    except Exception as e:
        logger.warning(f"Embedding failed for {record.case_id}: {e}")


# ── RAG Chat ──
class ChatRequest(BaseModel):
    message: str
    context_case_ids: list[str] = []

class ChatResponse(BaseModel):
    reply: str
    model: str = ""
    latency_ms: float = 0
    rag_cases_found: int = 0
    rag_case_ids: list[str] = []

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_llm(req: ChatRequest):
    """RAG-enhanced chat with AI about fraud analysis data."""
    rag = get_rag_engine()
    if not rag:
        return ChatResponse(reply="LLM not configured. Set GEMINI_API_KEY in .env", model="none")

    result = rag.chat(message=req.message, context_case_ids=req.context_case_ids)

    return ChatResponse(
        reply=result.get("reply", ""),
        model=result.get("model", ""),
        latency_ms=result.get("latency_ms", 0),
        rag_cases_found=result.get("rag_cases_found", 0),
        rag_case_ids=result.get("rag_case_ids", []),
    )


# ── Health ──
@app.get("/health")
async def health():
    stats = case_repo.get_stats()
    db_url = os.getenv("DATABASE_URL", "sqlite:///fraud_doc.db")
    db_type = "PostgreSQL" if "postgres" in db_url else "SQLite"
    return {
        "status": "ok",
        "version": "1.0.0",
        "database": db_type,
        "cases_stored": stats["total"],
        "rag_enabled": get_rag_engine() is not None,
        "llm_enabled": get_settings().llm_enabled,
    }


# ── Demo Cases ──
def _load_demo_cases():
    """Pre-load demo cases into DB."""
    stats = case_repo.get_stats()
    if stats["total"] > 0:
        logger.info(f"DB already has {stats['total']} cases, skipping demo load")
        return

    demos = [
        {
            "case_id": "demo-aze-001",
            "final_decision": "APPROVED", "final_score": 0.95,
            "rejection_reasons": [],
            "pipeline_version": "1.0.0", "total_latency_ms": 3240,
            "stage_latencies": {"quality_ms": 45, "ocr_ms": 2100, "rules_ms": 2, "llm_ms": 1093},
            "quality": {"quality_ok": True, "quality_score": 0.92, "reasons": [], "recommendation": "ACCEPT",
                        "details": {"blur_score": 1507, "brightness_mean": 142, "doc_area_ratio": 0.82}},
            "ocr": {"raw_text": "PCAZEAGALAROVA<<AYSU...", "fields": [
                {"name": "primary_identifier", "value": "AGALAROVA", "confidence": 0.99},
                {"name": "secondary_identifier", "value": "AYSU", "confidence": 0.98},
                {"name": "document_number", "value": "C63774367", "confidence": 0.99},
                {"name": "nationality", "value": "AZE", "confidence": 0.99},
                {"name": "date_of_birth", "value": "30.11.1974", "confidence": 0.97},
                {"name": "sex", "value": "F", "confidence": 0.99},
                {"name": "date_of_expiry", "value": "06.08.2021", "confidence": 0.98},
            ], "avg_confidence": 0.985, "doc_type_detected": "PASSPORT",
                "ocr_engine": "Hybrid (PaddleOCR v5 + EasyOCR)"},
            "rules": {"rules_passed": 10, "rules_failed": 0, "rules_total": 10, "violations": [],
                      "risk_score": 0.0, "risk_level": "LOW", "rules_version": "passport-v1.0"},
            "llm": {"fraud_probability": 0.05, "risk_level": "LOW",
                    "assessment": "All checks passed. Document appears genuine.",
                    "anomalies": [], "recommendation": "APPROVE",
                    "reasoning": "MRZ checksums valid, fields consistent.",
                    "latency_ms": 1093, "model": "gemini-2.0-flash"},
        },
        {
            "case_id": "demo-fraud-002",
            "final_decision": "REJECTED", "final_score": 0.12,
            "rejection_reasons": ["DOC_NUM_CHECK", "COMPOSITE_CHECK", "CROSS_CHECK"],
            "pipeline_version": "1.0.0", "total_latency_ms": 4100,
            "stage_latencies": {"quality_ms": 38, "ocr_ms": 2500, "rules_ms": 3, "llm_ms": 1559},
            "quality": {"quality_ok": True, "quality_score": 0.88, "reasons": [], "recommendation": "ACCEPT",
                        "details": {"blur_score": 1200, "brightness_mean": 130, "doc_area_ratio": 0.75}},
            "ocr": {"raw_text": "Tampered passport...", "fields": [
                {"name": "primary_identifier", "value": "SMITH", "confidence": 0.95},
                {"name": "document_number", "value": "X12345678", "confidence": 0.90},
                {"name": "nationality", "value": "GBR", "confidence": 0.92},
                {"name": "date_of_birth", "value": "15.03.1985", "confidence": 0.88},
                {"name": "sex", "value": "M", "confidence": 0.97},
            ], "avg_confidence": 0.924, "doc_type_detected": "PASSPORT",
                "ocr_engine": "Hybrid (PaddleOCR v5 + EasyOCR)"},
            "rules": {"rules_passed": 7, "rules_failed": 3, "rules_total": 10,
                      "violations": [
                          {"rule_id": "DOC_NUM_CHECK", "rule_name": "Document Number Checksum", "severity": "CRITICAL", "detail": "Checksum mismatch"},
                          {"rule_id": "COMPOSITE_CHECK", "rule_name": "Composite Checksum", "severity": "CRITICAL", "detail": "Composite check failed"},
                          {"rule_id": "CROSS_CHECK", "rule_name": "VIZ-MRZ Cross-Check", "severity": "HIGH", "detail": "Surname mismatch VIZ vs MRZ"},
                      ],
                      "risk_score": 0.88, "risk_level": "CRITICAL", "rules_version": "passport-v1.0"},
            "llm": {"fraud_probability": 0.92, "risk_level": "CRITICAL",
                    "assessment": "Multiple critical violations. Document number checksum fails, name mismatch between VIZ and MRZ.",
                    "anomalies": ["Checksum failure", "Name mismatch", "Possible photo substitution"],
                    "recommendation": "REJECT", "reasoning": "3 critical rule violations indicate document tampering.",
                    "latency_ms": 1559, "model": "gemini-2.0-flash"},
        },
        {
            "case_id": "demo-review-003",
            "final_decision": "REVIEW", "final_score": 0.55,
            "rejection_reasons": ["DATE_PLAUSIBILITY"],
            "pipeline_version": "1.0.0", "total_latency_ms": 3800,
            "stage_latencies": {"quality_ms": 42, "ocr_ms": 2300, "rules_ms": 2, "llm_ms": 1456},
            "quality": {"quality_ok": True, "quality_score": 0.85, "reasons": [], "recommendation": "ACCEPT",
                        "details": {"blur_score": 980, "brightness_mean": 155, "doc_area_ratio": 0.70}},
            "ocr": {"raw_text": "Expired passport...", "fields": [
                {"name": "primary_identifier", "value": "MUELLER", "confidence": 0.96},
                {"name": "document_number", "value": "T22334455", "confidence": 0.94},
                {"name": "nationality", "value": "DEU", "confidence": 0.98},
                {"name": "date_of_birth", "value": "22.06.1960", "confidence": 0.95},
                {"name": "date_of_expiry", "value": "15.01.2020", "confidence": 0.97},
                {"name": "sex", "value": "M", "confidence": 0.99},
            ], "avg_confidence": 0.965, "doc_type_detected": "PASSPORT",
                "ocr_engine": "Hybrid (PaddleOCR v5 + EasyOCR)"},
            "rules": {"rules_passed": 9, "rules_failed": 1, "rules_total": 10,
                      "violations": [
                          {"rule_id": "DATE_PLAUSIBILITY", "rule_name": "Date Plausibility", "severity": "CRITICAL", "detail": "Document expired: 2020-01-15"},
                      ],
                      "risk_score": 0.2, "risk_level": "MEDIUM", "rules_version": "passport-v1.0"},
            "llm": {"fraud_probability": 0.15, "risk_level": "LOW",
                    "assessment": "Document is expired but otherwise appears genuine. All checksums pass.",
                    "anomalies": ["Expired document"], "recommendation": "REVIEW",
                    "reasoning": "Only issue is expiration. No signs of tampering.",
                    "latency_ms": 1456, "model": "gemini-2.0-flash"},
        },
        {
            "case_id": "demo-blur-004",
            "final_decision": "REVIEW", "final_score": 0.60,
            "rejection_reasons": ["BLUR_HIGH"],
            "pipeline_version": "1.0.0", "total_latency_ms": 2900,
            "stage_latencies": {"quality_ms": 55, "ocr_ms": 1800, "rules_ms": 2, "llm_ms": 1043},
            "quality": {"quality_ok": False, "quality_score": 0.45, "reasons": ["BLUR_HIGH"], "recommendation": "REVIEW",
                        "details": {"blur_score": 65, "brightness_mean": 120, "doc_area_ratio": 0.60}},
            "ocr": {"raw_text": "Blurry capture...", "fields": [
                {"name": "primary_identifier", "value": "TANAKA", "confidence": 0.72},
                {"name": "document_number", "value": "TK9988776", "confidence": 0.68},
                {"name": "nationality", "value": "JPN", "confidence": 0.80},
                {"name": "date_of_birth", "value": "01.04.1990", "confidence": 0.65},
                {"name": "sex", "value": "F", "confidence": 0.85},
            ], "avg_confidence": 0.740, "doc_type_detected": "PASSPORT",
                "ocr_engine": "Hybrid (PaddleOCR v5 + EasyOCR)"},
            "rules": {"rules_passed": 10, "rules_failed": 0, "rules_total": 10, "violations": [],
                      "risk_score": 0.0, "risk_level": "LOW", "rules_version": "passport-v1.0"},
            "llm": {"fraud_probability": 0.20, "risk_level": "LOW",
                    "assessment": "Image quality is poor (blurry). Low OCR confidence. Recommend recapture.",
                    "anomalies": ["Low image quality", "Low OCR confidence"],
                    "recommendation": "REVIEW", "reasoning": "Cannot reliably assess due to image quality.",
                    "latency_ms": 1043, "model": "gemini-2.0-flash"},
        },
        {
            "case_id": "demo-perfect-005",
            "final_decision": "APPROVED", "final_score": 0.98,
            "rejection_reasons": [],
            "pipeline_version": "1.0.0", "total_latency_ms": 3100,
            "stage_latencies": {"quality_ms": 40, "ocr_ms": 1900, "rules_ms": 2, "llm_ms": 1158},
            "quality": {"quality_ok": True, "quality_score": 0.96, "reasons": [], "recommendation": "ACCEPT",
                        "details": {"blur_score": 2100, "brightness_mean": 135, "doc_area_ratio": 0.88}},
            "ocr": {"raw_text": "Perfect passport...", "fields": [
                {"name": "primary_identifier", "value": "SILVA", "confidence": 0.99},
                {"name": "secondary_identifier", "value": "MARIA", "confidence": 0.98},
                {"name": "document_number", "value": "BR1234567", "confidence": 0.99},
                {"name": "nationality", "value": "BRA", "confidence": 0.99},
                {"name": "date_of_birth", "value": "10.05.1988", "confidence": 0.98},
                {"name": "date_of_expiry", "value": "10.05.2028", "confidence": 0.99},
                {"name": "sex", "value": "F", "confidence": 0.99},
            ], "avg_confidence": 0.987, "doc_type_detected": "PASSPORT",
                "ocr_engine": "Hybrid (PaddleOCR v5 + EasyOCR)"},
            "rules": {"rules_passed": 10, "rules_failed": 0, "rules_total": 10, "violations": [],
                      "risk_score": 0.0, "risk_level": "LOW", "rules_version": "passport-v1.0"},
            "llm": {"fraud_probability": 0.03, "risk_level": "LOW",
                    "assessment": "Perfect document. All checksums pass, all fields consistent, high confidence.",
                    "anomalies": [], "recommendation": "APPROVE",
                    "reasoning": "10/10 rules passed, 98.7% OCR confidence, no anomalies.",
                    "latency_ms": 1158, "model": "gemini-2.0-flash"},
        },
    ]

    for demo in demos:
        try:
            case_repo.save(demo)
        except Exception as e:
            logger.warning(f"Failed to save demo case: {e}")

    logger.info(f"Loaded {len(demos)} demo cases into DB")


def _embed_existing_cases():
    """Embed all cases that don't have embeddings yet."""
    try:
        rag = get_rag_engine()
        if rag:
            count = rag.embed_all_cases()
            logger.info(f"Embedded {count} cases on startup")
    except Exception as e:
        logger.warning(f"Startup embedding failed: {e}")


# ── Serve static & UI ──
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = static_dir / "index.html"
    if ui_path.exists():
        content = ui_path.read_text(encoding="utf-8")
        content = content.replace('href="style.css"', 'href="/static/style.css"')
        content = content.replace('src="app.js"', 'src="/static/app.js"')
        return HTMLResponse(content)
    return HTMLResponse("<h1>Fraud-Doc Pipeline</h1><p>Go to <a href='/docs'>/docs</a></p>")


# ── Make store_case available to analyze route ──
app.state.store_case = store_case
