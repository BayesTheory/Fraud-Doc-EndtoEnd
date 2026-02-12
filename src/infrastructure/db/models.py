"""
Database Models — SQLAlchemy + pgvector.

Tables:
  - cases: Analysis results (structured data)
  - case_embeddings: Vector embeddings for RAG search
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text, JSON,
    ForeignKey, Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class CaseRecord(Base):
    """Stores every analysis run."""
    __tablename__ = "cases"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(36), unique=True, nullable=False, index=True)
    run_id = Column(String(8), index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Decision
    final_decision = Column(String(20), nullable=False, index=True)
    final_score = Column(Float, default=0.0)
    rejection_reasons = Column(JSON, default=list)

    # Pipeline
    pipeline_version = Column(String(20), default="1.0.0")
    total_latency_ms = Column(Float, default=0.0)
    stage_latencies = Column(JSON, default=dict)

    # Quality Gate
    quality_ok = Column(Boolean, default=False)
    quality_score = Column(Float, default=0.0)
    quality_details = Column(JSON, default=dict)

    # OCR
    ocr_engine = Column(String(50), default="")
    ocr_avg_confidence = Column(Float, default=0.0)
    ocr_doc_type = Column(String(30), default="")
    ocr_fields = Column(JSON, default=dict)
    ocr_raw_text = Column(Text, default="")

    # Rules
    rules_passed = Column(Integer, default=0)
    rules_failed = Column(Integer, default=0)
    rules_total = Column(Integer, default=0)
    rules_risk_score = Column(Float, default=0.0)
    rules_risk_level = Column(String(20), default="LOW")
    rules_violations = Column(JSON, default=list)

    # LLM
    llm_fraud_probability = Column(Float, default=0.0)
    llm_risk_level = Column(String(20), default="")
    llm_assessment = Column(Text, default="")
    llm_anomalies = Column(JSON, default=list)
    llm_recommendation = Column(String(20), default="")
    llm_reasoning = Column(Text, default="")
    llm_model = Column(String(50), default="")
    llm_latency_ms = Column(Float, default=0.0)

    # Full JSON (for backward compat)
    raw_json = Column(JSON, default=dict)

    # Relationship to embedding
    embedding = relationship("CaseEmbedding", back_populates="case", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Case {self.case_id} [{self.final_decision}] score={self.final_score}>"

    @classmethod
    def from_analysis_dict(cls, data: dict) -> "CaseRecord":
        """Create CaseRecord from analysis response dict."""
        quality = data.get("quality") or {}
        ocr = data.get("ocr") or {}
        rules = data.get("rules") or {}
        llm = data.get("llm") or {}

        return cls(
            case_id=data.get("case_id", str(uuid.uuid4())),
            run_id=data.get("run_id", str(uuid.uuid4())[:8]),
            final_decision=data.get("final_decision", "UNKNOWN"),
            final_score=data.get("final_score", 0.0),
            rejection_reasons=data.get("rejection_reasons", []),
            pipeline_version=data.get("pipeline_version", "1.0.0"),
            total_latency_ms=data.get("total_latency_ms", 0.0),
            stage_latencies=data.get("stage_latencies", {}),
            # Quality
            quality_ok=quality.get("quality_ok", False),
            quality_score=quality.get("quality_score", 0.0),
            quality_details=quality.get("details", {}),
            # OCR
            ocr_engine=ocr.get("ocr_engine", ""),
            ocr_avg_confidence=ocr.get("avg_confidence", 0.0),
            ocr_doc_type=ocr.get("doc_type_detected", ""),
            ocr_fields={f["name"]: f["value"] for f in ocr.get("fields", []) if isinstance(f, dict)},
            ocr_raw_text=ocr.get("raw_text", ""),
            # Rules
            rules_passed=rules.get("rules_passed", 0),
            rules_failed=rules.get("rules_failed", 0),
            rules_total=rules.get("rules_total", 0),
            rules_risk_score=rules.get("risk_score", 0.0),
            rules_risk_level=rules.get("risk_level", "LOW"),
            rules_violations=rules.get("violations", []),
            # LLM
            llm_fraud_probability=llm.get("fraud_probability", 0.0),
            llm_risk_level=llm.get("risk_level", ""),
            llm_assessment=llm.get("assessment", ""),
            llm_anomalies=llm.get("anomalies", []),
            llm_recommendation=llm.get("recommendation", ""),
            llm_reasoning=llm.get("reasoning", ""),
            llm_model=llm.get("model", ""),
            llm_latency_ms=llm.get("latency_ms", 0.0),
            # Raw
            raw_json=data,
        )

    def to_summary_text(self) -> str:
        """Generate text for embedding — captures the semantic content of this case."""
        parts = [
            f"Document analysis case {self.case_id}.",
            f"Decision: {self.final_decision}, score: {self.final_score:.2f}.",
        ]
        if self.ocr_fields:
            for k, v in self.ocr_fields.items():
                if v and v not in ("[BBOX_PRESENT]", ""):
                    parts.append(f"{k}: {v}")
        if self.rules_risk_level:
            parts.append(f"Rules risk: {self.rules_risk_level} ({self.rules_risk_score:.2f})")
        if self.rules_violations:
            for v in self.rules_violations[:5]:
                if isinstance(v, dict):
                    parts.append(f"Violation: [{v.get('severity')}] {v.get('detail', '')}")
        if self.llm_assessment:
            parts.append(f"AI assessment: {self.llm_assessment}")
        if self.llm_anomalies:
            parts.append(f"Anomalies: {', '.join(self.llm_anomalies[:5])}")
        return " ".join(parts)


class CaseEmbedding(Base):
    """Stores vector embeddings for RAG similarity search."""
    __tablename__ = "case_embeddings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(36), ForeignKey("cases.case_id", ondelete="CASCADE"), unique=True, nullable=False)
    embedding_model = Column(String(50), default="")
    embedding_dim = Column(Integer, default=768)
    # Store as JSON array — works with SQLite and PostgreSQL
    # For pgvector, we add a VECTOR column via migration
    embedding_vector = Column(JSON, nullable=True)

    case = relationship("CaseRecord", back_populates="embedding")

    def __repr__(self):
        return f"<Embedding case={self.case_id} dim={self.embedding_dim}>"
