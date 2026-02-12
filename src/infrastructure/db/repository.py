"""
Case Repository — CRUD + vector search.

Handles:
  - Storing analysis results
  - Listing/filtering cases
  - Vector similarity search for RAG
"""

import json
import math
import logging
from typing import Optional
from datetime import datetime

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from src.infrastructure.db.models import CaseRecord, CaseEmbedding
from src.infrastructure.db.database import get_db

logger = logging.getLogger(__name__)


class CaseRepository:
    """Repository for analysis cases."""

    def save(self, data: dict) -> CaseRecord:
        """Save an analysis result to the database."""
        with get_db() as db:
            record = CaseRecord.from_analysis_dict(data)
            # Check if already exists
            existing = db.query(CaseRecord).filter_by(case_id=record.case_id).first()
            if existing:
                logger.debug(f"Case {record.case_id} already exists, skipping")
                return existing
            db.add(record)
            db.flush()
            logger.info(f"Saved case {record.case_id} [{record.final_decision}]")
            return record

    def get_by_id(self, case_id: str) -> Optional[dict]:
        """Get a case by its case_id."""
        with get_db() as db:
            record = db.query(CaseRecord).filter_by(case_id=case_id).first()
            if record:
                return record.raw_json
            return None

    def list_cases(self, limit: int = 50, offset: int = 0, decision: str = None) -> dict:
        """List cases with optional filtering."""
        with get_db() as db:
            query = db.query(CaseRecord)
            if decision:
                query = query.filter_by(final_decision=decision)
            total = query.count()
            cases = query.order_by(desc(CaseRecord.created_at)).offset(offset).limit(limit).all()
            return {
                "total": total,
                "cases": [c.raw_json for c in cases],
            }

    def get_stats(self) -> dict:
        """Get aggregated statistics."""
        with get_db() as db:
            total = db.query(CaseRecord).count()
            approved = db.query(CaseRecord).filter_by(final_decision="APPROVED").count()
            rejected = db.query(CaseRecord).filter_by(final_decision="REJECTED").count()
            review = db.query(CaseRecord).filter_by(final_decision="REVIEW").count()
            avg_score = db.query(func.avg(CaseRecord.final_score)).scalar() or 0
            avg_latency = db.query(func.avg(CaseRecord.total_latency_ms)).scalar() or 0
            return {
                "total": total,
                "approved": approved,
                "rejected": rejected,
                "review": review,
                "avg_score": round(float(avg_score), 3),
                "avg_latency_ms": round(float(avg_latency), 1),
            }

    def save_embedding(self, case_id: str, vector: list[float], model: str = ""):
        """Save embedding vector for a case."""
        with get_db() as db:
            existing = db.query(CaseEmbedding).filter_by(case_id=case_id).first()
            if existing:
                existing.embedding_vector = vector
                existing.embedding_model = model
                existing.embedding_dim = len(vector)
            else:
                emb = CaseEmbedding(
                    case_id=case_id,
                    embedding_vector=vector,
                    embedding_model=model,
                    embedding_dim=len(vector),
                )
                db.add(emb)
            logger.info(f"Saved embedding for case {case_id} (dim={len(vector)})")

    def search_similar(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """
        Find most similar cases by cosine similarity.
        Works with JSON-stored vectors (SQLite) and pgvector (PostgreSQL).
        """
        with get_db() as db:
            embeddings = db.query(CaseEmbedding).filter(
                CaseEmbedding.embedding_vector.isnot(None)
            ).all()

            if not embeddings:
                return []

            # Compute cosine similarity in Python (works with any DB)
            scored = []
            for emb in embeddings:
                vec = emb.embedding_vector
                if not vec or len(vec) != len(query_vector):
                    continue
                sim = self._cosine_similarity(query_vector, vec)
                scored.append((emb.case_id, sim))

            # Sort by similarity (highest first)
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:top_k]

            # Fetch full case data
            results = []
            for case_id, score in top:
                case = db.query(CaseRecord).filter_by(case_id=case_id).first()
                if case:
                    result = case.raw_json.copy() if case.raw_json else {}
                    result["similarity_score"] = round(score, 4)
                    results.append(result)

            return results

    def get_case_text_for_embedding(self, case_id: str) -> str:
        """Get the text representation of a case for embedding."""
        with get_db() as db:
            case = db.query(CaseRecord).filter_by(case_id=case_id).first()
            if case:
                return case.to_summary_text()
            return ""

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
