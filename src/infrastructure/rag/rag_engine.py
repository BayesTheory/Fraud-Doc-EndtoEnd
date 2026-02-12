"""
RAG Chat Engine — Retrieval-Augmented Generation.

Flow:
  1. User asks a question
  2. Embed the question with Gemini
  3. Search similar cases in vector DB
  4. Build context from top-K similar cases
  5. Send question + context to Gemini LLM
  6. Return answer
"""

import time
import json
import logging
from typing import Optional

from src.infrastructure.db.repository import CaseRepository
from src.infrastructure.embeddings.gemini_embeddings import GeminiEmbeddingService

logger = logging.getLogger(__name__)


class RAGChatEngine:
    """RAG-powered chat for fraud analysis."""

    SYSTEM_PROMPT = """You are a senior fraud analysis AI assistant for the Fraud-Doc Pipeline.
You help analysts understand document fraud detection results and patterns.

You have access to a VECTOR DATABASE of past document analysis cases.
When provided with similar cases from the database, use them to give context-aware answers.

Your capabilities:
- Analyze patterns across multiple fraud cases
- Explain why documents were APPROVED, REJECTED, or sent to REVIEW
- Identify common fraud indicators (checksum failures, field mismatches, etc.)
- Compare cases and find similarities
- Provide statistical insights about fraud detection accuracy

Respond in the same language the user writes (Portuguese or English).
Be concise but thorough. Use bullet points and structured formatting.
When referencing specific cases, mention their case_id."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.repository = CaseRepository()
        self.embedding_service = GeminiEmbeddingService(api_key)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def chat(self, message: str, context_case_ids: list[str] = None) -> dict:
        """
        RAG-enhanced chat.

        Returns: {reply, model, latency_ms, rag_context_cases}
        """
        t0 = time.perf_counter()

        # ── Step 1: Embed the user's question ──
        query_vector = self.embedding_service.embed_text(message)

        # ── Step 2: Search similar cases ──
        similar_cases = []
        if query_vector:
            similar_cases = self.repository.search_similar(query_vector, top_k=5)
            logger.info(f"RAG found {len(similar_cases)} similar cases")

        # ── Step 3: Add explicit context cases ──
        explicit_cases = []
        if context_case_ids:
            for cid in context_case_ids:
                case = self.repository.get_by_id(cid)
                if case:
                    explicit_cases.append(case)

        # ── Step 4: Build context ──
        context_parts = []

        if similar_cases:
            context_parts.append("=== SIMILAR CASES FROM VECTOR DATABASE (RAG) ===")
            for i, case in enumerate(similar_cases, 1):
                sim_score = case.get("similarity_score", 0)
                summary = self._case_summary(case)
                context_parts.append(f"\n--- Similar Case #{i} (similarity: {sim_score:.3f}) ---\n{summary}")

        if explicit_cases:
            context_parts.append("\n=== EXPLICITLY REFERENCED CASES ===")
            for case in explicit_cases:
                context_parts.append(self._case_summary(case))

        # If no RAG results, use recent cases
        if not similar_cases and not explicit_cases:
            recent = self.repository.list_cases(limit=5)
            if recent.get("cases"):
                context_parts.append("=== RECENT CASES (no RAG match) ===")
                for case in recent["cases"]:
                    context_parts.append(self._case_summary(case))

        # Stats
        stats = self.repository.get_stats()
        context_parts.append(f"\n=== DATABASE STATS ===\nTotal cases: {stats['total']}, "
                           f"Approved: {stats['approved']}, Rejected: {stats['rejected']}, "
                           f"Review: {stats['review']}, Avg Score: {stats['avg_score']}")

        # ── Step 5: Call LLM ──
        context_text = "\n".join(context_parts) if context_parts else "No cases in database yet."

        user_prompt = f"""CONTEXT FROM RAG VECTOR DATABASE:
{context_text}

USER QUESTION: {message}"""

        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.model,
                contents=[self.SYSTEM_PROMPT, user_prompt],
            )
            latency = (time.perf_counter() - t0) * 1000

            return {
                "reply": response.text,
                "model": self.model,
                "latency_ms": round(latency, 1),
                "rag_cases_found": len(similar_cases),
                "rag_case_ids": [c.get("case_id") for c in similar_cases],
            }

        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            logger.error(f"RAG chat failed: {e}")
            return {
                "reply": f"Error: {e}",
                "model": "error",
                "latency_ms": round(latency, 1),
                "rag_cases_found": 0,
                "rag_case_ids": [],
            }

    def embed_case(self, case_id: str) -> bool:
        """Generate and store embedding for a case."""
        text = self.repository.get_case_text_for_embedding(case_id)
        if not text:
            logger.warning(f"No text to embed for case {case_id}")
            return False

        vector = self.embedding_service.embed_text(text)
        if not vector:
            logger.warning(f"Embedding generation failed for case {case_id}")
            return False

        self.repository.save_embedding(case_id, vector, model=self.embedding_service.MODEL)
        logger.info(f"Embedded case {case_id} (dim={len(vector)})")
        return True

    def embed_all_cases(self) -> int:
        """Embed all cases that don't have embeddings yet."""
        from src.infrastructure.db.database import get_db
        from src.infrastructure.db.models import CaseRecord, CaseEmbedding

        count = 0
        with get_db() as db:
            # Find cases without embeddings
            cases_with_emb = db.query(CaseEmbedding.case_id).subquery()
            unembedded = db.query(CaseRecord.case_id).filter(
                ~CaseRecord.case_id.in_(db.query(cases_with_emb.c.case_id))
            ).all()

        for (case_id,) in unembedded:
            if self.embed_case(case_id):
                count += 1

        logger.info(f"Embedded {count} cases")
        return count

    @staticmethod
    def _case_summary(case: dict) -> str:
        """Create a compact text summary of a case for the LLM prompt."""
        parts = [f"Case ID: {case.get('case_id', 'unknown')}"]
        parts.append(f"Decision: {case.get('final_decision', '?')} (score: {case.get('final_score', 0):.2f})")

        ocr = case.get("ocr") or {}
        fields = ocr.get("fields", [])
        if fields:
            for f in fields[:8]:
                if isinstance(f, dict):
                    parts.append(f"  {f.get('name')}: {f.get('value')}")

        rules = case.get("rules") or {}
        if rules.get("violations"):
            for v in rules["violations"][:3]:
                if isinstance(v, dict):
                    parts.append(f"  ⚠ [{v.get('severity')}] {v.get('rule_name')}: {v.get('detail')}")
        else:
            parts.append(f"  Rules: {rules.get('rules_passed', 0)}/{rules.get('rules_total', 0)} passed")

        llm = case.get("llm") or {}
        if llm.get("assessment"):
            parts.append(f"  AI: {llm['assessment'][:200]}")

        return "\n".join(parts)
