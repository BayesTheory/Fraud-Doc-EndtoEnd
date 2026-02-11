"""
Use Case: Analyze Document — Implementação COMPLETA.

Orquestra: Quality Gate → OCR → Rules → (Fraud) → Resultado
Mede latência de cada etapa.
"""

import time
import uuid

from src.core.interfaces.quality_gate import IQualityGate
from src.core.interfaces.ocr_engine import IOCREngine
from src.core.interfaces.rules_engine import IRulesEngine
from src.core.interfaces.fraud_classifier import IFraudClassifier
from src.core.interfaces.embedding_service import IEmbeddingService
from src.core.interfaces.storage_service import IStorageService
from src.core.entities.analysis_result import AnalysisResult


class AnalyzeDocumentUseCase:
    """
    Use Case: recebe imagem → roda pipeline → retorna resultado.

    Dependency Injection: todas as dependências vêm pelo construtor.
    Dependências opcionais (fraud, embeddings, storage) podem ser None.
    """

    PIPELINE_VERSION = "0.1.0"

    def __init__(
        self,
        quality_gate: IQualityGate,
        ocr_engine: IOCREngine,
        rules_engine: IRulesEngine,
        fraud_classifier: IFraudClassifier | None = None,
        embedding_service: IEmbeddingService | None = None,
        storage_service: IStorageService | None = None,
    ):
        self._quality = quality_gate
        self._ocr = ocr_engine
        self._rules = rules_engine
        self._fraud = fraud_classifier
        self._embeddings = embedding_service
        self._storage = storage_service

    def execute(self, image_bytes: bytes, case_id: str | None = None) -> AnalysisResult:
        """
        Executa o pipeline completo.

        1. Quality gate — se falhar, retorna RECAPTURE
        2. OCR — extrai campos
        3. Regras de negócio — valida campos
        4. [Opcional] Classificador de fraude
        5. Consolida resultado
        """
        cid = case_id or str(uuid.uuid4())
        stage_latencies: dict[str, float] = {}
        t_start = time.perf_counter()

        result = AnalysisResult(case_id=cid, pipeline_version=self.PIPELINE_VERSION)

        # ── 1. Quality Gate ────────────────────────────────
        t0 = time.perf_counter()
        quality = self._quality.evaluate(image_bytes)
        stage_latencies["quality_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result.quality = quality

        if not quality.quality_ok:
            result.final_decision = "REJECTED"
            result.final_score = quality.quality_score
            result.rejection_reasons = quality.reasons
            result.stage_latencies = stage_latencies
            result.total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return result

        # ── 2. OCR ─────────────────────────────────────────
        t0 = time.perf_counter()
        ocr = self._ocr.extract(image_bytes)
        stage_latencies["ocr_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result.ocr = ocr

        # ── 3. Regras de Negócio ───────────────────────────
        t0 = time.perf_counter()
        rules = self._rules.apply(ocr, doc_type=ocr.doc_type_detected)
        stage_latencies["rules_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result.rules = rules

        # ── 4. Fraud Classifier (Opcional) ─────────────────
        if self._fraud is not None:
            t0 = time.perf_counter()
            try:
                fraud = self._fraud.classify(image_bytes)
                stage_latencies["fraud_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                result.fraud = fraud
            except NotImplementedError:
                stage_latencies["fraud_ms"] = 0.0

        # ── 5. Decisão Final ───────────────────────────────
        result.rejection_reasons = []

        # Quality
        if not quality.quality_ok:
            result.rejection_reasons.extend(quality.reasons)

        # Rules
        critical_violations = [
            v for v in rules.violations if v.severity in ("HIGH", "CRITICAL")
        ]
        if critical_violations:
            result.rejection_reasons.extend([v.rule_id for v in critical_violations])

        # Fraud
        if result.fraud and result.fraud.fraud_score > result.fraud.threshold_used:
            result.rejection_reasons.append("FRAUD_DETECTED")

        # Decisão
        if not result.rejection_reasons:
            result.final_decision = "APPROVED"
            result.final_score = 1.0 - rules.risk_score
        elif any(r in ("FRAUD_DETECTED",) for r in result.rejection_reasons):
            result.final_decision = "REJECTED"
            result.final_score = rules.risk_score
        elif len(critical_violations) > 0:
            result.final_decision = "REJECTED"
            result.final_score = rules.risk_score
        else:
            result.final_decision = "REVIEW"
            result.final_score = rules.risk_score

        result.stage_latencies = stage_latencies
        result.total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return result
