"""
Use Case: Analyze Document

Orquestra o pipeline completo:
  Quality Gate → OCR → Rules → Fraud → Embedding → Persist

Esta é a lógica de aplicação — depende apenas dos contratos
(interfaces), nunca de implementações concretas.
"""

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

    Dependency Injection: todas as dependências vêm pelo construtor
    como interfaces. Quem instancia escolhe as implementações.
    """

    def __init__(
        self,
        quality_gate: IQualityGate,
        ocr_engine: IOCREngine,
        rules_engine: IRulesEngine,
        fraud_classifier: IFraudClassifier,
        embedding_service: IEmbeddingService,
        storage_service: IStorageService,
    ):
        self._quality = quality_gate
        self._ocr = ocr_engine
        self._rules = rules_engine
        self._fraud = fraud_classifier
        self._embeddings = embedding_service
        self._storage = storage_service

    def execute(self, case_id: str, image_bytes: bytes) -> AnalysisResult:
        """
        Executa o pipeline completo de análise.

        Fluxo:
            1. Salva imagem no storage
            2. Quality gate — se falhar, retorna RECAPTURE
            3. OCR — extrai campos
            4. Regras de negócio — valida campos
            5. Classificador de fraude — score de autenticidade
            6. Gera e armazena embedding vetorial
            7. Consolida resultado final
        """
        # TODO: Implementar orquestração com timing por etapa
        raise NotImplementedError("Pipeline orchestration not yet implemented")
