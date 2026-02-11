"""
Contract: Rules Engine

Aplica regras de negócio determinísticas sobre os campos
extraídos pelo OCR. Valida formato, consistência e layout.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from src.core.interfaces.ocr_engine import OCRResult


@dataclass
class RuleViolation:
    """Uma violação de regra detectada."""
    rule_id: str              # ex: "CPF_CHECKSUM"
    rule_name: str            # ex: "Validação de dígitos verificadores do CPF"
    severity: str             # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    detail: str               # ex: "Dígito verificador não bate: esperado 09, encontrado 11"


@dataclass
class RulesResult:
    """Resultado da aplicação de regras de negócio."""
    rules_passed: int
    rules_failed: int
    rules_total: int
    violations: list[RuleViolation] = field(default_factory=list)
    risk_score: float = 0.0          # 0.0 (limpo) a 1.0 (alto risco)
    risk_level: str = "LOW"          # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    rules_version: str = ""


class IRulesEngine(ABC):
    """
    Port: Rules Engine

    Aplica um conjunto de regras determinísticas sobre o resultado
    do OCR para validar a conformidade e detectar inconsistências.
    """

    @abstractmethod
    def apply(self, ocr_result: OCRResult, doc_type: str | None = None) -> RulesResult:
        """
        Aplica regras de negócio sobre os campos extraídos.

        Args:
            ocr_result: Resultado do OCR (campos + confiança).
            doc_type: Tipo do documento (afeta quais regras aplicar).

        Returns:
            RulesResult com violações, score e nível de risco.
        """
        ...
