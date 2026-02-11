"""
Use Case: Submit Feedback

Registra ground truth (verdade) para um caso já analisado.
Essencial para re-treino e cálculo de métricas em produção.
"""

from dataclasses import dataclass


@dataclass
class FeedbackInput:
    """Input do feedback."""
    case_id: str
    ground_truth_label: str     # "BONA_FIDE", "FORGED"
    attack_type: str | None = None
    reviewer_id: str | None = None
    notes: str | None = None


class SubmitFeedbackUseCase:
    """
    Use Case: registra feedback humano sobre um caso.
    Permite calcular métricas reais (precision, recall)
    e alimentar pipelines de re-treino.
    """

    def execute(self, feedback: FeedbackInput) -> dict:
        """Registra feedback e retorna confirmação."""
        # TODO: persistir feedback no banco
        raise NotImplementedError
