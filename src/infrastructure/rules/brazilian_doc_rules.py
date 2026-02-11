"""
Adapter: Brazilian Document Rules Engine — Implementação COMPLETA.

Motor de regras determinísticas para documentos brasileiros.
Cada regra é uma função pura — fácil de adicionar/remover/testar.
"""

import re
from datetime import datetime, date

from src.core.interfaces.rules_engine import IRulesEngine, RulesResult, RuleViolation
from src.core.interfaces.ocr_engine import OCRResult


class BrazilianDocRulesEngine(IRulesEngine):
    """
    Motor de regras para documentos brasileiros.

    Regras implementadas:
        1. CPF — dígitos verificadores (mod 11)
        2. Data nascimento — formato + plausibilidade
        3. Data emissão — posterior ao nascimento
        4. Campos obrigatórios — nome, CPF ou RG
        5. Confiança OCR — média mínima
        6. Nome — caracteres válidos
        7. Idade — plausível (0-130 anos)
        8. Cross-field — emissão vs nascimento
    """

    RULES_VERSION = "1.0.0"

    def __init__(self, rules_version: str | None = None):
        self._rules_version = rules_version or self.RULES_VERSION

    def apply(self, ocr_result: OCRResult, doc_type: str | None = None) -> RulesResult:
        """Aplica todas as regras sobre os campos do OCR."""
        violations: list[RuleViolation] = []
        fields_map = {f.name: f.value for f in ocr_result.fields}
        conf_map = {f.name: f.confidence for f in ocr_result.fields}

        # Lista de regras a aplicar (cada uma retorna violação ou None)
        rules = [
            self._rule_cpf_checksum(fields_map),
            self._rule_required_fields(fields_map),
            self._rule_date_format(fields_map),
            self._rule_date_plausibility(fields_map),
            self._rule_name_valid(fields_map),
            self._rule_age_plausible(fields_map),
            self._rule_emission_after_birth(fields_map),
            self._rule_ocr_confidence(ocr_result.avg_confidence, conf_map),
        ]

        for result in rules:
            if result is not None:
                if isinstance(result, list):
                    violations.extend(result)
                else:
                    violations.append(result)

        total = 8  # total de regras
        failed = len(violations)
        passed = total - failed

        risk_score = self._compute_risk_score(violations)
        risk_level = self._risk_level(risk_score)

        return RulesResult(
            rules_passed=passed,
            rules_failed=failed,
            rules_total=total,
            violations=violations,
            risk_score=round(risk_score, 3),
            risk_level=risk_level,
            rules_version=self._rules_version,
        )

    # ─── REGRAS ─────────────────────────────────────────────

    def _rule_cpf_checksum(self, fields: dict) -> RuleViolation | None:
        """Regra 1: CPF com dígitos verificadores válidos."""
        cpf_raw = fields.get("cpf")
        if not cpf_raw:
            return None  # Sem CPF → regra não se aplica aqui

        digits = re.sub(r"\D", "", cpf_raw)
        if len(digits) != 11:
            return RuleViolation(
                rule_id="CPF_LENGTH",
                rule_name="CPF deve ter 11 dígitos",
                severity="HIGH",
                detail=f"CPF encontrado tem {len(digits)} dígitos: {cpf_raw}",
            )

        # Rejeita CPFs com todos os dígitos iguais
        if digits == digits[0] * 11:
            return RuleViolation(
                rule_id="CPF_ALL_SAME",
                rule_name="CPF com dígitos todos iguais",
                severity="CRITICAL",
                detail=f"CPF inválido (todos iguais): {cpf_raw}",
            )

        # Calcula dígitos verificadores
        if not self._validate_cpf_digits(digits):
            return RuleViolation(
                rule_id="CPF_CHECKSUM",
                rule_name="Dígitos verificadores do CPF inválidos",
                severity="CRITICAL",
                detail=f"CPF não passa na validação mod-11: {cpf_raw}",
            )

        return None

    def _rule_required_fields(self, fields: dict) -> list[RuleViolation]:
        """Regra 2: Campos obrigatórios presentes."""
        violations = []
        has_id = "cpf" in fields or "rg" in fields

        if not has_id:
            violations.append(RuleViolation(
                rule_id="MISSING_ID_FIELD",
                rule_name="Campo de identificação ausente",
                severity="HIGH",
                detail="Nem CPF nem RG foram detectados no documento",
            ))

        if "nome" not in fields:
            violations.append(RuleViolation(
                rule_id="MISSING_NAME",
                rule_name="Nome não detectado",
                severity="MEDIUM",
                detail="O campo 'nome' não foi encontrado pelo OCR",
            ))

        return violations if violations else None

    def _rule_date_format(self, fields: dict) -> RuleViolation | None:
        """Regra 3: Datas em formato válido."""
        for key, value in fields.items():
            if "data" in key:
                parsed = self._parse_date(value)
                if parsed is None:
                    return RuleViolation(
                        rule_id="INVALID_DATE_FORMAT",
                        rule_name="Formato de data inválido",
                        severity="MEDIUM",
                        detail=f"Campo '{key}' com formato inválido: {value}",
                    )
        return None

    def _rule_date_plausibility(self, fields: dict) -> RuleViolation | None:
        """Regra 4: Data de nascimento plausível."""
        dn = fields.get("data_nascimento")
        if not dn:
            return None

        parsed = self._parse_date(dn)
        if parsed is None:
            return None  # Já coberto pela regra de formato

        today = date.today()
        if parsed > today:
            return RuleViolation(
                rule_id="FUTURE_BIRTH_DATE",
                rule_name="Data de nascimento no futuro",
                severity="CRITICAL",
                detail=f"Data de nascimento impossível: {dn}",
            )

        if parsed.year < 1900:
            return RuleViolation(
                rule_id="ANCIENT_BIRTH_DATE",
                rule_name="Data de nascimento muito antiga",
                severity="HIGH",
                detail=f"Data anterior a 1900: {dn}",
            )

        return None

    def _rule_name_valid(self, fields: dict) -> RuleViolation | None:
        """Regra 5: Nome contém apenas caracteres válidos."""
        nome = fields.get("nome")
        if not nome:
            return None

        # Nome deve ter pelo menos 2 palavras e só letras/espaços/acentos
        if not re.match(r"^[A-ZÀ-Ü\s\.\-']+$", nome.upper()):
            return RuleViolation(
                rule_id="INVALID_NAME_CHARS",
                rule_name="Nome com caracteres inválidos",
                severity="MEDIUM",
                detail=f"Nome contém caracteres inesperados: {nome}",
            )

        words = nome.strip().split()
        if len(words) < 2:
            return RuleViolation(
                rule_id="NAME_TOO_SHORT",
                rule_name="Nome incompleto",
                severity="LOW",
                detail=f"Nome com menos de 2 palavras: {nome}",
            )

        return None

    def _rule_age_plausible(self, fields: dict) -> RuleViolation | None:
        """Regra 6: Idade derivada é plausível (0-130)."""
        dn = fields.get("data_nascimento")
        if not dn:
            return None

        parsed = self._parse_date(dn)
        if parsed is None:
            return None

        age = (date.today() - parsed).days // 365
        if age < 0 or age > 130:
            return RuleViolation(
                rule_id="IMPLAUSIBLE_AGE",
                rule_name="Idade implausível",
                severity="HIGH",
                detail=f"Idade calculada: {age} anos (data: {dn})",
            )

        return None

    def _rule_emission_after_birth(self, fields: dict) -> RuleViolation | None:
        """Regra 7: Data de emissão ≥ data de nascimento."""
        dn = fields.get("data_nascimento")
        de = fields.get("data_1") or fields.get("data_emissao")
        if not dn or not de:
            return None

        parsed_dn = self._parse_date(dn)
        parsed_de = self._parse_date(de)
        if parsed_dn is None or parsed_de is None:
            return None

        if parsed_de < parsed_dn:
            return RuleViolation(
                rule_id="EMISSION_BEFORE_BIRTH",
                rule_name="Data de emissão anterior ao nascimento",
                severity="CRITICAL",
                detail=f"Emissão {de} é anterior ao nascimento {dn}",
            )

        return None

    def _rule_ocr_confidence(self, avg_conf: float, conf_map: dict) -> RuleViolation | None:
        """Regra 8: Confiança do OCR acima do mínimo."""
        if avg_conf < 0.5:
            return RuleViolation(
                rule_id="LOW_OCR_CONFIDENCE",
                rule_name="Confiança do OCR muito baixa",
                severity="MEDIUM",
                detail=f"Confiança média: {avg_conf:.2f} (mínimo: 0.50)",
            )
        return None

    # ─── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _validate_cpf_digits(digits: str) -> bool:
        """Valida últimos 2 dígitos do CPF (algoritmo mod-11)."""
        nums = [int(d) for d in digits]

        # Primeiro dígito verificador
        weights_1 = list(range(10, 1, -1))
        sum_1 = sum(n * w for n, w in zip(nums[:9], weights_1))
        d1 = 11 - (sum_1 % 11)
        d1 = 0 if d1 >= 10 else d1

        # Segundo dígito verificador
        weights_2 = list(range(11, 1, -1))
        sum_2 = sum(n * w for n, w in zip(nums[:10], weights_2))
        d2 = 11 - (sum_2 % 11)
        d2 = 0 if d2 >= 10 else d2

        return nums[9] == d1 and nums[10] == d2

    @staticmethod
    def _parse_date(date_str: str) -> date | None:
        """Tenta parsear data em formatos brasileiros comuns."""
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        return None

    @staticmethod
    def _compute_risk_score(violations: list[RuleViolation]) -> float:
        """Calcula score de risco baseado nas violações."""
        if not violations:
            return 0.0

        severity_weights = {
            "LOW": 0.1,
            "MEDIUM": 0.25,
            "HIGH": 0.5,
            "CRITICAL": 1.0,
        }

        total_weight = sum(
            severity_weights.get(v.severity, 0.25) for v in violations
        )
        return min(total_weight, 1.0)

    @staticmethod
    def _risk_level(score: float) -> str:
        """Converte score numérico em nível textual."""
        if score < 0.2:
            return "LOW"
        elif score < 0.5:
            return "MEDIUM"
        elif score < 0.8:
            return "HIGH"
        return "CRITICAL"
