"""
LLM Fraud Analyzer — Gemini-powered semantic analysis.

Receives OCR fields + rules violations and produces a human-readable
fraud assessment with structured JSON output.

Uses the new `google-genai` SDK (not deprecated `google-generativeai`).
"""
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from google import genai


@dataclass
class LLMAnalysis:
    """Structured output from LLM fraud analysis."""
    fraud_probability: float = 0.0          # 0.0 to 1.0
    risk_level: str = "LOW"                 # LOW, MEDIUM, HIGH, CRITICAL
    assessment: str = ""                    # Human-readable summary
    anomalies: List[str] = field(default_factory=list)
    recommendation: str = "APPROVE"         # APPROVE, REVIEW, REJECT
    reasoning: str = ""                     # Step-by-step reasoning
    latency_ms: float = 0.0
    model: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM_PROMPT = """You are a document fraud detection expert. You analyze passport OCR data and rule engine results to detect potential fraud.

IMPORTANT: Respond ONLY with a JSON object, no markdown, no backticks, no extra text.

Input: OCR-extracted fields from a passport and results from a deterministic rules engine.

Your job:
1. Analyze the OCR fields for semantic inconsistencies
2. Cross-reference the rules engine violations
3. Look for patterns that rules can't catch (semantic anomalies)
4. Provide a fraud probability score

Output JSON format:
{
    "fraud_probability": 0.0 to 1.0,
    "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
    "assessment": "Brief human-readable summary",
    "anomalies": ["list of specific anomalies found"],
    "recommendation": "APPROVE" | "REVIEW" | "REJECT",
    "reasoning": "Step-by-step analysis reasoning"
}

Rules:
- If rules engine found NO violations and fields look consistent → fraud_probability < 0.2, APPROVE
- If rules engine found checksum violations → fraud_probability > 0.7, likely REJECT
- If semantic inconsistencies → fraud_probability 0.3-0.7, REVIEW
- Always explain your reasoning
- Consider: name formatting, date consistency, country codes, document patterns
"""


class LLMFraudAnalyzer:
    """Gemini-powered fraud analysis for passport documents."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def analyze(
        self,
        ocr_fields: Dict[str, str],
        rules_violations: List[Dict] = None,
        risk_score: float = 0.0,
        risk_level: str = "LOW",
    ) -> LLMAnalysis:
        """
        Analyze passport data for fraud using Gemini.

        Args:
            ocr_fields: Dict of field_name → extracted_text
            rules_violations: List of rule violation dicts
            risk_score: Current risk score from rules engine
            risk_level: Current risk level from rules engine

        Returns:
            LLMAnalysis with fraud assessment
        """
        t0 = time.perf_counter()

        # Build the prompt
        user_prompt = self._build_prompt(ocr_fields, rules_violations, risk_score, risk_level)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[SYSTEM_PROMPT + "\n\n" + user_prompt],
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 1024,
                },
            )

            # Parse JSON response
            raw = response.text.strip()
            # Handle markdown code blocks
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:])  # remove first line
                if raw.rstrip().endswith("```"):
                    raw = raw.rstrip()[:-3]
                raw = raw.strip()

            data = json.loads(raw)
            latency = (time.perf_counter() - t0) * 1000

            return LLMAnalysis(
                fraud_probability=float(data.get("fraud_probability", 0)),
                risk_level=data.get("risk_level", "LOW"),
                assessment=data.get("assessment", ""),
                anomalies=data.get("anomalies", []),
                recommendation=data.get("recommendation", "APPROVE"),
                reasoning=data.get("reasoning", ""),
                latency_ms=round(latency, 1),
                model=self.model_name,
            )

        except json.JSONDecodeError as e:
            latency = (time.perf_counter() - t0) * 1000
            return LLMAnalysis(
                error=f"JSON parse error: {e}. Raw: {raw[:200]}",
                latency_ms=round(latency, 1),
                model=self.model_name,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            return LLMAnalysis(
                error=f"LLM error: {e}",
                latency_ms=round(latency, 1),
                model=self.model_name,
            )

    def _build_prompt(
        self,
        ocr_fields: Dict[str, str],
        rules_violations: List[Dict],
        risk_score: float,
        risk_level: str,
    ) -> str:
        """Build the user prompt with document data."""
        parts = ["Analyze this passport for potential fraud:\n"]

        # OCR fields
        parts.append("## OCR Extracted Fields")
        for name, value in sorted(ocr_fields.items()):
            if value and value not in ("[BBOX_PRESENT]", ""):
                parts.append(f"  {name}: {value}")

        # Rules result
        parts.append(f"\n## Rules Engine Result")
        parts.append(f"  Risk Score: {risk_score}")
        parts.append(f"  Risk Level: {risk_level}")

        if rules_violations:
            parts.append(f"  Violations ({len(rules_violations)}):")
            for v in rules_violations:
                parts.append(f"    - [{v.get('severity', '?')}] {v.get('rule_name', '?')}: {v.get('detail', '?')}")
        else:
            parts.append("  Violations: None (all rules passed)")

        return "\n".join(parts)
