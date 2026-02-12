"""Test LLM fraud analyzer with real and fraudulent passport data."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from src.infrastructure.llm.llm_analyzer import LLMFraudAnalyzer
from src.infrastructure.rules.passport_rules import PassportRulesEngine

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not set in .env")
    sys.exit(1)

analyzer = LLMFraudAnalyzer(api_key=api_key, model_name="gemini-2.0-flash")
engine = PassportRulesEngine()

# ── Real passport (should be APPROVED) ──
real_passport = {
    "mrz_upper_line": "PCAZEKALKAN<<FIMAR<<<<<<<<<<<<<<<<<<<<<<<<<<",
    "mrz_lower_line": "C092555921AZE5910058F261123929108E0<<<<<<<08",
    "primary_identifier": "KALKAN",
    "secondary_identifier": "FIMAR",
    "document_number": "C09255592",
    "date_of_birth": "05.10.1959",
    "date_of_expiry": "23.11.2026",
    "nationality": "AZE",
    "sex": "F",
    "issuing_state_code": "AZE",
}

# ── Fraudulent passport (name tampered + digit swap) ──
fraud_passport = {
    "mrz_upper_line": "PCAZEKALKAN<<FIMAR<<<<<<<<<<<<<<<<<<<<<<<<<<",
    "mrz_lower_line": "C002555921AZE5910058F261123929108E0<<<<<<<08",  # changed digit
    "primary_identifier": "SMITH",  # doesn't match MRZ
    "secondary_identifier": "FIMAR",
    "document_number": "C00255592",  # changed
    "date_of_birth": "05.10.1959",
    "date_of_expiry": "23.11.2026",
    "nationality": "AZE",
    "sex": "M",  # doesn't match MRZ (F)
    "issuing_state_code": "AZE",
}

print("=" * 70)
print("  LLM Fraud Analyzer — Gemini Test")
print("=" * 70)

for label, data in [("REAL PASSPORT", real_passport), ("FRAUD PASSPORT", fraud_passport)]:
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")

    # Run rules engine first
    rules_result = engine.apply(data)
    violations = [
        {"rule_id": v.rule_id, "rule_name": v.rule_name,
         "severity": v.severity, "detail": v.detail}
        for v in rules_result.violations
    ]

    print(f"  Rules: {rules_result.rules_passed}/{rules_result.rules_total} passed | "
          f"score={rules_result.risk_score} | level={rules_result.risk_level}")
    for v in rules_result.violations:
        print(f"    ⚠️ [{v.severity}] {v.detail}")

    # Run LLM
    print(f"\n  Calling Gemini ({analyzer.model_name})...")
    result = analyzer.analyze(
        ocr_fields=data,
        rules_violations=violations,
        risk_score=rules_result.risk_score,
        risk_level=rules_result.risk_level,
    )

    if result.error:
        print(f"  ❌ Error: {result.error}")
    else:
        print(f"  ✅ Response in {result.latency_ms:.0f}ms")
        print(f"  Fraud probability: {result.fraud_probability:.1%}")
        print(f"  Risk level: {result.risk_level}")
        print(f"  Recommendation: {result.recommendation}")
        print(f"  Assessment: {result.assessment}")
        if result.anomalies:
            print(f"  Anomalies:")
            for a in result.anomalies:
                print(f"    • {a}")
        print(f"  Reasoning: {result.reasoning[:200]}...")

print(f"\n{'='*70}")
print("DONE")
