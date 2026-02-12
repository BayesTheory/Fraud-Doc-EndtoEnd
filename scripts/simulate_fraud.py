"""
Fraud Simulation — Generate 20 fraudulent passport variants and test rules engine.

Takes real OCR output from MIDV-2020, alters specific fields to simulate fraud,
then validates that the rules engine correctly detects each tampering.
"""
import os
import sys
import copy
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.rules.passport_rules import (
    PassportRulesEngine, mrz_check_digit, parse_mrz_td3
)

# ── Real passport data extracted via OCR (AZE passport_89) ──
REAL_AZE = {
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

REAL_GRC = {
    "mrz_upper_line": "P<GRCGRIGORIADOU<<POLYXENI<<<<<<<<<<<<<<<<<<",
    "mrz_lower_line": "AK27302336GRC9002177F2003038<<<<<<<<<<<<<<04",
    "primary_identifier": "GRIGORIADOU",
    "secondary_identifier": "POLYXENI",
    "document_number": "AK2730233",
    "date_of_birth": "17.02.1990",
    "date_of_expiry": "03.03.2020",
    "nationality": "GRC",
    "sex": "F",
    "issuing_state_code": "GRC",
}

REAL_SRB = {
    "mrz_upper_line": "P<SRBNASTASIC<<SIMO<<<<<<<<<<<<<<<<<<<<<<<<<",
    "mrz_lower_line": "1891858334SRB7512191M26011351912975804649<54",
    "primary_identifier": "NASTASIC",
    "secondary_identifier": "SIMO",
    "document_number": "189185833",
    "date_of_birth": "19.12.1975",
    "date_of_expiry": "13.01.2026",
    "nationality": "SRB",
    "sex": "M",
    "issuing_state_code": "SRB",
}


def build_mrz_line2(doc_num, nationality, dob_yymmdd, sex, doe_yymmdd, personal="<<<<<<<<<<<<<<"):
    """Build a valid MRZ line 2 with correct check digits."""
    doc_num = doc_num.ljust(9, "<")[:9]
    dc1 = str(mrz_check_digit(doc_num))
    nat = nationality.ljust(3, "<")[:3]
    dc2 = str(mrz_check_digit(dob_yymmdd))
    dc3 = str(mrz_check_digit(doe_yymmdd))
    personal = personal.ljust(14, "<")[:14]
    dc4 = str(mrz_check_digit(personal))

    line2_no_composite = doc_num + dc1 + nat + dob_yymmdd + dc2 + sex + doe_yymmdd + dc3 + personal + dc4
    composite_data = line2_no_composite[0:10] + line2_no_composite[13:20] + line2_no_composite[21:43]
    dc5 = str(mrz_check_digit(composite_data))

    return line2_no_composite + dc5


# ── Fraud generators ──

def fraud_mrz_digit_swap(base, name_suffix=""):
    """Tamper: change 1 digit in MRZ line 2 (breaks checksum)."""
    f = copy.deepcopy(base)
    line2 = list(f["mrz_lower_line"])
    # Find a digit position and change it
    digit_positions = [i for i, c in enumerate(line2) if c.isdigit() and i < 9]
    if digit_positions:
        pos = random.choice(digit_positions)
        old = line2[pos]
        new = str((int(old) + random.randint(1, 9)) % 10)
        line2[pos] = new
        f["mrz_lower_line"] = "".join(line2)
    f["_fraud_type"] = f"MRZ digit swap (pos {pos}: {old}→{new}){name_suffix}"
    f["_expected_rules"] = ["DOC_NUMBER_CHECK", "COMPOSITE_CHECK"]
    return f


def fraud_name_mismatch(base, name_suffix=""):
    """Tamper: change VIZ name but not MRZ (cross-check should fail)."""
    f = copy.deepcopy(base)
    f["primary_identifier"] = "SMITH"  # different from MRZ
    f["_fraud_type"] = f"Name mismatch VIZ≠MRZ{name_suffix}"
    f["_expected_rules"] = ["CROSS_CHECK"]
    return f


def fraud_dob_mismatch(base, name_suffix=""):
    """Tamper: change DOB in VIZ but not MRZ."""
    f = copy.deepcopy(base)
    f["date_of_birth"] = "01.01.2000"  # clearly different
    f["_fraud_type"] = f"DOB mismatch VIZ≠MRZ{name_suffix}"
    f["_expected_rules"] = ["CROSS_CHECK"]
    return f


def fraud_expired_doc(base, name_suffix=""):
    """Tamper: set expiry date to past (forged extension)."""
    f = copy.deepcopy(base)
    # Modify both VIZ and MRZ to have expired DOE
    f["date_of_expiry"] = "01.01.2020"
    # Rebuild MRZ with expired DOE
    line2 = f["mrz_lower_line"]
    doc_num = line2[0:9]
    nat = line2[10:13]
    dob = line2[13:19]
    sex = line2[20]
    personal = line2[28:42]
    f["mrz_lower_line"] = build_mrz_line2(doc_num, nat, dob, sex, "200101", personal)
    f["_fraud_type"] = f"Expired document{name_suffix}"
    f["_expected_rules"] = ["DATE_PLAUSIBILITY"]
    return f


def fraud_invalid_country(base, name_suffix=""):
    """Tamper: replace nationality with invalid code."""
    f = copy.deepcopy(base)
    line2 = list(f["mrz_lower_line"])
    line2[10:13] = list("XXX")
    f["mrz_lower_line"] = "".join(line2)
    f["nationality"] = "XXX"
    f["_fraud_type"] = f"Invalid country code{name_suffix}"
    f["_expected_rules"] = ["COUNTRY_CODE", "COMPOSITE_CHECK"]
    return f


def fraud_missing_field(base, name_suffix=""):
    """Tamper: remove required field."""
    f = copy.deepcopy(base)
    f.pop("document_number", None)
    f["_fraud_type"] = f"Missing document_number{name_suffix}"
    f["_expected_rules"] = ["REQUIRED_FIELDS"]
    return f


def fraud_sex_mismatch(base, name_suffix=""):
    """Tamper: change sex in VIZ but not MRZ."""
    f = copy.deepcopy(base)
    f["sex"] = "M" if base.get("sex") == "F" else "F"
    f["_fraud_type"] = f"Sex mismatch VIZ≠MRZ{name_suffix}"
    f["_expected_rules"] = ["CROSS_CHECK"]
    return f


def fraud_future_dob(base, name_suffix=""):
    """Tamper: DOB in the future (impossible)."""
    f = copy.deepcopy(base)
    line2 = f["mrz_lower_line"]
    doc_num = line2[0:9]
    nat = line2[10:13]
    sex = line2[20]
    doe = line2[21:27]
    personal = line2[28:42]
    f["mrz_lower_line"] = build_mrz_line2(doc_num, nat, "350101", sex, doe, personal)
    f["date_of_birth"] = "01.01.2035"
    f["_fraud_type"] = f"Future DOB{name_suffix}"
    f["_expected_rules"] = ["DATE_PLAUSIBILITY"]
    return f


def fraud_all_empty_mrz(base, name_suffix=""):
    """Tamper: blank MRZ lines."""
    f = copy.deepcopy(base)
    f["mrz_upper_line"] = ""
    f["mrz_lower_line"] = ""
    f["_fraud_type"] = f"Empty MRZ lines{name_suffix}"
    f["_expected_rules"] = ["MRZ_FORMAT", "REQUIRED_FIELDS"]
    return f


def fraud_wrong_mrz_length(base, name_suffix=""):
    """Tamper: MRZ with wrong length."""
    f = copy.deepcopy(base)
    f["mrz_lower_line"] = f["mrz_lower_line"][:30]  # truncated
    f["_fraud_type"] = f"Truncated MRZ (30 chars){name_suffix}"
    f["_expected_rules"] = ["MRZ_FORMAT"]
    return f


def main():
    print("=" * 70)
    print("  Fraud Simulation — 20 Variants vs Rules Engine")
    print("=" * 70)

    engine = PassportRulesEngine()
    print(f"  Rules engine: {engine.RULES_VERSION} ({len(engine._rules)} rules)")

    # ── First: verify that REAL passports pass ──
    print(f"\n{'─'*70}")
    print("  BASELINE: Real passports (should PASS)")
    print(f"{'─'*70}")

    for label, data in [("AZE", REAL_AZE), ("GRC", REAL_GRC), ("SRB", REAL_SRB)]:
        result = engine.apply(data)
        status = "✅ PASS" if result.risk_level == "LOW" else f"⚠️ {result.risk_level}"
        fails = [v.rule_id for v in result.violations]
        print(f"  {label}: {status} | score={result.risk_score} | "
              f"rules={result.rules_passed}/{result.rules_total} | "
              f"fails={fails if fails else 'none'}")

    # ── Generate 20 fraud variants ──
    print(f"\n{'─'*70}")
    print("  FRAUD: 20 tampered variants (should FAIL)")
    print(f"{'─'*70}")

    frauds = []
    bases = [("AZE", REAL_AZE), ("GRC", REAL_GRC), ("SRB", REAL_SRB)]

    # Generate fraud types across different passports
    generators = [
        fraud_mrz_digit_swap,
        fraud_name_mismatch,
        fraud_dob_mismatch,
        fraud_expired_doc,
        fraud_invalid_country,
        fraud_missing_field,
        fraud_sex_mismatch,
        fraud_future_dob,
        fraud_all_empty_mrz,
        fraud_wrong_mrz_length,
    ]

    random.seed(42)
    idx = 0
    while len(frauds) < 20:
        base_label, base_data = bases[idx % len(bases)]
        gen = generators[idx % len(generators)]
        fraud = gen(base_data, f" [{base_label}]")
        frauds.append(fraud)
        idx += 1

    # ── Test each fraud ──
    detected = 0
    missed = 0
    results_detail = []

    for i, fraud in enumerate(frauds):
        fraud_type = fraud.pop("_fraud_type")
        expected = fraud.pop("_expected_rules")

        result = engine.apply(fraud)
        triggered_rules = {v.rule_id for v in result.violations}

        # Check if ANY expected rule was triggered
        caught = bool(triggered_rules & set(expected))
        if caught:
            detected += 1
            emoji = "🚨"
        else:
            missed += 1
            emoji = "❌ MISSED"

        all_violations = [f"{v.rule_id}({v.severity[0]})" for v in result.violations]

        print(f"  {emoji} [{i+1:2d}/20] {fraud_type:45s} | "
              f"score={result.risk_score:.3f} | "
              f"triggered={all_violations if all_violations else 'NONE'}")

        results_detail.append({
            "fraud_type": fraud_type,
            "detected": caught,
            "risk_score": result.risk_score,
            "risk_level": result.risk_level,
            "expected": expected,
            "triggered": list(triggered_rules),
        })

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Total fraud variants:   20")
    print(f"  Detected (true pos):    {detected} ({detected/20*100:.0f}%)")
    print(f"  Missed (false neg):     {missed} ({missed/20*100:.0f}%)")
    print(f"  Detection rate:         {detected/20*100:.0f}%")
    print()

    # By fraud type
    type_stats = {}
    for r in results_detail:
        t = r["fraud_type"].split("[")[0].strip()
        if t not in type_stats:
            type_stats[t] = {"total": 0, "detected": 0}
        type_stats[t]["total"] += 1
        type_stats[t]["detected"] += int(r["detected"])

    print(f"  {'Fraud Type':40s} | {'Detected':10s}")
    print(f"  {'-'*40} | {'-'*10}")
    for t, s in type_stats.items():
        rate = f"{s['detected']}/{s['total']}"
        print(f"  {t:40s} | {rate:10s}")

    print(f"\n{'='*70}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
