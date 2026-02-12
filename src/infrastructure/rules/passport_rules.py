"""
Passport-specific Rules Engine.

Implements ICAO Doc 9303 (Machine Readable Travel Documents) validation:
- MRZ checksum verification (check digits)
- Cross-validation between VIZ (Visual Inspection Zone) and MRZ
- Date plausibility checks
- Required field presence
- Country code validation (ISO 3166-1 alpha-3)
"""
import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

from src.core.interfaces.ocr_engine import OCRResult
from src.core.interfaces.rules_engine import IRulesEngine, RulesResult, RuleViolation


# ── ICAO 9303 MRZ Character Weights ─────────────────────────────────
MRZ_CHAR_VALUES = {}
for i in range(10):
    MRZ_CHAR_VALUES[str(i)] = i
for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    MRZ_CHAR_VALUES[c] = 10 + i
MRZ_CHAR_VALUES["<"] = 0

MRZ_WEIGHTS = [7, 3, 1]

# Valid ISO 3166-1 alpha-3 country codes (subset)
VALID_COUNTRY_CODES = {
    "AFG", "ALB", "DZA", "AND", "AGO", "ARG", "ARM", "AUS", "AUT",
    "AZE", "BHS", "BHR", "BGD", "BRB", "BLR", "BEL", "BLZ", "BEN",
    "BTN", "BOL", "BIH", "BWA", "BRA", "BRN", "BGR", "BFA", "BDI",
    "KHM", "CMR", "CAN", "CPV", "CAF", "TCD", "CHL", "CHN", "COL",
    "COG", "CRI", "HRV", "CUB", "CYP", "CZE", "DNK", "DJI", "DOM",
    "ECU", "EGY", "SLV", "GNQ", "ERI", "EST", "ETH", "FIN", "FRA",
    "GAB", "GMB", "GEO", "DEU", "GHA", "GRC", "GTM", "GIN", "GUY",
    "HTI", "HND", "HUN", "ISL", "IND", "IDN", "IRN", "IRQ", "IRL",
    "ISR", "ITA", "JAM", "JPN", "JOR", "KAZ", "KEN", "KWT", "KGZ",
    "LAO", "LVA", "LBN", "LSO", "LBR", "LBY", "LIE", "LTU", "LUX",
    "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MRT", "MUS", "MEX",
    "MDA", "MCO", "MNG", "MNE", "MAR", "MOZ", "MMR", "NAM", "NPL",
    "NLD", "NZL", "NIC", "NER", "NGA", "NOR", "OMN", "PAK", "PAN",
    "PRY", "PER", "PHL", "POL", "PRT", "QAT", "ROU", "RUS", "RWA",
    "SAU", "SEN", "SRB", "SGP", "SVK", "SVN", "SOM", "ZAF", "KOR",
    "ESP", "LKA", "SDN", "SUR", "SWZ", "SWE", "CHE", "SYR", "TWN",
    "TJK", "TZA", "THA", "TGO", "TTO", "TUN", "TUR", "TKM", "UGA",
    "UKR", "ARE", "GBR", "USA", "URY", "UZB", "VEN", "VNM", "YEM",
    "ZMB", "ZWE", "UTO",  # UTO = ICAO test nationality
}


def mrz_check_digit(data: str) -> int:
    """Calculate ICAO 9303 check digit (mod-10 weighted sum)."""
    total = 0
    for i, char in enumerate(data):
        value = MRZ_CHAR_VALUES.get(char.upper(), 0)
        weight = MRZ_WEIGHTS[i % 3]
        total += value * weight
    return total % 10


def parse_mrz_date(date_str: str) -> Optional[date]:
    """Parse MRZ date YYMMDD → Python date. 00-29→2000s, 30-99→1900s."""
    if len(date_str) != 6 or not date_str.isdigit():
        return None
    try:
        yy, mm, dd = int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6])
        year = 2000 + yy if yy < 30 else 1900 + yy
        return date(year, mm, dd)
    except ValueError:
        return None


@dataclass
class MRZParsed:
    """Parsed TD3 MRZ (2 lines × 44 chars)."""
    document_code: str = ""
    issuing_country: str = ""
    primary_identifier: str = ""
    secondary_identifier: str = ""
    document_number: str = ""
    document_number_check: int = -1
    nationality: str = ""
    date_of_birth: str = ""
    dob_check: int = -1
    sex: str = ""
    date_of_expiry: str = ""
    doe_check: int = -1
    personal_number: str = ""
    personal_number_check: int = -1
    composite_check: int = -1
    raw_line1: str = ""
    raw_line2: str = ""
    is_valid_format: bool = False


def parse_mrz_td3(line1: str, line2: str) -> MRZParsed:
    """Parse TD3 passport MRZ."""
    result = MRZParsed(raw_line1=line1, raw_line2=line2)
    l1 = line1.strip().upper().replace(" ", "")
    l2 = line2.strip().upper().replace(" ", "")

    if len(l1) < 40 or len(l2) < 40:
        return result

    result.is_valid_format = True
    result.document_code = l1[0:2].replace("<", "")
    result.issuing_country = l1[2:5]

    names_section = l1[5:]
    name_parts = names_section.split("<<")
    if len(name_parts) >= 2:
        result.primary_identifier = name_parts[0].replace("<", " ").strip()
        result.secondary_identifier = name_parts[1].replace("<", " ").strip()
    elif len(name_parts) == 1:
        result.primary_identifier = name_parts[0].replace("<", " ").strip()

    result.document_number = l2[0:9].replace("<", "")
    result.document_number_check = int(l2[9]) if l2[9].isdigit() else -1
    result.nationality = l2[10:13]
    result.date_of_birth = l2[13:19]
    result.dob_check = int(l2[19]) if l2[19].isdigit() else -1
    result.sex = l2[20:21]
    result.date_of_expiry = l2[21:27]
    result.doe_check = int(l2[27]) if l2[27].isdigit() else -1
    result.personal_number = l2[28:42].replace("<", "")
    result.personal_number_check = int(l2[42]) if len(l2) > 42 and l2[42].isdigit() else -1
    result.composite_check = int(l2[43]) if len(l2) > 43 and l2[43].isdigit() else -1

    return result


class PassportRulesEngine(IRulesEngine):
    """
    ICAO 9303 passport validation — 10 rules:
    1. MRZ format (TD3)
    2. Document number check digit
    3. DOB check digit
    4. DOE check digit
    5. Personal number check digit
    6. Composite check digit
    7. Country code (ISO 3166)
    8. Date plausibility
    9. Required fields
    10. VIZ↔MRZ cross-check
    """

    RULES_VERSION = "passport-v1.0"

    def __init__(self):
        self._rules = [
            ("MRZ_FORMAT", "MRZ Format Validation", self._rule_mrz_format),
            ("DOC_NUM_CHECK", "Document Number Checksum", self._rule_doc_number_check),
            ("DOB_CHECK", "Date of Birth Checksum", self._rule_dob_check),
            ("DOE_CHECK", "Date of Expiry Checksum", self._rule_doe_check),
            ("PN_CHECK", "Personal Number Checksum", self._rule_personal_number_check),
            ("COMPOSITE_CHECK", "Composite Checksum", self._rule_composite_check),
            ("COUNTRY_CODE", "Country Code Validation", self._rule_country_code),
            ("DATE_PLAUSIBILITY", "Date Plausibility", self._rule_date_plausibility),
            ("REQUIRED_FIELDS", "Required Fields Presence", self._rule_required_fields),
            ("CROSS_CHECK", "VIZ ↔ MRZ Cross-Check", self._rule_cross_check),
        ]

    def apply(self, ocr_result: OCRResult, doc_type: str | None = None) -> RulesResult:
        """Apply all passport rules to OCR result."""
        # Convert OCRResult fields to dict
        if hasattr(ocr_result, 'fields') and isinstance(ocr_result.fields, list):
            # OCRResult.fields is a list of OCRField objects
            fields = {}
            for f in ocr_result.fields:
                name = f.name if hasattr(f, 'name') else str(f.get('name', ''))
                value = f.value if hasattr(f, 'value') else str(f.get('value', ''))
                fields[name] = value
        elif hasattr(ocr_result, 'extracted_fields'):
            fields = ocr_result.extracted_fields
        elif isinstance(ocr_result, dict):
            fields = ocr_result
        else:
            fields = {}

        # Parse MRZ
        mrz = None
        l1 = fields.get("mrz_upper_line", "")
        l2 = fields.get("mrz_lower_line", "")
        if l1 and l2:
            mrz = parse_mrz_td3(l1, l2)

        violations = []
        rules_failed_set = set()

        for rule_id, rule_name, rule_fn in self._rules:
            try:
                rule_violations = rule_fn(fields, mrz)
                for sev, detail in rule_violations:
                    violations.append(RuleViolation(
                        rule_id=rule_id,
                        rule_name=rule_name,
                        severity=sev,
                        detail=detail,
                    ))
                    rules_failed_set.add(rule_id)
            except Exception as e:
                violations.append(RuleViolation(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    severity="LOW",
                    detail=f"Rule execution error: {e}",
                ))

        rules_total = len(self._rules)
        rules_failed = len(rules_failed_set)
        rules_passed = rules_total - rules_failed

        # Risk score: weight by severity
        sev_weights = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0.5}
        total_weight = sum(sev_weights.get(v.severity, 0) for v in violations)
        risk_score = min(1.0, total_weight / 15.0)

        if risk_score >= 0.7:
            risk_level = "CRITICAL"
        elif risk_score >= 0.4:
            risk_level = "HIGH"
        elif risk_score >= 0.2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return RulesResult(
            rules_passed=rules_passed,
            rules_failed=rules_failed,
            rules_total=rules_total,
            violations=violations,
            risk_score=round(risk_score, 3),
            risk_level=risk_level,
            rules_version=self.RULES_VERSION,
        )

    # ── Rules (each returns list of (severity, detail) tuples) ───────

    def _rule_mrz_format(self, fields: Dict, mrz: Optional[MRZParsed]) -> List[Tuple[str, str]]:
        v = []
        l1 = fields.get("mrz_upper_line", "")
        l2 = fields.get("mrz_lower_line", "")
        if not l1:
            v.append(("CRITICAL", "MRZ line 1 not found"))
        elif len(l1.strip()) < 40:
            v.append(("HIGH", f"MRZ line 1 too short: {len(l1.strip())} chars (expected 44)"))
        if not l2:
            v.append(("CRITICAL", "MRZ line 2 not found"))
        elif len(l2.strip()) < 40:
            v.append(("HIGH", f"MRZ line 2 too short: {len(l2.strip())} chars (expected 44)"))
        if l1 and not l1.strip().upper().startswith("P"):
            v.append(("MEDIUM", f"MRZ line 1 should start with 'P', got '{l1[:2]}'"))
        return v

    def _rule_doc_number_check(self, fields: Dict, mrz: Optional[MRZParsed]) -> List[Tuple[str, str]]:
        if not mrz or not mrz.is_valid_format:
            return []
        l2 = mrz.raw_line2.strip().upper()
        if len(l2) < 10:
            return []
        expected = mrz_check_digit(l2[0:9])
        if mrz.document_number_check != expected:
            return [("CRITICAL", f"Doc number check: got {mrz.document_number_check}, expected {expected}")]
        return []

    def _rule_dob_check(self, fields: Dict, mrz: Optional[MRZParsed]) -> List[Tuple[str, str]]:
        if not mrz or not mrz.is_valid_format:
            return []
        l2 = mrz.raw_line2.strip().upper()
        if len(l2) < 20:
            return []
        expected = mrz_check_digit(l2[13:19])
        if mrz.dob_check != expected:
            return [("CRITICAL", f"DOB check: got {mrz.dob_check}, expected {expected}")]
        return []

    def _rule_doe_check(self, fields: Dict, mrz: Optional[MRZParsed]) -> List[Tuple[str, str]]:
        if not mrz or not mrz.is_valid_format:
            return []
        l2 = mrz.raw_line2.strip().upper()
        if len(l2) < 28:
            return []
        expected = mrz_check_digit(l2[21:27])
        if mrz.doe_check != expected:
            return [("CRITICAL", f"DOE check: got {mrz.doe_check}, expected {expected}")]
        return []

    def _rule_personal_number_check(self, fields: Dict, mrz: Optional[MRZParsed]) -> List[Tuple[str, str]]:
        if not mrz or not mrz.is_valid_format:
            return []
        l2 = mrz.raw_line2.strip().upper()
        if len(l2) < 43:
            return []
        pn = l2[28:42]
        if pn.replace("<", "") == "":
            return []
        expected = mrz_check_digit(pn)
        if mrz.personal_number_check != expected:
            return [("HIGH", f"Personal number check: got {mrz.personal_number_check}, expected {expected}")]
        return []

    def _rule_composite_check(self, fields: Dict, mrz: Optional[MRZParsed]) -> List[Tuple[str, str]]:
        if not mrz or not mrz.is_valid_format:
            return []
        l2 = mrz.raw_line2.strip().upper()
        if len(l2) < 44:
            return []
        composite_data = l2[0:10] + l2[13:20] + l2[21:43]
        expected = mrz_check_digit(composite_data)
        if mrz.composite_check != expected:
            return [("CRITICAL", f"Composite check: got {mrz.composite_check}, expected {expected}")]
        return []

    def _rule_country_code(self, fields: Dict, mrz: Optional[MRZParsed]) -> List[Tuple[str, str]]:
        if not mrz or not mrz.is_valid_format:
            return []
        v = []
        issuing = mrz.issuing_country.replace("<", "")
        if issuing and issuing not in VALID_COUNTRY_CODES:
            v.append(("HIGH", f"Invalid issuing country: '{issuing}'"))
        nat = mrz.nationality.replace("<", "")
        if nat and nat not in VALID_COUNTRY_CODES:
            v.append(("HIGH", f"Invalid nationality: '{nat}'"))
        return v

    def _rule_date_plausibility(self, fields: Dict, mrz: Optional[MRZParsed]) -> List[Tuple[str, str]]:
        if not mrz or not mrz.is_valid_format:
            return []
        v = []
        today = date.today()

        dob = parse_mrz_date(mrz.date_of_birth)
        if dob:
            if dob > today:
                v.append(("CRITICAL", f"DOB in future: {dob}"))
            age = (today - dob).days / 365.25
            if age > 150:
                v.append(("HIGH", f"Implausible age: {age:.0f} years"))

        doe = parse_mrz_date(mrz.date_of_expiry)
        if doe:
            if doe < today:
                v.append(("CRITICAL", f"Document expired: {doe}"))
            years = (doe - today).days / 365.25
            if years > 15:
                v.append(("HIGH", f"Expiry too far: {doe} ({years:.0f}y)"))
        if dob and doe and dob >= doe:
            v.append(("CRITICAL", "DOB after DOE"))
        return v

    def _rule_required_fields(self, fields: Dict, mrz: Optional[MRZParsed]) -> List[Tuple[str, str]]:
        v = []
        for f in ["mrz_upper_line", "mrz_lower_line", "primary_identifier",
                   "date_of_birth", "document_number"]:
            val = fields.get(f, "")
            if not val or (isinstance(val, str) and val.strip() in ("", "[bbox_present]")):
                v.append(("HIGH", f"Required field missing: {f}"))
        return v

    def _rule_cross_check(self, fields: Dict, mrz: Optional[MRZParsed]) -> List[Tuple[str, str]]:
        if not mrz or not mrz.is_valid_format:
            return []
        v = []
        viz_doc = fields.get("document_number", "").strip().replace(" ", "").upper()
        mrz_doc = mrz.document_number.replace("<", "").upper()
        if viz_doc and viz_doc not in ("[BBOX_PRESENT]", "") and mrz_doc:
            if viz_doc != mrz_doc:
                v.append(("CRITICAL", f"Doc# mismatch: VIZ='{viz_doc}' vs MRZ='{mrz_doc}'"))

        viz_name = fields.get("primary_identifier", "").strip().upper()
        mrz_name = mrz.primary_identifier.upper()
        if viz_name and viz_name not in ("[BBOX_PRESENT]", "") and mrz_name:
            if viz_name[:3] != mrz_name[:3]:
                v.append(("HIGH", f"Surname mismatch: VIZ='{viz_name}' vs MRZ='{mrz_name}'"))

        viz_sex = fields.get("sex", "").strip().upper()
        if viz_sex and viz_sex not in ("[BBOX_PRESENT]", "") and mrz.sex:
            if viz_sex[0] != mrz.sex[0]:
                v.append(("HIGH", f"Sex mismatch: VIZ='{viz_sex}' vs MRZ='{mrz.sex}'"))

        # DOB cross-check: VIZ date vs MRZ date
        viz_dob = fields.get("date_of_birth", "").strip()
        if viz_dob and viz_dob not in ("[BBOX_PRESENT]", "") and mrz.date_of_birth:
            mrz_dob = parse_mrz_date(mrz.date_of_birth)
            if mrz_dob:
                # Extract day/month/year from VIZ (various formats)
                dob_nums = re.findall(r'\d+', viz_dob)
                if len(dob_nums) >= 3:
                    try:
                        # Try DD.MM.YYYY or DD MM YYYY
                        d, m, y = int(dob_nums[0]), int(dob_nums[1]), int(dob_nums[2])
                        if y < 100:
                            y = 2000 + y if y < 30 else 1900 + y
                        viz_date = date(y, m, d)
                        if viz_date != mrz_dob:
                            v.append(("CRITICAL", f"DOB mismatch: VIZ='{viz_dob}' vs MRZ={mrz_dob}"))
                    except (ValueError, TypeError):
                        pass  # Can't parse VIZ date, skip
        return v
