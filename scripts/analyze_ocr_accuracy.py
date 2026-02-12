"""
Deep OCR Analysis — test accuracy across multiple images with preprocessing.
Compares raw vs preprocessed OCR, validates MRZ checksums, identifies patterns.
"""
import os
import sys
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from src.infrastructure.data.coco_loader import load_coco_split


# ── MRZ Character Map (common OCR confusions) ──
MRZ_CHAR_FIX = str.maketrans({
    " ": "<",     # space → filler
    "«": "<",     # guillemet → filler
    "‹": "<",     # single guillemet
    "č": "<",     # czech c-caron
    "Č": "<",     # czech C-caron
    "c": "C",     # lowercase
    "é": "<",     # accented e
})

# MRZ only allows: A-Z, 0-9, <
MRZ_VALID = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")


def clean_mrz_text(raw: str) -> str:
    """Post-process MRZ text: fix common OCR errors."""
    text = raw.upper().strip()
    text = text.translate(MRZ_CHAR_FIX)

    # Replace remaining invalid chars with best guess
    cleaned = []
    for ch in text:
        if ch in MRZ_VALID:
            cleaned.append(ch)
        elif ch in "()[]{}":
            cleaned.append("<")
        elif ch == "O" and len(cleaned) > 0 and cleaned[-1].isdigit():
            cleaned.append("0")  # O after digit → 0
        else:
            cleaned.append("<")  # fallback
    return "".join(cleaned)


def mrz_check_digit(data: str) -> int:
    """ICAO 9303 check digit calculation."""
    weights = [7, 3, 1]
    values = {str(i): i for i in range(10)}
    for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        values[c] = i + 10
    values["<"] = 0

    total = 0
    for i, ch in enumerate(data):
        v = values.get(ch, 0)
        total += v * weights[i % 3]
    return total % 10


def validate_mrz_line2(line2: str) -> dict:
    """Parse and validate MRZ line 2 checksums."""
    result = {"raw": line2, "length": len(line2), "valid": False}

    if len(line2) != 44:
        result["error"] = f"Expected 44 chars, got {len(line2)}"
        return result

    doc_num = line2[0:9]
    doc_check = line2[9]
    nationality = line2[10:13]
    dob = line2[13:19]
    dob_check = line2[19]
    sex = line2[20]
    doe = line2[21:27]
    doe_check = line2[27]
    personal = line2[28:42]
    personal_check = line2[42]
    composite_check = line2[43]

    result["fields"] = {
        "doc_num": doc_num,
        "nationality": nationality,
        "dob": dob,
        "sex": sex,
        "doe": doe,
        "personal": personal,
    }

    # Verify check digits
    checks = {}
    expected_doc = mrz_check_digit(doc_num)
    checks["doc_num"] = {
        "expected": expected_doc,
        "got": int(doc_check) if doc_check.isdigit() else -1,
        "ok": doc_check.isdigit() and int(doc_check) == expected_doc,
    }

    expected_dob = mrz_check_digit(dob)
    checks["dob"] = {
        "expected": expected_dob,
        "got": int(dob_check) if dob_check.isdigit() else -1,
        "ok": dob_check.isdigit() and int(dob_check) == expected_dob,
    }

    expected_doe = mrz_check_digit(doe)
    checks["doe"] = {
        "expected": expected_doe,
        "got": int(doe_check) if doe_check.isdigit() else -1,
        "ok": doe_check.isdigit() and int(doe_check) == expected_doe,
    }

    # Composite
    composite_data = line2[0:10] + line2[13:20] + line2[21:43]
    expected_composite = mrz_check_digit(composite_data)
    checks["composite"] = {
        "expected": expected_composite,
        "got": int(composite_check) if composite_check.isdigit() else -1,
        "ok": composite_check.isdigit() and int(composite_check) == expected_composite,
    }

    result["checks"] = checks
    result["valid"] = all(c["ok"] for c in checks.values())
    return result


def preprocess_for_ocr(crop: np.ndarray, mode: str = "raw") -> np.ndarray:
    """Apply image preprocessing before OCR."""
    if mode == "raw":
        return crop

    if mode == "gray_thresh":
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold → binary
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    if mode == "contrast":
        # CLAHE contrast enhancement
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced

    if mode == "upscale":
        # 2x upscale → sharpen
        h, w = crop.shape[:2]
        big = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp = cv2.filter2D(big, -1, kernel)
        return sharp

    if mode == "mrz_optimized":
        # Best combo for MRZ: upscale + grayscale + CLAHE + Otsu
        h, w = crop.shape[:2]
        big = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return crop


def main():
    print("=" * 70)
    print("  Deep OCR Analysis — EasyOCR on MIDV-2020 Passports")
    print("=" * 70)

    # Load
    ds = load_coco_split("data/raw", "train")
    print(f"  Dataset: {len(ds.samples)} images")

    # Init EasyOCR
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    print("  EasyOCR: OK")

    # ── Test 1: Raw vs Preprocessed on MRZ (5 images) ──
    print(f"\n{'='*70}")
    print("  TEST 1: MRZ Accuracy — Raw vs Preprocessed (5 images)")
    print(f"{'='*70}")

    modes = ["raw", "contrast", "upscale", "mrz_optimized"]
    mrz_results = {m: {"checksums_ok": 0, "total": 0, "line_len_ok": 0} for m in modes}

    for idx in range(min(5, len(ds.samples))):
        sample = ds.samples[idx]
        img_path = os.path.join(ds.base_dir, sample.file_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        print(f"\n  ── Image {idx+1}: {sample.original_name} ({sample.country_code}) ──")

        for mode in modes:
            # Get MRZ lower line bbox
            if "mrz_lower_line" not in sample.fields:
                continue

            region = sample.fields["mrz_lower_line"][0]
            x1, y1, x2, y2 = region.to_xyxy()
            pad_x = max(5, int((x2 - x1) * 0.05))
            pad_y = max(3, int((y2 - y1) * 0.15))
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
            crop = image[y1:y2, x1:x2]

            processed = preprocess_for_ocr(crop, mode)

            # EasyOCR with allowlist for MRZ
            t0 = time.perf_counter()
            results = reader.readtext(
                processed,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
                paragraph=True,  # merge into single line
            )
            elapsed = time.perf_counter() - t0

            raw_text = " ".join([r[1] for r in results]) if results else ""
            cleaned = clean_mrz_text(raw_text)

            mrz_results[mode]["total"] += 1
            if len(cleaned) == 44:
                mrz_results[mode]["line_len_ok"] += 1
                validation = validate_mrz_line2(cleaned)
                if validation["valid"]:
                    mrz_results[mode]["checksums_ok"] += 1
                check_str = " | ".join(
                    f"{k}:{'✓' if v['ok'] else '✗'}"
                    for k, v in validation["checks"].items()
                )
                print(f"    [{mode:15s}] {elapsed*1000:5.0f}ms | "
                      f"len={len(cleaned)} | {check_str}")
            else:
                print(f"    [{mode:15s}] {elapsed*1000:5.0f}ms | "
                      f"len={len(cleaned)} (need 44) | '{cleaned[:50]}...'")

    # Summary
    print(f"\n  ── MRZ Preprocessing Summary ──")
    print(f"  {'Mode':15s} | {'Length OK':10s} | {'Checksums OK':12s}")
    print(f"  {'-'*15} | {'-'*10} | {'-'*12}")
    for mode in modes:
        r = mrz_results[mode]
        total = max(r["total"], 1)
        len_str = f"{r['line_len_ok']}/{r['total']}"
        chk_str = f"{r['checksums_ok']}/{r['total']}"
        print(f"  {mode:15s} | {len_str:10s} | {chk_str:12s}")

    # ── Test 2: Full field accuracy on 5 images ──
    print(f"\n{'='*70}")
    print("  TEST 2: Full Field OCR Accuracy (5 images)")
    print(f"{'='*70}")

    key_fields = ["mrz_upper_line", "mrz_lower_line", "primary_identifier",
                   "secondary_identifier", "document_number", "date_of_birth",
                   "date_of_expiry", "nationality", "sex", "issuing_state_code"]

    for idx in range(min(5, len(ds.samples))):
        sample = ds.samples[idx]
        img_path = os.path.join(ds.base_dir, sample.file_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        h, w = image.shape[:2]

        print(f"\n  ── Image {idx+1}: {sample.original_name} ({sample.country_code}) ──")

        for field_name in key_fields:
            if field_name not in sample.fields:
                print(f"    {field_name:25s}: [no bbox]")
                continue

            region = sample.fields[field_name][0]
            x1, y1, x2, y2 = region.to_xyxy()
            pad_x = max(5, int((x2 - x1) * 0.05))
            pad_y = max(3, int((y2 - y1) * 0.1))
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
            crop = image[y1:y2, x1:x2]

            # Use preprocessing for MRZ fields
            is_mrz = "mrz" in field_name
            if is_mrz:
                processed = preprocess_for_ocr(crop, "raw")
                results = reader.readtext(
                    processed,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
                    paragraph=True,
                )
            else:
                # Contrast enhancement for VIZ fields
                processed = preprocess_for_ocr(crop, "contrast")
                results = reader.readtext(processed)

            text_parts = []
            confs = []
            for r in (results or []):
                if len(r) >= 3:
                    text_parts.append(r[1])
                    confs.append(r[2])
                elif len(r) >= 2:
                    text_parts.append(r[1])
                    confs.append(0.5)
            text = " ".join(text_parts)
            avg_conf = sum(confs) / len(confs) if confs else 0

            if is_mrz:
                text = clean_mrz_text(text)

            print(f"    {field_name:25s}: conf={avg_conf:.2f} | '{text[:60]}'")

    # ── Test 3: MRZ Checksum validation with cleaned text ──
    print(f"\n{'='*70}")
    print("  TEST 3: MRZ Checksum Validation (optimized pipeline, 10 images)")
    print(f"{'='*70}")

    passed = 0
    failed = 0
    length_wrong = 0

    for idx in range(min(10, len(ds.samples))):
        sample = ds.samples[idx]
        img_path = os.path.join(ds.base_dir, sample.file_name)
        image = cv2.imread(img_path)
        if image is None or "mrz_lower_line" not in sample.fields:
            continue
        h, w = image.shape[:2]

        region = sample.fields["mrz_lower_line"][0]
        x1, y1, x2, y2 = region.to_xyxy()
        pad_x = max(5, int((x2 - x1) * 0.05))
        pad_y = max(3, int((y2 - y1) * 0.15))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        crop = image[y1:y2, x1:x2]

        processed = preprocess_for_ocr(crop, "raw")
        results = reader.readtext(
            processed,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
            paragraph=True,
        )

        raw_text = " ".join([r[1] for r in results]) if results else ""
        cleaned = clean_mrz_text(raw_text)

        if len(cleaned) != 44:
            length_wrong += 1
            emoji = "📏"
            detail = f"len={len(cleaned)}"
        else:
            v = validate_mrz_line2(cleaned)
            if v["valid"]:
                passed += 1
                emoji = "✅"
                detail = "ALL checksums OK"
            else:
                failed += 1
                emoji = "❌"
                bad = [k for k, c in v["checks"].items() if not c["ok"]]
                detail = f"FAILED: {', '.join(bad)}"

        print(f"  {emoji} {sample.original_name:40s} | {detail}")

    total = passed + failed + length_wrong
    print(f"\n  Summary: {passed}/{total} checksums OK, "
          f"{failed}/{total} failed, {length_wrong}/{total} wrong length")

    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
