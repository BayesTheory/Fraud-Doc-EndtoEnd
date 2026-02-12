"""
EasyOCR test on MIDV-2020 passport — crop each annotated field and OCR it.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.infrastructure.data.coco_loader import load_coco_split


def main():
    print("=" * 60)
    print("  EasyOCR Test — Single Passport (MIDV-2020)")
    print("=" * 60)

    # Load dataset
    ds = load_coco_split("data/raw", "train")
    sample = ds.samples[0]
    img_path = os.path.join(ds.base_dir, sample.file_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"ERROR: cannot read {img_path}")
        return
    h, w = image.shape[:2]
    print(f"  Image: {sample.file_name}")
    print(f"  Country: {sample.country_code}")
    print(f"  Size: {w}x{h}")
    print(f"  Fields: {len(sample.fields)} types")

    # Init EasyOCR
    print("\n[1] Init EasyOCR (CPU, English)...")
    t0 = time.perf_counter()
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    print(f"  Init: {time.perf_counter()-t0:.1f}s")

    # OCR each field
    print("\n[2] OCR'ing each annotated field...")
    print("-" * 70)

    extracted = {}
    total_time = 0

    for field_name, regions in sorted(sample.fields.items()):
        for region in regions:
            x1, y1, x2, y2 = region.to_xyxy()

            # Add padding
            pad_x = max(5, int((x2 - x1) * 0.05))
            pad_y = max(3, int((y2 - y1) * 0.1))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            t0 = time.perf_counter()
            results = reader.readtext(crop)
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            texts = [r[1] for r in results]
            confs = [r[2] for r in results]
            text = " ".join(texts)
            avg_conf = sum(confs) / len(confs) if confs else 0

            # Store first occurrence per field
            if field_name not in extracted or text:
                extracted[field_name] = text

            status = "✓" if text else "∅"
            print(f"  {field_name:25s} | {elapsed*1000:6.0f}ms | "
                  f"conf={avg_conf:.2f} | {status} | '{text[:60]}'")

    print("-" * 70)
    n_filled = len([v for v in extracted.values() if v])
    print(f"\n  Total OCR time: {total_time:.1f}s")
    print(f"  Fields extracted: {n_filled}/{len(extracted)}")
    print(f"  Avg per field: {total_time/max(len(extracted),1)*1000:.0f}ms")

    # Show key fields
    print("\n[3] Key Fields:")
    for key in ["mrz_upper_line", "mrz_lower_line", "primary_identifier",
                 "secondary_identifier", "document_number", "date_of_birth",
                 "date_of_expiry", "nationality", "sex"]:
        val = extracted.get(key, "[not found]")
        print(f"  {key:25s}: {val}")

    # Run rules engine
    print("\n[4] Running Passport Rules Engine...")
    from src.infrastructure.rules.passport_rules import PassportRulesEngine
    engine = PassportRulesEngine()

    class FieldHolder:
        def __init__(self, fields):
            self.extracted_fields = fields

    rules_result = engine.apply(FieldHolder(extracted))
    print(f"  Risk score: {rules_result.risk_score}")
    print(f"  Risk level: {rules_result.risk_level}")
    print(f"  Rules: {rules_result.rules_passed}/{rules_result.rules_total} passed")
    if rules_result.violations:
        print(f"  Violations ({len(rules_result.violations)}):")
        for v in rules_result.violations:
            print(f"    [{v.severity:8s}] {v.rule_id}: {v.detail[:80]}")

    print(f"\n{'='*60}")
    print(f"  TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
