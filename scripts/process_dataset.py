"""
Batch Dataset Processor — MIDV-2020 MRP Pipeline

Runs the full document validation pipeline on every image in the dataset:
  Image → Quality Gate → OCR (bbox-guided) → Passport Rules → Report

Usage:
    python -m scripts.process_dataset [--split train] [--limit 10] [--no-ocr]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infrastructure.data.coco_loader import load_coco_split, PassportSample
from src.infrastructure.quality.opencv_quality_gate import OpenCVQualityGate
from src.infrastructure.rules.passport_rules import (
    PassportRulesEngine,
    parse_mrz_td3,
)


def process_single_image(
    image_path: str,
    sample: PassportSample,
    quality_gate: OpenCVQualityGate,
    rules_engine: PassportRulesEngine,
    ocr_engine=None,
) -> dict:
    """Process a single passport image through the pipeline."""
    result = {
        "file": sample.original_name,
        "country": sample.country_code,
        "image_id": sample.image_id,
        "dimensions": f"{sample.width}x{sample.height}",
        "annotated_fields": list(sample.fields.keys()),
        "num_fields": len(sample.fields),
        "stages": {},
        "timings": {},
    }

    # ── Stage 1: Load Image ──
    t0 = time.perf_counter()
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        result["error"] = f"Cannot read image: {image_path}"
        return result
    result["timings"]["load"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Stage 2: Quality Gate ──
    t0 = time.perf_counter()
    try:
        quality = quality_gate.evaluate(image_bytes)
        result["stages"]["quality"] = {
            "passed": quality.quality_ok,
            "score": quality.quality_score,
            "reasons": quality.reasons,
            "recommendation": quality.recommendation,
            "details": quality.details,
        }
    except Exception as e:
        result["stages"]["quality"] = {"error": str(e), "passed": True}
    result["timings"]["quality"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Stage 3: OCR (optional, bbox-guided) ──
    t0 = time.perf_counter()
    extracted_fields = {}

    if ocr_engine is not None:
        try:
            ocr_result = ocr_engine.extract_with_regions(image, sample)
            extracted_fields = ocr_result.extracted_fields
            result["stages"]["ocr"] = {
                "num_fields_extracted": len(extracted_fields),
                "fields": {k: v[:50] for k, v in extracted_fields.items()},
                "confidence": ocr_result.confidence,
            }
        except Exception as e:
            result["stages"]["ocr"] = {"error": str(e)}
    else:
        # Without OCR, we still know WHICH fields exist from annotations
        # We just can't read their text content
        result["stages"]["ocr"] = {
            "skipped": True,
            "reason": "OCR disabled (--no-ocr flag or PaddleOCR not installed)",
            "fields_with_bbox": list(sample.fields.keys()),
        }
        # Mark fields as present but empty
        for field_name in sample.fields:
            extracted_fields[field_name] = "[bbox_present]"

    result["timings"]["ocr"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Stage 4: Passport Rules ──
    t0 = time.perf_counter()
    try:
        # Create a simple object with extracted_fields attribute
        class FieldHolder:
            def __init__(self, fields):
                self.extracted_fields = fields

        rules_result = rules_engine.apply(FieldHolder(extracted_fields))
        result["stages"]["rules"] = {
            "risk_score": rules_result.risk_score,
            "risk_level": rules_result.risk_level,
            "rules_total": rules_result.rules_total,
            "rules_passed": rules_result.rules_passed,
            "rules_failed": rules_result.rules_failed,
            "num_violations": len(rules_result.violations),
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "rule_name": v.rule_name,
                    "severity": v.severity,
                    "detail": v.detail,
                }
                for v in rules_result.violations
            ],
        }
    except Exception as e:
        result["stages"]["rules"] = {"error": str(e)}
    result["timings"]["rules"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Final Decision ──
    quality_ok = result["stages"].get("quality", {}).get("passed", True)
    risk_score = result["stages"].get("rules", {}).get("risk_score", 0)
    risk_level = result["stages"].get("rules", {}).get("risk_level", "LOW")

    # Has critical violations?
    critical_violations = [
        v for v in result["stages"].get("rules", {}).get("violations", [])
        if v.get("severity") == "CRITICAL"
    ]

    if not quality_ok:
        decision = "REJECTED_QUALITY"
    elif len(critical_violations) > 0:
        decision = "SUSPICIOUS"
    elif risk_level in ("HIGH", "CRITICAL"):
        decision = "REVIEW"
    else:
        decision = "APPROVED"

    result["decision"] = decision
    result["total_time_ms"] = round(
        sum(result["timings"].values()), 1
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Process MIDV-2020 dataset")
    parser.add_argument("--data-dir", default="data/raw", help="Dataset directory")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--limit", type=int, default=0, help="Limit images (0=all)")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR (faster)")
    parser.add_argument("--output", default="data/results", help="Output directory")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  MIDV-2020 MRP Pipeline — Batch Processor")
    print(f"{'='*60}")
    print(f"  Split: {args.split}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  OCR: {'enabled' if not args.no_ocr else 'DISABLED'}")
    print(f"{'='*60}\n")

    # ── Load dataset ──
    print("[1/4] Loading COCO dataset...")
    dataset = load_coco_split(args.data_dir, args.split)
    stats = dataset.stats()
    print(f"  → {stats['total_images']} images")
    print(f"  → Countries: {stats['countries']}")
    print(f"  → By country: {stats['by_country']}")

    # ── Initialize components ──
    print("\n[2/4] Initializing pipeline components...")
    quality_gate = OpenCVQualityGate()
    rules_engine = PassportRulesEngine()
    print("  → Quality Gate: OK")
    print("  → Passport Rules Engine: OK (10 rules)")

    ocr_engine = None
    if not args.no_ocr:
        try:
            from src.infrastructure.ocr.passport_ocr_engine import PassportOCREngine
            ocr_engine = PassportOCREngine(lang="en", use_gpu=False)
            print("  → Passport OCR Engine: OK")
        except Exception as e:
            print(f"  → Passport OCR Engine: FAILED ({e}), continuing without OCR")

    # ── Process images ──
    samples = dataset.samples
    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"\n[3/4] Processing {len(samples)} images...")

    results = []
    decisions = {"APPROVED": 0, "SUSPICIOUS": 0, "REVIEW": 0, "REJECTED_QUALITY": 0}
    total_time = 0

    for i, sample in enumerate(samples):
        image_path = os.path.join(dataset.base_dir, sample.file_name)

        result = process_single_image(
            image_path, sample, quality_gate, rules_engine, ocr_engine
        )
        results.append(result)

        decision = result.get("decision", "ERROR")
        decisions[decision] = decisions.get(decision, 0) + 1
        total_time += result.get("total_time_ms", 0)

        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            pct = (i + 1) / len(samples) * 100
            avg_ms = total_time / (i + 1)
            print(f"  [{i+1:4d}/{len(samples)}] {pct:5.1f}% | "
                  f"avg {avg_ms:.0f}ms/img | "
                  f"✓{decisions['APPROVED']} "
                  f"?{decisions['SUSPICIOUS']} "
                  f"⚠{decisions['REVIEW']} "
                  f"✗{decisions['REJECTED_QUALITY']}")

    # ── Save results ──
    print(f"\n[4/4] Saving results...")
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, f"pipeline_{args.split}.json")

    report = {
        "metadata": {
            "split": args.split,
            "total_images": len(samples),
            "ocr_enabled": not args.no_ocr,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {
            "decisions": decisions,
            "avg_time_ms": round(total_time / max(len(samples), 1), 1),
            "total_time_s": round(total_time / 1000, 1),
            "by_country": {},
        },
        "results": results,
    }

    # Country breakdown
    for r in results:
        country = r.get("country", "unknown")
        if country not in report["summary"]["by_country"]:
            report["summary"]["by_country"][country] = {
                "total": 0, "approved": 0, "suspicious": 0
            }
        report["summary"]["by_country"][country]["total"] += 1
        if r.get("decision") == "APPROVED":
            report["summary"]["by_country"][country]["approved"] += 1
        elif r.get("decision") == "SUSPICIOUS":
            report["summary"]["by_country"][country]["suspicious"] += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  → Saved to: {output_file}")

    # ── Print Summary ──
    print(f"\n{'='*60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total images:  {len(samples)}")
    print(f"  Avg time/img:  {total_time / max(len(samples), 1):.0f} ms")
    print(f"  Total time:    {total_time / 1000:.1f} s")
    print(f"")
    print(f"  Decisions:")
    for dec, count in sorted(decisions.items()):
        pct = count / max(len(samples), 1) * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"    {dec:20s} {count:4d} ({pct:5.1f}%) {bar}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
