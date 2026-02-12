"""
PaddleOCR v5 vs EasyOCR — proper comparison on MRZ fields.
Extracts rec_texts/rec_scores from PaddleOCR OCRResult dict.
"""
import os, sys, time
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.infrastructure.data.coco_loader import load_coco_split

# ── Setup ──
ds = load_coco_split("data/raw", "train")

print("=" * 70)
print("  PaddleOCR v5 vs EasyOCR — MRZ Comparison (5 images)")
print("=" * 70)

# Init both engines
print("\n[Init]")
t0 = time.perf_counter()
import easyocr
easy = easyocr.Reader(["en"], gpu=False, verbose=False)
print(f"  EasyOCR:    {time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
from paddleocr import PaddleOCR
paddle = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
print(f"  PaddleOCR:  {time.perf_counter()-t0:.1f}s")

# ── Compare on MRZ fields ──
fields_to_test = ["mrz_lower_line", "mrz_upper_line", "document_number",
                   "primary_identifier", "date_of_birth"]

for idx in range(min(5, len(ds.samples))):
    sample = ds.samples[idx]
    img = cv2.imread(os.path.join(ds.base_dir, sample.file_name))
    if img is None:
        continue
    h, w = img.shape[:2]

    print(f"\n── Image {idx+1}: {sample.original_name} ({sample.country_code}) ──")

    for field_name in fields_to_test:
        if field_name not in sample.fields:
            continue

        region = sample.fields[field_name][0]
        x1, y1, x2, y2 = region.to_xyxy()
        pad_x = max(5, int((x2 - x1) * 0.05))
        pad_y = max(3, int((y2 - y1) * 0.15))
        crop = img[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]

        # ── EasyOCR ──
        t0 = time.perf_counter()
        is_mrz = "mrz" in field_name
        if is_mrz:
            eres = easy.readtext(crop, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<", paragraph=True)
        else:
            eres = easy.readtext(crop)
        e_time = (time.perf_counter() - t0) * 1000
        e_text = " ".join([r[1] for r in eres]) if eres else ""
        e_conf = sum(r[2] for r in eres if len(r) >= 3) / max(len(eres), 1) if eres and len(eres[0]) >= 3 else 0.5

        # ── PaddleOCR ──
        t0 = time.perf_counter()
        pres = paddle.predict(crop)
        p_time = (time.perf_counter() - t0) * 1000
        p_text = ""
        p_conf = 0
        if pres and len(pres) > 0:
            r = pres[0]
            if hasattr(r, 'keys') and 'rec_texts' in r:
                texts = r['rec_texts']
                scores = r['rec_scores']
                p_text = " ".join(texts) if isinstance(texts, list) else str(texts)
                p_conf = sum(scores) / len(scores) if scores else 0

        # ── Compare ──
        match = "==" if e_text.replace(" ", "") == p_text.replace(" ", "") else "!="
        print(f"  {field_name:25s}")
        print(f"    Easy: {e_time:5.0f}ms | conf={e_conf:.2f} | '{e_text[:60]}'")
        print(f"    Padl: {p_time:5.0f}ms | conf={p_conf:.2f} | '{p_text[:60]}'")
        print(f"    {match}")

print(f"\n{'='*70}")
print("DONE")
