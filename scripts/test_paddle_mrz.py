"""Quick test: PaddleOCR v5 on the actual passport image MRZ zone."""
import os, sys, cv2, numpy as np
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

img_path = r"data\raw\valid\midv2020-aze-passport_45_jpg.rf.653d1f1f642197ae32f963fec8e70d5c.jpg"
image = cv2.imread(img_path)
h, w = image.shape[:2]
print(f"Image: {w}x{h}")

# Crop bottom 40% for MRZ
mrz_crop = image[int(h * 0.60):, :]
cv2.imwrite("_mrz_crop.jpg", mrz_crop)
print(f"MRZ crop: {mrz_crop.shape[1]}x{mrz_crop.shape[0]}")

from paddleocr import PaddleOCR
paddle = PaddleOCR(lang="en")

# v3.4+ API: use predict() 
print("\n=== PaddleOCR .predict() on MRZ crop ===")
result = paddle.predict(mrz_crop)

MRZ_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")

for res in result:
    if hasattr(res, 'rec_texts'):
        texts = res.rec_texts
        scores = res.rec_scores
        boxes = res.dt_polys
        for text, score, box in zip(texts, scores, boxes):
            y = box[0][1] if len(box) > 0 else 0
            clean = text.upper().replace(" ", "")
            print(f"  Y={y:6.1f} | conf={score:.3f} | len={len(clean):2d} | {text}")
    else:
        # Try dict-like access
        print(f"  Result type: {type(res)}")
        print(f"  Result dir: {[a for a in dir(res) if not a.startswith('_')]}")
        if hasattr(res, 'keys'):
            for k in res.keys():
                print(f"    {k}: {res[k]}")
