# 🔍 Fraud-Doc EndtoEnd

End-to-end document fraud detection pipeline.  
Validates identity documents (passports) using computer vision, OCR, and deterministic rule engines.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      API Layer                          │
│  FastAPI endpoints · Dark-mode Web UI · Pydantic DTOs   │
├─────────────────────────────────────────────────────────┤
│                     Core Layer                          │
│  Interfaces (Ports) · Use Cases · Domain Entities       │
├─────────────────────────────────────────────────────────┤
│                Infrastructure Layer                     │
│  OpenCV Quality Gate · PaddleOCR Engine · Rules Engines │
│  PostgreSQL + pgvector · MinIO Storage · COCO Loader    │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Pipeline

Each document flows through 4 stages:

| Stage | Component | Description |
|-------|-----------|-------------|
| **1. Quality Gate** | `OpenCVQualityGate` | Blur, brightness, resolution, framing checks (~5ms) |
| **2. OCR** | `EasyOCR` + `PaddleOCR v5` | Dual-engine: EasyOCR primary (~1s/field), Paddle fallback (~3s/field) |
| **3. Rules Engine** | `PassportRulesEngine` / `BrazilianDocRulesEngine` | Deterministic validation with severity scoring |
| **4. Decision** | `AnalyzeDocumentUseCase` | Multi-signal aggregation → APPROVED / REVIEW / SUSPICIOUS / REJECTED |

## 📑 Passport Rules (ICAO 9303)

10 rules for Machine Readable Travel Documents:

| # | Rule | Severity |
|---|------|----------|
| 1 | MRZ format validation (TD3, 2×44 chars) | CRITICAL |
| 2 | Document number check digit | CRITICAL |
| 3 | Date of birth check digit | CRITICAL |
| 4 | Date of expiry check digit | CRITICAL |
| 5 | Personal number check digit | HIGH |
| 6 | Composite check digit | CRITICAL |
| 7 | Country code (ISO 3166-1 alpha-3) | HIGH |
| 8 | Date plausibility (DOB past, DOE reasonable) | CRITICAL/HIGH |
| 9 | Required fields presence | HIGH |
| 10 | VIZ ↔ MRZ cross-check (tampering detection) | CRITICAL |

## 📊 Dataset: MIDV-2020 MRP

Using [MIDV-2020](https://arxiv.org/abs/2107.00396) passport subset in COCO format:

| Split | Images | Countries |
|-------|--------|-----------|
| Train | 320 | AZE, GRC, LVA, SRB |
| Valid | 80 | AZE, GRC, LVA, SRB |
| Test | 81 | Mixed |
| **Total** | **481** | **4 nationalities** |

**34 annotated field categories** including MRZ lines, dates, document numbers, face, signature.

## 🔬 OCR Test Results

Tested on MIDV-2020 passports (4 countries):

| Engine | MRZ Checksum Accuracy | Avg Confidence | Speed/field |
|--------|----------------------|----------------|-------------|
| **EasyOCR** (raw + allowlist) | 9/10 (90%) | 0.50-1.00 | ~1.5s |
| **PaddleOCR v5** (enable_mkldnn=False) | TBD (higher conf) | 0.94-1.00 | ~3-10s |

**Strategy**: EasyOCR primary (fast) → PaddleOCR fallback (accurate) → Manual review if both fail.

## ⚡ Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -e .
pip install easyocr           # Primary OCR engine

# Run OCR test on single passport
python scripts/test_ocr_single.py

# Run dual-engine comparison
python scripts/test_dual_ocr.py

# Run accuracy analysis (MRZ checksum validation)
python scripts/analyze_ocr_accuracy.py

# Run batch pipeline (no OCR, fast)
python scripts/process_dataset.py --split train --no-ocr

# Start API
uvicorn src.api.main:app --reload
```

## 📂 Project Structure

```
src/
├── core/                    # Domain layer (zero dependencies)
│   ├── interfaces/          # Ports: IQualityGate, IOCREngine, IRulesEngine...
│   ├── entities/            # Document, AnalysisResult
│   └── use_cases/           # AnalyzeDocumentUseCase
├── infrastructure/          # Adapters
│   ├── quality/             # OpenCV quality gate
│   ├── ocr/                 # EasyOCR (primary) + PaddleOCR v5 (fallback)
│   ├── rules/               # Brazilian doc rules + Passport ICAO rules
│   ├── data/                # COCO dataset loader (MIDV-2020)
│   ├── db/                  # SQLAlchemy + pgvector
│   ├── embeddings/          # PgVector similarity search
│   └── storage/             # MinIO object storage
├── api/                     # FastAPI application
│   ├── routes/              # /analyze, /cases, /feedback
│   └── schemas/             # Pydantic request/response models
└── config/                  # Settings (pydantic-settings)

scripts/
├── process_dataset.py       # Batch pipeline processor
├── test_ocr_single.py       # EasyOCR single passport test
├── test_dual_ocr.py         # PaddleOCR vs EasyOCR comparison
└── analyze_ocr_accuracy.py  # MRZ checksum accuracy analysis

data/
├── raw/                     # MIDV-2020 dataset (train/valid/test)
└── results/                 # Pipeline output JSONs

static/
└── index.html               # Dark-mode Web UI
```

## 🔬 Current Status

| Component | Code | Tested | Status |
|-----------|------|--------|--------|
| COCO DataLoader | ✅ | ✅ | ✅ Production-ready |
| Quality Gate | ✅ | ✅ | ✅ Production-ready |
| Passport Rules | ✅ | ✅ | ✅ Checksums validated via OCR |
| EasyOCR Engine | ✅ | ✅ | ✅ 90% MRZ accuracy |
| PaddleOCR v5 Fallback | ✅ | ✅ | ✅ Higher confidence, slower |
| Batch Processor | ✅ | ✅ | ✅ With OCR integration |
| API + Web UI | ✅ | ✅ | ✅ BR docs only |

## 📍 Roadmap

- [x] Clean Architecture skeleton
- [x] Quality Gate (OpenCV)
- [x] COCO DataLoader (MIDV-2020)
- [x] Passport Rules Engine (ICAO 9303)
- [x] Batch Pipeline (no OCR)
- [x] EasyOCR integration + MRZ validation (9/10 checksums OK)
- [x] PaddleOCR v5 dual-engine comparison
- [x] Integrate dual-OCR into PassportOCREngine
- [x] Run batch pipeline WITH OCR
- [x] Validate rules engine with real OCR output
- [x] Fraud Simulation — synthetic tampering on MIDV-2020
- [x] Fraud Classifier — binary model
- [x] LLM Integration — semantic anomaly analysis
- [ ] Docker Compose deployment
=
