# 🔍 Fraud-Doc EndtoEnd

End-to-end document fraud detection pipeline built with **Clean Architecture (Ports & Adapters)**.  
Validates identity documents (passports, Brazilian IDs) using computer vision, OCR, and deterministic rule engines.

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
| **2. OCR** | `PaddleOCREngine` / `PassportOCREngine` | Field extraction — bbox-guided for annotated data |
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

## ⚡ Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -e .

# Run batch pipeline (no OCR, fast)
python scripts/process_dataset.py --split train --no-ocr

# Run batch pipeline (with OCR)
python scripts/process_dataset.py --split train --limit 10

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
│   ├── ocr/                 # PaddleOCR + Passport OCR engines
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
└── process_dataset.py       # Batch pipeline processor

data/
├── raw/                     # MIDV-2020 dataset (train/valid/test)
└── results/                 # Pipeline output JSONs

static/
└── index.html               # Dark-mode Web UI
```

## 🔬 Current Status

| Component | Code | Tested | Production-Ready |
|-----------|------|--------|-------------------|
| COCO DataLoader | ✅ | ✅ | ✅ |
| Quality Gate | ✅ | ✅ | ✅ |
| Passport Rules | ✅ | ⚠️ partial | ❌ needs OCR data |
| Passport OCR | ✅ | ❌ | ❌ not yet tested |
| Batch Processor | ✅ | ✅ | ⚠️ without OCR |
| API + Web UI | ✅ | ✅ | ⚠️ BR docs only |

## 📍 Roadmap

- [x] Clean Architecture skeleton
- [x] Quality Gate (OpenCV)
- [x] COCO DataLoader (MIDV-2020)
- [x] Passport Rules Engine (ICAO 9303)
- [x] Batch Pipeline (no OCR)
- [ ] **OCR Integration** — PaddleOCR on passport fields
- [ ] Fraud Simulation — synthetic tampering on MIDV-2020
- [ ] Fraud Classifier — EfficientNet-B0 binary model
- [ ] LLM Integration — semantic anomaly analysis
- [ ] Docker Compose deployment

## 📄 License

MIT
