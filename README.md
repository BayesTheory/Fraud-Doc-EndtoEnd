# Fraud-Doc-EndtoEnd

Pipeline end-to-end de validação e prevenção de fraudes em documentos de identidade.

## Arquitetura

```
Upload Imagem → Quality Gate → OCR → Regras de Negócio → Classificador Fraude → Busca Vetorial → Resultado JSON
```

## Stack

| Camada | Tecnologia |
|---|---|
| API | FastAPI |
| OCR | PaddleOCR |
| CV / Quality | OpenCV |
| ML | PyTorch (EfficientNet-B0) |
| Banco | Postgres 16 + pgvector |
| Storage | MinIO (local) / S3 (prod) |
| Container | Docker Compose |

## Estrutura do Projeto

Segue **Clean Architecture (Ports & Adapters)**:

- `src/core/interfaces/` — Contratos (ABCs) — o que cada serviço DEVE fazer
- `src/core/entities/` — Modelos de domínio puros (sem dependências externas)
- `src/core/use_cases/` — Orquestração de regras de negócio
- `src/infrastructure/` — Implementações concretas dos contratos
- `src/api/` — Adapter HTTP (FastAPI)
- `src/pipeline/` — Orquestrador do pipeline completo

## Quick Start

```bash
docker compose up --build
# API disponível em http://localhost:8000/docs
```
