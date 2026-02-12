# ── Fraud-Doc Pipeline — Production Dockerfile ──
# Optimized for GCP Cloud Run
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV + PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ src/
COPY static/ static/

# Environment defaults
ENV PORT=8000
ENV DATABASE_URL=sqlite:///fraud_doc.db
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Cloud Run expects PORT env var
EXPOSE $PORT

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port $PORT"]
