"""
FastAPI Application — Entry Point.

Registra rotas, serve a interface de teste, e configura CORS.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

from src.api.routes.analyze import router as analyze_router

app = FastAPI(
    title="Fraud-Doc Pipeline",
    description="Pipeline de validação e prevenção de fraudes em documentos de identidade",
    version="0.1.0",
)

# Registrar rotas
app.include_router(analyze_router, prefix="/api/v1", tags=["Análise"])


# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# Servir interface de teste
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve a interface web de teste."""
    ui_path = Path(__file__).parent.parent.parent / "static" / "index.html"
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Fraud-Doc Pipeline</h1><p>API rodando. Acesse <a href='/docs'>/docs</a></p>")
