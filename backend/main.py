"""
FastAPI Main Application - AI Resume Screening System API.
"""
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from backend.routers import resume, matching, skills, prediction
from backend.model_loader import model_store
from backend.schemas import HealthResponse


# ── Logging Setup ─────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/api.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
)


# ── Lifespan: Load models on startup ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting AI Resume Screening API...")
    model_store.load_all()
    yield
    logger.info("🛑 Shutting down API...")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Resume Screening & ATS System",
    description="""
## AI-Powered Resume Screening and ATS System

A production-ready API for automated resume analysis, ATS scoring, job matching,
skill gap detection, and hiring prediction.

### Features
- 📄 **Resume Upload** – PDF, DOCX, TXT support
- 📊 **ATS Score** – 6-component weighted scoring (0–100)
- 🎯 **Job Role Prediction** – Multi-class ML classifier (LR/RF/SVM)
- 🔗 **Job Matching** – Cosine similarity ranking
- 🔍 **Skill Gap Analysis** – Domain-aware skill comparison
- 📚 **Learning Recommendations** – Personalized roadmap
- 🤖 **Hiring Prediction** – XGBoost probability score
    """,
    version="1.0.0",
    contact={"name": "AI Resume Screening System", "email": "support@atsai.com"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(resume.router)
app.include_router(matching.router)
app.include_router(skills.router)
app.include_router(prediction.router)


# ── Root & Health ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "AI Resume Screening & ATS System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """API health check and model loading status."""
    return HealthResponse(
        status="healthy" if model_store.models_loaded else "degraded",
        version="1.0.0",
        models_loaded=model_store.models_loaded,
    )


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
