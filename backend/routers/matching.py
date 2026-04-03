"""
Matching Router - Resume-to-job cosine similarity matching.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from backend.schemas import MatchRequest, MatchResponse, JobMatchResult
from backend.model_loader import model_store

router = APIRouter(prefix="/match", tags=["Job Matching"])


def _get_matcher():
    if not model_store.models_loaded:
        raise HTTPException(503, "Models not loaded.")
    return model_store.matcher


@router.post("/jobs", response_model=MatchResponse,
             summary="Rank job descriptions by resume similarity")
async def match_jobs(request: MatchRequest, matcher=Depends(_get_matcher)):
    """
    Computes cosine similarity between the resume and all available job descriptions.
    Returns ranked list of top matching jobs.
    """
    try:
        if not request.resume_text.strip():
            raise HTTPException(400, "Resume text cannot be empty.")

        matches = matcher.match(request.resume_text, top_n=request.top_n)
        total_jobs = len(matcher.job_metadata)

        # Build a brief resume summary (first 200 chars of cleaned text)
        from preprocessing.nlp_pipeline import NLPPipeline
        pipeline = NLPPipeline()
        summary = " ".join(pipeline.get_keywords(request.resume_text, top_n=10))

        return MatchResponse(
            resume_summary=summary,
            total_jobs_scanned=total_jobs,
            matches=[JobMatchResult(**m) for m in matches],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job matching error: {e}")
        raise HTTPException(500, str(e))
