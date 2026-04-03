"""
Skills Router - Skill extraction, gap analysis, and recommendations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.schemas import (
    SkillGapRequest, SkillGapResponse, SkillInfo,
    ResumeTextRequest, RecommendationResponse, LearningResource, CareerPath,
)
from utils.skill_extractor import extract_skills, extract_skills_by_domain
from utils.skill_gap import compute_skill_gap
from utils.recommender import recommend_learning_path, suggest_career_path

router = APIRouter(prefix="/skills", tags=["Skills & Recommendations"])


@router.post("/extract", summary="Extract skills from resume or job description text")
async def extract_skills_endpoint(request: ResumeTextRequest):
    """
    Extract skills from text using the hybrid keyword-dictionary matcher.
    Returns skills grouped by domain.
    """
    try:
        by_domain = extract_skills_by_domain(request.resume_text)
        all_skills = extract_skills(request.resume_text)
        return {
            "all_skills": sorted(all_skills),
            "total_count": len(all_skills),
            "by_domain": {d: sorted(s) for d, s in by_domain.items()},
        }
    except Exception as e:
        logger.error(f"Skill extraction error: {e}")
        raise HTTPException(500, str(e))


@router.post("/gap", response_model=SkillGapResponse,
             summary="Compute skill gap between resume and job description")
async def skill_gap(request: SkillGapRequest):
    """
    Compare skills in a resume vs. job description.
    Returns matched skills, missing skills ranked by importance, and gap score.
    """
    try:
        result = compute_skill_gap(
            resume_text=request.resume_text,
            job_text=request.job_text,
        )
        ranked_missing = [
            SkillInfo(**s) for s in result["ranked_missing"]
        ]
        return SkillGapResponse(
            matched_skills=result["matched_skills"],
            missing_skills=result["missing_skills"],
            extra_skills=result["extra_skills"],
            skill_match_ratio=result["skill_match_ratio"],
            gap_score=result["gap_score"],
            match_score=result["match_score"],
            total_required=result["total_required"],
            total_matched=result["total_matched"],
            ranked_missing=ranked_missing,
        )
    except Exception as e:
        logger.error(f"Skill gap error: {e}")
        raise HTTPException(500, str(e))


@router.post("/recommend", response_model=RecommendationResponse,
             summary="Get personalized learning recommendations")
async def recommend(request: SkillGapRequest):
    """
    Based on skill gap between resume and job, returns:
    - Learning roadmap for missing skills
    - Career path progression suggestions
    """
    try:
        gap = compute_skill_gap(
            resume_text=request.resume_text,
            job_text=request.job_text,
        )
        missing = [s["skill"] for s in gap["ranked_missing"]]
        roadmap = recommend_learning_path(missing)
        career = suggest_career_path(
            predicted_role=request.domain or "Software Engineering",
            skill_match_score=gap["match_score"],
        )
        return RecommendationResponse(
            learning_roadmap=[LearningResource(**r) for r in roadmap],
            career_path=CareerPath(**career),
        )
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(500, str(e))
