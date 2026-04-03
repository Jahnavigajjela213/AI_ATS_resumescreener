"""
Resume Router - File upload, role prediction, and ATS scoring endpoints.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from loguru import logger

from backend.schemas import (
    ATSRequest, ATSResponse, ATSBreakdown,
    PredictedRole, ResumeTextRequest,
)
from backend.model_loader import model_store
from data.data_loader import extract_text_from_bytes
from preprocessing.nlp_pipeline import NLPPipeline
from preprocessing.text_cleaner import extract_years_experience, extract_education_level
from utils.skill_extractor import extract_skills
from utils.skill_gap import compute_ats_score
from feature_engineering.feature_builder import build_structured_features
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

router = APIRouter(prefix="/resume", tags=["Resume"])
pipeline = NLPPipeline()


def _get_classifier():
    if not model_store.models_loaded:
        raise HTTPException(503, "Models not loaded. Please wait or restart the server.")
    return model_store.classifier


@router.post("/upload", summary="Upload a resume file and extract text")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a PDF, DOCX, or TXT resume file.
    Returns the extracted text and basic metadata.
    """
    allowed = {".pdf", ".docx", ".doc", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}")

    try:
        content = await file.read()
        text = extract_text_from_bytes(content, file.filename)

        if not text.strip():
            raise HTTPException(422, "Could not extract text from the uploaded file.")

        exp = extract_years_experience(text)
        edu = extract_education_level(text)
        skills = extract_skills(text)

        logger.info(f"Uploaded resume: {file.filename} | {len(text)} chars | {exp} yrs | {edu}")

        return {
            "filename": file.filename,
            "text": text,
            "text_length": len(text),
            "extracted_years_experience": exp,
            "detected_education": edu,
            "extracted_skills": sorted(skills),
            "skill_count": len(skills),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, f"Error processing file: {str(e)}")


@router.post("/predict-role", response_model=PredictedRole,
             summary="Predict the job role from resume text")
async def predict_role(request: ResumeTextRequest,
                       classifier=Depends(_get_classifier)):
    """
    Classify the resume into a job role/category.
    Returns predicted role and confidence scores.
    """
    try:
        clean = pipeline.transform(request.resume_text)
        vec = model_store.vectorizer.transform([clean])

        # Build structured features
        df_single = pd.DataFrame([{
            "text": request.resume_text,
            "years_experience": extract_years_experience(request.resume_text),
            "education_level": extract_education_level(request.resume_text),
            "skills": ",".join(extract_skills(request.resume_text)),
        }])
        structured = build_structured_features(df_single)
        X = hstack([vec, csr_matrix(structured)])

        probs = classifier.predict_proba(X)
        top_role = max(probs, key=probs.get)
        keywords = pipeline.get_keywords(request.resume_text, top_n=15)

        return PredictedRole(
            predicted_role=top_role,
            confidence_scores=probs,
            top_keywords=keywords,
        )
    except Exception as e:
        logger.error(f"Role prediction error: {e}")
        raise HTTPException(500, str(e))


@router.post("/ats-score", response_model=ATSResponse,
             summary="Calculate comprehensive ATS score for a resume")
async def calculate_ats_score(request: ATSRequest):
    """
    Compute ATS score (0–100) across 6 dimensions:
    keyword match, skill match, experience, structure, education, readability.
    Returns breakdown, suggestions, and skill gaps.
    """
    try:
        years_exp = extract_years_experience(request.resume_text)
        edu_level = extract_education_level(request.resume_text)
        resume_skills = extract_skills(request.resume_text)

        job_skills = set()
        if request.required_skills:
            job_skills = {s.strip().lower() for s in request.required_skills.split(",") if s.strip()}
        if not job_skills:
            job_skills = extract_skills(request.job_description)

        result = compute_ats_score(
            resume_text=request.resume_text,
            job_text=request.job_description,
            resume_skills=resume_skills,
            job_skills=job_skills,
            years_exp=years_exp,
            min_exp=float(request.min_experience or 0),
            education_level=edu_level,
            education_required=request.education_required or "Bachelors",
        )

        missing_skills = [s["skill"] for s in result["skill_gap"]["ranked_missing"]]

        return ATSResponse(
            ats_score=result["ats_score"],
            grade=result["grade"],
            breakdown=ATSBreakdown(**result["breakdown"]),
            sections_found=result["sections_found"],
            suggestions=result["suggestions"],
            extracted_skills=sorted(resume_skills),
            missing_skills=missing_skills,
        )
    except Exception as e:
        logger.error(f"ATS score error: {e}")
        raise HTTPException(500, str(e))
