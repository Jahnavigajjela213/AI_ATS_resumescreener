"""
Pydantic Schemas for FastAPI request/response models.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────

class JobDescriptionRequest(BaseModel):
    job_title: str = Field(..., example="Senior Data Scientist")
    job_description: str = Field(..., example="We are looking for a Data Scientist...")
    required_skills: Optional[str] = Field(None, example="Python,ML,SQL")
    min_experience: Optional[float] = Field(0, example=4)
    education_required: Optional[str] = Field("Bachelors", example="Masters")


class ResumeTextRequest(BaseModel):
    resume_text: str = Field(..., example="Experienced Data Scientist with 5 years...")


class ATSRequest(BaseModel):
    resume_text: str = Field(..., example="Experienced Data Scientist...")
    job_description: str = Field(..., example="We are looking for a Data Scientist...")
    job_title: Optional[str] = Field("", example="Data Scientist")
    required_skills: Optional[str] = Field("", example="Python,ML,SQL")
    min_experience: Optional[float] = Field(0, example=4)
    education_required: Optional[str] = Field("Bachelors", example="Masters")


class MatchRequest(BaseModel):
    resume_text: str = Field(..., example="Experienced Python developer...")
    top_n: Optional[int] = Field(5, ge=1, le=15, example=5)


class SkillGapRequest(BaseModel):
    resume_text: str
    job_text: str
    domain: Optional[str] = Field(None, example="Data Science")


class HiringPredictionRequest(BaseModel):
    ats_score: float = Field(..., ge=0, le=100, example=72.5)
    skill_match_score: float = Field(..., ge=0, le=100, example=65.0)
    years_experience: float = Field(..., ge=0, example=4.0)
    education_level: str = Field(..., example="Masters")
    skill_count: int = Field(..., ge=0, example=12)
    keyword_match_score: float = Field(..., ge=0, le=100, example=70.0)
    structure_score: float = Field(..., ge=0, le=100, example=80.0)


# ── Response Models ────────────────────────────────────────────────────────────

class ATSBreakdown(BaseModel):
    keyword_match: float
    skill_match: float
    experience: float
    structure: float
    education: float
    readability: float


class ATSResponse(BaseModel):
    ats_score: float
    grade: str
    breakdown: ATSBreakdown
    sections_found: Dict[str, bool]
    suggestions: List[str]
    extracted_skills: List[str]
    missing_skills: List[str]


class PredictedRole(BaseModel):
    predicted_role: str
    confidence_scores: Dict[str, float]
    top_keywords: List[str]


class JobMatchResult(BaseModel):
    rank: int
    job_title: str
    job_id: Any
    similarity_score: float
    match_percentage: float
    required_skills: str
    min_experience: float
    education_required: str
    salary_range: str
    description_snippet: str


class MatchResponse(BaseModel):
    resume_summary: str
    total_jobs_scanned: int
    matches: List[JobMatchResult]


class SkillInfo(BaseModel):
    skill: str
    importance: int
    domain: str
    priority: str


class SkillGapResponse(BaseModel):
    matched_skills: List[str]
    missing_skills: List[str]
    extra_skills: List[str]
    skill_match_ratio: float
    gap_score: float
    match_score: float
    total_required: int
    total_matched: int
    ranked_missing: List[SkillInfo]


class LearningResource(BaseModel):
    skill: str
    courses: List[str]
    estimated_time: str
    difficulty: str
    certifications: List[str]


class CareerPath(BaseModel):
    category: str
    career_stages: List[str]
    estimated_current_level: str
    next_step: str
    recommended_actions: List[str]


class RecommendationResponse(BaseModel):
    learning_roadmap: List[LearningResource]
    career_path: CareerPath


class HiringPredictionResponse(BaseModel):
    hiring_probability: float
    prediction: str
    confidence: str
    feature_importance: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
