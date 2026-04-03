"""
Feature Builder Module - Combines TF-IDF and structured features.
"""
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from typing import List, Tuple, Optional
from loguru import logger
from preprocessing.text_cleaner import extract_years_experience, extract_education_level
from feature_engineering.tfidf_vectorizer import ResumeVectorizer
from preprocessing.nlp_pipeline import NLPPipeline

EDUCATION_MAP = {
    "Unknown": 0,
    "High School": 1,
    "Associate": 2,
    "Bachelors": 3,
    "Masters": 4,
    "PhD": 5,
}


def education_to_int(level: str) -> int:
    return EDUCATION_MAP.get(level, 0)


def build_structured_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build structured numeric features from resume DataFrame.
    Columns expected: years_experience, education_level or derived from text.
    """
    features = []
    for _, row in df.iterrows():
        exp = float(row.get("years_experience", 0) or 0)
        edu_str = str(row.get("education_level", "Unknown"))
        edu = education_to_int(edu_str)
        # Skills count
        skills_str = str(row.get("skills", "")) or ""
        skill_count = len([s for s in skills_str.split(",") if s.strip()])
        features.append([exp, edu, skill_count])
    return np.array(features, dtype=float)


def build_features(df: pd.DataFrame, vectorizer: Optional[ResumeVectorizer] = None,
                   fit: bool = True) -> Tuple[object, np.ndarray, ResumeVectorizer]:
    """
    Build full feature matrix combining TF-IDF text + structured features.

    Args:
        df: DataFrame with 'text', 'category', 'years_experience', 'education_level'
        vectorizer: existing vectorizer; if None, new one is created
        fit: whether to fit the vectorizer (False if already fitted)

    Returns:
        X: combined feature matrix (sparse + dense)
        y: label array
        vectorizer: fitted vectorizer
    """
    pipeline = NLPPipeline()
    texts = [pipeline.transform(str(t)) for t in df["text"].fillna("").tolist()]

    if vectorizer is None:
        vectorizer = ResumeVectorizer(max_features=5000)

    if fit:
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)

    structured = build_structured_features(df)
    structured_sparse = csr_matrix(structured)

    X = hstack([tfidf_matrix, structured_sparse])

    # Labels
    y = df["category"].fillna("Unknown").values if "category" in df.columns else None

    logger.info(f"Feature matrix shape: {X.shape}")
    return X, y, vectorizer


def compute_keyword_overlap(resume_text: str, job_text: str) -> float:
    """
    Simple keyword overlap ratio between resume and job description.
    Returns value 0.0–1.0.
    """
    pipeline = NLPPipeline()
    resume_words = set(pipeline.transform(resume_text).split())
    job_words = set(pipeline.transform(job_text).split())
    if not job_words:
        return 0.0
    overlap = resume_words.intersection(job_words)
    return len(overlap) / len(job_words)
