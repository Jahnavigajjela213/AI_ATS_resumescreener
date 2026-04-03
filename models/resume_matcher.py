"""
Resume-Job Matcher - Cosine similarity ranking system.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from feature_engineering.tfidf_vectorizer import ResumeVectorizer
from preprocessing.nlp_pipeline import NLPPipeline


class ResumeMatcher:
    """
    Matches a resume against a list of job descriptions using TF-IDF cosine similarity.
    Returns a ranked list of jobs with similarity scores.
    """

    def __init__(self, vectorizer: Optional[ResumeVectorizer] = None):
        self.vectorizer = vectorizer or ResumeVectorizer(max_features=5000)
        self.pipeline = NLPPipeline()
        self.job_vectors = None
        self.job_metadata: List[Dict] = []
        self.is_fitted = False

    def fit_jobs(self, jobs_df: pd.DataFrame) -> "ResumeMatcher":
        """
        Fit the vectorizer on job descriptions and store job vectors.

        Args:
            jobs_df: DataFrame with columns 'title', 'description', and optionally others
        """
        texts = [self.pipeline.transform(str(t))
                 for t in jobs_df["description"].fillna("").tolist()]

        if not self.vectorizer.is_fitted:
            self.job_vectors = self.vectorizer.fit_transform(texts)
        else:
            self.job_vectors = self.vectorizer.transform(texts)

        self.job_metadata = jobs_df.to_dict("records")
        self.is_fitted = True
        logger.info(f"ResumeMatcher fitted on {len(texts)} job descriptions")
        return self

    def match(self, resume_text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Compute cosine similarity between resume and all jobs.

        Args:
            resume_text: Raw resume text
            top_n: Number of top matches to return

        Returns:
            List of ranked matches, each with job metadata and score
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit_jobs() first.")

        cleaned = self.pipeline.transform(resume_text)
        resume_vec = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(resume_vec, self.job_vectors)[0]

        top_indices = np.argsort(similarities)[::-1][:top_n]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            job = self.job_metadata[idx].copy()
            score = float(similarities[idx])
            results.append({
                "rank": rank,
                "job_title": job.get("title", "Unknown"),
                "job_id": job.get("id", idx),
                "similarity_score": round(score, 4),
                "match_percentage": round(score * 100, 2),
                "required_skills": job.get("required_skills", ""),
                "min_experience": job.get("min_experience", 0),
                "education_required": job.get("education_required", ""),
                "salary_range": job.get("salary_range", ""),
                "description_snippet": str(job.get("description", ""))[:200] + "...",
            })

        logger.info(f"Matched resume against {len(self.job_metadata)} jobs. "
                    f"Top match: {results[0]['job_title']} ({results[0]['match_percentage']}%)")
        return results

    def get_similarity_score(self, resume_text: str, job_text: str) -> float:
        """
        Direct cosine similarity between one resume and one job description.
        """
        r_clean = self.pipeline.transform(resume_text)
        j_clean = self.pipeline.transform(job_text)

        # Use a fresh vectorizer fitted on both texts
        temp_vec = ResumeVectorizer(max_features=2000)
        temp_vec.fit([r_clean, j_clean])
        vecs = temp_vec.transform([r_clean, j_clean])
        sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        return round(float(sim), 4)
