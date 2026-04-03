"""
TF-IDF Vectorizer Module - Reusable vectorizer for resume/job text.
"""
import os
import joblib
import numpy as np
from pathlib import Path
from scipy.sparse import spmatrix
from typing import List, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger


class ResumeVectorizer:
    """
    Wrapper around sklearn TfidfVectorizer with save/load support.

    Usage:
        vectorizer = ResumeVectorizer()
        vectorizer.fit(corpus)
        vectors = vectorizer.transform(texts)
        vectorizer.save("models/saved/tfidf.pkl")
    """

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2),
                 min_df: int = 1, max_df: float = 0.95, sublinear_tf: bool = True):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self._vectorizer: Optional[TfidfVectorizer] = None
        self.is_fitted = False

    def _build_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
        )

    def fit(self, corpus: List[str]) -> "ResumeVectorizer":
        """Fit the vectorizer on a corpus of texts."""
        self._vectorizer = self._build_vectorizer()
        self._vectorizer.fit(corpus)
        self.is_fitted = True
        logger.info(f"TF-IDF vectorizer fitted on {len(corpus)} documents, "
                    f"{len(self.get_feature_names())} features")
        return self

    def transform(self, texts: List[str]):
        """Transform texts to TF-IDF matrix."""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer not fitted. Call .fit() first.")
        return self._vectorizer.transform(texts)

    def fit_transform(self, corpus: List[str]):
        """Fit and transform in one step."""
        self.fit(corpus)
        return self.transform(corpus)

    def get_feature_names(self) -> List[str]:
        """Return feature names (vocabulary terms)."""
        if not self.is_fitted:
            return []
        return self._vectorizer.get_feature_names_out().tolist()

    def get_vector_for_text(self, text: str) -> np.ndarray:
        """Transform a single text string to dense vector."""
        vec = self.transform([text])
        return vec.toarray()[0]

    def save(self, path: str) -> None:
        """Save the fitted vectorizer to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted vectorizer.")
        os.makedirs(Path(path).parent, exist_ok=True)
        joblib.dump(self._vectorizer, path)
        logger.info(f"Vectorizer saved to {path}")

    def load(self, path: str) -> "ResumeVectorizer":
        """Load a fitted vectorizer from disk."""
        self._vectorizer = joblib.load(path)
        self.is_fitted = True
        logger.info(f"Vectorizer loaded from {path}")
        return self

    @classmethod
    def from_file(cls, path: str) -> "ResumeVectorizer":
        """Class method to create instance from saved file."""
        instance = cls()
        return instance.load(path)
