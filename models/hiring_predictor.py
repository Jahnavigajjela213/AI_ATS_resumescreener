"""
Hiring Success Predictor - XGBoost model with feature importance.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not installed; falling back to GradientBoosting.")

FEATURE_NAMES = [
    "ats_score",
    "skill_match_score",
    "years_experience",
    "education_level",
    "skill_count",
    "keyword_match_score",
    "structure_score",
]

EDUCATION_MAP = {
    "Unknown": 0, "High School": 1, "Associate": 2,
    "Bachelors": 3, "Masters": 4, "PhD": 5,
}


class HiringPredictor:
    """
    Binary predictor for hiring success (1 = likely hired, 0 = not hired).
    Uses XGBoost (or GradientBoosting fallback).
    """

    def __init__(self, use_xgboost: bool = True):
        self.use_xgboost = use_xgboost and HAS_XGB
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = FEATURE_NAMES

    def _build_model(self):
        if self.use_xgboost and HAS_XGB:
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42,
        )

    @staticmethod
    def build_features_from_dict(data: Dict[str, Any]) -> np.ndarray:
        """
        Build feature vector from a dict of resume/job attributes.

        Expected keys: ats_score, skill_match_score, years_experience,
                       education_level (str), skill_count, keyword_match_score,
                       structure_score
        """
        edu = EDUCATION_MAP.get(str(data.get("education_level", "Unknown")), 0)
        features = [
            float(data.get("ats_score", 50)),
            float(data.get("skill_match_score", 0)),
            float(data.get("years_experience", 0)),
            float(edu),
            float(data.get("skill_count", 0)),
            float(data.get("keyword_match_score", 0)),
            float(data.get("structure_score", 0)),
        ]
        return np.array([features])

    def generate_synthetic_training_data(self, n_samples: int = 500) -> tuple:
        """
        Generate synthetic labeled training data for the hiring predictor.
        In a real system this would come from historical hiring records.
        """
        np.random.seed(42)
        ats_scores = np.random.uniform(20, 100, n_samples)
        skill_scores = np.random.uniform(0, 100, n_samples)
        experience = np.random.uniform(0, 15, n_samples)
        education = np.random.randint(0, 6, n_samples).astype(float)
        skill_count = np.random.randint(2, 20, n_samples).astype(float)
        keyword_score = np.random.uniform(0, 100, n_samples)
        structure_score = np.random.uniform(40, 100, n_samples)

        X = np.column_stack([
            ats_scores, skill_scores, experience, education,
            skill_count, keyword_score, structure_score,
        ])

        # Hiring probability based on weighted score
        score = (
            ats_scores * 0.35 + skill_scores * 0.30 +
            np.clip(experience / 10, 0, 1) * 100 * 0.15 +
            education / 5 * 100 * 0.10 + keyword_score * 0.10
        )
        prob = 1 / (1 + np.exp(-(score - 65) / 10))
        y = (prob > 0.5).astype(int)

        return X, y

    def train(self, X: np.ndarray = None, y: np.ndarray = None) -> "HiringPredictor":
        """Train the predictor. If X/y not provided, generates synthetic data."""
        if X is None:
            X, y = self.generate_synthetic_training_data()

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info(f"HiringPredictor trained with {X.shape[0]} samples "
                    f"({'XGBoost' if self.use_xgboost else 'GradientBoosting'})")
        return self

    def predict_proba(self, X: np.ndarray) -> float:
        """Return probability of being hired (0.0–1.0)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        X_scaled = self.scaler.transform(X)
        prob = self.model.predict_proba(X_scaled)[0][1]
        return round(float(prob), 4)

    def predict(self, X: np.ndarray) -> int:
        """Return binary prediction (1 = hire, 0 = reject)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        X_scaled = self.scaler.transform(X)
        return int(self.model.predict(X_scaled)[0])

    def feature_importance(self) -> List[Dict[str, Any]]:
        """Return feature importance scores sorted descending."""
        if not self.is_trained:
            return []
        importances = self.model.feature_importances_
        ranked = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True,
        )
        return [{"feature": f, "importance": round(float(i), 4)}
                for f, i in ranked]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate on a test set."""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        return {
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4),
            "classification_report": report,
        }

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, "hiring_predictor.pkl"))
        joblib.dump(self.scaler, os.path.join(directory, "hiring_scaler.pkl"))
        logger.info(f"HiringPredictor saved to {directory}")

    def load(self, directory: str) -> "HiringPredictor":
        self.model = joblib.load(os.path.join(directory, "hiring_predictor.pkl"))
        self.scaler = joblib.load(os.path.join(directory, "hiring_scaler.pkl"))
        self.is_trained = True
        logger.info(f"HiringPredictor loaded from {directory}")
        return self
