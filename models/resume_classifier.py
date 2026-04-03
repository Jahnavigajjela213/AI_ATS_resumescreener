"""
Resume Classifier - Multi-model classification with comparison and evaluation.
"""
import os
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score,
)


SUPPORTED_MODELS = ["logistic_regression", "random_forest", "svm"]


def _build_model(model_type: str):
    """Factory: create a model instance by name."""
    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", multi_class="auto",
            class_weight="balanced", random_state=42,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
    elif model_type == "svm":
        return LinearSVC(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=42,
        )
    raise ValueError(f"Unknown model type: {model_type}. Choose from {SUPPORTED_MODELS}")


class ResumeClassifier:
    """
    Resume job-role classifier supporting LR, RF, and SVM.
    Includes training, evaluation, cross-validation, save/load.
    """

    def __init__(self, model_type: str = "random_forest"):
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {SUPPORTED_MODELS}")
        self.model_type = model_type
        self.model = _build_model(model_type)
        self.label_encoder = LabelEncoder()
        self.classes_: List[str] = []
        self.is_trained = False

    def train(self, X, y: np.ndarray) -> "ResumeClassifier":
        """Train the classifier."""
        y_enc = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_.tolist()
        self.model.fit(X, y_enc)
        self.is_trained = True
        logger.info(f"{self.model_type} trained on {X.shape[0]} samples, "
                    f"{len(self.classes_)} classes")
        return self

    def predict(self, X) -> np.ndarray:
        """Predict job role labels."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")
        y_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X) -> Dict[str, float]:
        """
        Predict class probabilities (if supported).
        Returns dict mapping class_name → probability.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            return {cls: round(float(p), 4)
                    for cls, p in zip(self.classes_, probs)}
        # SVM: use decision function, softmax-normalize
        scores = self.model.decision_function(X)[0]
        if scores.ndim == 0:
            scores = np.array([scores])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        return {cls: round(float(p), 4)
                for cls, p in zip(self.classes_, probs)}

    def evaluate(self, X_test, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate on test set. Returns metrics dict."""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        logger.info(f"{self.model_type} → Accuracy: {acc:.4f}, F1: {f1:.4f}")
        return {
            "model": self.model_type,
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1, 4),
            "precision_weighted": round(precision, 4),
            "recall_weighted": round(recall, 4),
            "classification_report": report,
        }

    def cross_validate(self, X, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Run k-fold cross-validation."""
        y_enc = self.label_encoder.transform(y) if self.is_trained else \
                self.label_encoder.fit_transform(y)
        scores = cross_val_score(self.model, X, y_enc, cv=cv, scoring="f1_weighted")
        return {"mean_f1": round(scores.mean(), 4), "std_f1": round(scores.std(), 4)}

    def save(self, directory: str) -> None:
        """Save model and label encoder to directory."""
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"classifier_{self.model_type}.pkl")
        encoder_path = os.path.join(directory, "label_encoder.pkl")
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        logger.info(f"Classifier saved to {directory}")

    def load(self, directory: str) -> "ResumeClassifier":
        """Load model and label encoder from directory."""
        model_path = os.path.join(directory, f"classifier_{self.model_type}.pkl")
        encoder_path = os.path.join(directory, "label_encoder.pkl")
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.classes_ = self.label_encoder.classes_.tolist()
        self.is_trained = True
        logger.info(f"Classifier loaded from {directory}")
        return self


def compare_all_models(X_train, y_train, X_test, y_test) -> List[Dict[str, Any]]:
    """
    Train and evaluate all supported classifiers.
    Returns sorted list of results (best accuracy first).
    """
    results = []
    for model_type in SUPPORTED_MODELS:
        clf = ResumeClassifier(model_type=model_type)
        clf.train(X_train, y_train)
        metrics = clf.evaluate(X_test, y_test)
        results.append(metrics)
    return sorted(results, key=lambda x: x["accuracy"], reverse=True)
