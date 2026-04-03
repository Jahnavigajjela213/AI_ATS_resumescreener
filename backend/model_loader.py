"""
Model Loader - Singleton that loads all saved models on startup.
"""
import os
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.resume_classifier import ResumeClassifier
from models.resume_matcher import ResumeMatcher
from models.hiring_predictor import HiringPredictor
from feature_engineering.tfidf_vectorizer import ResumeVectorizer
from data.data_loader import load_jobs

SAVE_DIR = str(ROOT / "models" / "saved")


class ModelStore:
    """Singleton model store loaded at app startup."""

    _instance: Optional["ModelStore"] = None

    def __init__(self):
        self.classifier: Optional[ResumeClassifier] = None
        self.matcher: Optional[ResumeMatcher] = None
        self.predictor: Optional[HiringPredictor] = None
        self.vectorizer: Optional[ResumeVectorizer] = None
        self.models_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ModelStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_all(self, save_dir: str = SAVE_DIR) -> bool:
        """
        Load all saved models. If models not found, run training first.
        Returns True if successful.
        """
        try:
            logger.info(f"Loading models from: {save_dir}")
            os.makedirs(save_dir, exist_ok=True)

            # Check if models exist; if not, train them automatically
            classifier_path = os.path.join(save_dir, "classifier_random_forest.pkl")
            if not os.path.exists(classifier_path):
                logger.warning("Models not found. Running training pipeline...")
                self._run_training(save_dir)

            # Load Classifier
            self.classifier = ResumeClassifier(model_type="random_forest")
            self.classifier.load(save_dir)
            logger.info("✅ Classifier loaded")

            # Load TF-IDF Vectorizer
            vec_path = os.path.join(save_dir, "tfidf_vectorizer.pkl")
            self.vectorizer = ResumeVectorizer()
            self.vectorizer.load(vec_path)
            logger.info("✅ TF-IDF Vectorizer loaded")

            # Load Matcher with job vectors
            matcher_tfidf_path = os.path.join(save_dir, "matcher_tfidf.pkl")
            jobs_df = load_jobs()
            self.matcher = ResumeMatcher()
            if os.path.exists(matcher_tfidf_path):
                self.matcher.vectorizer.load(matcher_tfidf_path)
                # Re-fit job vectors using loaded vectorizer
                from preprocessing.nlp_pipeline import NLPPipeline
                pipeline = NLPPipeline()
                texts = [pipeline.transform(str(t)) for t in jobs_df["description"].fillna("").tolist()]
                self.matcher.job_vectors = self.matcher.vectorizer.transform(texts)
                self.matcher.job_metadata = jobs_df.to_dict("records")
                self.matcher.is_fitted = True
            else:
                self.matcher.fit_jobs(jobs_df)
            logger.info("✅ Resume Matcher loaded")

            # Load Hiring Predictor
            self.predictor = HiringPredictor()
            self.predictor.load(save_dir)
            logger.info("✅ Hiring Predictor loaded")

            self.models_loaded = True
            logger.info("🚀 All models loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
            return False

    def _run_training(self, save_dir: str):
        """Run the training pipeline to generate model files."""
        import subprocess
        train_script = str(ROOT / "models" / "train_models.py")
        result = subprocess.run(
            [sys.executable, train_script],
            capture_output=True, text=True,
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            logger.error(f"Training failed:\n{result.stderr}")
            raise RuntimeError("Model training failed. Please run 'python models/train_models.py' manually.")
        logger.info("Training completed successfully.")


# Global instance
model_store = ModelStore.get_instance()
