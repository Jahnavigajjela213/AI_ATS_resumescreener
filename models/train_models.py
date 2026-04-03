"""
Model Training Script - End-to-end training pipeline.
Trains all models and saves them to models/saved/.
"""
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from sklearn.model_selection import train_test_split

from data.data_loader import load_resumes, load_jobs
from preprocessing.nlp_pipeline import NLPPipeline
from feature_engineering.tfidf_vectorizer import ResumeVectorizer
from feature_engineering.feature_builder import build_features
from models.resume_classifier import ResumeClassifier, compare_all_models
from models.resume_matcher import ResumeMatcher
from models.hiring_predictor import HiringPredictor

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")


def setup_logging():
    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/training.log", level="DEBUG", rotation="10 MB")


def train_classifier(X_train, y_train, X_test, y_test, save_dir: str):
    """Train all classifiers, compare, and save the best one."""
    logger.info("=" * 60)
    logger.info("TRAINING RESUME CLASSIFIERS")
    logger.info("=" * 60)

    results = compare_all_models(X_train, y_train, X_test, y_test)

    logger.info("\n📊 CLASSIFIER COMPARISON RESULTS:")
    for r in results:
        logger.info(f"  {r['model']:25s} | Accuracy: {r['accuracy']:.4f} | F1: {r['f1_weighted']:.4f}")

    best = results[0]
    logger.info(f"\n✅ Best model: {best['model']} (Accuracy: {best['accuracy']:.4f})")

    # Save the best model
    best_clf = ResumeClassifier(model_type=best["model"])
    best_clf.train(X_train, y_train)
    best_clf.save(save_dir)

    # Save comparison results
    results_path = os.path.join(save_dir, "classifier_comparison.json")
    with open(results_path, "w") as f:
        # Remove non-serializable parts
        clean = [{k: v for k, v in r.items() if k != "classification_report"} for r in results]
        json.dump(clean, f, indent=2)

    return best_clf, results


def train_matcher(jobs_df, save_dir: str):
    """Fit ResumeMatcher on job descriptions and save vectorizer."""
    logger.info("=" * 60)
    logger.info("TRAINING RESUME MATCHER")
    logger.info("=" * 60)

    matcher = ResumeMatcher()
    matcher.fit_jobs(jobs_df)
    matcher.vectorizer.save(os.path.join(save_dir, "matcher_tfidf.pkl"))
    logger.info(f"✅ ResumeMatcher fitted on {len(jobs_df)} jobs")
    return matcher


def train_hiring_predictor(save_dir: str):
    """Train hiring success predictor and save."""
    logger.info("=" * 60)
    logger.info("TRAINING HIRING SUCCESS PREDICTOR")
    logger.info("=" * 60)

    predictor = HiringPredictor(use_xgboost=True)
    X, y = predictor.generate_synthetic_training_data(n_samples=1000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    predictor.train(X_train, y_train)
    metrics = predictor.evaluate(X_test, y_test)

    logger.info(f"✅ HiringPredictor trained | Accuracy: {metrics['accuracy']:.4f} | AUC: {metrics['roc_auc']:.4f}")

    feat_imp = predictor.feature_importance()
    logger.info("\n🔍 FEATURE IMPORTANCE:")
    for f in feat_imp:
        logger.info(f"  {f['feature']:25s} | {f['importance']:.4f}")

    predictor.save(save_dir)
    return predictor, metrics


def main():
    setup_logging()
    os.makedirs(SAVE_DIR, exist_ok=True)

    logger.info("🚀 Starting AI Resume Screening - Model Training Pipeline")
    logger.info(f"📁 Models will be saved to: {SAVE_DIR}")

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    logger.info("\n📂 Loading datasets...")
    resumes_df = load_resumes()
    jobs_df = load_jobs()
    logger.info(f"   Loaded {len(resumes_df)} resumes, {len(jobs_df)} job descriptions")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    logger.info("\n🧹 Preprocessing text...")
    pipeline = NLPPipeline()
    resumes_df["clean_text"] = resumes_df["text"].apply(pipeline.transform)

    # ── 3. Feature Engineering ────────────────────────────────────────────────
    logger.info("\n⚙️  Building features...")
    X, y, vectorizer = build_features(resumes_df)
    vectorizer.save(os.path.join(SAVE_DIR, "tfidf_vectorizer.pkl"))
    logger.info(f"   Feature matrix: {X.shape}, Labels: {len(set(y))} classes")

    # ── 4. Train/Test Split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 5. Train Classifier ───────────────────────────────────────────────────
    best_clf, clf_results = train_classifier(X_train, y_train, X_test, y_test, SAVE_DIR)

    # ── 6. Train Matcher ──────────────────────────────────────────────────────
    matcher = train_matcher(jobs_df, SAVE_DIR)

    # ── 7. Train Hiring Predictor ─────────────────────────────────────────────
    predictor, pred_metrics = train_hiring_predictor(SAVE_DIR)

    # ── 8. Done ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    logger.info(f"📁 Saved models to: {SAVE_DIR}")
    logger.info("=" * 60)
    logger.info("\nSaved files:")
    for f in os.listdir(SAVE_DIR):
        logger.info(f"  📄 {f}")


if __name__ == "__main__":
    main()
