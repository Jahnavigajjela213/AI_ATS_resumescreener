"""
Prediction Router - Hiring success probability prediction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from backend.schemas import HiringPredictionRequest, HiringPredictionResponse
from backend.model_loader import model_store

router = APIRouter(prefix="/predict", tags=["Hiring Prediction"])


def _get_predictor():
    if not model_store.models_loaded:
        raise HTTPException(503, "Models not loaded.")
    return model_store.predictor


@router.post("/hiring", response_model=HiringPredictionResponse,
             summary="Predict hiring success probability")
async def predict_hiring(
    request: HiringPredictionRequest,
    predictor=Depends(_get_predictor),
):
    """
    Predict the probability that a candidate will be selected for hiring.
    Based on ATS score, skill match, experience, education, and other features.
    """
    try:
        features = predictor.build_features_from_dict(request.model_dump())
        prob = predictor.predict_proba(features)
        pred = predictor.predict(features)
        feat_imp = predictor.feature_importance()

        if prob >= 0.75:
            confidence = "High"
        elif prob >= 0.50:
            confidence = "Moderate"
        else:
            confidence = "Low"

        return HiringPredictionResponse(
            hiring_probability=round(prob * 100, 2),
            prediction="Likely to be Hired" if pred == 1 else "Unlikely to be Hired",
            confidence=confidence,
            feature_importance=feat_imp,
        )
    except Exception as e:
        logger.error(f"Hiring prediction error: {e}")
        raise HTTPException(500, str(e))
