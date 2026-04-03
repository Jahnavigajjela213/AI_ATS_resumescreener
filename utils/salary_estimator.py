"""
Salary Estimator — estimates expected salary range based on skills, experience, and education.
Uses a rule-based model with market-rate salary bands (USD/year).
"""
from typing import Dict, Any, List, Tuple


# ── Base salary bands by role (USD/year) ─────────────────────────────────────
ROLE_SALARY_BANDS: Dict[str, Tuple[int, int]] = {
    "Data Science":          (75_000, 175_000),
    "AI/ML":                 (95_000, 210_000),
    "Software Engineering":  (80_000, 200_000),
    "DevOps/Cloud":          (85_000, 185_000),
    "Data Engineering":      (85_000, 185_000),
    "Cybersecurity":         (80_000, 175_000),
    "Data Analytics":        (60_000, 130_000),
    "General":               (55_000, 140_000),
}

# ── High-value skill premium bonuses (USD, additive) ─────────────────────────
SKILL_PREMIUMS: Dict[str, int] = {
    "machine learning": 15_000, "deep learning": 15_000,
    "llm": 20_000, "generative ai": 20_000, "langchain": 12_000,
    "rag": 12_000, "mlflow": 8_000, "transformers": 15_000,
    "kubernetes": 12_000, "aws": 10_000, "azure": 10_000, "gcp": 10_000,
    "terraform": 8_000, "spark": 10_000, "kafka": 10_000,
    "snowflake": 8_000, "dbt": 7_000, "pytorch": 10_000,
    "tensorflow": 10_000, "xgboost": 5_000, "docker": 6_000,
    "fastapi": 5_000, "react": 6_000, "typescript": 6_000,
    "golang": 10_000, "rust": 10_000, "scala": 8_000,
    "ci/cd": 6_000, "microservices": 8_000,
}

# ── Education bonuses ─────────────────────────────────────────────────────────
EDUCATION_BONUS: Dict[str, int] = {
    "PhD":        20_000,
    "Masters":    12_000,
    "Bachelors":  0,
    "Associate":  -5_000,
    "High School":-10_000,
    "Unknown":    -3_000,
}

# ── Experience multipliers (applied to band range) ────────────────────────────
def _exp_multiplier(years: float) -> float:
    if years >= 10:
        return 1.0          # top of band
    elif years >= 7:
        return 0.85
    elif years >= 5:
        return 0.73
    elif years >= 3:
        return 0.60
    elif years >= 1:
        return 0.48
    else:
        return 0.38         # entry level


def estimate_salary(
    matched_skills: List[str],
    years_exp: float,
    education_level: str,
    predicted_role: str = "Software Engineering",
    location: str = "United States",
) -> Dict[str, Any]:
    """
    Estimate salary range based on skills, experience, education, and role.

    Returns:
        dict with low, mid, high (USD), skill_premium, education_bonus,
        breakdown explanation, and location-adjusted notes.
    """
    # Base band for role
    band_low, band_high = ROLE_SALARY_BANDS.get(predicted_role, ROLE_SALARY_BANDS["General"])
    band_range = band_high - band_low

    # Experience position within band
    pos = _exp_multiplier(years_exp)
    base_salary = round(band_low + pos * band_range)

    # Skill premium (cap at 35k)
    skill_premium = min(
        sum(SKILL_PREMIUMS.get(s.lower(), 0) for s in matched_skills),
        35_000
    )

    # Education adjustment
    edu_bonus = EDUCATION_BONUS.get(education_level, 0)

    # Final range
    mid = base_salary + skill_premium + edu_bonus
    low = round(mid * 0.88)
    high = round(mid * 1.18)

    # Location factor
    loc_factors = {
        "United States": 1.0,
        "United Kingdom": 0.72,
        "Germany": 0.68,
        "Canada": 0.78,
        "Australia": 0.75,
        "India": 0.22,
        "Singapore": 0.85,
        "Netherlands": 0.72,
        "Remote": 0.95,
    }
    loc_factor = loc_factors.get(location, 0.80)

    # Top skills contributing to premium
    contributing_skills = [
        {"skill": s, "bonus": SKILL_PREMIUMS[s.lower()]}
        for s in matched_skills
        if s.lower() in SKILL_PREMIUMS
    ]
    contributing_skills.sort(key=lambda x: x["bonus"], reverse=True)

    return {
        "low_usd": low,
        "mid_usd": mid,
        "high_usd": high,
        "low_local": round(low * loc_factor),
        "mid_local": round(mid * loc_factor),
        "high_local": round(high * loc_factor),
        "location": location,
        "location_factor": loc_factor,
        "currency_note": "USD" if location == "United States" else f"~USD × {loc_factor}",
        "skill_premium": skill_premium,
        "education_bonus": edu_bonus,
        "base_estimate": base_salary,
        "top_contributing_skills": contributing_skills[:5],
        "role": predicted_role,
        "experience_percentile": f"{round(pos * 100)}th",
    }


def get_location_options() -> List[str]:
    return ["United States", "United Kingdom", "Germany", "Canada",
            "Australia", "India", "Singapore", "Netherlands", "Remote", "Other"]
