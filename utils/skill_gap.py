"""
Skill Gap Module - Detects missing skills and ranks them by importance.
Enhanced with weighted ATS scoring and better suggestions.
"""
from typing import List, Set, Dict, Any, Optional
from loguru import logger
from utils.skill_dictionary import get_skill_importance, DOMAIN_SKILLS
from utils.skill_extractor import extract_skills, get_skill_overlap


def compute_skill_gap(resume_text: str, job_text: str,
                      resume_skills: Optional[Set[str]] = None,
                      job_skills: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Compute the skill gap between a resume and a job description.

    Returns dict with matched, missing, extra, gap_score, match_score.
    """
    if resume_skills is None:
        resume_skills = extract_skills(resume_text)
    if job_skills is None:
        job_skills = extract_skills(job_text)

    overlap = get_skill_overlap(resume_skills, job_skills)
    missing = set(overlap["missing"])
    matched = set(overlap["matched"])

    # Gap score: weighted by importance
    total_importance = sum(get_skill_importance(s) for s in job_skills) or 1
    missing_importance = sum(get_skill_importance(s) for s in missing)
    matched_importance = sum(get_skill_importance(s) for s in matched)

    gap_score = round((missing_importance / total_importance) * 100, 2)
    match_score = round((matched_importance / total_importance) * 100, 2)

    ranked_missing = rank_missing_skills(missing)

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "extra_skills": sorted(overlap["extra"]),
        "skill_match_ratio": overlap["match_ratio"],
        "gap_score": gap_score,
        "match_score": match_score,
        "total_required": len(job_skills),
        "total_matched": len(matched),
        "ranked_missing": ranked_missing,
    }


def rank_missing_skills(missing_skills: Set[str]) -> List[Dict[str, Any]]:
    """
    Rank missing skills by importance score (descending).
    Returns: List of dicts [{skill, importance, domain, priority}]
    """
    ranked = []
    for skill in missing_skills:
        importance = get_skill_importance(skill)
        domain = _find_domain(skill)
        ranked.append({
            "skill": skill,
            "importance": importance,
            "domain": domain,
            "priority": _importance_to_priority(importance),
        })
    return sorted(ranked, key=lambda x: x["importance"], reverse=True)


def _find_domain(skill: str) -> str:
    """Find which domain a skill belongs to."""
    skill_lower = skill.lower()
    for domain, skills in DOMAIN_SKILLS.items():
        if skill_lower in [s.lower() for s in skills]:
            return domain
    return "General"


def _importance_to_priority(score: int) -> str:
    if score >= 9:
        return "Critical"
    elif score >= 7:
        return "High"
    elif score >= 5:
        return "Medium"
    return "Low"


def compute_ats_score(resume_text: str, job_text: str,
                      resume_skills: Optional[Set[str]] = None,
                      job_skills: Optional[Set[str]] = None,
                      years_exp: float = 0.0,
                      min_exp: float = 0.0,
                      education_level: str = "Unknown",
                      education_required: str = "Bachelors") -> Dict[str, Any]:
    """
    Compute comprehensive ATS score (0–100) from multiple signals.

    Scoring breakdown:
    - Keyword Match Score:    30%
    - Skill Match Score:      30%
    - Experience Relevance:   15%
    - Resume Structure:       10%
    - Education Match:        10%
    - Readability:             5%
    """
    from preprocessing.text_cleaner import (
        clean_text, extract_sections, calculate_readability_score
    )
    from feature_engineering.feature_builder import compute_keyword_overlap

    # 1. Keyword Match Score (0-100)
    keyword_ratio = compute_keyword_overlap(resume_text, job_text)
    keyword_score = round(min(keyword_ratio * 200, 100), 2)  # scale up since ratio < 0.5 typical

    # 2. Skill Match Score (0-100)
    gap_data = compute_skill_gap(resume_text, job_text, resume_skills, job_skills)
    skill_score = gap_data["match_score"]

    # 3. Experience Relevance (0-100)
    if min_exp <= 0:
        exp_score = 100.0
    elif years_exp >= min_exp:
        exp_score = 100.0
    else:
        ratio = years_exp / min_exp
        exp_score = round(min(ratio * 100, 100), 2)

    # 4. Resume Structure Check (0-100)
    sections = extract_sections(resume_text)
    section_score = round((sum(sections.values()) / max(len(sections), 1)) * 100, 2)

    # 5. Education Match (0-100)
    edu_map = {"Unknown": 0, "High School": 1, "Associate": 2,
               "Bachelors": 3, "Masters": 4, "PhD": 5}
    edu_val = edu_map.get(education_level, 0)
    req_val = edu_map.get(education_required, 3)
    edu_score = 100.0 if edu_val >= req_val else round((edu_val / max(req_val, 1)) * 100, 2)

    # 6. Readability Score (already 0-100)
    readability = calculate_readability_score(resume_text)

    # Weighted final ATS score
    ats_score = (
        keyword_score * 0.30 +
        skill_score   * 0.30 +
        exp_score     * 0.15 +
        section_score * 0.10 +
        edu_score     * 0.10 +
        readability   * 0.05
    )
    ats_score = round(min(ats_score, 100.0), 2)

    # Build improvement suggestions
    suggestions = []
    if keyword_score < 60:
        suggestions.append("🔑 Add more keywords from the job description (industry terms, tools, buzzwords).")
    if skill_score < 60:
        miss = gap_data["ranked_missing"][:3]
        if miss:
            top_skills = ", ".join(s["skill"] for s in miss)
            suggestions.append(f"🎯 Learn high-priority missing skills: {top_skills}.")
    if exp_score < 80:
        suggestions.append(f"⏳ Gain more experience — this job requires {min_exp:.0f} yrs; you have {years_exp:.0f} yrs.")
    if section_score < 70:
        missing_secs = [k for k, v in sections.items() if not v]
        if missing_secs:
            suggestions.append(f"📄 Add these missing resume sections: {', '.join(missing_secs)}.")
    if edu_score < 100:
        suggestions.append(f"🎓 The job prefers {education_required}+ level education.")
    if readability < 50:
        suggestions.append("✍️ Improve readability: use shorter, action-verb-led bullet points.")

    if not suggestions and ats_score >= 80:
        suggestions.append("✅ Excellent resume — focus on tailoring your cover letter!")

    return {
        "ats_score": ats_score,
        "breakdown": {
            "keyword_match": keyword_score,
            "skill_match": skill_score,
            "experience": exp_score,
            "structure": section_score,
            "education": edu_score,
            "readability": readability,
        },
        "sections_found": sections,
        "skill_gap": gap_data,
        "suggestions": suggestions,
        "grade": _score_to_grade(ats_score),
    }


def _score_to_grade(score: float) -> str:
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 55:
        return "Fair"
    elif score >= 40:
        return "Weak"
    return "Poor"
