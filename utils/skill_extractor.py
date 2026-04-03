"""
Skill Extractor Module - Hybrid keyword + regex skill extraction.
Improved multi-word phrase matching and case-insensitive boundary matching.
"""
import re
from typing import List, Set, Optional, Dict
from loguru import logger
from utils.skill_dictionary import DOMAIN_SKILLS, get_all_skills, get_skills_by_domain


def _normalize(text: str) -> str:
    """Lowercase and normalize whitespace for matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


# Pre-compile boundary pattern for single-word skills
def _make_pattern(skill_norm: str) -> re.Pattern:
    if " " in skill_norm:
        # Phrase match: allow flexible whitespace
        parts = [re.escape(p) for p in skill_norm.split()]
        return re.compile(r"\b" + r"\s+".join(parts) + r"\b")
    else:
        return re.compile(r"\b" + re.escape(skill_norm) + r"\b")


# Build a cached lookup: normalized skill → display skill
_SKILL_PATTERNS: Optional[Dict[str, tuple]] = None  # norm → (display, pattern)


def _get_skill_patterns() -> Dict[str, tuple]:
    global _SKILL_PATTERNS
    if _SKILL_PATTERNS is None:
        _SKILL_PATTERNS = {}
        for skill in get_all_skills():
            norm = _normalize(skill)
            if norm not in _SKILL_PATTERNS:
                _SKILL_PATTERNS[norm] = (skill.lower(), _make_pattern(norm))
    return _SKILL_PATTERNS


def extract_skills(text: str, domain: Optional[str] = None) -> Set[str]:
    """
    Extract skills from text using keyword matching.

    Args:
        text: Raw resume or job description text
        domain: Optional domain filter (e.g., "Data Science", "AI/ML")

    Returns:
        Set of matched skill strings (lowercase)
    """
    if not text:
        return set()

    norm_text = _normalize(text)

    if domain:
        skill_pool = get_skills_by_domain(domain)
        patterns = {_normalize(s): (s.lower(), _make_pattern(_normalize(s))) for s in skill_pool}
    else:
        patterns = _get_skill_patterns()

    found: Set[str] = set()
    for norm_skill, (display, pattern) in patterns.items():
        if pattern.search(norm_text):
            found.add(display)

    logger.debug(f"Extracted {len(found)} skills from text (domain={domain})")
    return found


def extract_skills_by_domain(text: str) -> Dict[str, Set[str]]:
    """
    Extract skills from text grouped by domain.

    Returns:
        Dict mapping domain name → set of found skills
    """
    results: Dict[str, Set[str]] = {}
    for domain in DOMAIN_SKILLS:
        found = extract_skills(text, domain)
        if found:
            results[domain] = found
    return results


def extract_skills_from_csv_column(skills_str: str) -> Set[str]:
    """
    Parse comma-separated skills string from CSV data.
    E.g. "Python,Machine Learning,SQL" → {'python', 'machine learning', 'sql'}
    """
    if not skills_str or not isinstance(skills_str, str):
        return set()
    return {s.strip().lower() for s in skills_str.split(",") if s.strip()}


def get_skill_overlap(resume_skills: Set[str], job_skills: Set[str]) -> Dict[str, any]:
    """
    Compute overlap between resume skills and job skills.

    Returns:
        dict with 'matched', 'missing', 'extra', 'match_ratio'
    """
    # Normalize both sets for fair comparison
    resume_norm = {s.lower().strip() for s in resume_skills}
    job_norm = {s.lower().strip() for s in job_skills}

    matched = resume_norm.intersection(job_norm)
    missing = job_norm - resume_norm
    extra = resume_norm - job_norm
    match_ratio = len(matched) / len(job_norm) if job_norm else 0.0
    return {
        "matched": sorted(matched),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "match_ratio": round(match_ratio, 4),
    }
