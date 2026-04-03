"""
Text Cleaning Module - Reusable text preprocessing functions.
"""
import re
import string
import unicodedata
from loguru import logger


# Common resume noise patterns
URL_PATTERN = re.compile(r"http[s]?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]")
SPECIAL_CHARS = re.compile(r"[^a-zA-Z0-9\s]")
MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str, remove_urls: bool = True, remove_emails: bool = True,
               remove_phones: bool = True) -> str:
    """
    Full text cleaning pipeline.
    - Normalizes unicode characters
    - Removes URLs, emails, phone numbers
    - Removes special characters and digits
    - Strips extra whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Normalize unicode (handles accented chars)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Lowercase
    text = text.lower()

    # Remove patterns
    if remove_urls:
        text = URL_PATTERN.sub(" ", text)
    if remove_emails:
        text = EMAIL_PATTERN.sub(" ", text)
    if remove_phones:
        text = PHONE_PATTERN.sub(" ", text)

    # Remove special characters (keep spaces)
    text = SPECIAL_CHARS.sub(" ", text)

    # Remove standalone digits
    text = re.sub(r"\b\d+\b", " ", text)

    # Normalize whitespace
    text = MULTI_SPACE.sub(" ", text).strip()

    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single space."""
    return MULTI_SPACE.sub(" ", text).strip()


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters from text."""
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def extract_sections(text: str) -> dict:
    """
    Identify common resume sections.
    Returns a dict of section names mapped to whether they exist.
    """
    text_lower = text.lower()
    sections = {
        "skills": bool(re.search(r"\bskills?\b", text_lower)),
        "experience": bool(re.search(r"\bexperience\b|\bwork history\b|\bemployment\b", text_lower)),
        "education": bool(re.search(r"\beducation\b|\bdegree\b|\buniversity\b|\bcollege\b", text_lower)),
        "projects": bool(re.search(r"\bprojects?\b|\bportfolio\b", text_lower)),
        "summary": bool(re.search(r"\bsummary\b|\bobjective\b|\bprofile\b", text_lower)),
        "certifications": bool(re.search(r"\bcertif\b|\blicens\b|\bcoursera\b|\budemy\b", text_lower)),
    }
    return sections


def extract_years_experience(text: str) -> float:
    """
    Heuristic extraction of years of experience from text.
    Looks for patterns like '5 years', '3+ years', '4-5 years'.
    Returns estimated years (float).
    """
    patterns = [
        r"(\d+)\+?\s*years?\s*of\s*experience",
        r"(\d+)\+?\s*years?\s*experience",
        r"over\s+(\d+)\s*years?",
        r"(\d+)\s*yrs?\s*experience",
        r"(\d+)[–-](\d+)\s*years?",
    ]
    found = []
    for pat in patterns:
        matches = re.findall(pat, text.lower())
        for m in matches:
            if isinstance(m, tuple):
                # Range: take average
                nums = [float(x) for x in m if x]
                found.append(sum(nums) / len(nums))
            else:
                found.append(float(m))
    return max(found) if found else 0.0


def extract_education_level(text: str) -> str:
    """
    Detect highest education level from text.
    Returns one of: PhD, Masters, Bachelors, Associate, High School, Unknown.
    """
    text_lower = text.lower()
    if re.search(r"\bph\.?d\b|\bdoctor", text_lower):
        return "PhD"
    if re.search(r"\bmaster[s]?\b|\bm\.s\b|\bmsc\b|\bmba\b|\bm\.eng", text_lower):
        return "Masters"
    if re.search(r"\bbachelor[s]?\b|\bb\.s\b|\bb\.e\b|\bb\.tech\b|\bundergrad", text_lower):
        return "Bachelors"
    if re.search(r"\bassociate", text_lower):
        return "Associate"
    if re.search(r"\bhigh school\b|\bhsc\b|\bssc\b", text_lower):
        return "High School"
    return "Unknown"


def calculate_readability_score(text: str) -> float:
    """
    Simple readability score based on avg sentence length and word length.
    Returns score 0-100 (higher = more readable).
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0

    words_per_sentence = []
    avg_word_len = []
    for sent in sentences:
        words = sent.split()
        if words:
            words_per_sentence.append(len(words))
            avg_word_len.extend([len(w) for w in words])

    if not words_per_sentence:
        return 0.0

    avg_sent_len = sum(words_per_sentence) / len(words_per_sentence)
    avg_wl = sum(avg_word_len) / len(avg_word_len) if avg_word_len else 0

    # Ideal: 15-20 words/sentence, 5-7 chars/word
    sent_score = max(0, 100 - abs(avg_sent_len - 17) * 3)
    word_score = max(0, 100 - abs(avg_wl - 6) * 10)

    return round((sent_score + word_score) / 2, 2)
