"""
NLP Pipeline Module - Reusable tokenization and lemmatization pipeline.
Handles newer NLTK versions that require 'punkt_tab' instead of 'punkt'.
"""
import re
import nltk
from typing import List, Optional
from loguru import logger
from preprocessing.text_cleaner import clean_text

# Download required NLTK data on first use
_NLTK_INITIALIZED = False


def _init_nltk():
    global _NLTK_INITIALIZED
    if not _NLTK_INITIALIZED:
        resources = [
            "punkt", "punkt_tab", "stopwords", "wordnet",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "omw-1.4"
        ]
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
        _NLTK_INITIALIZED = True


def get_stopwords() -> set:
    """Return default English stopwords set."""
    _init_nltk()
    from nltk.corpus import stopwords
    stop = set(stopwords.words("english"))
    # Add domain-specific noise words
    stop.update([
        "etc", "also", "using", "used", "use", "work", "working",
        "worked", "year", "years", "experience", "strong", "good",
        "well", "able", "knowledge", "understanding", "include",
        "including", "skillset", "proven", "excellent", "ability",
        "seek", "seeking", "objective", "summary", "responsible",
    ])
    return stop


def tokenize(text: str) -> List[str]:
    """Tokenize text into word tokens with fallback."""
    _init_nltk()
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except Exception:
        # Robust fallback
        return re.findall(r"\b[a-zA-Z]+\b", text)


def remove_stopwords(tokens: List[str], stopwords_set: Optional[set] = None) -> List[str]:
    """Remove stopwords from token list."""
    stop = stopwords_set or get_stopwords()
    return [t for t in tokens if t.lower() not in stop and len(t) > 2]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize a list of tokens using WordNetLemmatizer."""
    _init_nltk()
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(t.lower()) for t in tokens]
    except Exception:
        return [t.lower() for t in tokens]


class NLPPipeline:
    """
    End-to-end NLP text preprocessing pipeline.

    Steps:
    1. Clean text (remove noise, normalize)
    2. Tokenize
    3. Remove stopwords
    4. Lemmatize
    5. Rejoin to clean string
    """

    def __init__(self,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 use_lemmatization: bool = True,
                 custom_stopwords: Optional[set] = None):
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.use_lemmatization = use_lemmatization
        self.custom_stopwords = custom_stopwords
        self._stopwords = None
        _init_nltk()

    @property
    def stopwords(self) -> set:
        if self._stopwords is None:
            self._stopwords = get_stopwords()
            if self.custom_stopwords:
                self._stopwords.update(self.custom_stopwords)
        return self._stopwords

    def transform(self, text: str) -> str:
        """
        Transform raw text into cleaned, lemmatized string.
        Returns space-joined processed tokens.
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Clean
        text = clean_text(text, self.remove_urls, self.remove_emails)

        # Step 2: Tokenize
        tokens = tokenize(text)

        # Step 3: Remove stopwords (only alpha tokens)
        tokens = [t for t in tokens if t.isalpha()]
        tokens = remove_stopwords(tokens, self.stopwords)

        # Step 4: Lemmatize
        if self.use_lemmatization:
            tokens = lemmatize_tokens(tokens)

        return " ".join(tokens)

    def transform_batch(self, texts: List[str]) -> List[str]:
        """Transform a list of texts."""
        return [self.transform(t) for t in texts]

    def get_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract top N keywords from text after cleaning.
        Returns sorted list of unique keywords.
        """
        processed = self.transform(text)
        tokens = processed.split()
        freq: dict = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_tokens[:top_n]]
