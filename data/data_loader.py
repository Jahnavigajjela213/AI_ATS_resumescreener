"""
Data Loader Module - loads and validates resume and job description data.
Supports PDF (PyPDF2 + pdfplumber fallback), DOCX, and TXT files.
"""
import io
import os
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent


def load_resumes(path: Optional[str] = None) -> pd.DataFrame:
    """Load resume data from CSV."""
    fpath = path or str(DATA_DIR / "sample_resumes.csv")
    try:
        df = pd.read_csv(fpath)
        logger.info(f"Loaded {len(df)} resumes from {fpath}")
        return df
    except FileNotFoundError:
        logger.error(f"Resume file not found: {fpath}")
        raise


def load_jobs(path: Optional[str] = None) -> pd.DataFrame:
    """Load job descriptions from CSV."""
    fpath = path or str(DATA_DIR / "sample_jobs.csv")
    try:
        df = pd.read_csv(fpath)
        logger.info(f"Loaded {len(df)} job descriptions from {fpath}")
        return df
    except FileNotFoundError:
        logger.error(f"Jobs file not found: {fpath}")
        raise


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Extract text from PDF bytes — tries PyPDF2 first, then pdfplumber."""
    text = ""

    # Primary: PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}")

    # Fallback: pdfplumber (handles complex layouts better)
    if len(text) < 50:
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n".join(pages).strip()
        except Exception as e:
            logger.warning(f"pdfplumber also failed: {e}")

    return text


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Extract text from file bytes (for upload handling)."""
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".pdf":
            return extract_text_from_pdf_bytes(file_bytes)
        elif ext in (".docx", ".doc"):
            import docx2txt
            return docx2txt.process(io.BytesIO(file_bytes))
        elif ext == ".txt":
            return file_bytes.decode("utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        raise


def load_raw_text(path: str) -> str:
    """Load raw text from .txt, .pdf, or .docx file."""
    ext = Path(path).suffix.lower()
    try:
        if ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            with open(path, "rb") as f:
                return extract_text_from_pdf_bytes(f.read())
        elif ext in (".docx", ".doc"):
            import docx2txt
            return docx2txt.process(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logger.error(f"Error loading file {path}: {e}")
        raise


def get_resume_corpus(path: Optional[str] = None) -> list:
    """Return list of resume text strings."""
    df = load_resumes(path)
    return df["text"].fillna("").tolist()


def get_job_corpus(path: Optional[str] = None) -> list:
    """Return list of job description strings."""
    df = load_jobs(path)
    return df["description"].fillna("").tolist()
