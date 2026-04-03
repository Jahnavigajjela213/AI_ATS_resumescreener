"""
PDF Exporter — generates a downloadable ATS analysis report using fpdf2.
"""
from __future__ import annotations
from typing import Dict, Any
import io
from datetime import datetime


def _safe(text: str, max_len: int = 200) -> str:
    """Sanitise text for fpdf (remove non-latin-1 chars)."""
    return str(text).encode("latin-1", errors="replace").decode("latin-1")[:max_len]


def generate_pdf_report(ats_result: Dict[str, Any], resume_text: str,
                        job_title: str = "Target Role") -> bytes:
    """
    Generate a full ATS analysis PDF report.

    Returns: bytes of the PDF file.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 not installed. Run: pip install fpdf2")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(102, 126, 234)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(10)
    pdf.cell(0, 12, "AI ATS Resume Screening Report", align="C", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%d %b %Y  %H:%M')}  |  Job: {_safe(job_title, 60)}", align="C", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(12)

    score = ats_result.get("ats_score", 0)
    grade = ats_result.get("grade", "N/A")
    breakdown = ats_result.get("breakdown", {})
    suggestions = ats_result.get("suggestions", [])
    gap = ats_result.get("skill_gap", {})

    # ── ATS Score ─────────────────────────────────────────────────────────────
    grade_colors = {
        "Excellent": (0, 201, 167),
        "Good": (102, 126, 234),
        "Fair": (247, 151, 30),
        "Weak": (235, 51, 73),
        "Poor": (204, 0, 0),
    }
    r, g, b = grade_colors.get(grade, (150, 150, 150))

    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12, f"  ATS Score: {score}/100   |   Grade: {grade}", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Score Breakdown ───────────────────────────────────────────────────────
    _section_title(pdf, "Score Breakdown")
    weights = ["30%", "30%", "15%", "10%", "10%", "5%"]
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(230, 230, 240)
    pdf.cell(70, 8, "Component", border=1, fill=True)
    pdf.cell(30, 8, "Score", border=1, fill=True, align="C")
    pdf.cell(20, 8, "Weight", border=1, fill=True, align="C")
    pdf.cell(70, 8, "Progress", border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 10)
    for (k, v), w in zip(breakdown.items(), weights):
        label = k.replace("_", " ").title()
        bar_fill = int(v * 0.62)  # scale to ~62mm
        pdf.cell(70, 8, _safe(label), border=1)
        pdf.cell(30, 8, f"{v:.1f}", border=1, align="C")
        pdf.cell(20, 8, w, border=1, align="C")
        # mini progress bar
        x, y, h = pdf.get_x(), pdf.get_y(), 6
        pdf.set_fill_color(220, 220, 235)
        pdf.rect(x, y + 1, 68, h, "F")
        bar_r, bar_g, bar_b = (0, 201, 167) if v >= 70 else (247, 151, 30) if v >= 50 else (235, 51, 73)
        pdf.set_fill_color(bar_r, bar_g, bar_b)
        pdf.rect(x, y + 1, max(bar_fill, 2), h, "F")
        pdf.cell(70, 8, "", border=1)
        pdf.ln()

    pdf.ln(4)

    # ── Skill Summary ─────────────────────────────────────────────────────────
    _section_title(pdf, "Skill Analysis")
    matched = gap.get("matched_skills", [])
    missing = gap.get("missing_skills", [])
    extra   = gap.get("extra_skills", [])

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Total Required: {gap.get('total_required', 0)}   "
                   f"Matched: {gap.get('total_matched', 0)}   "
                   f"Missing: {len(missing)}   "
                   f"Extra: {len(extra)}   "
                   f"Match Score: {gap.get('match_score', 0):.1f}%", ln=True)
    pdf.ln(2)

    # Matched skills
    if matched:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(0, 150, 120)
        pdf.cell(0, 7, "Matched Skills:", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 6, _safe(", ".join(sorted(matched)), 600))
        pdf.ln(2)

    # Missing skills
    if missing:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(200, 0, 50)
        pdf.cell(0, 7, "Missing Skills:", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 6, _safe(", ".join(sorted(missing)), 600))
        pdf.ln(2)

    # ── Suggestions ──────────────────────────────────────────────────────────
    if suggestions:
        pdf.ln(2)
        _section_title(pdf, "AI Improvement Suggestions")
        pdf.set_font("Helvetica", "", 10)
        for i, s in enumerate(suggestions, 1):
            pdf.multi_cell(0, 7, _safe(f"{i}. {s}", 300))

    # ── Sections Found ────────────────────────────────────────────────────────
    sections = ats_result.get("sections_found", {})
    if sections:
        pdf.ln(2)
        _section_title(pdf, "Resume Sections Detected")
        pdf.set_font("Helvetica", "", 10)
        for sec, found in sections.items():
            status = "FOUND" if found else "MISSING"
            cr, cg, cb = (0, 150, 100) if found else (200, 0, 50)
            pdf.set_text_color(cr, cg, cb)
            pdf.cell(60, 7, f"  {'[OK]' if found else '[X]'} {sec.title()}", ln=False)
            pdf.set_text_color(0, 0, 0)
        pdf.ln(10)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, "AI Resume ATS Screener v2.0 — Powered by Streamlit + Scikit-learn", align="C")

    return pdf.output()


def _section_title(pdf, title: str):
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(240, 240, 255)
    pdf.cell(0, 9, f"  {title}", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
