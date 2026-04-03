"""
Cover Letter Generator — creates a tailored cover letter from resume + JD gaps.
"""
from typing import List, Dict, Any
import re
from datetime import datetime


# ── Templates ────────────────────────────────────────────────────────────────

TEMPLATES = {
    "Professional": """\
{date}

Hiring Manager
{company_name}

Dear Hiring Manager,

I am writing to express my strong interest in the {job_title} position at {company_name}. \
With {years_exp} years of experience in {primary_domain} and a proven track record in \
{top_matched_skills}, I am confident that I can contribute meaningfully to your team from day one.

In my recent work, I have developed deep expertise in {matched_skill_list}. \
These experiences have honed my ability to deliver impactful results in fast-paced environments \
and collaborate effectively across cross-functional teams.

I am particularly excited about the opportunity to work with {job_title_short} technologies \
and look forward to expanding my expertise in {missing_skill_highlight} as I grow within your organisation.

I would welcome the opportunity to discuss how my background aligns with your team's goals. \
Thank you for considering my application — I look forward to hearing from you.

Sincerely,
[Your Name]
[Email] | [Phone] | [LinkedIn]
""",
    "Creative": """\
{date}

Hi {company_name} Team 👋,

When I read the job posting for **{job_title}**, I immediately thought: *this is exactly where I want to be.*

Over the past {years_exp} years, I have been building real-world expertise in {primary_domain} — \
working with {matched_skill_list} to solve complex problems and ship products that matter. \
My background in {top_matched_skills} translates directly to the challenges your team faces daily.

I'm particularly drawn to the focus on {job_title_short} and the emphasis on {missing_skill_highlight}. \
I'm actively upskilling in this area and already have foundational knowledge to hit the ground running.

I'd love to chat about how I can bring value to {company_name} — no fluff, just results.

Cheers,
[Your Name]
[Email] | [Phone] | [LinkedIn]
""",
    "Technical": """\
{date}

Re: Application for {job_title} — {company_name}

Dear Hiring Manager,

I am a {primary_domain} professional with {years_exp}+ years of hands-on experience. \
My technical toolkit includes {matched_skill_list}, which directly aligns with your stated requirements.

Key highlights:
• Proficient in: {top_matched_skills}
• Currently enhancing expertise in: {missing_skill_highlight}
• Education: {education_level}
• Demonstrated ability to architect, build, and deploy {job_title_short} solutions at scale

I thrive in data-driven environments and have a strong commitment to code quality, system design, \
and delivering measurable outcomes. I am eager to bring this to the {job_title} role at {company_name}.

Please find my resume attached. I would welcome a technical discussion at your convenience.

Best regards,
[Your Name]
[GitHub: github.com/] | [Email] | [Phone]
""",
}


def generate_cover_letter(
    job_title: str,
    company_name: str,
    years_exp: float,
    education_level: str,
    matched_skills: List[str],
    missing_skills: List[str],
    predicted_role: str = "Software Engineering",
    style: str = "Professional",
) -> str:
    """
    Generate a tailored cover letter from resume analysis data.

    Args:
        job_title: Target job title
        company_name: Target company name
        years_exp: Detected years of experience
        education_level: Detected education level
        matched_skills: Skills already in the resume
        missing_skills: Skills missing from the resume
        predicted_role: Domain / predicted role category
        style: "Professional" | "Creative" | "Technical"

    Returns:
        Formatted cover letter string
    """
    template = TEMPLATES.get(style, TEMPLATES["Professional"])

    # Build snippets
    top_matched = matched_skills[:4] if matched_skills else ["Python", "problem-solving"]
    top_missing = missing_skills[:2] if missing_skills else ["emerging technologies"]

    matched_skill_list = ", ".join(top_matched)
    top_matched_skills = matched_skills[0] if matched_skills else "core technologies"
    missing_highlight = " and ".join(top_missing)
    job_title_short = _shorten_title(job_title)
    primary_domain = predicted_role or "technology"

    cover = template.format(
        date=datetime.now().strftime("%d %B %Y"),
        job_title=job_title or "the advertised position",
        company_name=company_name or "your company",
        years_exp=f"{years_exp:.0f}" if years_exp > 0 else "several",
        education_level=education_level,
        matched_skill_list=matched_skill_list,
        top_matched_skills=top_matched_skills,
        missing_skill_highlight=missing_highlight,
        job_title_short=job_title_short,
        primary_domain=primary_domain,
    )
    return cover


def _shorten_title(title: str) -> str:
    """Extract first 2 meaningful words from a job title."""
    if not title:
        return "technical"
    words = [w for w in title.split() if w.lower() not in
             {"senior", "junior", "lead", "principal", "staff", "associate"}]
    return " ".join(words[:2]) if words else title
