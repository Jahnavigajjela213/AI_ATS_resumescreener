"""
Recommendation Engine - Learning paths, certifications, and career roadmap.
Expanded with more modern skills: LLMs, GenAI, cloud-native, etc.
"""
from typing import List, Dict, Any

# ── Learning Resource Database ──────────────────────────────────────────────

LEARNING_RESOURCES: Dict[str, Dict[str, Any]] = {
    "python": {
        "courses": ["Python Bootcamp – Udemy (Jose Portilla)", "Python for Everybody – Coursera"],
        "certifications": ["PCEP – Python Certified Entry-Level"],
        "time": "4–6 weeks", "difficulty": "Beginner",
    },
    "machine learning": {
        "courses": ["ML Specialization – Andrew Ng (Coursera)", "Fast.ai Practical ML"],
        "certifications": ["Google Professional ML Engineer", "AWS Certified ML Specialty"],
        "time": "8–12 weeks", "difficulty": "Intermediate",
    },
    "deep learning": {
        "courses": ["Deep Learning Specialization – Coursera", "PyTorch Fundamentals – Microsoft"],
        "certifications": ["NVIDIA Deep Learning Institute"],
        "time": "10–14 weeks", "difficulty": "Advanced",
    },
    "sql": {
        "courses": ["SQL for Data Science – Coursera", "Mode Analytics SQL Tutorial (free)"],
        "certifications": ["Oracle Database SQL Certified Associate"],
        "time": "2–4 weeks", "difficulty": "Beginner",
    },
    "tensorflow": {
        "courses": ["TensorFlow Developer Certificate – Google", "TF in Practice – Coursera"],
        "certifications": ["TensorFlow Developer Certificate"],
        "time": "6–8 weeks", "difficulty": "Intermediate",
    },
    "pytorch": {
        "courses": ["PyTorch for Deep Learning – Zero to Mastery", "Fast.ai Course (free)"],
        "time": "6–8 weeks", "difficulty": "Intermediate",
    },
    "aws": {
        "courses": ["AWS Cloud Practitioner – A Cloud Guru", "AWS Solutions Architect – Udemy (Stephane Maarek)"],
        "certifications": ["AWS Cloud Practitioner", "AWS Solutions Architect Associate", "AWS ML Specialty"],
        "time": "6–10 weeks", "difficulty": "Intermediate",
    },
    "azure": {
        "courses": ["AZ-900: Azure Fundamentals – Microsoft Learn (free)", "Azure DevOps – Udemy"],
        "certifications": ["AZ-900 Azure Fundamentals", "AZ-204 Azure Developer"],
        "time": "4–8 weeks", "difficulty": "Intermediate",
    },
    "gcp": {
        "courses": ["GCP Professional Cloud Architect – Coursera", "Google Cloud Skills Boost (free)"],
        "certifications": ["Google Associate Cloud Engineer", "Google Professional Data Engineer"],
        "time": "6–10 weeks", "difficulty": "Intermediate",
    },
    "docker": {
        "courses": ["Docker Mastery – Udemy (Bret Fisher)", "Docker Official Docs Tutorial"],
        "certifications": ["Docker Certified Associate (DCA)"],
        "time": "2–3 weeks", "difficulty": "Beginner",
    },
    "kubernetes": {
        "courses": ["Kubernetes for Developers – Linux Foundation", "CKA Certification Prep – KodeKloud"],
        "certifications": ["CKA – Certified Kubernetes Administrator", "CKAD – Certified Kubernetes Developer"],
        "time": "6–8 weeks", "difficulty": "Advanced",
    },
    "nlp": {
        "courses": ["NLP Specialization – Coursera (deeplearning.ai)", "Hugging Face NLP Course (free)"],
        "certifications": ["DeepLearning.AI specialization"],
        "time": "8–10 weeks", "difficulty": "Advanced",
    },
    "transformers": {
        "courses": ["Hugging Face NLP Course (free)", "Transformers for NLP – Coursera"],
        "time": "4–6 weeks", "difficulty": "Advanced",
    },
    "llm": {
        "courses": ["LLMOps Specialization – Coursera", "Building LLM Applications – DeepLearning.AI (free)"],
        "certifications": ["DeepLearning.AI LLM Specialization"],
        "time": "4–8 weeks", "difficulty": "Advanced",
    },
    "langchain": {
        "courses": ["LangChain for LLM Application Development – DeepLearning.AI (free)", "LangChain Docs"],
        "time": "2–4 weeks", "difficulty": "Intermediate",
    },
    "generative ai": {
        "courses": ["Generative AI with LLMs – Coursera (AWS + DeepLearning.AI)", "Google Generative AI Path"],
        "certifications": ["Google Generative AI Professional"],
        "time": "4–6 weeks", "difficulty": "Intermediate",
    },
    "rag": {
        "courses": ["Building RAG Systems – DeepLearning.AI (free)", "LlamaIndex Docs + Tutorials"],
        "time": "2–3 weeks", "difficulty": "Intermediate",
    },
    "prompt engineering": {
        "courses": ["Prompt Engineering for Developers – DeepLearning.AI (free)", "Promptingguide.ai (free)"],
        "certifications": ["Certified Prompt Engineer"],
        "time": "1–2 weeks", "difficulty": "Beginner",
    },
    "react": {
        "courses": ["React – The Complete Guide – Udemy (Maximilian Schwarzmüller)", "React Official Docs Tutorial"],
        "certifications": ["Meta Front-End Developer Professional (Coursera)"],
        "time": "4–6 weeks", "difficulty": "Intermediate",
    },
    "spark": {
        "courses": ["Spark & Python for Big Data – Udemy", "Databricks Academy (free tier)"],
        "certifications": ["Databricks Certified Associate Developer for Apache Spark"],
        "time": "4–6 weeks", "difficulty": "Intermediate",
    },
    "mlflow": {
        "courses": ["MLflow Official Documentation", "ML Engineering for Production – Coursera"],
        "time": "2–3 weeks", "difficulty": "Intermediate",
    },
    "scikit-learn": {
        "courses": ["Hands-On ML with Scikit-Learn – O'Reilly Book", "Scikit-learn User Guide (free)"],
        "time": "3–4 weeks", "difficulty": "Beginner",
    },
    "xgboost": {
        "courses": ["XGBoost for Data Scientists – Kaggle Learn (free)", "Various Kaggle competition notebooks"],
        "time": "1–2 weeks", "difficulty": "Intermediate",
    },
    "fastapi": {
        "courses": ["FastAPI Official Tutorial (free)", "Building APIs with FastAPI – TestDriven.io"],
        "time": "1–2 weeks", "difficulty": "Beginner",
    },
    "airflow": {
        "courses": ["Apache Airflow: The Hands-On Guide – Udemy", "Airflow Official Docs (free)"],
        "certifications": ["Astronomer Certification Apache Airflow Fundamentals"],
        "time": "3–4 weeks", "difficulty": "Intermediate",
    },
    "kafka": {
        "courses": ["Apache Kafka for Beginners – Udemy (Stephane Maarek)", "Confluent Kafka Training"],
        "certifications": ["Confluent Certified Developer for Apache Kafka"],
        "time": "3–4 weeks", "difficulty": "Intermediate",
    },
    "ci/cd": {
        "courses": ["DevOps Fundamentals – Udemy", "GitHub Actions Official Docs (free)"],
        "time": "2–3 weeks", "difficulty": "Intermediate",
    },
    "terraform": {
        "courses": ["HashiCorp Terraform Associate Prep – Udemy (Zeal Vora)"],
        "certifications": ["HashiCorp Terraform Associate"],
        "time": "3–4 weeks", "difficulty": "Intermediate",
    },
    "dbt": {
        "courses": ["dbt Fundamentals – dbt Learn (free)", "Analytics Engineering Bootcamp"],
        "certifications": ["dbt Analytics Engineering Certification"],
        "time": "2–3 weeks", "difficulty": "Intermediate",
    },
    "snowflake": {
        "courses": ["Snowflake Hands-On Essentials – Snowflake Academy (free)"],
        "certifications": ["SnowPro Core Certification"],
        "time": "2–4 weeks", "difficulty": "Intermediate",
    },
    "power bi": {
        "courses": ["Power BI From Beginner to Pro – Udemy", "Microsoft Learn Power BI Path (free)"],
        "certifications": ["Microsoft Power BI Data Analyst PL-300"],
        "time": "3–5 weeks", "difficulty": "Beginner",
    },
    "tableau": {
        "courses": ["Tableau 2023 A-Z – Udemy", "Tableau Public Gallery + Docs"],
        "certifications": ["Tableau Desktop Specialist"],
        "time": "2–4 weeks", "difficulty": "Beginner",
    },
    "git": {
        "courses": ["Git & GitHub Bootcamp – Udemy (Colt Steele)", "Pro Git Book (free)"],
        "time": "1–2 weeks", "difficulty": "Beginner",
    },
    "linux": {
        "courses": ["Linux Command Line Basics – Udemy", "Linux Foundation Essentials (free)"],
        "certifications": ["CompTIA Linux+", "LFCS Linux Foundation Certified SysAdmin"],
        "time": "2–4 weeks", "difficulty": "Beginner",
    },
}

DEFAULT_RESOURCE = {
    "courses": ["Search on Coursera, Udemy, or YouTube for tutorials"],
    "time": "4–6 weeks", "difficulty": "Intermediate",
}

# ── Career Path Definitions ──────────────────────────────────────────────────

CAREER_PATHS: Dict[str, List[str]] = {
    "Data Science": [
        "Junior Data Analyst",
        "Data Scientist",
        "Senior Data Scientist",
        "Lead Data Scientist / ML Engineer",
        "Principal Data Scientist / Head of Data",
    ],
    "AI/ML": [
        "ML Engineer",
        "Senior ML Engineer",
        "AI Research Scientist",
        "Principal AI Scientist",
        "AI Research Director",
    ],
    "Software Engineering": [
        "Junior Software Engineer",
        "Software Engineer",
        "Senior Software Engineer",
        "Staff Engineer / Tech Lead",
        "Principal Engineer / Engineering Manager",
    ],
    "DevOps/Cloud": [
        "Junior DevOps Engineer",
        "DevOps Engineer",
        "Senior DevOps / Site Reliability Engineer",
        "Cloud Architect",
        "Principal Cloud Architect",
    ],
    "Data Engineering": [
        "Data Analyst",
        "Data Engineer",
        "Senior Data Engineer",
        "Lead Data Engineer",
        "Principal Data Engineer / Head of Data Platform",
    ],
    "Cybersecurity": [
        "Security Analyst",
        "Penetration Tester",
        "Senior Security Engineer",
        "Security Architect",
        "Chief Information Security Officer (CISO)",
    ],
}


def recommend_learning_path(missing_skills: List[str]) -> List[Dict[str, Any]]:
    """
    Build a structured learning roadmap for missing skills.
    Returns: List of dicts with skill, courses, time, difficulty, certifications
    """
    roadmap = []
    for skill in missing_skills:
        resource = LEARNING_RESOURCES.get(skill.lower(), DEFAULT_RESOURCE)
        roadmap.append({
            "skill": skill,
            "courses": resource.get("courses", []),
            "estimated_time": resource.get("time", "4–6 weeks"),
            "difficulty": resource.get("difficulty", "Intermediate"),
            "certifications": resource.get("certifications", []),
        })
    return roadmap


def suggest_career_path(predicted_role: str, skill_match_score: float) -> Dict[str, Any]:
    """
    Suggest a career progression path based on predicted role and skill score.
    Returns dict with career_stages, current_level, next_steps.
    """
    role_map = {
        "Data Science": "Data Science",
        "AI/ML": "AI/ML",
        "Software Engineering": "Software Engineering",
        "DevOps": "DevOps/Cloud",
        "Cloud": "DevOps/Cloud",
        "DevOps/Cloud": "DevOps/Cloud",
        "Data Engineering": "Data Engineering",
        "Data Analytics": "Data Science",
        "Cybersecurity": "Cybersecurity",
    }
    category = role_map.get(predicted_role, "Software Engineering")
    stages = CAREER_PATHS.get(category, CAREER_PATHS["Software Engineering"])

    # Estimate current level from skill score
    if skill_match_score >= 80:
        level_idx = min(2, len(stages) - 1)
    elif skill_match_score >= 55:
        level_idx = 1
    else:
        level_idx = 0

    return {
        "category": category,
        "career_stages": stages,
        "estimated_current_level": stages[level_idx],
        "next_step": stages[min(level_idx + 1, len(stages) - 1)],
        "recommended_actions": [
            f"Strengthen core {category} skills and fill identified gaps",
            "Build 2–3 portfolio projects that demonstrate real-world impact",
            "Contribute to open source projects or Kaggle competitions",
            "Obtain at least one recognized industry certification",
            "Network actively on LinkedIn — post your projects, engage with community",
            "Practice system design and behavioral interviews for senior roles",
        ],
    }
