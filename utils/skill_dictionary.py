"""
Skill Dictionary - Expanded domain skill lists with ATS-relevant terms.
"""
from typing import Dict, List

# ─────────────────────────────────────────────────────────────
# Domain-specific skill dictionaries
# ─────────────────────────────────────────────────────────────

DATA_SCIENCE_SKILLS = [
    "python", "r", "statistics", "machine learning", "deep learning",
    "scikit-learn", "tensorflow", "keras", "pytorch", "xgboost", "lightgbm",
    "catboost", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "sql", "nosql", "spark", "hadoop", "databricks", "airflow",
    "feature engineering", "model evaluation", "a/b testing", "hypothesis testing",
    "regression", "classification", "clustering", "reinforcement learning",
    "time series", "forecasting", "mlflow", "dvc", "data wrangling",
    "data visualization", "statistical analysis", "bayesian", "probability",
    "power bi", "tableau", "looker", "data analysis", "exploratory data analysis",
    "random forest", "gradient boosting", "svm", "naive bayes", "knn",
    "principal component analysis", "dimensionality reduction", "cross validation",
    "hyperparameter tuning", "model deployment", "model monitoring",
]

AI_ML_SKILLS = [
    "nlp", "natural language processing", "computer vision", "openai",
    "transformers", "bert", "gpt", "llm", "hugging face", "spacy", "nltk",
    "cv", "yolo", "opencv", "cnn", "rnn", "lstm", "attention mechanism",
    "generative ai", "stable diffusion", "langchain", "vector database",
    "embeddings", "rag", "fine tuning", "prompt engineering", "gans",
    "vae", "diffusion models", "multimodal", "zero shot", "few shot",
    "reinforcement learning", "rlhf", "reward modeling", "llama",
    "mistral", "claude", "gemini", "openai api", "pinecone", "weaviate",
    "chroma", "faiss", "semantic search", "sentence transformers",
]

SOFTWARE_ENGINEERING_SKILLS = [
    "java", "c++", "c#", "golang", "rust", "scala", "kotlin", "swift",
    "javascript", "typescript", "react", "angular", "vue", "next.js",
    "node.js", "django", "flask", "fastapi", "spring boot", "express",
    "REST API", "graphql", "grpc", "microservices", "event driven",
    "design patterns", "solid principles", "clean code", "tdd", "bdd",
    "git", "github", "gitlab", "jira", "agile", "scrum", "kanban",
    "html", "css", "sass", "webpack", "vite", "software architecture",
    "object oriented programming", "functional programming", "api development",
    "unit testing", "integration testing", "code review", "debugging",
    "redis", "rabbitmq", "celery", "websockets", "oauth", "jwt",
]

DEVOPS_CLOUD_SKILLS = [
    "aws", "azure", "gcp", "google cloud", "cloud computing",
    "docker", "kubernetes", "helm", "terraform", "ansible", "puppet", "chef",
    "ci/cd", "jenkins", "github actions", "gitlab ci", "circleci",
    "linux", "bash", "shell scripting", "networking", "security",
    "prometheus", "grafana", "elk stack", "monitoring", "logging",
    "serverless", "lambda", "s3", "ec2", "rds", "vpc", "iam",
    "site reliability engineering", "sre", "devops", "infrastructure as code",
    "azure devops", "cloudformation", "pulumi", "argocd", "istio",
]

DATA_ENGINEERING_SKILLS = [
    "etl", "data pipeline", "apache spark", "apache kafka", "apache flink",
    "airflow", "luigi", "prefect", "dbt", "snowflake", "redshift", "bigquery",
    "postgresql", "mysql", "oracle", "mongodb", "redis", "cassandra",
    "data warehouse", "data lake", "data modeling", "schema design",
    "stream processing", "batch processing", "delta lake", "apache iceberg",
    "data quality", "data governance", "metadata management",
    "azure data factory", "aws glue", "google dataflow",
]

DATABASES_SKILLS = [
    "sql", "postgresql", "mysql", "sqlite", "oracle", "sql server", "mariadb",
    "mongodb", "redis", "cassandra", "dynamodb", "elasticsearch",
    "firebase", "supabase", "neo4j", "graph database", "influxdb",
    "clickhouse", "duckdb", "snowflake", "redshift", "bigquery",
]

SOFT_SKILLS = [
    "leadership", "communication", "teamwork", "problem solving", "critical thinking",
    "project management", "time management", "agile", "scrum", "mentoring",
    "presentation", "collaboration", "analytical", "creative thinking",
    "attention to detail", "adaptability", "stakeholder management",
]

CYBERSECURITY_SKILLS = [
    "penetration testing", "vulnerability assessment", "siem", "soc",
    "firewalls", "intrusion detection", "encryption", "zero trust",
    "devsecops", "owasp", "compliance", "gdpr", "soc2", "iso 27001",
    "threat modeling", "incident response", "forensics",
]

# ─────────────────────────────────────────────────────────────
# Master skill registry
# ─────────────────────────────────────────────────────────────

DOMAIN_SKILLS: Dict[str, List[str]] = {
    "Data Science": DATA_SCIENCE_SKILLS,
    "AI/ML": AI_ML_SKILLS,
    "Software Engineering": SOFTWARE_ENGINEERING_SKILLS,
    "DevOps/Cloud": DEVOPS_CLOUD_SKILLS,
    "Data Engineering": DATA_ENGINEERING_SKILLS,
    "Databases": DATABASES_SKILLS,
    "Cybersecurity": CYBERSECURITY_SKILLS,
    "Soft Skills": SOFT_SKILLS,
}

# Skill importance weights (1-10, higher = more in demand)
SKILL_IMPORTANCE: Dict[str, int] = {
    "python": 10, "machine learning": 10, "deep learning": 9,
    "tensorflow": 8, "pytorch": 9, "sql": 9, "aws": 9,
    "kubernetes": 8, "docker": 8, "react": 8, "node.js": 7,
    "spark": 8, "nlp": 9, "transformers": 9, "bert": 8,
    "fastapi": 7, "scikit-learn": 8, "xgboost": 7,
    "airflow": 7, "kafka": 8, "git": 7, "ci/cd": 7,
    "typescript": 7, "llm": 9, "langchain": 8, "openai": 8,
    "rag": 8, "generative ai": 9, "dbt": 7, "snowflake": 8,
    "azure": 8, "gcp": 7, "terraform": 7, "linux": 7,
    "data pipeline": 8, "fine tuning": 8, "prompt engineering": 8,
    "power bi": 6, "tableau": 6, "postgresql": 7, "mongodb": 7,
}


def get_all_skills() -> List[str]:
    """Return a flat list of all unique skills across all domains."""
    all_skills: list = []
    for skills in DOMAIN_SKILLS.values():
        all_skills.extend(skills)
    return list(set(all_skills))


def get_skills_by_domain(domain: str) -> List[str]:
    """Return skills for a specific domain."""
    return DOMAIN_SKILLS.get(domain, [])


def get_skill_importance(skill: str) -> int:
    """Return importance score of a skill (default 5)."""
    return SKILL_IMPORTANCE.get(skill.lower(), 5)
