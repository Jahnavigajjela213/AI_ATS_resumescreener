# 🤖 AI Smart Resume Screening & Skill Gap Prediction System

A **production-ready, end-to-end Machine Learning system** for automated resume screening,
ATS scoring, job matching, skill gap detection, and hiring prediction — served via a FastAPI REST API.

---

## 🚀 Quick Start

```bash
# 1. Clone and enter directory
cd resume_screening

# 2. Install all dependencies (Python 3.10+)
pip install -r requirements.txt
python -c "import nltk; [nltk.download(r, quiet=True) for r in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"

# 3. Train all ML models
python run.py --train-only

# 4. Start the API server
python run.py

# 5. Open interactive API docs
#    http://localhost:8000/docs
```

---

## 📁 Project Structure

```
resume_screening/
│
├── backend/                    # FastAPI application
│   ├── main.py                 # App factory, lifespan, middleware
│   ├── model_loader.py         # Singleton model store (loads on startup)
│   ├── schemas.py              # All Pydantic request/response models
│   └── routers/
│       ├── resume.py           # /resume — upload, classify, ATS score
│       ├── matching.py         # /match  — cosine similarity job ranking
│       ├── skills.py           # /skills — extract, gap, recommend
│       └── prediction.py       # /predict — hiring success probability
│
├── models/                     # ML model classes
│   ├── resume_classifier.py    # LR / RF / SVM multi-class classifier
│   ├── resume_matcher.py       # TF-IDF cosine similarity matcher
│   ├── hiring_predictor.py     # XGBoost hiring success predictor
│   ├── train_models.py         # End-to-end training pipeline
│   └── saved/                  # Serialized model files (auto-generated)
│
├── preprocessing/              # NLP pipeline
│   ├── text_cleaner.py         # Clean, normalize, extract metadata
│   └── nlp_pipeline.py         # Tokenize → stopwords → lemmatize
│
├── feature_engineering/        # Feature extraction
│   ├── tfidf_vectorizer.py     # Reusable TF-IDF vectorizer (save/load)
│   └── feature_builder.py      # Combine TF-IDF + structured features
│
├── utils/                      # Skill intelligence
│   ├── skill_dictionary.py     # Domain skill lists + importance weights
│   ├── skill_extractor.py      # Hybrid keyword/regex skill extraction
│   ├── skill_gap.py            # ATS scoring + gap computation
│   └── recommender.py          # Learning paths + career roadmap
│
├── data/                       # Data layer
│   ├── data_loader.py          # CSV / PDF / DOCX / TXT loaders
│   ├── sample_resumes.csv      # Labeled resume training data
│   └── sample_jobs.csv         # Job description database
│
├── config/
│   └── settings.py             # Pydantic settings (env-configurable)
│
├── logs/                       # Rotating log files
├── run.py                      # CLI entry point
├── Makefile                    # Developer workflow shortcuts
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | API info |
| `GET`  | `/health` | Health check + model status |
| `POST` | `/resume/upload` | Upload PDF/DOCX/TXT, extract text |
| `POST` | `/resume/predict-role` | Classify resume into a job role |
| `POST` | `/resume/ats-score` | Generate 6-component ATS score |
| `POST` | `/match/jobs` | Rank jobs by cosine similarity |
| `POST` | `/skills/extract` | Extract skills by domain |
| `POST` | `/skills/gap` | Detect skill gaps vs. job requirements |
| `POST` | `/skills/recommend` | Get learning roadmap + career path |
| `POST` | `/predict/hiring` | Predict hiring success probability |

Interactive docs: **http://localhost:8000/docs**

---

## 🧠 ML Pipeline

```
Raw Text (Resume / Job Description)
      │
      ▼
┌─────────────────────────────────┐
│   NLP Preprocessing             │
│   • URL/email/phone removal     │
│   • Tokenization (NLTK)         │
│   • Stopword removal            │
│   • Lemmatization               │
└────────────────┬────────────────┘
                 │
      ┌──────────▼──────────┐
      │  Feature Engineering │
      │  • TF-IDF (5000 dim) │
      │  • Structured feats  │
      │    (exp, edu, skills)│
      └──────────┬──────────┘
                 │
     ┌───────────┼───────────────┐
     ▼           ▼               ▼
┌─────────┐ ┌─────────┐  ┌──────────────┐
│Classifier│ │ Matcher │  │    XGBoost   │
│ LR/RF/  │ │ Cosine  │  │  Hiring      │
│  SVM    │ │Similarity│  │  Predictor  │
└────┬────┘ └────┬────┘  └──────┬───────┘
     │           │               │
     ▼           ▼               ▼
 Job Role    Ranked Jobs    Hire/No-Hire
 + Proba    + Match %      + Probability
```

### Models Trained

| Model | Purpose | Algorithm |
|-------|---------|-----------|
| Resume Classifier | Predict job role | LR / RF / SVM (best selected) |
| Resume Matcher | Rank jobs by fit | TF-IDF + Cosine Similarity |
| Hiring Predictor | Predict hire success | XGBoost (GradientBoosting fallback) |

---

## 🔍 Skill Gap Engine

The skill gap engine compares skills extracted from a resume against a job description using:

- **300+ skills** across 7 domains: Data Science, AI/ML, Software Engineering, DevOps/Cloud, Data Engineering, Databases, Soft Skills
- **Importance-weighted scoring** (1–10 scale)
- **Priority ranking**: Critical → High → Medium → Low
- **Learning path recommendations** with estimated time and certifications

---

## 📊 ATS Score Breakdown

| Component | Weight | How It's Computed |
|-----------|--------|-------------------|
| Keyword Match | 30% | Overlap of cleaned tokens between resume and JD |
| Skill Match | 30% | Importance-weighted skill overlap |
| Experience | 15% | `years_found / years_required × 100` |
| Structure | 10% | Presence of: Skills, Experience, Education, Projects, Summary, Certifications |
| Education | 10% | Detected vs. required education level |
| Readability | 5% | Avg sentence length & word length heuristic |

**Grade Scale**: Excellent (85+) · Good (70+) · Fair (55+) · Weak (40+) · Poor

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and adjust:

```env
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MAX_FEATURES=5000
TEST_SIZE=0.2
RANDOM_STATE=42
```

---

## 🐳 Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# API will be available at http://localhost:8000
```

---

## 🛠️ Developer Commands

```bash
make install    # Install dependencies + download NLTK data
make train      # Train all ML models
make api        # Start dev server with auto-reload
make serve      # Start production server
make health     # Test /health endpoint
make clean      # Remove saved models and logs
```

---

## 📈 Sample API Calls

### Predict Job Role
```bash
curl -X POST "http://localhost:8000/resume/predict-role" \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Experienced Data Scientist with 5 years in Python, ML, and TensorFlow..."}'
```

### ATS Score
```bash
curl -X POST "http://localhost:8000/resume/ats-score" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Senior Python developer with 4 years experience...",
    "job_description": "We are looking for a Python backend engineer...",
    "job_title": "Backend Engineer",
    "min_experience": 3,
    "education_required": "Bachelors"
  }'
```

### Skill Gap Analysis
```bash
curl -X POST "http://localhost:8000/skills/gap" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Python, pandas, scikit-learn, SQL...",
    "job_text": "Must know Python, TensorFlow, Kubernetes, Spark, SQL..."
  }'
```

### Hiring Prediction
```bash
curl -X POST "http://localhost:8000/predict/hiring" \
  -H "Content-Type: application/json" \
  -d '{
    "ats_score": 78.5,
    "skill_match_score": 65.0,
    "years_experience": 4.5,
    "education_level": "Masters",
    "skill_count": 14,
    "keyword_match_score": 72.0,
    "structure_score": 85.0
  }'
```

---

## 📋 Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| ML Models | Scikit-learn, XGBoost |
| NLP | NLTK (tokenize, lemmatize, stopwords) |
| Feature Extraction | TF-IDF (sklearn) |
| Explainability | Feature Importance (XGBoost) |
| File Parsing | PyPDF2, docx2txt |
| Logging | Loguru |
| Serialization | Joblib |
| Containerization | Docker + Docker Compose |

---

## 📝 License

MIT License — free for personal and commercial use.
