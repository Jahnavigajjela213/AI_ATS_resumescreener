.PHONY: install train api test clean

# ── Install dependencies ───────────────────────────────────────
install:
	pip install -r requirements.txt
	python -c "import nltk; [nltk.download(r, quiet=True) for r in ['punkt','punkt_tab','stopwords','wordnet','averaged_perceptron_tagger','omw-1.4']]"

# ── Train all ML models ────────────────────────────────────────
train:
	python run.py --train-only

# ── Start API server (development) ────────────────────────────
api:
	python run.py --reload

# ── Start API server (production) ─────────────────────────────
serve:
	python run.py

# ── Train then start server ────────────────────────────────────
start: train api

# ── Run a quick API health check ──────────────────────────────
health:
	curl -s http://localhost:8000/health | python -m json.tool

# ── Clean generated model files ───────────────────────────────
clean:
	rm -rf models/saved/*.pkl models/saved/*.json logs/*.log

# ── Show project structure ─────────────────────────────────────
tree:
	find . -not -path './.git/*' -not -path './__pycache__/*' \
	       -not -name '*.pyc' -not -name '*.egg-info' \
	       | sort | sed -e "s/[^-][^\/]*\//  |/g" -e "s/|\([^ ]\)/|-\1/"
