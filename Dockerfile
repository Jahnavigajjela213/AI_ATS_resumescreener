# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; [nltk.download(r, quiet=True) for r in ['punkt','stopwords','wordnet','averaged_perceptron_tagger','omw-1.4']]"

# Copy source code
COPY . .

# Train models on image build (optional — comment out for faster builds)
RUN python models/train_models.py || echo "Training failed, will train at runtime."

# Create log dir
RUN mkdir -p logs

# Expose port (Render assigns $PORT dynamically)
EXPOSE 8501

# Default: start Streamlit frontend (contains all logic)
# Render injects $PORT at runtime
CMD streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
