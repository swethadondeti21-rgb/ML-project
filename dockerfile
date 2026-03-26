# Dockerfile
# =============================================================================
# Multi-stage build for the Fraud Detection API
#
# WHY MULTI-STAGE?
#   Stage 1 (trainer): Has all build dependencies installed to train the model.
#                      This stage produces models/model.pkl.
#   Stage 2 (server):  Only has runtime dependencies (smaller image).
#                      It copies model.pkl from Stage 1 and runs the API.
# =============================================================================

# ── STAGE 1: Training ────────────────────────────────────────────────────────
FROM python:3.11-slim AS trainer

# Set working directory inside the container
WORKDIR /app

# Install dependencies first — Docker caches this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files needed for training
COPY src/generate_data.py ./src/generate_data.py
COPY src/train.py         ./src/train.py

# Generate data and train the model
RUN python src/generate_data.py
RUN python src/train.py


# ── STAGE 2: Serving ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS server

WORKDIR /app

# Install only the packages needed to run the API
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    scikit-learn==1.4.0 \
    pandas==2.2.0 \
    numpy==1.26.3 \
    pydantic==2.6.0

# Copy only the serving code
COPY src/serve.py ./src/serve.py

# Copy the trained model from the trainer stage
COPY --from=trainer /app/models/model.pkl  ./models/model.pkl
# COPY --from=trainer /app/encoder.pkl       ./encoder.pkl

# Tell Docker that the container listens on port 8000
EXPOSE 8000

# Health check using native Python instead of curl
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=15s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# The command to run when the container starts
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]