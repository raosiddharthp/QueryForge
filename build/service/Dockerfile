FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the self-hosted models at build time (ADR-003) so cold starts
# don't pay the download cost on Cloud Run's Always-Free CPU.
RUN python3 -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY config.py .
COPY models.py .
COPY budget_guard.py .
COPY gemini_client.py .
COPY embeddings.py .
COPY chunker.py .
COPY retrieval.py .
COPY fusion.py .
COPY recommender.py .
COPY main.py .

ENV PORT=8080
EXPOSE 8080

CMD exec python3 -m uvicorn main:app --host 0.0.0.0 --port ${PORT}
