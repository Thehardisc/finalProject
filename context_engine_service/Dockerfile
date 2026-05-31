FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY context_engine_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -m spacy download en_core_web_sm

# Copy shared utilities (needed for shared/constants.py)
COPY shared/ ./shared/

# Copy service code
COPY context_engine_service/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008"]
