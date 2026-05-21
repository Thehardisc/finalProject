"""
trainer/config.py — Centralised configuration for the trainer package.
All env-vars and derived paths are resolved here once at import time.
"""
import os
from pathlib import Path

# ── Database ──────────────────────────────────────────────────────────────────
DB_USER     = os.getenv("POSTGRES_USER",  "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME     = os.getenv("POSTGRES_DB",    "emotion_db")
DB_HOST     = os.getenv("DB_HOST",        "db")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

# ── Trainer knobs ─────────────────────────────────────────────────────────────
RETRAIN_INTERVAL = int(os.environ.get("RETRAIN_INTERVAL_SECONDS", 1800))
ACCURACY_GATE    = float(os.environ.get("ACCURACY_GATE", 0.60))
MAX_SAMPLES      = int(os.environ.get("MAX_SAMPLES", 2500))

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))
META_PATH  = MODEL_PATH.with_name("meta_weights_meta.json")
