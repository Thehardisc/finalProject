"""Add missing columns to existing tables

Revision ID: 002_add_missing_columns
Revises: 001_initial_base
Create Date: 2026-04-19

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError

# revision identifiers, used by Alembic.
revision: str = '002_add_missing_columns'
down_revision: Union[str, None] = '001_initial_base'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Add escalation_score to conversation_states if missing
    try:
        conn.execute(sa.text(
            "ALTER TABLE conversation_states ADD COLUMN escalation_score FLOAT DEFAULT 0.0;"
        ))
    except Exception:
        pass  # Column already exists — safe to ignore

    # Add ground_truth_emotion to emotion_analysis if missing (safety net for old volumes)
    try:
        conn.execute(sa.text(
            "ALTER TABLE emotion_analysis ADD COLUMN IF NOT EXISTS ground_truth_emotion VARCHAR(50);"
        ))
    except Exception:
        pass  # Column already exists — safe to ignore

    # Add is_verified to emotion_analysis if missing (safety net for old volumes)
    try:
        conn.execute(sa.text(
            "ALTER TABLE emotion_analysis ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT FALSE;"
        ))
    except Exception:
        pass  # Column already exists — safe to ignore


def downgrade() -> None:
    conn = op.get_bind()
    try:
        conn.execute(sa.text("ALTER TABLE conversation_states DROP COLUMN IF EXISTS escalation_score;"))
    except Exception:
        pass
