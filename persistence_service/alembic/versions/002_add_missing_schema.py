"""Add missing schema changes

Revision ID: 002_add_missing_schema
Revises: 001_initial_base
Create Date: 2026-04-19 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002_add_missing_schema'
down_revision: Union[str, None] = '002_add_missing_columns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    pass

def downgrade() -> None:
    pass
