"""Initial Base Migration"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001_initial_base'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.create_table(
        'messages',
        sa.Column('message_id', sa.String(), nullable=False),
        sa.Column('conversation_id', sa.String(), nullable=True),
        sa.Column('user_id', sa.String(), nullable=True),
        sa.Column('text', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('message_id')
    )
    op.create_index(op.f('ix_messages_conversation_id'), 'messages', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_messages_timestamp'), 'messages', ['timestamp'], unique=False)

    op.create_table(
        'emotion_analysis',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('message_id', sa.String(), sa.ForeignKey('messages.message_id'), nullable=True),
        sa.Column('emotions_json', sa.Text(), nullable=True),
        sa.Column('reasoning_json', sa.Text(), nullable=True),
        sa.Column('pipeline_log_json', sa.Text(), nullable=True),
        sa.Column('ground_truth_emotion', sa.String(), nullable=True),
        sa.Column('is_verified', sa.Boolean(), nullable=True, default=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_emotion_analysis_message_id'), 'emotion_analysis', ['message_id'], unique=False)

    op.create_table(
        'conversation_states',
        sa.Column('conversation_id', sa.String(), nullable=False),
        sa.Column('state_json', sa.Text(), nullable=True),
        sa.Column('escalation_score', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('last_updated', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('conversation_id')
    )

def downgrade() -> None:
    op.drop_table('conversation_states')
    op.drop_index(op.f('ix_emotion_analysis_message_id'), table_name='emotion_analysis')
    op.drop_table('emotion_analysis')
    op.drop_index(op.f('ix_messages_timestamp'), table_name='messages')
    op.drop_index(op.f('ix_messages_conversation_id'), table_name='messages')
    op.drop_table('messages')
