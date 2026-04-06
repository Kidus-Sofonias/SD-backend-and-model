# File role: Alembic migration script describing a database schema change for this revision.
# Connects to: sqlalchemy.
# Key symbols/vars: revision, down_revision, branch_labels, depends_on.
"""add ml fields to trips

Revision ID: 5e2ece5421a1
Revises: 905c113c790f
Create Date: 2026-03-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5e2ece5421a1'
down_revision: Union[str, Sequence[str], None] = '905c113c790f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add ML-related columns to trips table
    op.add_column('trips', sa.Column('score', sa.Integer(), nullable=True))
    op.add_column('trips', sa.Column('score_breakdown', sa.Text(), nullable=True))
    op.add_column('trips', sa.Column('feature_version', sa.String(), nullable=True))
    op.add_column('trips', sa.Column('model_version', sa.String(), nullable=True))
    op.add_column('trips', sa.Column('confidence', sa.Float(), nullable=True))
    op.add_column('trips', sa.Column('processed_at', sa.DateTime(), nullable=True))
    op.add_column('trips', sa.Column('raw_deleted', sa.Boolean(), nullable=False, default=False))


def downgrade() -> None:
    """Downgrade schema."""
    # Remove the added columns
    op.drop_column('trips', 'raw_deleted')
    op.drop_column('trips', 'processed_at')
    op.drop_column('trips', 'confidence')
    op.drop_column('trips', 'model_version')
    op.drop_column('trips', 'feature_version')
    op.drop_column('trips', 'score_breakdown')
    op.drop_column('trips', 'score')