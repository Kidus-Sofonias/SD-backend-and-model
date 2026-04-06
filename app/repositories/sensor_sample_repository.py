# File role: Data-access repository encapsulating SQLAlchemy queries used by service and route layers.
# Connects to: sqlalchemy, app.db.models.sensor_sample.
# Key symbols/vars: SensorSampleRepository.
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models.sensor_sample import SensorSample
from app.db.session import commit_with_retry


class SensorSampleRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_many(self, *, user_id: str, trip_id: str, rows: List[dict]) -> int:
        objs = [SensorSample(user_id=user_id, trip_id=trip_id, **row) for row in rows]
        self.db.add_all(objs)
        commit_with_retry(self.db)
        return len(objs)

    def list_by_trip(
        self,
        *,
        user_id: str,
        trip_id: str,
        limit: int = 1000,
        after_ts: Optional[datetime] = None,
    ) -> list[SensorSample]:
        stmt = select(SensorSample).where(
            SensorSample.trip_id == trip_id,
            SensorSample.user_id == user_id,
        )
        if after_ts:
            stmt = stmt.where(SensorSample.ts > after_ts)

        stmt = stmt.order_by(SensorSample.ts.asc()).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def count_by_trip(self, *, user_id: str, trip_id: str) -> int:
        stmt = select(func.count()).select_from(SensorSample).where(
            SensorSample.trip_id == trip_id,
            SensorSample.user_id == user_id,
        )
        return int(self.db.execute(stmt).scalar_one())
