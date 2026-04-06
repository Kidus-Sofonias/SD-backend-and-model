# File role: ORM model package bootstrap.
# Imports all mapped classes so SQLAlchemy string-based relationships resolve
# when callers import models from this package.
# Key symbols/vars: User, Trip, DrivingEvent, SensorSample.

from app.db.models.user import User
from app.db.models.trip import Trip
from app.db.models.driving_event import DrivingEvent
from app.db.models.sensor_sample import SensorSample

__all__ = ["User", "Trip", "DrivingEvent", "SensorSample"]
