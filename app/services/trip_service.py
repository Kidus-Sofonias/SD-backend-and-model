# File role: Business-logic service that coordinates repositories/schemas and enforces use-case rules.
# Connects to: app.repositories.trip_repository.
# Key symbols/vars: TripService.
from app.repositories.trip_repository import SqlTripRepository, TripRecord


class TripService:
    def __init__(self, repo: SqlTripRepository):
        self.repo = repo

    def start_trip(self, user_id: str) -> TripRecord:
        return self.repo.create_trip(user_id=user_id)

    def end_trip(self, trip_id: str, user_id: str) -> TripRecord:
        return self.repo.end_trip(trip_id=trip_id, user_id=user_id)
