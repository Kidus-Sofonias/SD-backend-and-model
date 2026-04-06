# File role: Backward-compatible wrapper for the typo-named service module.
# New code should import from app.services.trip_processing_service.

from app.services.trip_processing_service import TripProcessingService

__all__ = ["TripProcessingService"]
