# File role: Shared core utilities for configuration, security, JWT handling, logging, and typed application errors.
# Connects to: app.core.config.
# Key symbols/vars: RequestIdFilter, setup_logging.
import logging
from app.core.config import settings


class RequestIdFilter(logging.Filter):
    """""
    Ensures logs always have reuest_id field.
    If not provided via logger 'extra', it will show '-'.
    """
    def filter(self, record:logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True

def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(request_id)s %(name)s %(message)s")
    )

    root.handlers.clear()
    root.addHandler(handler)