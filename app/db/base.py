# File role: Database bootstrapping/session module used by repositories and route dependency injection.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: Base.
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
