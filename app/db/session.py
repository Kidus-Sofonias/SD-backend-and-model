# File role: Database bootstrapping/session module used by repositories and route dependency injection.
# Connects to: sqlalchemy, app.core.config.
# Key symbols/vars: connect_args, engine, SessionLocal, get_db.
from __future__ import annotations

from collections.abc import Generator
import sqlite3
import time

from sqlalchemy import create_engine, inspect as sa_inspect
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {
        "check_same_thread": False,
        "timeout": 30,
    }

engine = create_engine(settings.database_url, connect_args=connect_args)

if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA busy_timeout = 30000")
            cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.execute("PRAGMA foreign_keys = ON")
            try:
                # Switching journal mode can transiently fail if another process
                # already has the database open during reload/startup.
                cursor.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError as exc:
                if "database is locked" not in str(exc).lower():
                    raise
        finally:
            cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=Session)

SQLITE_LOCK_RETRY_ATTEMPTS = 6
SQLITE_LOCK_RETRY_BASE_DELAY_SECONDS = 0.25


def _is_sqlite_lock_error(error: Exception) -> bool:
    return settings.database_url.startswith("sqlite") and "database is locked" in str(error).lower()


def _column_snapshot(instance: object) -> dict[str, object]:
    mapper = sa_inspect(instance).mapper
    snapshot: dict[str, object] = {}
    for attr in mapper.column_attrs:
        key = attr.key
        snapshot[key] = getattr(instance, key)
    return snapshot


def commit_with_retry(db: Session) -> None:
    for attempt in range(SQLITE_LOCK_RETRY_ATTEMPTS):
        try:
            db.commit()
            return
        except OperationalError as exc:
            new_instances = list(db.new)
            dirty_instances = [instance for instance in db.dirty if db.is_modified(instance, include_collections=False)]
            deleted_instances = list(db.deleted)
            dirty_snapshots = {id(instance): _column_snapshot(instance) for instance in dirty_instances}

            db.rollback()
            is_last_attempt = attempt == SQLITE_LOCK_RETRY_ATTEMPTS - 1
            if not _is_sqlite_lock_error(exc) or is_last_attempt:
                raise

            # rollback clears pending/dirty state; restore pending work before retrying
            for instance in new_instances:
                db.add(instance)

            for instance in dirty_instances:
                for key, value in dirty_snapshots[id(instance)].items():
                    setattr(instance, key, value)
                db.add(instance)

            for instance in deleted_instances:
                try:
                    db.delete(instance)
                except Exception:
                    pass

            time.sleep(SQLITE_LOCK_RETRY_BASE_DELAY_SECONDS * (attempt + 1))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
