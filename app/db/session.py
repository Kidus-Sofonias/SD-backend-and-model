# File role: Database bootstrapping/session module used by repositories and route dependency injection.
# Connects to: sqlalchemy, app.core.config.
# Key symbols/vars: connect_args, engine, SessionLocal, get_db.
from __future__ import annotations

from collections.abc import Generator
import socket
import sqlite3
import time
from urllib.parse import urlparse, urlunparse

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


def _resolve_ipv4(url: str) -> str:
    """Resolve a PostgreSQL hostname to an IPv4 address to avoid IPv6 routing issues (e.g., on Render)."""
    if not url.startswith("postgresql"):
        return url
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return url
    try:
        addrs = socket.getaddrinfo(hostname, None, socket.AF_INET)
        if addrs:
            ipv4 = addrs[0][4][0]
            if ipv4 == hostname:
                return url
            # Reconstruct the netloc safely to avoid mangling credentials that might
            # contain the hostname as a substring.
            userinfo = ""
            if parsed.username:
                userinfo = parsed.username
                if parsed.password:
                    userinfo += ":" + parsed.password
                userinfo += "@"
            port_part = f":{parsed.port}" if parsed.port else ""
            new_netloc = f"{userinfo}{ipv4}{port_part}"
            url = urlunparse(parsed._replace(netloc=new_netloc))
    except socket.gaierror:
        pass
    return url


_database_url = _resolve_ipv4(settings.database_url)

if _database_url.startswith("postgresql"):
    connect_args.setdefault("sslmode", "require")
    connect_args.setdefault("connect_timeout", 10)

engine = create_engine(_database_url, connect_args=connect_args)

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
