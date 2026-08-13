"""Migration chain smoke test.

The app's unit tests use ``Base.metadata.create_all`` which bypasses alembic
entirely, so a broken migration used to go completely unnoticed. This test runs
the REAL migration chain (`alembic upgrade head`) against a scratch SQLite DB
and asserts the Phase 10 schema is present.

Phase 10 also fixed the pre-existing fresh-DB chain gaps:
- ``sensor_samples`` was only created by init_db's create_all, never by the
  init migration (so a fresh alembic upgrade broke at 20260723 on SQLite).
- ``if_not_exists=True`` in op.add_column is a SQLite syntax error; both
  20260723 and 20260813 now inspect the live schema before adding columns.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]

HEAD_REVISION = "20260813_add_event_confidence_severity"


def _run_alembic(db_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # Use forward slashes so sqlite:///<abs path> parses on every platform.
    env["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_alembic_upgrade_head_on_fresh_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "migrate.db"

    result = _run_alembic(db_path, "upgrade", "head")
    assert result.returncode == 0, f"alembic upgrade head failed:\n{result.stderr[-3000:]}"

    conn = sqlite3.connect(db_path)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"users", "trips", "driving_events", "sensor_samples"} <= tables, tables

        event_columns = {r[1] for r in conn.execute("PRAGMA table_info(driving_events)")}
        assert {"confidence", "severity", "duration_s", "occurred_at", "lat", "lon"} <= event_columns, event_columns

        sample_columns = {r[1] for r in conn.execute("PRAGMA table_info(sensor_samples)")}
        assert {"speed_mps", "altitude_m", "ts", "trip_id"} <= sample_columns, sample_columns

        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        assert version == HEAD_REVISION, version
    finally:
        conn.close()


def test_alembic_downgrade_removes_phase10_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "migrate-dn.db"

    assert _run_alembic(db_path, "upgrade", "head").returncode == 0
    assert _run_alembic(db_path, "downgrade", "-1").returncode == 0

    conn = sqlite3.connect(db_path)
    try:
        event_columns = {r[1] for r in conn.execute("PRAGMA table_info(driving_events)")}
        assert not ({"confidence", "severity", "duration_s"} <= event_columns), event_columns
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        assert version == "20260723_add_sensor_sample_altitude", version
    finally:
        conn.close()
