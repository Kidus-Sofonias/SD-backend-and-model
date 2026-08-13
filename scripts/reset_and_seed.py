"""Reset the database and seed a fresh sample dataset (hackathon feature).

Use after upgrading the schema (run ``alembic upgrade head`` first). This
script:

1. Deletes EVERY user, trip, sensor sample, driving event (and any hackathon
   tables that exist: ``critical_events``, ``vehicle_profiles``) in FK-safe
   order, with Postgres sequence resync / SQLite sequence reset.
2. Clears the auto-retrain state so retrain milestones restart cleanly.
3. Seeds a brand-new admin + demo drivers + finalized trips (with demo review
   labels) via ``scripts.seed_demo_data``.

DANGER: this is destructive. It refuses to run against a database whose
``APP_ENV`` is ``production`` unless ``--yes`` is passed.

USAGE
-----
# Preview what will be wiped (no changes)
    python -m scripts.reset_and_seed --dry-run

# Wipe + seed the configured (dev/staging) database
    python -m scripts.reset_and_seed --yes

# Wipe + seed a local SQLite file (safest for demos)
    python -m scripts.reset_and_seed --local-db demo.db

# Custom sample set
    python -m scripts.reset_and_seed --yes --drivers 5 --trips-per-driver 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from scripts.seed_demo_data import seed_demo_data


# Children first — FK-safe delete order.
TABLE_ORDER = [
    "critical_events",
    "driving_events",
    "sensor_samples",
    "trips",
    "vehicle_profiles",
    "users",
]

STATE_PATHS = [
    BACKEND_ROOT / "artifacts" / "reports" / "auto_retrain_state.json",
]


def _existing_tables(engine) -> list[str]:
    return set(inspect(engine).get_table_names())


def _row_counts(engine, existing: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    with engine.connect() as conn:
        for table in TABLE_ORDER:
            if table in existing:
                counts[table] = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar_one()
    return counts


def _wipe(engine, existing: set[str], dry_run: bool) -> dict:
    targets = [t for t in TABLE_ORDER if t in existing]
    dialect = engine.dialect.name
    summary = {"dialect": dialect, "tables_to_wipe": targets}

    if dry_run:
        return summary

    if dialect == "postgresql":
        with engine.begin() as conn:
            conn.execute(
                text(
                    "TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE".format(
                        tables=", ".join(targets)
                    )
                )
            )
        # TRUNCATE ... RESTART IDENTITY already resyncs sequences (fixes the
        # historical explicit-ID import desync that caused UniqueViolations).
    else:
        with engine.begin() as conn:
            for table in targets:
                conn.execute(text(f'DELETE FROM "{table}"'))
            if "sqlite_sequence" in existing:
                conn.execute(
                    text(
                        "DELETE FROM sqlite_sequence WHERE name IN ({names})".format(
                            names=", ".join(f"'{t}'" for t in targets)
                        )
                    )
                )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="Confirm the destructive wipe (required outside local SQLite)")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be wiped and seeded")
    parser.add_argument("--local-db", type=str, default=None, metavar="FILE", help="Use a local SQLite file instead of the configured DB")
    parser.add_argument("--drivers", type=int, default=3, help="Demo drivers to seed (1-5, default 3)")
    parser.add_argument("--trips-per-driver", type=int, default=4, help="Trips per driver (default 4)")
    parser.add_argument("--password", type=str, default="demo1234", help="Password for seeded demo accounts")
    parser.add_argument("--admin-password", type=str, default=None, help="Admin password for the seeded admin account")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the demo data generator")
    args = parser.parse_args()

    from app.core.config import settings

    if args.local_db:
        db_path = Path(args.local_db)
        if db_path.exists():
            db_path.unlink()
        db_url = f"sqlite:///{db_path.resolve()}"
        engine = create_engine(db_url, connect_args={"check_same_thread": False})
        from app.db.base import Base
        Base.metadata.create_all(bind=engine)
    else:
        db_url = settings.database_url
        engine = create_engine(db_url)

    app_env = settings.app_env.strip().lower()
    print("=" * 64)
    print("  DrivePulse  –  Reset & Re-seed (DESTRUCTIVE)")
    print("=" * 64)
    print(f"  Database:     {db_url}")
    print(f"  App env:      {app_env or 'local'}")
    print()

    existing = _existing_tables(engine)
    missing = [t for t in TABLE_ORDER if t not in existing]
    wipe = _wipe(engine, existing, dry_run=True)
    print(f"  Tables to wipe: {wipe['tables_to_wipe'] or '(none found)'}")
    if missing:
        print(f"  Not present (skipped): {missing}")
    counts = _row_counts(engine, existing)
    if counts:
        print(f"  Rows to delete: {counts}")
    print()

    if not args.dry_run and not args.local_db:
        is_production = app_env == "production"
        if not args.yes:
            print("  [ABORT] This is a DESTRUCTIVE wipe of your configured database.")
            print("  Pass --yes to confirm, or use --local-db FILE for a scratch SQLite DB.")
            sys.exit(1)
        if is_production:
            print("  [ABORT] Refusing to reset a production database even with --yes.")
            print("  Run this against a dev/staging DB, or set APP_ENV=development.")
            sys.exit(1)

    if args.dry_run:
        print("  [DRY-RUN] No changes made. Re-run without --dry-run to execute.")
        print(f"  Would then seed: {args.drivers} drivers, {args.drivers * args.trips_per_driver} trips")
        return

    print("  Wiping existing data ...")
    _wipe(engine, existing, dry_run=False)
    for state_path in STATE_PATHS:
        if state_path.exists():
            state_path.unlink()
            print(f"  Cleared {state_path.name}")
    print()

    # Seed the fresh sample set (same engine so admin/drivers/trips land here).
    summary = seed_demo_data(
        driver_count=min(args.drivers, 5),
        trips_per_driver=args.trips_per_driver,
        password=args.password,
        admin_password=args.admin_password,
        local_db=str(Path(args.local_db).resolve()) if args.local_db else None,
        seed=args.seed,
    )

    # Fail loudly instead of printing success when the seed produced nothing
    # (e.g. migrations were never run and the schema is missing).
    if not summary.get("trips_created") or not summary.get("trips_finalized"):
        print()
        print("  [FAILED]  Seeding produced no trips.")
        print("  Did you run `alembic upgrade head` first? Check the schema and retry.")
        sys.exit(1)

    print()
    print("-" * 60)
    print("  [SUCCESS]  RESET & SEED COMPLETE")
    print("-" * 60)
    print(f"  Admin:     {summary['admin_email']} / {args.admin_password or args.password}")
    for driver in summary["drivers"]:
        print(f"  Driver:    {driver['email']}  ({driver['name']} – {driver['profile']}) / {args.password}")
    print(f"  Trips:     {summary['trips_created']} created, {summary['trips_finalized']} finalized, "
          f"{summary['trips_reviewed']} demo-reviewed")
    print("  Review labels are 'demo_review' — the admin UI is the source of real human reviews.")


if __name__ == "__main__":
    main()
