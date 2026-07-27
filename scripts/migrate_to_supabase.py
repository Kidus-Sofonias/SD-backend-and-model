"""Migrate SQLite data to Supabase PostgreSQL with retry logic."""
import sys, time, random
from datetime import datetime
sys.path.insert(0, ".")
from sqlalchemy import create_engine, text

SQLITE = "sqlite:///./demo_v3.db"
SUPA = "postgresql://postgres.ifvngxypgubpuabbevyg:5puCYSaFLT4PWMYt@aws-0-eu-west-1.pooler.supabase.com:6543/postgres"
BATCH = 500

def log(m):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {m}")
    sys.stdout.flush()

def retry_insert(pg_conn, stmt, params, max_retries=5):
    for attempt in range(max_retries):
        try:
            pg_conn.execute(stmt, params)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)
            log(f"  retry {attempt+1}/{max_retries} after {wait:.0f}s: {e}")
            time.sleep(wait)
            # Reconnect
            pg_conn.connection.invalidate()
            pg_conn.connection.dbapi_connection.reconnect()

t0 = time.time()
sq = create_engine(SQLITE, connect_args={"check_same_thread": False, "timeout": 30})
pg = create_engine(SUPA, connect_args={
    "sslmode": "require",
    "connect_timeout": 60,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5,
})
log("Starting migration...")

# Clean Supabase sensor_samples and re-migrate all
with pg.begin() as c:
    c.execute(text("DELETE FROM sensor_samples"))
    log("Cleared sensor_samples in Supabase")

# --- driving_events: resume from max id ---
with pg.connect() as c:
    max_done = c.execute(text("SELECT COALESCE(MAX(id),0) FROM driving_events")).scalar()
log(f"driving_events: resuming from id > {max_done}")

with sq.connect() as c:
    total = c.execute(text("SELECT COUNT(*) FROM driving_events WHERE id > :mid"), {"mid": max_done}).scalar()
log(f"{total} driving_events rows remaining")

INSERT_EVENT = text("""
    INSERT INTO driving_events(id,user_id,trip_id,event_type,value,occurred_at,lat,lon,created_at)
    VALUES(:id,:uid,:tid,:et,:v,:oc,:lat,:lon,:ca) ON CONFLICT DO NOTHING
""")

offset = 0
while offset < total:
    with sq.connect() as c:
        rows = c.execute(
            text("SELECT id,user_id,trip_id,event_type,value,occurred_at,lat,lon,created_at FROM driving_events WHERE id > :mid ORDER BY id LIMIT :lim OFFSET :off"),
            {"mid": max_done, "lim": BATCH, "off": offset}
        ).fetchall()
    if not rows:
        break
    params = [{"id": r.id, "uid": r.user_id, "tid": r.trip_id, "et": r.event_type, "v": r.value,
               "oc": r.occurred_at, "lat": r.lat, "lon": r.lon, "ca": r.created_at} for r in rows]
    with pg.begin() as c:
        retry_insert(c, INSERT_EVENT, params)
    offset += len(rows)
    if offset % 2000 == 0 or offset >= total:
        log(f"  driving_events: {min(offset,total)}/{total}")
log("driving_events done!")

# --- sensor_samples: migrate all ---
with sq.connect() as c:
    total = c.execute(text("SELECT COUNT(*) FROM sensor_samples")).scalar()
log(f"sensor_samples to migrate: {total}")

INSERT_SAMPLE = text("""
    INSERT INTO sensor_samples(id,user_id,trip_id,ts,speed_mps,lat,lon,accuracy_m,altitude_m,ax,ay,az,gx,gy,gz)
    VALUES(:id,:uid,:tid,:ts,:sp,:lat,:lon,:acc,:alt,:ax,:ay,:az,:gx,:gy,:gz) ON CONFLICT DO NOTHING
""")

offset = 0
while offset < total:
    with sq.connect() as c:
        rows = c.execute(
            text("SELECT id,user_id,trip_id,ts,speed_mps,lat,lon,accuracy_m,altitude_m,ax,ay,az,gx,gy,gz FROM sensor_samples ORDER BY id LIMIT :lim OFFSET :off"),
            {"lim": BATCH, "off": offset}
        ).fetchall()
    if not rows:
        break
    params = [{"id": r.id, "uid": r.user_id, "tid": r.trip_id, "ts": r.ts, "sp": r.speed_mps,
               "lat": r.lat, "lon": r.lon, "acc": r.accuracy_m, "alt": r.altitude_m,
               "ax": r.ax, "ay": r.ay, "az": r.az, "gx": r.gx, "gy": r.gy, "gz": r.gz} for r in rows]
    with pg.begin() as c:
        retry_insert(c, INSERT_SAMPLE, params)
    offset += len(rows)
    if offset % 5000 == 0 or offset >= total:
        log(f"  sensor_samples: {min(offset,total)}/{total}")
log("sensor_samples done!")

# Verify
log("Verifying counts...")
with sq.connect() as sc, pg.connect() as pc:
    for t in ["users", "trips", "driving_events", "sensor_samples"]:
        s = sc.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
        p = pc.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
        ok = "OK" if s == p else f"MISMATCH (diff={s-p})"
        log(f"  {ok} {t}: src={s} dst={p}")

log(f"Total time: {time.time()-t0:.0f}s")
