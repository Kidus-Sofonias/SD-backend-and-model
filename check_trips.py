from sqlalchemy import text
from app.db.session import SessionLocal

db = SessionLocal()
result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
tables = result.fetchall()
print("Tables:", tables)
db.close()