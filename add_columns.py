import sqlite3

conn = sqlite3.connect('sdbackend.db')
cursor = conn.cursor()

# Add the missing columns
cursor.execute("ALTER TABLE trips ADD COLUMN score INTEGER")
cursor.execute("ALTER TABLE trips ADD COLUMN score_breakdown TEXT")
cursor.execute("ALTER TABLE trips ADD COLUMN feature_version TEXT")
cursor.execute("ALTER TABLE trips ADD COLUMN model_version TEXT")
cursor.execute("ALTER TABLE trips ADD COLUMN confidence REAL")
cursor.execute("ALTER TABLE trips ADD COLUMN processed_at DATETIME")
cursor.execute("ALTER TABLE trips ADD COLUMN raw_deleted BOOLEAN DEFAULT 0")

conn.commit()
conn.close()

print("Columns added to trips table")