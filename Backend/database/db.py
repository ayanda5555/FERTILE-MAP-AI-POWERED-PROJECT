import sqlite3
import os
from config import Config

def get_db():
    # Debug: show the exact path Python is trying to open
    print("Connecting to:", Config.DATABASE)
    db = sqlite3.connect(Config.DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    # Schema file is inside the same 'database' folder
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    print("Loading schema from:", schema_path)
    with open(schema_path, 'r') as f:
        db.executescript(f.read())
    db.commit()
    db.close()
    print("Database initialized successfully")

if __name__ == '__main__':
    init_db()