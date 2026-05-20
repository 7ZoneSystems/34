"""
Creates the empty Per-Session Memory DB: session_memory.db
Tables are defined but contain no data.
Run once: python setup_session_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_memory.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS session_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,          -- 'student', 'mentor', 'system'
    content     TEXT NOT NULL,
    metadata    TEXT,                   -- JSON blob for extras
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS session_state (
    session_id  TEXT PRIMARY KEY,
    student_id  INTEGER,
    topic_id    INTEGER,
    status      TEXT DEFAULT 'active',  -- active, paused, completed
    started_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);
"""


def create_empty():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA)
    conn.commit()

    print(f"Session Memory DB created at {DB_PATH} (empty — 0 rows in all tables)")
    conn.close()


if __name__ == "__main__":
    create_empty()
