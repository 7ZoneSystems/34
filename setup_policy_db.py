"""
Creates and populates the Policy DB: policy.db
Tables: content_rules, blocked_patterns, safety_config, review_queue

Run once: .env/bin/python setup_policy_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS content_rules (
    rule_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_name    TEXT NOT NULL,
    rule_type    TEXT NOT NULL,          -- 'allow', 'block', 'flag'
    pattern      TEXT NOT NULL,          -- keyword or phrase (lowercase)
    description  TEXT,
    severity     INTEGER DEFAULT 1       -- 1=low, 2=medium, 3=high
);

CREATE TABLE IF NOT EXISTS blocked_patterns (
    pattern_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category     TEXT NOT NULL,          -- 'unsafe_content', 'off_syllabus_risk', 'harmful_advice'
    pattern      TEXT NOT NULL,          -- keyword/phrase (lowercase)
    action       TEXT NOT NULL DEFAULT 'block'  -- 'block', 'flag', 'retry'
);

CREATE TABLE IF NOT EXISTS safety_config (
    config_key   TEXT PRIMARY KEY,
    config_value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_queue (
    review_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   INTEGER NOT NULL,
    question     TEXT NOT NULL,
    failed_output TEXT NOT NULL,
    fail_reason  TEXT NOT NULL,
    retry_count  INTEGER DEFAULT 0,
    status       TEXT DEFAULT 'pending',  -- 'pending', 'reviewed', 'approved', 'rejected'
    assigned_mentor_id INTEGER,
    created_at   TEXT DEFAULT (datetime('now')),
    resolved_at  TEXT
);
"""

# --- Safety configuration ---
SAFETY_CONFIG = {
    "max_retries":           "2",
    "min_output_length":     "10",
    "max_output_length":     "5000",
    "require_mentor_for":    "off_syllabus",
    "block_harmful_content": "true",
    "flag_speculative":      "true",
}

# --- Blocked patterns ---
BLOCKED_PATTERNS = [
    # unsafe content
    ("unsafe_content",   "how to hack",           "block"),
    ("unsafe_content",   "how to cheat",          "block"),
    ("unsafe_content",   "how to steal",          "block"),
    ("unsafe_content",   "how to harm",           "block"),
    ("unsafe_content",   "how to make a weapon",  "block"),
    ("unsafe_content",   "how to make drugs",     "block"),
    ("unsafe_content",   "suicide",               "flag"),
    ("unsafe_content",   "self-harm",             "flag"),

    # harmful academic advice
    ("harmful_advice",   "you don't need to study",  "block"),
    ("harmful_advice",   "school is useless",        "flag"),
    ("harmful_advice",   "drop out",                 "flag"),

    # off-syllabus risk (speculative / misleading)
    ("off_syllabus_risk", "this is definitely true without evidence", "flag"),
    ("off_syllabus_risk", "scientists are wrong",     "flag"),
    ("off_syllabus_risk", "conspiracy",               "flag"),
    ("off_syllabus_risk", "pseudoscience",            "flag"),
    ("off_syllabus_risk", "astrology",                "flag"),
    ("off_syllabus_risk", "flat earth",               "block"),
]

# --- Content rules ---
CONTENT_RULES = [
    # allow rules
    ("safe_academic",      "allow", "mathematics",    "Standard math content",   1),
    ("safe_physics",       "allow", "physics",        "Standard physics content", 1),
    ("safe_cs",            "allow", "computer science","Standard CS content",     1),
    ("safe_science",       "allow", "science",        "General science content",  1),

    # flag rules (needs review)
    ("speculative",        "flag",  "might be",       "Speculative language",     2),
    ("unverified",         "flag",  "unverified",     "Unverified claims",        2),
    ("opinion",            "flag",  "i believe",      "Opinion presented as fact", 2),
    ("off_topic_drift",    "flag",  "anyway",         "Off-topic drift detected", 1),

    # block rules
    ("dangerous",          "block", "dangerous",      "Dangerous content marker", 3),
    ("illegal",            "block", "illegal",        "Illegal activity reference", 3),
]


def create_and_populate():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript(SCHEMA)

    # safety config
    for key, val in SAFETY_CONFIG.items():
        cur.execute(
            "INSERT INTO safety_config (config_key, config_value) VALUES (?, ?)",
            (key, val),
        )

    # blocked patterns
    for cat, pattern, action in BLOCKED_PATTERNS:
        cur.execute(
            "INSERT INTO blocked_patterns (category, pattern, action) VALUES (?, ?, ?)",
            (cat, pattern, action),
        )

    # content rules
    for name, rtype, pattern, desc, severity in CONTENT_RULES:
        cur.execute(
            "INSERT INTO content_rules (rule_name, rule_type, pattern, description, severity) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, rtype, pattern, desc, severity),
        )

    conn.commit()

    print(f"Policy DB created at {DB_PATH}")
    for table in ["content_rules", "blocked_patterns", "safety_config"]:
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    conn.close()


if __name__ == "__main__":
    create_and_populate()
