"""
Creates and populates the Central DB: central.db
Tables: students, subjects, chapters, topics, mentors, mentor_subjects
Run once: python setup_central_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "central.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    email           TEXT,
    grade           TEXT,
    enrolled_at     TEXT DEFAULT (datetime('now')),
    review_status   TEXT DEFAULT 'active',
    pending_reviews INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS subjects (
    subject_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS chapters (
    chapter_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id   INTEGER NOT NULL,
    chapter_name TEXT NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

CREATE TABLE IF NOT EXISTS topics (
    topic_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id   INTEGER NOT NULL,
    topic_name   TEXT NOT NULL,
    FOREIGN KEY (chapter_id) REFERENCES chapters(chapter_id)
);

CREATE TABLE IF NOT EXISTS mentors (
    mentor_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    mentor_name  TEXT NOT NULL,
    bio          TEXT
);

CREATE TABLE IF NOT EXISTS mentor_subjects (
    mentor_id    INTEGER NOT NULL,
    subject_id   INTEGER NOT NULL,
    PRIMARY KEY (mentor_id, subject_id),
    FOREIGN KEY (mentor_id)  REFERENCES mentors(mentor_id),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

CREATE TABLE IF NOT EXISTS mentor_reviews (
    review_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id         INTEGER NOT NULL,
    question           TEXT NOT NULL,
    failed_output      TEXT NOT NULL,
    fail_reason        TEXT NOT NULL,
    retry_count        INTEGER DEFAULT 0,
    assigned_mentor_id INTEGER,
    status             TEXT DEFAULT 'pending',
    created_at         TEXT,
    resolved_at        TEXT,
    FOREIGN KEY (student_id)         REFERENCES students(student_id),
    FOREIGN KEY (assigned_mentor_id) REFERENCES mentors(mentor_id)
);
"""

STUDENTS = [
    ("Aarav Patel",   "aarav@example.com",   "10"),
    ("Priya Sharma",  "priya@example.com",   "12"),
    ("Rohan Gupta",   "rohan@example.com",   "11"),
    ("Ananya Singh",  "ananya@example.com",  "10"),
    ("Vikram Reddy",  "vikram@example.com",  "12"),
]

SUBJECTS = ["Mathematics", "Physics", "Computer Science"]

CHAPTERS = {
    "Mathematics": {
        "Algebra":   ["Linear Equations", "Quadratic Equations", "Polynomials"],
        "Calculus":  ["Limits", "Derivatives", "Integrals"],
        "Geometry":  ["Triangles", "Circles", "Coordinate Geometry"],
    },
    "Physics": {
        "Mechanics":     ["Newton's Laws", "Work & Energy", "Momentum"],
        "Thermodynamics":["Heat Transfer", "Laws of Thermodynamics", "Entropy"],
        "Optics":        ["Reflection", "Refraction", "Lenses"],
    },
    "Computer Science": {
        "Programming Fundamentals": ["Variables & Data Types", "Control Flow", "Functions"],
        "Data Structures":          ["Arrays", "Linked Lists", "Trees & Graphs"],
        "Algorithms":               ["Sorting", "Searching", "Dynamic Programming"],
    },
}

MENTORS = [
    ("Dr. Meera Iyer",    "Expert in algebra and calculus with 15 years experience",        ["Mathematics"]),
    ("Prof. Arjun Nair",  "Physics researcher specialising in mechanics and thermodynamics", ["Physics"]),
    ("Ms. Kavya Das",     "Full-stack developer and CS educator",                           ["Computer Science"]),
    ("Mr. Rahul Menon",   "Mathematician with focus on geometry and number theory",         ["Mathematics"]),
    ("Dr. Sneha Kapoor",  "Interdisciplinary mentor covering physics and CS",               ["Physics", "Computer Science"]),
]


def create_and_populate():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # --- schema ---
    cur.executescript(SCHEMA)

    # --- students ---
    cur.executemany(
        "INSERT INTO students (name, email, grade) VALUES (?, ?, ?)", STUDENTS
    )

    # --- subjects, chapters, topics ---
    subj_ids = {}
    for subj_name in SUBJECTS:
        cur.execute("INSERT INTO subjects (subject_name) VALUES (?)", (subj_name,))
        subj_ids[subj_name] = cur.lastrowid

    for subj_name, chapters in CHAPTERS.items():
        sid = subj_ids[subj_name]
        for chap_name, topics in chapters.items():
            cur.execute(
                "INSERT INTO chapters (subject_id, chapter_name) VALUES (?, ?)",
                (sid, chap_name),
            )
            chap_id = cur.lastrowid
            for topic_name in topics:
                cur.execute(
                    "INSERT INTO topics (chapter_id, topic_name) VALUES (?, ?)",
                    (chap_id, topic_name),
                )

    # --- mentors + mentor_subjects ---
    for mname, bio, subject_list in MENTORS:
        cur.execute(
            "INSERT INTO mentors (mentor_name, bio) VALUES (?, ?)", (mname, bio)
        )
        mid = cur.lastrowid
        for sname in subject_list:
            cur.execute(
                "INSERT INTO mentor_subjects (mentor_id, subject_id) VALUES (?, ?)",
                (mid, subj_ids[sname]),
            )

    conn.commit()

    # --- quick summary ---
    print(f"Central DB created at {DB_PATH}")
    for table in ["students", "subjects", "chapters", "topics", "mentors"]:
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    conn.close()


if __name__ == "__main__":
    create_and_populate()
