"""
Mock Caller — invokes mock_mentor.py for relevant mentors.

Usage:
    python mock_caller.py --question "Is this algebra solution correct?" --topic_id 1
    python mock_caller.py --question "Explain Newton's third law"

The caller:
  1. Looks up which mentors cover the given topic (or all mentors if no topic_id)
  2. Calls mock_mentor.py for each mentor via subprocess
  3. Collects and prints the combined responses as JSON
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "central.db")
MENTOR_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_mentor.py")
PYTHON = sys.executable  # use the same python interpreter


def get_mentors_for_topic(topic_id: int) -> list[dict]:
    """Find all mentors whose subject covers the given topic_id."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT DISTINCT m.mentor_id, m.mentor_name
        FROM mentors m
        JOIN mentor_subjects ms ON m.mentor_id = ms.mentor_id
        JOIN chapters c         ON c.subject_id = ms.subject_id
        JOIN topics t           ON t.chapter_id  = c.chapter_id
        WHERE t.topic_id = ?
        ORDER BY m.mentor_id
    """, (topic_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_mentors() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT mentor_id, mentor_name FROM mentors ORDER BY mentor_id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def call_mentor(mentor_id: int, question: str, topic_id: int | None = None) -> dict:
    """Invoke mock_mentor.py as a subprocess and return its JSON response."""
    cmd = [PYTHON, MENTOR_SCRIPT, "--mentor_id", str(mentor_id), "--question", question]
    if topic_id is not None:
        cmd += ["--topic_id", str(topic_id)]

    print(f"\n>>> Calling mentor id={mentor_id} ...")
    result = subprocess.run(cmd, capture_output=False)  # stdin/stdout go to terminal

    # The mentor script prints JSON at the end; we also re-read from its stdout.
    # Since we passed capture_output=False, the human interacts directly.
    # We need to re-parse — use a second call that captures, but that blocks input.
    # Instead, return a placeholder; the real output is already on screen.
    return {"mentor_id": mentor_id, "note": "Response printed above by mock_mentor.py"}


def call_mentor_captured(mentor_id: int, question: str, topic_id: int | None = None) -> dict:
    """
    Alternative: call mock_mentor.py and capture its JSON output.
    Requires piping the yes/no answer through stdin.
    Not used in interactive mode — included for future automation.
    """
    cmd = [PYTHON, MENTOR_SCRIPT, "--mentor_id", str(mentor_id), "--question", question]
    if topic_id is not None:
        cmd += ["--topic_id", str(topic_id)]

    result = subprocess.run(cmd, capture_output=True, text=True, input="")
    # parse last JSON block from stdout
    try:
        json_start = result.stdout.rindex("--- RESPONSE (JSON) ---")
        json_blob = result.stdout[json_start + len("--- RESPONSE (JSON) ---"):].strip()
        return json.loads(json_blob)
    except (ValueError, json.JSONDecodeError):
        return {"mentor_id": mentor_id, "raw_stdout": result.stdout, "error": "Could not parse JSON"}


def main():
    parser = argparse.ArgumentParser(description="Mock Mentor Caller")
    parser.add_argument("--question", type=str, required=True, help="The question to ask mentors")
    parser.add_argument("--topic_id", type=int, default=None, help="Topic ID (filters mentors)")
    args = parser.parse_args()

    if args.topic_id is not None:
        mentors = get_mentors_for_topic(args.topic_id)
        if not mentors:
            print(f"No mentors found for topic_id={args.topic_id}")
            sys.exit(1)
        print(f"Found {len(mentors)} mentor(s) for topic_id={args.topic_id}:")
    else:
        mentors = get_all_mentors()
        print(f"No topic specified — calling all {len(mentors)} mentor(s):")

    for m in mentors:
        print(f"  - {m['mentor_name']} (id={m['mentor_id']})")

    # --- call each mentor interactively ---
    responses = []
    for m in mentors:
        print(f"\n{'='*60}")
        print(f"Calling: {m['mentor_name']}")
        print(f"{'='*60}")
        resp = call_mentor(m["mentor_id"], args.question, args.topic_id)
        responses.append(resp)

    # --- summary ---
    print(f"\n{'='*60}")
    print("ALL RESPONSES COLLECTED")
    print(f"{'='*60}")
    print(json.dumps(responses, indent=2))


if __name__ == "__main__":
    main()
