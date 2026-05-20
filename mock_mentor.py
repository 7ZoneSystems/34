"""
Mock Mentor Endpoint — CLI-based.
Usage:
    python mock_mentor.py --mentor_id 1 --question "Is this answer correct?"
    python mock_mentor.py --mentor_id 3 --topic_id 7 --question "Should the student proceed?"

The script prints the question, waits for you (the human) to type yes/no
and optional reasoning, then prints a JSON response to stdout.

Exit codes: 0 = success, 1 = bad args, 2 = mentor not found
"""

import argparse
import json
import sqlite3
import sys
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "central.db")


def get_mentor(mentor_id: int) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT mentor_id, mentor_name, bio FROM mentors WHERE mentor_id = ?",
        (mentor_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def main():
    parser = argparse.ArgumentParser(description="Mock Mentor CLI Endpoint")
    parser.add_argument("--mentor_id", type=int, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--topic_id", type=int, default=None, help="Optional topic context")
    args = parser.parse_args()

    mentor = get_mentor(args.mentor_id)
    if mentor is None:
        print(f"Error: mentor_id {args.mentor_id} not found in central.db", file=sys.stderr)
        sys.exit(2)

    # --- present the "incoming request" to the human operator ---
    print("=" * 50)
    print(f"MENTOR CALL  →  {mentor['mentor_name']} (id={mentor['mentor_id']})")
    if args.topic_id is not None:
        print(f"TOPIC ID     →  {args.topic_id}")
    print(f"QUESTION     →  {args.question}")
    print("=" * 50)

    # --- wait for human input ---
    while True:
        answer = input("\nYour decision (yes / no): ").strip().lower()
        if answer in ("yes", "no", "y", "n"):
            answer = "yes" if answer in ("yes", "y") else "no"
            break
        print("  Please type 'yes' or 'no'.")

    reasoning = input("Reasoning (optional, press Enter to skip): ").strip()

    # --- build response ---
    response = {
        "mentor_id":   mentor["mentor_id"],
        "mentor_name": mentor["mentor_name"],
        "topic_id":    args.topic_id,
        "question":    args.question,
        "decision":    answer,
        "reasoning":   reasoning if reasoning else None,
    }

    print("\n--- RESPONSE (JSON) ---")
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
