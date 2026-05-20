"""
Student Learning Pipeline — clean student-facing interface.

Usage:
    .env/bin/python main.py

The student picks their ID, asks questions naturally, and sees
friendly responses. All internal pipeline logic (D2, D2.1, D4, D1,
mentor routing) runs silently in the background.
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import Pipeline


def welcome():
    print()
    print("=" * 56)
    print("          Welcome to the Learning Assistant")
    print("=" * 56)
    print()
    print("  Ask me any question about your studies.")
    print("  I'll find the right topic and connect you")
    print("  with a mentor when needed.")
    print()
    print("  Type 'quit' to exit, 'switch' to change student.")
    print()


def pick_student(pipeline):
    students = pipeline.handler.list_students()
    print("  Who are you?")
    print()
    for s in students:
        print(f"    [{s['student_id']}]  {s['name']}  (Grade {s['grade']})")
    print()
    while True:
        raw = input("  Enter your number: ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            return None
        try:
            sid = int(raw)
            student = pipeline.handler.get_student(sid)
            if student:
                return student
            print(f"  No student with ID {sid}. Try again.")
        except ValueError:
            print("  Please enter a number.")


def format_response(result: dict) -> str:
    """Turn pipeline result dict into a student-friendly message."""
    d2 = result["d2"]
    d2_1 = result.get("d2_1")
    d4 = result.get("d4")
    d1 = result.get("d1")
    d3 = result.get("d3")
    mentors = result.get("mentors", [])
    lines = []

    # D3 safety gate — if queued for mentor review, show that and stop
    if d3 and d3.get("status") == "mentor_review":
        return (
            "Your query is being reviewed by a mentor for safety.\n"
            "  You'll receive a response once it's been checked.\n"
            "  In the meantime, feel free to ask another question!"
        )

    classification = d2["classification"]

    # --- syllabus match (new or repeat) ---
    if classification in ("MATCHES_SYLLABUS_NEW", "MATCHES_SYLLABUS_REPEAT"):
        topic = d2["syllabus_topic"]
        topic_short = topic.split(" > ")[-1] if " > " in topic else topic
        subject = topic.split(" > ")[0] if " > " in topic else topic

        if classification == "MATCHES_SYLLABUS_REPEAT" and d1:
            pct = d1["coverage_pct"]
            covered = len(d1["coverage"])
            total = covered + len(d1["missing_topics"])
            lines.append(
                f"Welcome back! We've covered {covered} of {total} "
                f"subtopics in {subject} ({pct}% so far)."
            )
            if d1["missing_topics"]:
                lines.append(
                    "  Topics still to explore:"
                )
                for t in d1["missing_topics"][:5]:
                    lines.append(f"    - {t}")
        else:
            lines.append(
                f"Great question! This falls under {subject}."
            )

        # mentor decisions
        if mentors:
            approved = [m for m in mentors if m.get("decision") == "yes"]
            rejected = [m for m in mentors if m.get("decision") == "no"]
            errors  = [m for m in mentors if "error" in m]

            if approved:
                names = ", ".join(m.get("mentor_name", "?") for m in approved)
                lines.append(
                    f"  Mentor approved: {names} — you're on the right track!"
                )
            if rejected:
                names = ", ".join(m.get("mentor_name", "?") for m in rejected)
                lines.append(
                    f"  Mentor suggested review: {names} recommends "
                    f"revisiting the basics first."
                )
                for m in rejected:
                    if m.get("reasoning"):
                        lines.append(f"    \"{m['reasoning']}\"")
            if errors:
                lines.append(
                    "  (Mentor unavailable right now — try again later.)"
                )
        else:
            lines.append("  No mentors available for this topic right now.")

    # --- off-syllabus: D2.1 matched to syllabus ---
    elif d2_1 and d2_1.get("status") == "syllabus_match":
        topic = d2_1["topic"]
        topic_short = topic.split(" > ")[-1] if " > " in topic else topic
        lines.append(
            f"Interesting question! I found a connection to {topic_short} "
            f"in your syllabus."
        )
        if d2_1.get("subtopics"):
            lines.append("  Related syllabus topics:")
            for t in d2_1["subtopics"][:4]:
                short = t.split(" > ")[-1] if " > " in t else t
                lines.append(f"    - {short}")

    # --- off-syllabus: D2.1 → D4 redirect ---
    elif d4:
        tag = d4.get("classification_tag", "unknown")
        status = d4.get("d4_status", "")

        if status == "guide_mode":
            lines.append(
                "I can see you're really curious about this topic!"
            )
            lines.append(
                "  Here's what I found to help you explore further:"
            )
            result_text = d4.get("result", "")
            for line in result_text.split("\n"):
                line = line.strip()
                if line:
                    lines.append(f"    {line}")
        elif tag == "genuinely_novel":
            lines.append(
                "This is a great question that goes beyond the syllabus!"
            )
            if d4.get("result"):
                lines.append("  Here's some enrichment material:")
                for line in d4["result"].split("\n"):
                    line = line.strip()
                    if line:
                        lines.append(f"    {line}")
        else:
            lines.append(
                "That's an interesting question, but it's outside your "
                "current syllabus."
            )
            lines.append(
                "  Try focusing on your current topics first, "
                "and come back to this later!"
            )

    # --- off-syllabus: D2.1 no match, no D4 ---
    elif d2_1 and d2_1.get("status") == "redirect_to_mentor":
        closest = d2_1.get("closest_syllabus_topic", "")
        closest_short = (closest.split(" > ")[-1]
                         if " > " in closest else closest)
        lines.append(
            "This topic isn't in your syllabus yet, "
            "but it might relate to "
            f"{closest_short or 'your studies'}."
        )
        lines.append(
            "  I'd recommend asking a mentor about this."
        )

    # --- fallback ---
    else:
        lines.append(
            "I'm not sure how to help with that one. "
            "Try rephrasing or ask about a specific subject!"
        )

    return "\n".join(lines)


def main():
    pipeline = Pipeline()
    try:
        welcome()
        student = pick_student(pipeline)
        if student is None:
            print("  Goodbye!")
            return

        print(f"\n  Hello, {student['name']}!")
        today = datetime.now().strftime("%Y-%m-%d")

        while True:
            print()
            question = input("  You: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("\n  Goodbye! Keep studying!\n")
                break
            if question.lower() == "switch":
                student = pick_student(pipeline)
                if student is None:
                    print("\n  Goodbye!")
                    break
                print(f"\n  Hello, {student['name']}!")
                continue

            # run pipeline silently
            result = pipeline.run_question_silent(
                student["student_id"], today, question
            )

            # show student-friendly response
            response = format_response(result)
            print()
            print(f"  Assistant: {response}")

    except KeyboardInterrupt:
        print("\n\n  Goodbye!")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
