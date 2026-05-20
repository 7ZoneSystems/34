"""
Student Learning Pipeline — clean student-facing interface with AI Tutor.

Usage:
    .env/bin/python main.py

Flow: Question → Memory DB → D2 → D2.1 → D4 → D3 → AI Tutor (guided learning)
The AI tutor teaches subtopics one by one until the student says "new topic".
"""

import sys
import os
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import Pipeline

EXIT_PATTERNS = re.compile(
    r"^(new topic|next topic|another topic|different topic|"
    r"done|stop|quit|exit|next|skip|that's all|i'm done|"
    r"learn something new|something else|move on)",
    re.IGNORECASE,
)


def welcome():
    print()
    print("=" * 56)
    print("          Welcome to the Learning Assistant")
    print("=" * 56)
    print()
    print("  Ask me any question about your studies.")
    print("  I'll teach you the topic step by step.")
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


def is_non_question(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 3:
        return True
    non_q = [
        r"^(thanks?|thank you|thankyou)",
        r"^(sorry|my bad|oops)",
        r"^(ok|okay|sure|alright|fine)",
        r"^(hi|hello|hey|yo)\b",
        r"^(bye|goodbye|see you|later)",
        r"^(cool|nice|great|wow|awesome)",
    ]
    for pat in non_q:
        if re.match(pat, t):
            return True
    return False


def wants_new_topic(text: str) -> bool:
    return bool(EXIT_PATTERNS.match(text.strip()))


def guided_learning(pipeline, student, topic, subtopics, date):
    """Enter guided learning mode — teach subtopics one by one."""
    tutor = pipeline.tutor
    history = []

    topic_short = topic.split(" > ")[-1] if " > " in topic else topic
    n = len(subtopics)

    print()
    print(f"  Let's learn about {topic_short}!")
    print(f"  We'll go through {n} subtopic{'s' if n > 1 else ''}.")
    print(f"  Say 'new topic' anytime to stop.")
    print()

    for i, subtopic in enumerate(subtopics, 1):
        # teach this subtopic
        print(f"  --- Subtopic {i}/{n}: {subtopic} ---")
        print()

        explanation = tutor.teach_subtopic(topic, subtopic, history)
        history.append({"role": "assistant", "content": explanation})

        # D3 check on tutor output
        d3_check = pipeline.d3.check_output(explanation, subtopic, {})
        if not d3_check["safe"]:
            explanation = (
                f"  Let me explain {subtopic} briefly: "
                f"This is an important concept in {topic_short}. "
                f"Please ask your teacher for more details."
            )

        for line in explanation.split("\n"):
            print(f"    {line}")
        print()

        # if last subtopic, done
        if i == n:
            print(f"  That covers {topic_short}! Great job studying!")
            print()
            break

        # wait for student to continue or exit
        while True:
            answer = input("  You: ").strip()
            if not answer:
                continue
            if answer.lower() in ("quit", "exit", "q"):
                print("\n  Goodbye! Keep studying!\n")
                return "quit"
            if answer.lower() == "switch":
                return "switch"
            if wants_new_topic(answer):
                print(f"\n  Sure! Let's move on to something new.\n")
                return "continue"
            # any other input = continue to next subtopic
            # but also store it as a follow-up question in memory
            pipeline.handler.session_conn.execute(
                """INSERT INTO session_memory
                       (session_id, role, content, metadata)
                   VALUES (?, 'student', ?, ?)""",
                (
                    f"session_{student['student_id']}",
                    answer,
                    '{"date":"' + date + '","follow_up":true}',
                ),
            )
            pipeline.handler.session_conn.commit()
            history.append({"role": "user", "content": answer})
            # give a brief response to the follow-up, then continue
            followup = tutor.teach_subtopic(topic, subtopic, history)
            history.append({"role": "assistant", "content": followup})
            for line in followup.split("\n"):
                print(f"    {line}")
            print()
            break

    return "continue"


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

            if is_non_question(question):
                print()
                print("  Assistant: Feel free to ask me anything about your studies!")
                continue

            # run pipeline (D2 → D2.1 → D4 → D3)
            result = pipeline.run_question_silent(
                student["student_id"], today, question
            )

            d3 = result.get("d3")
            if d3 and d3.get("status") == "mentor_review":
                print()
                print("  Your query is being reviewed by a mentor for safety.")
                print("  You'll receive a response once it's been checked.")
                continue

            topic = result.get("topic", question)
            subtopics = result.get("subtopics", [question])

            # enter guided learning
            action = guided_learning(
                pipeline, student, topic, subtopics, today
            )
            if action == "quit":
                break
            if action == "switch":
                student = pick_student(pipeline)
                if student is None:
                    print("\n  Goodbye!")
                    break
                print(f"\n  Hello, {student['name']}!")

    except KeyboardInterrupt:
        print("\n\n  Goodbye!")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
