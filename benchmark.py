"""
Benchmark: Pipeline vs Plain Grok

Runs ALL 27 syllabus topics through:
  A) Our pipeline (D2→D2.1→D4→D3→AI Tutor guided learning)
  B) Plain xAI Grok (direct question→answer, no pipeline)

Uses Grok as judge, generates comparison graphs.

Usage:
    .env/bin/python benchmark.py
"""

import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import Pipeline, XAI_API_KEY, XAI_MODEL, OpenAI

# ---------------------------------------------------------------------------
# One question per topic — all 27 topics from central DB
# ---------------------------------------------------------------------------
QUESTIONS = [
    # Mathematics > Algebra
    {"id": 1,  "q": "How to solve a system of linear equations?",
     "subject": "Mathematics", "topic": "Linear Equations"},
    {"id": 2,  "q": "Explain the quadratic formula and the discriminant",
     "subject": "Mathematics", "topic": "Quadratic Equations"},
    {"id": 3,  "q": "What are polynomials and how do you factor them?",
     "subject": "Mathematics", "topic": "Polynomials"},
    # Mathematics > Calculus
    {"id": 4,  "q": "What is a limit in calculus and how do you evaluate it?",
     "subject": "Mathematics", "topic": "Limits"},
    {"id": 5,  "q": "How to find the derivative of a function using rules?",
     "subject": "Mathematics", "topic": "Derivatives"},
    {"id": 6,  "q": "What is integration and how do you compute a definite integral?",
     "subject": "Mathematics", "topic": "Integrals"},
    # Mathematics > Geometry
    {"id": 7,  "q": "What are the types of triangles and how to prove congruence?",
     "subject": "Mathematics", "topic": "Triangles"},
    {"id": 8,  "q": "What is the equation of a circle and how to find arc length?",
     "subject": "Mathematics", "topic": "Circles"},
    {"id": 9,  "q": "How to find distance and midpoint between two points?",
     "subject": "Mathematics", "topic": "Coordinate Geometry"},
    # Physics > Mechanics
    {"id": 10, "q": "Explain Newton's three laws of motion with examples",
     "subject": "Physics", "topic": "Newton's Laws"},
    {"id": 11, "q": "What is work in physics and explain conservation of energy",
     "subject": "Physics", "topic": "Work & Energy"},
    {"id": 12, "q": "What is momentum and explain elastic vs inelastic collision",
     "subject": "Physics", "topic": "Momentum"},
    # Physics > Thermodynamics
    {"id": 13, "q": "What are the three modes of heat transfer?",
     "subject": "Physics", "topic": "Heat Transfer"},
    {"id": 14, "q": "Explain the first and second laws of thermodynamics",
     "subject": "Physics", "topic": "Laws of Thermodynamics"},
    {"id": 15, "q": "What is entropy and how does it relate to disorder?",
     "subject": "Physics", "topic": "Entropy"},
    # Physics > Optics
    {"id": 16, "q": "What is the law of reflection and how do mirrors work?",
     "subject": "Physics", "topic": "Reflection"},
    {"id": 17, "q": "What is refraction and explain Snell's law",
     "subject": "Physics", "topic": "Refraction"},
    {"id": 18, "q": "How do convex and concave lenses work? What is focal length?",
     "subject": "Physics", "topic": "Lenses"},
    # CS > Programming Fundamentals
    {"id": 19, "q": "What are variables and data types in Python?",
     "subject": "Computer Science", "topic": "Variables & Data Types"},
    {"id": 20, "q": "How do for loops, while loops, and if-else work in Python?",
     "subject": "Computer Science", "topic": "Control Flow"},
    {"id": 21, "q": "What is a function and how to write a recursive function?",
     "subject": "Computer Science", "topic": "Functions"},
    # CS > Data Structures
    {"id": 22, "q": "What is an array and how to reverse one in-place?",
     "subject": "Computer Science", "topic": "Arrays"},
    {"id": 23, "q": "What is a linked list and how to detect a cycle in it?",
     "subject": "Computer Science", "topic": "Linked Lists"},
    {"id": 24, "q": "What is a binary tree and explain BFS vs DFS traversal",
     "subject": "Computer Science", "topic": "Trees & Graphs"},
    # CS > Algorithms
    {"id": 25, "q": "How does quicksort work and what is its time complexity?",
     "subject": "Computer Science", "topic": "Sorting"},
    {"id": 26, "q": "What is binary search and when to use it over linear search?",
     "subject": "Computer Science", "topic": "Searching"},
    {"id": 27, "q": "What is dynamic programming and explain memoization vs tabulation",
     "subject": "Computer Science", "topic": "Dynamic Programming"},
]

DIMENSIONS = ["relevance", "completeness", "structure", "pedagogy", "accuracy"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_plain_grok(client, question: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[
                {"role": "system",
                 "content": ("You are a helpful tutor. Answer clearly. "
                             "Cover key concepts, give examples, "
                             "explain step by step.")},
                {"role": "user", "content": question},
            ],
            temperature=0.7, max_tokens=1000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def get_pipeline_teaching(pipeline, question: str) -> dict:
    result = pipeline.run_question_silent(1, "2026-05-21", question)
    topic = result.get("topic", question)
    subtopics = result.get("subtopics", [question])
    teachings = []
    history = []
    for sub in subtopics[:3]:
        t = pipeline.tutor.teach_subtopic(topic, sub, history)
        history.append({"role": "assistant", "content": t})
        teachings.append({"subtopic": sub, "content": t})
    return {
        "topic": topic,
        "subtopics": subtopics,
        "teachings": teachings,
        "classification": result["d2"].get("classification", ""),
        "syllabus_match": result["d2"].get("syllabus_match", False),
    }


def judge(client, question: str, pipeline_out: dict, grok_answer: str) -> dict:
    pipe_text = "\n\n".join(
        f"[{t['subtopic']}]\n{t['content']}"
        for t in pipeline_out.get("teachings", [])
    )

    prompt = f"""You are an impartial judge comparing two tutoring approaches.

QUESTION: "{question}"

--- APPROACH A (Structured Pipeline with guided subtopic teaching) ---
Topic: {pipeline_out.get('topic', 'N/A')}
Subtopics: {', '.join(pipeline_out.get('subtopics', [])[:6])}

Teaching:
{pipe_text[:2500]}

--- APPROACH B (Direct answer) ---
{grok_answer[:2500]}

Score each 1-10 on: relevance, completeness, structure, pedagogy, accuracy.
Return ONLY JSON:
{{"A": {{"relevance":X,"completeness":X,"structure":X,"pedagogy":X,"accuracy":X}},
  "B": {{"relevance":X,"completeness":X,"structure":X,"pedagogy":X,"accuracy":X}},
  "winner":"A" or "B" or "tie","reason":"brief"}}"""

    resp = client.chat.completions.create(
        model=XAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=400,
    )
    text = resp.choices[0].message.content.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def generate_graphs(results: list):
    """Generate comparison charts."""
    subjects = ["Mathematics", "Physics", "Computer Science"]
    colors_a = "#2196F3"
    colors_b = "#FF9800"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Pipeline vs Plain Grok — Benchmark Results",
                 fontsize=16, fontweight="bold")

    # --- 1. Overall score comparison per question ---
    ax = axes[0][0]
    a_totals = [r["scores"]["A_total"] for r in results]
    b_totals = [r["scores"]["B_total"] for r in results]
    x = np.arange(len(results))
    w = 0.35
    ax.bar(x - w/2, a_totals, w, label="Pipeline", color=colors_a, alpha=0.85)
    ax.bar(x + w/2, b_totals, w, label="Plain Grok", color=colors_b, alpha=0.85)
    ax.set_xlabel("Question #")
    ax.set_ylabel("Total Score (out of 50)")
    ax.set_title("Total Score per Question")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["id"]) for r in results], fontsize=8)
    ax.legend()
    ax.set_ylim(0, 55)

    # --- 2. Average by dimension ---
    ax = axes[0][1]
    dim_labels = ["Relevance", "Completeness", "Structure", "Pedagogy", "Accuracy"]
    a_avgs = []
    b_avgs = []
    for dim in DIMENSIONS:
        a_avgs.append(np.mean([r["scores"]["A"][dim] for r in results]))
        b_avgs.append(np.mean([r["scores"]["B"][dim] for r in results]))
    x = np.arange(len(dim_labels))
    ax.bar(x - w/2, a_avgs, w, label="Pipeline", color=colors_a, alpha=0.85)
    ax.bar(x + w/2, b_avgs, w, label="Plain Grok", color=colors_b, alpha=0.85)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Average Score")
    ax.set_title("Average Score by Dimension")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, rotation=15)
    ax.legend()
    ax.set_ylim(0, 11)

    # --- 3. Win rate by subject ---
    ax = axes[1][0]
    subj_labels = ["Math", "Physics", "CS"]
    a_wins = []
    b_wins = []
    ties = []
    for subj in subjects:
        sr = [r for r in results if r["subject"] == subj]
        a_wins.append(sum(1 for r in sr if r["winner"] == "A"))
        b_wins.append(sum(1 for r in sr if r["winner"] == "B"))
        ties.append(sum(1 for r in sr if r["winner"] == "tie"))
    x = np.arange(len(subj_labels))
    ax.bar(x - w, a_wins, w, label="Pipeline", color=colors_a, alpha=0.85)
    ax.bar(x, b_wins, w, label="Plain Grok", color=colors_b, alpha=0.85)
    ax.bar(x + w, ties, w, label="Tie", color="#9E9E9E", alpha=0.85)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Wins")
    ax.set_title("Wins by Subject")
    ax.set_xticks(x)
    ax.set_xticklabels(subj_labels)
    ax.legend()

    # --- 4. Winner pie chart ---
    ax = axes[1][1]
    a_total = sum(1 for r in results if r["winner"] == "A")
    b_total = sum(1 for r in results if r["winner"] == "B")
    t_total = sum(1 for r in results if r["winner"] == "tie")
    sizes = [a_total, b_total, t_total]
    labels_pie = [f"Pipeline ({a_total})", f"Plain Grok ({b_total})",
                  f"Tie ({t_total})"]
    colors_pie = [colors_a, colors_b, "#9E9E9E"]
    ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct="%1.0f%%",
           startangle=90, textprops={"fontsize": 11})
    ax.set_title("Overall Win Distribution")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(BASE_DIR, "benchmark_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def run_benchmark():
    if not XAI_API_KEY or XAI_API_KEY == "your-xai-api-key-here":
        print("ERROR: XAI_API_KEY not set in api.env")
        sys.exit(1)

    client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    pipeline = Pipeline()

    print("=" * 64)
    print("  BENCHMARK: Pipeline vs Plain Grok")
    print("=" * 64)
    print(f"  Questions: {len(QUESTIONS)} (all 27 syllabus topics)")
    print()

    results = []

    for i, item in enumerate(QUESTIONS, 1):
        q = item["q"]
        print(f"[{i:2d}/{len(QUESTIONS)}] {item['topic']:25s} | {item['subject']}")

        # Pipeline
        t0 = time.time()
        pipe_out = get_pipeline_teaching(pipeline, q)
        t_pipe = time.time() - t0

        # Plain Grok
        t0 = time.time()
        grok_ans = get_plain_grok(client, q)
        t_grok = time.time() - t0

        # Judge
        try:
            j = judge(client, q, pipe_out, grok_ans)
            a_scores = j.get("A", {})
            b_scores = j.get("B", {})
            a_total = sum(a_scores.get(d, 0) for d in DIMENSIONS)
            b_total = sum(b_scores.get(d, 0) for d in DIMENSIONS)
            winner = j.get("winner", "?")
            reason = j.get("reason", "")
        except Exception as e:
            a_scores = b_scores = {}
            a_total = b_total = 0
            winner = "?"
            reason = str(e)

        print(f"         Pipeline: {a_total}/50 ({t_pipe:.0f}s)  |  "
              f"Grok: {b_total}/50 ({t_grok:.0f}s)  →  {winner}")

        results.append({
            "id": item["id"],
            "question": q,
            "subject": item["subject"],
            "topic": item["topic"],
            "scores": {"A": a_scores, "B": b_scores,
                       "A_total": a_total, "B_total": b_total},
            "winner": winner,
            "reason": reason,
            "time_pipeline": round(t_pipe, 1),
            "time_grok": round(t_grok, 1),
        })

    pipeline.close()

    # --- Summary ---
    a_wins = sum(1 for r in results if r["winner"] == "A")
    b_wins = sum(1 for r in results if r["winner"] == "B")
    ties = sum(1 for r in results if r["winner"] == "tie")
    a_avg = np.mean([r["scores"]["A_total"] for r in results])
    b_avg = np.mean([r["scores"]["B_total"] for r in results])

    print()
    print("=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print(f"  Pipeline wins:   {a_wins}/{len(results)}")
    print(f"  Plain Grok wins: {b_wins}/{len(results)}")
    print(f"  Ties:            {ties}/{len(results)}")
    print(f"  Avg Pipeline:    {a_avg:.1f}/50")
    print(f"  Avg Plain Grok:  {b_avg:.1f}/50")

    print("\n  Per-subject:")
    for subj in ["Mathematics", "Physics", "Computer Science"]:
        sr = [r for r in results if r["subject"] == subj]
        aw = sum(1 for r in sr if r["winner"] == "A")
        bw = sum(1 for r in sr if r["winner"] == "B")
        print(f"    {subj:20s}: Pipeline {aw} | Grok {bw}")

    # --- Graphs ---
    print("\n  Generating graphs...")
    graph_path = generate_graphs(results)
    print(f"  Saved: {graph_path}")

    # --- JSON ---
    json_path = os.path.join(BASE_DIR, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(results),
            "summary": {
                "pipeline_wins": a_wins,
                "grok_wins": b_wins,
                "ties": ties,
                "avg_pipeline": round(a_avg, 1),
                "avg_grok": round(b_avg, 1),
            },
            "results": results,
        }, f, indent=2)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    run_benchmark()
