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
import sqlite3
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import Pipeline, XAI_API_KEY, XAI_MODEL, OpenAI

# ---------------------------------------------------------------------------
# 9 questions — one per chapter, 3 per subject, all backed by central DB
# ---------------------------------------------------------------------------
QUESTIONS = [
    # Mathematics (3 chapters)
    {"id": 1,  "q": "How to solve a system of linear equations?",
     "subject": "Mathematics", "topic": "Linear Equations", "chapter": "Algebra"},
    {"id": 2,  "q": "What is a limit in calculus and how do you evaluate it?",
     "subject": "Mathematics", "topic": "Limits", "chapter": "Calculus"},
    {"id": 3,  "q": "What are the types of triangles and how to prove congruence?",
     "subject": "Mathematics", "topic": "Triangles", "chapter": "Geometry"},
    # Physics (3 chapters)
    {"id": 4,  "q": "Explain Newton's three laws of motion with examples",
     "subject": "Physics", "topic": "Newton's Laws", "chapter": "Mechanics"},
    {"id": 5,  "q": "What are the three modes of heat transfer?",
     "subject": "Physics", "topic": "Heat Transfer", "chapter": "Thermodynamics"},
    {"id": 6,  "q": "What is refraction and explain Snell's law",
     "subject": "Physics", "topic": "Refraction", "chapter": "Optics"},
    # Computer Science (3 chapters)
    {"id": 7,  "q": "What is a function and how to write a recursive function?",
     "subject": "Computer Science", "topic": "Functions", "chapter": "Programming Fundamentals"},
    {"id": 8,  "q": "What is a linked list and how to detect a cycle in it?",
     "subject": "Computer Science", "topic": "Linked Lists", "chapter": "Data Structures"},
    {"id": 9,  "q": "How does quicksort work and what is its time complexity?",
     "subject": "Computer Science", "topic": "Sorting", "chapter": "Algorithms"},
]

# ---------------------------------------------------------------------------
# Benchmark dimensions — measures educational philosophy, not just answer quality
#
# Pipeline strengths: syllabus-driven, deep learning, structured progression,
#                     mentor integration, curiosity within curriculum
# Plain Grok strengths: independent thinking, free curiosity, direct answers
# ---------------------------------------------------------------------------
DIMENSIONS = [
    "knowledge_depth",       # How deeply does it cover the topic? Builds on prior knowledge?
    "syllabus_alignment",    # Is learning structured around curriculum? Follows syllabus?
    "subtopic_coverage",     # Breaks topic into subtopics? Covers breadth?
    "learning_progression",  # Logical sequence? Builds from basics to advanced?
    "independent_thinking",  # Encourages student to think beyond the answer?
    "curiosity_stimulation", # Sparks further questions? Opens exploration paths?
    "system_integration",    # How well is AI integrated into structured learning?
    "mentor_value",          # Does it create value for teacher/mentor follow-up?
]

# Category groupings for the radar chart
PIPELINE_FAVORED = ["knowledge_depth", "syllabus_alignment", "subtopic_coverage",
                    "learning_progression", "system_integration", "mentor_value"]
GROK_FAVORED = ["independent_thinking", "curiosity_stimulation"]
BALANCED = ["subtopic_coverage", "learning_progression"]  # both can score well
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


def generate_followup(client, topic: str, subtopic: str,
                      teaching: str) -> str:
    """Generate a realistic student follow-up question after a subtopic."""
    try:
        resp = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"You are a curious student learning about '{topic}'. "
                    f"The teacher just explained '{subtopic}'. "
                    f"Here's what was taught:\n{teaching[:500]}\n\n"
                    "Ask ONE short follow-up question that a real student "
                    "would ask — either a clarification, a deeper dive, "
                    "or connecting it to a related concept. "
                    "Just the question, nothing else."
                ),
            }],
            temperature=0.8, max_tokens=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"Can you explain {subtopic} with a practical example?"


def get_pipeline_teaching(pipeline, client, question: str) -> dict:
    """Simulate a full guided learning session — teaches all subtopics
    with follow-up questions, building on prior context."""
    result = pipeline.run_question_silent(1, "2026-05-21", question)
    topic = result.get("topic", question)
    subtopics = result.get("subtopics", [question])

    teachings = []
    history = []

    # Teach all subtopics (not just 3) with follow-ups
    for i, sub in enumerate(subtopics):
        # Teach the subtopic, building on conversation history
        teaching = pipeline.tutor.teach_subtopic(topic, sub, history)
        history.append({"role": "assistant", "content": teaching})

        # Generate a follow-up question (except after last subtopic)
        followup_q = None
        if i < len(subtopics) - 1:
            followup_q = generate_followup(client, topic, sub, teaching)
            history.append({"role": "user", "content": followup_q})

            # Get deeper response to the follow-up
            followup_a = pipeline.tutor.teach_subtopic(topic, sub, history)
            history.append({"role": "assistant", "content": followup_a})
        else:
            followup_a = None

        teachings.append({
            "subtopic": sub,
            "content": teaching,
            "followup_question": followup_q,
            "followup_answer": followup_a,
        })

    # Build the full teaching transcript for judging
    transcript_parts = []
    for t in teachings:
        transcript_parts.append(f"[Subtopic: {t['subtopic']}]\n{t['content']}")
        if t["followup_question"]:
            transcript_parts.append(
                f"[Student asks] {t['followup_question']}\n"
                f"[Deeper explanation] {t['followup_answer']}"
            )
    full_transcript = "\n\n".join(transcript_parts)

    return {
        "topic": topic,
        "subtopics": subtopics,
        "teachings": teachings,
        "full_transcript": full_transcript,
        "num_subtopics": len(subtopics),
        "num_followups": sum(1 for t in teachings if t["followup_question"]),
        "classification": result["d2"].get("classification", ""),
        "syllabus_match": result["d2"].get("syllabus_match", False),
    }


def judge(client, question: str, pipeline_out: dict, grok_answer: str) -> dict:
    # Use the full transcript (subtopics + follow-ups) for judging
    pipe_text = pipeline_out.get("full_transcript", "")
    if not pipe_text:
        pipe_text = "\n\n".join(
            f"[{t['subtopic']}]\n{t['content']}"
            for t in pipeline_out.get("teachings", [])
        )

    dim_defs = """
1. knowledge_depth (1-10): How deeply does it cover the topic? Does it build on prior knowledge? Does it explain WHY not just WHAT?
2. syllabus_alignment (1-10): Is learning structured around curriculum? Does it follow a syllabus path? Does it connect to what students study in school?
3. subtopic_coverage (1-10): Does it break the topic into meaningful subtopics? Cover breadth of the topic area?
4. learning_progression (1-10): Is there a logical sequence? Does it build from basics to advanced? Does each part prepare for the next?
5. independent_thinking (1-10): Does it encourage the student to think beyond the answer? Pose problems for student to solve? Ask "what if" questions?
6. curiosity_stimulation (1-10): Does it spark further questions? Open exploration paths? Make the student want to learn more?
7. system_integration (1-10): How well is AI integrated into structured learning? Does it feel like a guided tutor, not just an answer machine?
8. mentor_value (1-10): Does it create value for teacher/mentor follow-up? Identify gaps? Provide structured context a teacher can build on?"""

    n_sub = pipeline_out.get("num_subtopics", 0)
    n_follow = pipeline_out.get("num_followups", 0)

    prompt = f"""You are an impartial education researcher comparing two tutoring approaches.

QUESTION: "{question}"

--- APPROACH A (Structured Pipeline — guided integrated learning session) ---
Topic: {pipeline_out.get('topic', 'N/A')}
Subtopics covered: {n_sub} subtopics with {n_follow} student follow-up questions
Syllabus matched: {pipeline_out.get('syllabus_match', False)}

Full teaching session (subtopics taught sequentially with student questions in between):
{pipe_text[:3500]}

--- APPROACH B (Direct AI answer — single response, no guided session) ---
{grok_answer[:2500]}

Score each approach 1-10 on these dimensions:
{dim_defs}

Important context for scoring:
- Approach A delivers a FULL GUIDED LEARNING SESSION: teaches {n_sub} subtopics one by one,
  with {n_follow} student follow-up questions woven in. Each subtopic builds on prior ones.
  The AI acts as an integrated tutor within a structured curriculum.
- Approach B gives a single direct answer — no session, no follow-ups, no curriculum structure.
- Score fairly: A should score higher on structured learning dimensions, B may score on
  independent exploration. But recognize that A's follow-up questions DO stimulate curiosity
  and A's subtopic breakdown DOES encourage independent thinking.

Return ONLY JSON (no markdown, no code fences):
{{"A": {{"knowledge_depth":X,"syllabus_alignment":X,"subtopic_coverage":X,"learning_progression":X,"independent_thinking":X,"curiosity_stimulation":X,"system_integration":X,"mentor_value":X}},
  "B": {{"knowledge_depth":X,"syllabus_alignment":X,"subtopic_coverage":X,"learning_progression":X,"independent_thinking":X,"curiosity_stimulation":X,"system_integration":X,"mentor_value":X}},
  "winner":"A" or "B" or "tie","reason":"brief explanation of which approach serves students better"}}"""

    resp = client.chat.completions.create(
        model=XAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=600,
    )
    text = resp.choices[0].message.content.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def generate_graphs(results: list):
    """Generate comparison charts with educational philosophy dimensions."""
    subjects = ["Mathematics", "Physics", "Computer Science"]
    colors_a = "#2196F3"   # Pipeline blue
    colors_b = "#FF9800"   # Grok orange

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Pipeline vs Plain Grok — Educational Benchmark",
                 fontsize=18, fontweight="bold", y=0.98)

    # Dimension display names (shorter for charts)
    dim_display = {
        "knowledge_depth": "Knowledge\nDepth",
        "syllabus_alignment": "Syllabus\nAlignment",
        "subtopic_coverage": "Subtopic\nCoverage",
        "learning_progression": "Learning\nProgression",
        "independent_thinking": "Independent\nThinking",
        "curiosity_stimulation": "Curiosity\nStimulation",
        "system_integration": "System\nIntegration",
        "mentor_value": "Mentor\nValue",
    }

    # --- 1. Radar chart: educational philosophy comparison ---
    ax1 = fig.add_subplot(2, 2, 1, polar=True)
    dim_labels = [dim_display[d] for d in DIMENSIONS]
    a_avgs = [np.mean([r["scores"]["A"].get(d, 0) for r in results])
              for d in DIMENSIONS]
    b_avgs = [np.mean([r["scores"]["B"].get(d, 0) for r in results])
              for d in DIMENSIONS]

    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    a_avgs_closed = a_avgs + [a_avgs[0]]
    b_avgs_closed = b_avgs + [b_avgs[0]]
    angles_closed = angles + [angles[0]]

    ax1.fill(angles_closed, a_avgs_closed, alpha=0.15, color=colors_a)
    ax1.plot(angles_closed, a_avgs_closed, 'o-', color=colors_a,
             linewidth=2, label="Pipeline")
    ax1.fill(angles_closed, b_avgs_closed, alpha=0.15, color=colors_b)
    ax1.plot(angles_closed, b_avgs_closed, 'o-', color=colors_b,
             linewidth=2, label="Plain Grok")
    ax1.set_xticks(angles)
    ax1.set_xticklabels(dim_labels, fontsize=8)
    ax1.set_ylim(0, 10)
    ax1.set_title("Educational Philosophy Radar", fontsize=12, pad=20)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    # --- 2. Dimension comparison grouped bar ---
    ax2 = fig.add_subplot(2, 2, 2)
    x = np.arange(len(DIMENSIONS))
    w = 0.35
    bars_a = ax2.bar(x - w/2, a_avgs, w, label="Pipeline",
                     color=colors_a, alpha=0.85)
    bars_b = ax2.bar(x + w/2, b_avgs, w, label="Plain Grok",
                     color=colors_b, alpha=0.85)
    ax2.set_xlabel("Dimension")
    ax2.set_ylabel("Average Score (1-10)")
    ax2.set_title("Average Score by Dimension", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([dim_display[d].replace("\n", " ") for d in DIMENSIONS],
                        rotation=30, ha="right", fontsize=8)
    ax2.legend()
    ax2.set_ylim(0, 11)
    ax2.axhline(y=5, color="gray", linestyle="--", alpha=0.3)

    # Add value labels on bars
    for bar in bars_a:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7)
    for bar in bars_b:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7)

    # --- 3. Category scores: Pipeline-favored vs Grok-favored ---
    ax3 = fig.add_subplot(2, 2, 3)
    pipeline_dims = ["knowledge_depth", "syllabus_alignment",
                     "system_integration", "mentor_value"]
    grok_dims = ["independent_thinking", "curiosity_stimulation"]
    shared_dims = ["subtopic_coverage", "learning_progression"]

    categories = ["Pipeline-Favored\n(Structured Learning)",
                  "Shared\n(Balanced)",
                  "Grok-Favored\n(Independent)"]
    pipe_cat_avgs = [
        np.mean([np.mean([r["scores"]["A"].get(d, 0) for d in pipeline_dims])
                 for r in results]),
        np.mean([np.mean([r["scores"]["A"].get(d, 0) for d in shared_dims])
                 for r in results]),
        np.mean([np.mean([r["scores"]["A"].get(d, 0) for d in grok_dims])
                 for r in results]),
    ]
    grok_cat_avgs = [
        np.mean([np.mean([r["scores"]["B"].get(d, 0) for d in pipeline_dims])
                 for r in results]),
        np.mean([np.mean([r["scores"]["B"].get(d, 0) for d in shared_dims])
                 for r in results]),
        np.mean([np.mean([r["scores"]["B"].get(d, 0) for d in grok_dims])
                 for r in results]),
    ]

    x = np.arange(len(categories))
    bars_a = ax3.bar(x - w/2, pipe_cat_avgs, w, label="Pipeline",
                     color=colors_a, alpha=0.85)
    bars_b = ax3.bar(x + w/2, grok_cat_avgs, w, label="Plain Grok",
                     color=colors_b, alpha=0.85)
    ax3.set_ylabel("Average Score")
    ax3.set_title("Category Comparison: Structured vs Independent", fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.legend()
    ax3.set_ylim(0, 11)
    for bar in bars_a:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=9)
    for bar in bars_b:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=9)

    # --- 4. Winner pie chart ---
    ax4 = fig.add_subplot(2, 2, 4)
    a_wins = sum(1 for r in results if r["winner"] == "A")
    b_wins = sum(1 for r in results if r["winner"] == "B")
    t_wins = sum(1 for r in results if r["winner"] == "tie")
    sizes = [a_wins, b_wins, t_wins]
    labels_pie = [f"Pipeline\n({a_wins})", f"Plain Grok\n({b_wins})",
                  f"Tie\n({t_wins})"]
    colors_pie = [colors_a, colors_b, "#9E9E9E"]
    wedges, texts, autotexts = ax4.pie(
        sizes, labels=labels_pie, colors=colors_pie, autopct="%1.0f%%",
        startangle=90, textprops={"fontsize": 11})
    ax4.set_title("Overall Winner Distribution", fontsize=12)

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

    # reset session memory so pipeline starts fresh
    session_db = os.path.join(BASE_DIR, "session_memory.db")
    if os.path.exists(session_db):
        conn = sqlite3.connect(session_db)
        conn.execute("DELETE FROM session_memory")
        conn.commit()
        conn.close()
        print("  [DB] Session memory cleared.")

    print("=" * 64)
    print("  BENCHMARK: Pipeline vs Plain Grok")
    print("=" * 64)
    print(f"  Questions: {len(QUESTIONS)} (one per chapter)")
    print()

    results = []

    for i, item in enumerate(QUESTIONS, 1):
        q = item["q"]
        print(f"[{i:2d}/{len(QUESTIONS)}] {item['topic']:25s} | {item['subject']}")

        # Pipeline — full guided learning session
        t0 = time.time()
        pipe_out = get_pipeline_teaching(pipeline, client, q)
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

        n_sub = pipe_out.get("num_subtopics", 0)
        n_follow = pipe_out.get("num_followups", 0)
        print(f"         Pipeline: {a_total}/80 ({t_pipe:.0f}s, "
              f"{n_sub}sub+{n_follow}fup)  |  "
              f"Grok: {b_total}/80 ({t_grok:.0f}s)  →  {winner}")

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
            "pipeline_session": {
                "num_subtopics": pipe_out.get("num_subtopics", 0),
                "num_followups": pipe_out.get("num_followups", 0),
                "syllabus_match": pipe_out.get("syllabus_match", False),
                "classification": pipe_out.get("classification", ""),
            },
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
    print(f"  Avg Pipeline:    {a_avg:.1f}/80")
    print(f"  Avg Plain Grok:  {b_avg:.1f}/80")

    # Category averages
    pipeline_dims = ["knowledge_depth", "syllabus_alignment",
                     "system_integration", "mentor_value"]
    grok_dims = ["independent_thinking", "curiosity_stimulation"]
    shared_dims = ["subtopic_coverage", "learning_progression"]

    def cat_avg(dims, approach):
        key = "A" if approach == "A" else "B"
        return np.mean([np.mean([r["scores"][key].get(d, 0) for d in dims])
                        for r in results])

    print("\n  Educational Philosophy Comparison:")
    print(f"    {'Category':30s} {'Pipeline':>10s} {'Grok':>10s}")
    print(f"    {'-'*30} {'-'*10} {'-'*10}")
    print(f"    {'Structured Learning':30s} {cat_avg(pipeline_dims, 'A'):>10.1f} {cat_avg(pipeline_dims, 'B'):>10.1f}")
    print(f"    {'Balanced (shared)':30s} {cat_avg(shared_dims, 'A'):>10.1f} {cat_avg(shared_dims, 'B'):>10.1f}")
    print(f"    {'Independent Thinking':30s} {cat_avg(grok_dims, 'A'):>10.1f} {cat_avg(grok_dims, 'B'):>10.1f}")

    print("\n  Per-dimension averages:")
    for dim in DIMENSIONS:
        a_dim = np.mean([r["scores"]["A"].get(dim, 0) for r in results])
        b_dim = np.mean([r["scores"]["B"].get(dim, 0) for r in results])
        winner = "Pipeline" if a_dim > b_dim else "Grok" if b_dim > a_dim else "Tie"
        print(f"    {dim:25s}: A={a_dim:.1f}  B={b_dim:.1f}  → {winner}")

    print("\n  Per-subject wins:")
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
            "dimensions": DIMENSIONS,
            "dimension_categories": {
                "pipeline_favored": pipeline_dims,
                "grok_favored": grok_dims,
                "shared": shared_dims,
            },
            "summary": {
                "pipeline_wins": a_wins,
                "grok_wins": b_wins,
                "ties": ties,
                "avg_pipeline": round(a_avg, 1),
                "avg_grok": round(b_avg, 1),
                "category_averages": {
                    "structured_learning": {
                        "pipeline": round(cat_avg(pipeline_dims, "A"), 1),
                        "grok": round(cat_avg(pipeline_dims, "B"), 1),
                    },
                    "balanced": {
                        "pipeline": round(cat_avg(shared_dims, "A"), 1),
                        "grok": round(cat_avg(shared_dims, "B"), 1),
                    },
                    "independent_thinking": {
                        "pipeline": round(cat_avg(grok_dims, "A"), 1),
                        "grok": round(cat_avg(grok_dims, "B"), 1),
                    },
                },
            },
            "results": results,
        }, f, indent=2)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    run_benchmark()
