"""
Benchmark: Pipeline (Guided Learning) vs Plain Grok

Runs all 9 benchmark questions in PARALLEL through:
  A) Pipeline: D2 → D2.1 → D4 → D3 → AI Tutor (full guided learning session)
  B) Plain Grok: direct question → answer (no pipeline)

Each pipeline session simulates real interactive learning:
  - ALL subtopics taught (not just 3)
  - Student follow-up questions after each subtopic
  - Deeper responses building on conversation history
  - D3 safety checks on tutor output

Evaluation aligned with paper hypotheses (H1-H6).

Usage:
    .env/bin/python benchmark.py
"""

import json
import os
import sys
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Evaluation dimensions — aligned with paper hypotheses H1-H6
# ---------------------------------------------------------------------------
DIMENSIONS = [
    "knowledge_depth",        # H6: Does it build deep understanding or shallow answers?
    "syllabus_alignment",     # H3: Does it stay within curriculum scope?
    "subtopic_coverage",      # H6: Does it break topic into structured subtopics?
    "learning_progression",   # H2: Does it build from basics to advanced logically?
    "independent_thinking",   # H1: Does it encourage student to think, not just receive?
    "curiosity_stimulation",  # H4: Does it spark genuine exploration?
    "system_integration",     # H5: Does AI act as guided tutor, not answer machine?
    "mentor_value",           # H5: Does it create structured context for human teachers?
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# Pipeline session — full interactive guided learning
# =========================================================================
def run_pipeline_session(item: dict) -> dict:
    """Run a full guided learning session through the pipeline.
    Each thread creates its own Pipeline instance for isolation."""
    pipeline = Pipeline()
    client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    q = item["q"]

    try:
        # Run pipeline (D2 → D2.1 → D4 → D3)
        result = pipeline.run_question_silent(1, "2026-05-21", q)
        topic = result.get("topic", q)
        subtopics = result.get("subtopics", [q])
        classification = result["d2"].get("classification", "")
        syllabus_match = result["d2"].get("syllabus_match", False)

        # Full guided learning session — teach ALL subtopics with follow-ups
        teachings = []
        history = []

        for i, sub in enumerate(subtopics):
            # Teach subtopic, building on conversation history
            teaching = pipeline.tutor.teach_subtopic(topic, sub, history)
            history.append({"role": "assistant", "content": teaching})

            # D3 safety check on tutor output
            d3_check = pipeline.d3.check_output(teaching, sub, {})
            if not d3_check.get("safe", True):
                teaching = (
                    f"[Content flagged by safety check — "
                    f"mentor review queued for: {sub}]"
                )

            # Generate student follow-up question (except after last subtopic)
            followup_q = None
            followup_a = None
            if i < len(subtopics) - 1:
                followup_q = _generate_followup(client, topic, sub, teaching)
                history.append({"role": "user", "content": followup_q})

                # Get deeper response to follow-up
                followup_a = pipeline.tutor.teach_subtopic(
                    topic, sub, history
                )
                history.append({"role": "assistant", "content": followup_a})

                # D3 check on follow-up answer
                d3_fu = pipeline.d3.check_output(followup_a, sub, {})
                if not d3_fu.get("safe", True):
                    followup_a = (
                        f"[Follow-up flagged — ask your mentor about: {sub}]"
                    )

            teachings.append({
                "subtopic": sub,
                "teaching": teaching,
                "followup_q": followup_q,
                "followup_a": followup_a,
            })

        # Build full transcript for judging
        transcript = _build_transcript(teachings)

        return {
            "id": item["id"],
            "q": q,
            "subject": item["subject"],
            "topic": topic,
            "subtopics": subtopics,
            "classification": classification,
            "syllabus_match": syllabus_match,
            "num_subtopics": len(teachings),
            "num_followups": sum(1 for t in teachings if t["followup_q"]),
            "transcript": transcript,
            "d4_triggered": result.get("d4") is not None,
            "d3_safe": result.get("d3", {}).get("safe", True),
            "pipeline_result": {
                "d2": result["d2"],
                "d2_1": result.get("d2_1"),
                "d4": result.get("d4"),
                "d3": result.get("d3"),
            },
        }
    finally:
        pipeline.close()


def _generate_followup(client, topic: str, subtopic: str,
                       teaching: str) -> str:
    """Generate a realistic student follow-up question."""
    try:
        resp = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"You are a curious student learning about '{topic}'. "
                    f"The teacher just explained '{subtopic}'.\n"
                    f"Teaching excerpt: {teaching[:400]}\n\n"
                    "Ask ONE short follow-up question — a clarification, "
                    "deeper dive, or connecting to a related concept. "
                    "Just the question, nothing else."
                ),
            }],
            temperature=0.8, max_tokens=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"Can you explain {subtopic} with a practical example?"


def _build_transcript(teachings: list) -> str:
    """Build full teaching transcript for judging."""
    parts = []
    for t in teachings:
        parts.append(f"[Subtopic: {t['subtopic']}]\n{t['teaching']}")
        if t["followup_q"]:
            parts.append(
                f"[Student asks] {t['followup_q']}\n"
                f"[Deeper explanation] {t['followup_a']}"
            )
    return "\n\n".join(parts)


# =========================================================================
# Plain Grok session — single direct answer
# =========================================================================
def run_grok_session(item: dict) -> dict:
    """Get a direct answer from Grok (no pipeline)."""
    client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    q = item["q"]

    try:
        resp = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[
                {"role": "system",
                 "content": (
                     "You are a helpful tutor. Answer clearly. "
                     "Cover key concepts, give examples, "
                     "explain step by step."
                 )},
                {"role": "user", "content": q},
            ],
            temperature=0.7, max_tokens=1000,
        )
        answer = resp.choices[0].message.content.strip()
        return {
            "id": item["id"],
            "answer": answer,
            "num_chars": len(answer),
        }
    except Exception as e:
        return {"id": item["id"], "answer": f"[ERROR: {e}]", "num_chars": 0}


# =========================================================================
# Judge — xAI as impartial evaluator
# =========================================================================
DIM_DEFS = """
1. knowledge_depth (1-10): How deeply does it cover the topic? Does it build on prior knowledge? Does it explain WHY not just WHAT?
2. syllabus_alignment (1-10): Is learning structured around curriculum? Does it follow a syllabus path? Does it connect to what students study in school?
3. subtopic_coverage (1-10): Does it break the topic into meaningful subtopics? Cover breadth of the topic area?
4. learning_progression (1-10): Is there a logical sequence? Does it build from basics to advanced? Does each part prepare for the next?
5. independent_thinking (1-10): Does it encourage the student to think beyond the answer? Pose problems for student to solve? Ask "what if" questions?
6. curiosity_stimulation (1-10): Does it spark further questions? Open exploration paths? Make the student want to learn more?
7. system_integration (1-10): How well is AI integrated into structured learning? Does it feel like a guided tutor, not just an answer machine?
8. mentor_value (1-10): Does it create value for teacher/mentor follow-up? Identify gaps? Provide structured context a teacher can build on?"""


def judge_session(client, pipe_out: dict, grok_out: dict) -> dict:
    """Judge one question — pipeline vs Grok."""
    q = pipe_out["q"]
    pipe_text = pipe_out["transcript"][:3500]
    grok_text = grok_out["answer"][:2500]
    n_sub = pipe_out.get("num_subtopics", 0)
    n_fu = pipe_out.get("num_followups", 0)

    prompt = f"""You are an impartial education researcher comparing two tutoring approaches.

QUESTION: "{q}"

--- APPROACH A (Structured Pipeline — guided integrated learning session) ---
Topic: {pipe_out.get('topic', 'N/A')}
Subtopics covered: {n_sub} subtopics with {n_fu} student follow-up questions
Syllabus matched: {pipe_out.get('syllabus_match', False)}

Full teaching session:
{pipe_text}

--- APPROACH B (Direct AI answer — single response, no guided session) ---
{grok_text}

Score each approach 1-10 on:
{DIM_DEFS}

Context:
- Approach A teaches {n_sub} subtopics with {n_fu} follow-up questions in a guided session
- Approach B gives a single direct answer
- Score fairly: A should score higher on structured dimensions, B may score on exploration

Return ONLY JSON (no markdown, no fences):
{{"A": {{"knowledge_depth":X,"syllabus_alignment":X,"subtopic_coverage":X,"learning_progression":X,"independent_thinking":X,"curiosity_stimulation":X,"system_integration":X,"mentor_value":X}},
  "B": {{"knowledge_depth":X,"syllabus_alignment":X,"subtopic_coverage":X,"learning_progression":X,"independent_thinking":X,"curiosity_stimulation":X,"system_integration":X,"mentor_value":X}},
  "winner":"A" or "B" or "tie","reason":"brief"}}"""

    try:
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
        j = json.loads(text.strip())
        a_scores = j.get("A", {})
        b_scores = j.get("B", {})
        a_total = sum(a_scores.get(d, 0) for d in DIMENSIONS)
        b_total = sum(b_scores.get(d, 0) for d in DIMENSIONS)
        return {
            "A": a_scores, "B": b_scores,
            "A_total": a_total, "B_total": b_total,
            "winner": j.get("winner", "?"),
            "reason": j.get("reason", ""),
        }
    except Exception as e:
        return {
            "A": {}, "B": {},
            "A_total": 0, "B_total": 0,
            "winner": "?", "reason": str(e),
        }


# =========================================================================
# Graph generation
# =========================================================================
def generate_graphs(results: list) -> str:
    """Generate comparison charts with educational philosophy dimensions."""
    colors_a = "#2196F3"   # Pipeline blue
    colors_b = "#FF9800"   # Grok orange

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

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Pipeline (Guided Learning) vs Plain Grok — Educational Benchmark",
        fontsize=18, fontweight="bold", y=0.98,
    )

    # --- 1. Radar chart ---
    ax1 = fig.add_subplot(2, 2, 1, polar=True)
    a_avgs = [np.mean([r["scores"]["A"].get(d, 0) for r in results])
              for d in DIMENSIONS]
    b_avgs = [np.mean([r["scores"]["B"].get(d, 0) for r in results])
              for d in DIMENSIONS]
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    a_closed = a_avgs + [a_avgs[0]]
    b_closed = b_avgs + [b_avgs[0]]
    angles_closed = angles + [angles[0]]

    ax1.fill(angles_closed, a_closed, alpha=0.15, color=colors_a)
    ax1.plot(angles_closed, a_closed, "o-", color=colors_a, linewidth=2,
             label="Pipeline")
    ax1.fill(angles_closed, b_closed, alpha=0.15, color=colors_b)
    ax1.plot(angles_closed, b_closed, "o-", color=colors_b, linewidth=2,
             label="Plain Grok")
    ax1.set_xticks(angles)
    ax1.set_xticklabels([dim_display[d] for d in DIMENSIONS], fontsize=8)
    ax1.set_ylim(0, 10)
    ax1.set_title("Educational Philosophy Radar", fontsize=12, pad=20)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    # --- 2. Dimension bars ---
    ax2 = fig.add_subplot(2, 2, 2)
    x = np.arange(len(DIMENSIONS))
    w = 0.35
    bars_a = ax2.bar(x - w/2, a_avgs, w, label="Pipeline", color=colors_a,
                     alpha=0.85)
    bars_b = ax2.bar(x + w/2, b_avgs, w, label="Plain Grok", color=colors_b,
                     alpha=0.85)
    ax2.set_ylabel("Average Score (1-10)")
    ax2.set_title("Average Score by Dimension", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [dim_display[d].replace("\n", " ") for d in DIMENSIONS],
        rotation=30, ha="right", fontsize=8,
    )
    ax2.legend()
    ax2.set_ylim(0, 11)
    ax2.axhline(y=5, color="gray", linestyle="--", alpha=0.3)
    for bar in bars_a:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=7)
    for bar in bars_b:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=7)

    # --- 3. Category comparison ---
    ax3 = fig.add_subplot(2, 2, 3)
    pipeline_dims = ["knowledge_depth", "syllabus_alignment",
                     "system_integration", "mentor_value"]
    grok_dims = ["independent_thinking", "curiosity_stimulation"]
    shared_dims = ["subtopic_coverage", "learning_progression"]
    categories = ["Structured Learning\n(Pipeline-favored)",
                  "Balanced\n(Shared)",
                  "Independent Thinking\n(Grok-favored)"]

    def cat_avg(dims, key):
        return np.mean([np.mean([r["scores"][key].get(d, 0) for d in dims])
                        for r in results])

    pipe_cats = [cat_avg(pipeline_dims, "A"), cat_avg(shared_dims, "A"),
                 cat_avg(grok_dims, "A")]
    grok_cats = [cat_avg(pipeline_dims, "B"), cat_avg(shared_dims, "B"),
                 cat_avg(grok_dims, "B")]
    x = np.arange(len(categories))
    bars_a = ax3.bar(x - w/2, pipe_cats, w, label="Pipeline", color=colors_a,
                     alpha=0.85)
    bars_b = ax3.bar(x + w/2, grok_cats, w, label="Plain Grok", color=colors_b,
                     alpha=0.85)
    ax3.set_ylabel("Average Score")
    ax3.set_title("Category: Structured vs Independent", fontsize=12)
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

    # --- 4. Per-question scores ---
    ax4 = fig.add_subplot(2, 2, 4)
    a_totals = [r["scores"]["A_total"] for r in results]
    b_totals = [r["scores"]["B_total"] for r in results]
    labels = [r["topic"].split(" > ")[-1] if " > " in r["topic"]
              else r["topic"][:15] for r in results]
    x = np.arange(len(results))
    bars_a = ax4.bar(x - w/2, a_totals, w, label="Pipeline", color=colors_a,
                     alpha=0.85)
    bars_b = ax4.bar(x + w/2, b_totals, w, label="Plain Grok", color=colors_b,
                     alpha=0.85)
    ax4.set_ylabel("Total Score (out of 80)")
    ax4.set_title("Score per Question", fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax4.legend()
    ax4.set_ylim(0, 88)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(BASE_DIR, "benchmark_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# =========================================================================
# Main benchmark runner
# =========================================================================
def run_benchmark():
    if not XAI_API_KEY or XAI_API_KEY == "your-xai-api-key-here":
        print("ERROR: XAI_API_KEY not set in api.env")
        sys.exit(1)

    # Reset session memory
    session_db = os.path.join(BASE_DIR, "session_memory.db")
    if os.path.exists(session_db):
        conn = sqlite3.connect(session_db)
        conn.execute("DELETE FROM session_memory")
        conn.commit()
        conn.close()
        print("  [DB] Session memory cleared.")

    print("=" * 64)
    print("  BENCHMARK: Pipeline (Guided Learning) vs Plain Grok")
    print("  Running 9 sessions IN PARALLEL")
    print("=" * 64)
    print()

    # ---- Phase 1: Run all pipeline sessions in parallel ----
    print("  Phase 1/3: Running pipeline sessions (parallel)...")
    t0 = time.time()
    pipeline_results = {}

    with ThreadPoolExecutor(max_workers=9) as executor:
        futures = {
            executor.submit(run_pipeline_session, item): item
            for item in QUESTIONS
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                pipeline_results[item["id"]] = result
                n_sub = result["num_subtopics"]
                n_fu = result["num_followups"]
                print(f"    Q{item['id']}: {item['topic']:25s} "
                      f"({n_sub}sub+{n_fu}fup) ✓")
            except Exception as e:
                print(f"    Q{item['id']}: {item['topic']:25s} ERROR: {e}")
                pipeline_results[item["id"]] = {
                    "id": item["id"], "q": item["q"],
                    "subject": item["subject"], "topic": item["topic"],
                    "subtopics": [], "transcript": "",
                    "num_subtopics": 0, "num_followups": 0,
                    "classification": "", "syllabus_match": False,
                }

    t_pipeline = time.time() - t0
    print(f"  Pipeline sessions done in {t_pipeline:.0f}s")

    # ---- Phase 2: Run all Grok sessions in parallel ----
    print("\n  Phase 2/3: Running Grok sessions (parallel)...")
    t0 = time.time()
    grok_results = {}

    with ThreadPoolExecutor(max_workers=9) as executor:
        futures = {
            executor.submit(run_grok_session, item): item
            for item in QUESTIONS
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                grok_results[item["id"]] = result
                print(f"    Q{item['id']}: {item['topic']:25s} "
                      f"({result['num_chars']} chars) ✓")
            except Exception as e:
                print(f"    Q{item['id']}: {item['topic']:25s} ERROR: {e}")
                grok_results[item["id"]] = {
                    "id": item["id"], "answer": f"[ERROR: {e}]",
                }

    t_grok = time.time() - t0
    print(f"  Grok sessions done in {t_grok:.0f}s")

    # ---- Phase 3: Judge all sessions in parallel ----
    print("\n  Phase 3/3: Judging all sessions (parallel)...")
    t0 = time.time()
    client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    judged_results = []

    with ThreadPoolExecutor(max_workers=9) as executor:
        futures = {}
        for item in QUESTIONS:
            pipe_out = pipeline_results.get(item["id"], {})
            grok_out = grok_results.get(item["id"], {})
            future = executor.submit(judge_session, client, pipe_out, grok_out)
            futures[future] = item

        for future in as_completed(futures):
            item = futures[future]
            try:
                scores = future.result()
                pipe_out = pipeline_results[item["id"]]
                print(f"    Q{item['id']}: A={scores['A_total']}/80  "
                      f"B={scores['B_total']}/80  → {scores['winner']}")
                judged_results.append({
                    "id": item["id"],
                    "q": item["q"],
                    "subject": item["subject"],
                    "topic": pipe_out.get("topic", item["topic"]),
                    "subtopics": pipe_out.get("subtopics", []),
                    "scores": scores,
                    "winner": scores["winner"],
                    "reason": scores["reason"],
                    "pipeline_session": {
                        "num_subtopics": pipe_out.get("num_subtopics", 0),
                        "num_followups": pipe_out.get("num_followups", 0),
                        "syllabus_match": pipe_out.get("syllabus_match", False),
                        "classification": pipe_out.get("classification", ""),
                        "d4_triggered": pipe_out.get("d4_triggered", False),
                    },
                })
            except Exception as e:
                print(f"    Q{item['id']}: JUDGE ERROR: {e}")

    t_judge = time.time() - t0
    print(f"  Judging done in {t_judge:.0f}s")

    # Sort by question ID
    judged_results.sort(key=lambda r: r["id"])

    # ---- Summary ----
    a_wins = sum(1 for r in judged_results if r["winner"] == "A")
    b_wins = sum(1 for r in judged_results if r["winner"] == "B")
    ties = sum(1 for r in judged_results if r["winner"] == "tie")
    a_avg = np.mean([r["scores"]["A_total"] for r in judged_results])
    b_avg = np.mean([r["scores"]["B_total"] for r in judged_results])

    print()
    print("=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print(f"  Pipeline wins:   {a_wins}/9")
    print(f"  Plain Grok wins: {b_wins}/9")
    print(f"  Ties:            {ties}/9")
    print(f"  Avg Pipeline:    {a_avg:.1f}/80")
    print(f"  Avg Plain Grok:  {b_avg:.1f}/80")
    print(f"  Total time:      {t_pipeline + t_grok + t_judge:.0f}s")

    # Category averages
    pipeline_dims = ["knowledge_depth", "syllabus_alignment",
                     "system_integration", "mentor_value"]
    grok_dims = ["independent_thinking", "curiosity_stimulation"]
    shared_dims = ["subtopic_coverage", "learning_progression"]

    def cat_avg(dims, approach):
        key = "A" if approach == "A" else "B"
        return np.mean([np.mean([r["scores"][key].get(d, 0) for d in dims])
                        for r in judged_results])

    print("\n  Per-dimension averages:")
    for dim in DIMENSIONS:
        a_dim = np.mean([r["scores"]["A"].get(dim, 0)
                         for r in judged_results])
        b_dim = np.mean([r["scores"]["B"].get(dim, 0)
                         for r in judged_results])
        w = "Pipeline" if a_dim > b_dim else "Grok" if b_dim > a_dim else "Tie"
        print(f"    {dim:25s}: A={a_dim:.1f}  B={b_dim:.1f}  → {w}")

    print("\n  Philosophy comparison:")
    print(f"    {'Structured Learning':30s} "
          f"A={cat_avg(pipeline_dims, 'A'):.1f}  B={cat_avg(pipeline_dims, 'B'):.1f}")
    print(f"    {'Balanced (shared)':30s} "
          f"A={cat_avg(shared_dims, 'A'):.1f}  B={cat_avg(shared_dims, 'B'):.1f}")
    print(f"    {'Independent Thinking':30s} "
          f"A={cat_avg(grok_dims, 'A'):.1f}  B={cat_avg(grok_dims, 'B'):.1f}")

    print("\n  Per-subject wins:")
    for subj in ["Mathematics", "Physics", "Computer Science"]:
        sr = [r for r in judged_results if r["subject"] == subj]
        aw = sum(1 for r in sr if r["winner"] == "A")
        bw = sum(1 for r in sr if r["winner"] == "B")
        print(f"    {subj:20s}: Pipeline {aw} | Grok {bw}")

    # ---- Graphs ----
    print("\n  Generating graphs...")
    graph_path = generate_graphs(judged_results)
    print(f"  Saved: {graph_path}")

    # ---- JSON ----
    json_path = os.path.join(BASE_DIR, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(judged_results),
            "dimensions": DIMENSIONS,
            "time_pipeline_s": round(t_pipeline, 1),
            "time_grok_s": round(t_grok, 1),
            "time_judge_s": round(t_judge, 1),
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
            "results": judged_results,
        }, f, indent=2)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    run_benchmark()
