"""
Main Pipeline — Student Input + D2 Classification + D1 Aggregation

Usage:
    python pipeline.py

D2 classifies each student question into:
  - MATCHES_SYLLABUS      : topic is in the syllabus (first time asked)
  - NO_MATCH_NEW          : topic is NOT in syllabus (first time asked)
  - MATCHES_SYLLABUS_REPEAT : topic is in syllabus AND student asked it before
  - NO_MATCH_REPEAT       : topic is NOT in syllabus AND student asked it before

D1 activates ONLY when D2 returns MATCHES_SYLLABUS_REPEAT and:
  - Scans session memory for all related past prompts
  - Builds a topic aggregation (covered vs missing subtopics)
  - Translates the raw prompt into structured intent
  - Outputs a compact reasoning packet for downstream systems
"""

import json
import os
import sqlite3
import sys
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CENTRAL_DB  = os.path.join(BASE_DIR, "central.db")
SESSION_DB  = os.path.join(BASE_DIR, "session_memory.db")
MODEL_NAME  = "all-MiniLM-L6-v2"


# =========================================================================
# 1. Student Input Handler
# =========================================================================
class StudentInputHandler:
    """Validates the student against central.db and stores each question
    into session_memory.db."""

    def __init__(self):
        self.central_conn = sqlite3.connect(CENTRAL_DB)
        self.central_conn.row_factory = sqlite3.Row
        self.session_conn = sqlite3.connect(SESSION_DB)
        self.session_conn.row_factory = sqlite3.Row

    # ---- student lookup ----
    def get_student(self, student_id: int) -> dict | None:
        row = self.central_conn.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_students(self) -> list[dict]:
        rows = self.central_conn.execute(
            "SELECT student_id, name, grade FROM students ORDER BY student_id"
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- intake ----
    def intake(self, student_id: int, date: str, question: str) -> dict:
        student = self.get_student(student_id)
        if student is None:
            raise ValueError(f"student_id={student_id} not found in central.db")

        self.session_conn.execute(
            """INSERT INTO session_memory
                   (session_id, role, content, metadata)
               VALUES (?, 'student', ?, ?)""",
            (
                f"session_{student_id}",
                question,
                json.dumps({"date": date, "student_id": student_id}),
            ),
        )
        self.session_conn.commit()

        row_id = self.session_conn.execute(
            "SELECT last_insert_rowid()"
        ).fetchone()[0]

        return {
            "memory_row_id": row_id,
            "student_name": student["name"],
            "student_grade": student["grade"],
            "question": question,
            "date": date,
        }

    # ---- history ----
    def get_recent_questions(self, student_id: int, limit: int = 50) -> list[dict]:
        rows = self.session_conn.execute(
            """SELECT content, metadata, created_at
               FROM session_memory
               WHERE session_id = ? AND role = 'student'
               ORDER BY id DESC LIMIT ?""",
            (f"session_{student_id}", limit),
        ).fetchall()
        results = []
        for r in rows:
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
            results.append({
                "content": r["content"],
                "date": meta.get("date", "unknown"),
                "created_at": r["created_at"],
            })
        return results

    # ---- store D2 analysis ----
    def store_d2_result(self, student_id: int, analysis: dict):
        self.session_conn.execute(
            """INSERT INTO session_memory
                   (session_id, role, content, metadata)
               VALUES (?, 'd2_analysis', ?, ?)""",
            (
                f"session_{student_id}",
                analysis["new_question"],
                json.dumps(analysis),
            ),
        )
        self.session_conn.commit()

    def store_d1_result(self, student_id: int, analysis: dict):
        self.session_conn.execute(
            """INSERT INTO session_memory
                   (session_id, role, content, metadata)
               VALUES (?, 'd1_analysis', ?, ?)""",
            (
                f"session_{student_id}",
                analysis["triggered_by_question"],
                json.dumps(analysis),
            ),
        )
        self.session_conn.commit()

    def close(self):
        self.central_conn.close()
        self.session_conn.close()


# =========================================================================
# 2. D2 — Embedding + Vector-Similarity Topic Classifier
# =========================================================================
class D2Classifier:
    """Loads syllabus from central.db + question history from session_memory.db,
    then classifies each new question by cosine similarity on embeddings."""

    SIMILARITY_THRESHOLD = 0.65   # above = "matches"

    def __init__(self):
        print(f"[D2] Loading embedding model '{MODEL_NAME}' ...")
        self.model = SentenceTransformer(MODEL_NAME)
        print("[D2] Model loaded.\n")

        self.central_conn = sqlite3.connect(CENTRAL_DB)
        self.central_conn.row_factory = sqlite3.Row
        self.session_conn = sqlite3.connect(SESSION_DB)
        self.session_conn.row_factory = sqlite3.Row

        # will be populated by load_context()
        self._syllabus_texts: list[str] = []
        self._syllabus_ids: list[int] = []
        self._syllabus_embs: np.ndarray | None = None

        self._past_texts: list[str] = []
        self._past_dates: list[str] = []
        self._past_embs: np.ndarray | None = None

    # ---- context loaders ----
    def load_syllabus(self):
        rows = self.central_conn.execute(
            """SELECT t.topic_id,
                      s.subject_name || ' > ' || c.chapter_name || ' > ' || t.topic_name
                          AS full_path
               FROM topics t
               JOIN chapters c ON t.chapter_id = c.chapter_id
               JOIN subjects s ON c.subject_id = s.subject_id
               ORDER BY t.topic_id"""
        ).fetchall()
        self._syllabus_ids   = [r["topic_id"] for r in rows]
        self._syllabus_texts = [r["full_path"]  for r in rows]
        self._syllabus_embs  = self._encode(self._syllabus_texts)
        print(f"[D2] Loaded {len(self._syllabus_texts)} syllabus topics.")

    def load_history(self, student_id: int):
        rows = self.session_conn.execute(
            """SELECT content, metadata
               FROM session_memory
               WHERE session_id = ? AND role = 'student'
               ORDER BY id""",
            (f"session_{student_id}",),
        ).fetchall()
        self._past_texts = [r["content"] for r in rows]
        self._past_dates = [
            (json.loads(r["metadata"]).get("date", "?") if r["metadata"] else "?")
            for r in rows
        ]
        if self._past_texts:
            self._past_embs = self._encode(self._past_texts)
            print(f"[D2] Loaded {len(self._past_texts)} past questions from session history.")
        else:
            self._past_embs = None
            print("[D2] No past questions in session history (first question).")

    def load_context(self, student_id: int):
        self.load_syllabus()
        self.load_history(student_id)

    # ---- encoding ----
    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_norm, b_norm.T)

    # ---- classify ----
    def classify(self, question: str) -> dict:
        q_emb = self._encode([question])[0]

        # --- syllabus match ---
        sims_syllabus = self._cosine(q_emb.reshape(1, -1), self._syllabus_embs)[0]
        best_idx  = int(np.argmax(sims_syllabus))
        best_score = float(sims_syllabus[best_idx])
        matches_syllabus = best_score >= self.SIMILARITY_THRESHOLD

        # --- history (repeat) match ---
        is_repeat = False
        repeat_score = 0.0
        repeat_date = None
        if self._past_embs is not None and len(self._past_texts) > 0:
            sims_past = self._cosine(q_emb.reshape(1, -1), self._past_embs)[0]
            best_past_idx  = int(np.argmax(sims_past))
            repeat_score   = float(sims_past[best_past_idx])
            is_repeat      = repeat_score >= self.SIMILARITY_THRESHOLD
            repeat_date    = self._past_dates[best_past_idx]

        # --- combined classification ---
        if matches_syllabus and is_repeat:
            classification = "MATCHES_SYLLABUS_REPEAT"
            detail = ("Topic IS in the syllabus AND the student asked about "
                      "a very similar topic before.")
        elif matches_syllabus and not is_repeat:
            classification = "MATCHES_SYLLABUS"
            detail = "Topic IS in the syllabus. First time the student asks about it."
        elif not matches_syllabus and is_repeat:
            classification = "NO_MATCH_REPEAT"
            detail = ("Topic is NOT in the syllabus, but the student asked about "
                      "something similar before.")
        else:
            classification = "NO_MATCH_NEW"
            detail = ("Topic is NOT in the syllabus and is brand new — "
                      "student hasn't asked about it before.")

        return {
            "new_question":     question,
            "classification":   classification,
            "detail":           detail,
            # syllabus
            "syllabus_match":   matches_syllabus,
            "syllabus_score":   round(best_score, 4),
            "syllabus_topic":   self._syllabus_texts[best_idx],
            "syllabus_topic_id": self._syllabus_ids[best_idx],
            # repeat
            "is_repeat":        is_repeat,
            "repeat_score":     round(repeat_score, 4),
            "repeat_matched_q": self._past_texts[int(np.argmax(
                                    self._cosine(q_emb.reshape(1,-1),
                                                 self._past_embs)[0]))]
                                if self._past_embs is not None else None,
            "repeat_date":      repeat_date,
        }

    def close(self):
        self.central_conn.close()
        self.session_conn.close()


# =========================================================================
# 3. D1 — Repeated-Topic Aggregation & Prompt Translation Layer
# =========================================================================
class D1Aggregator:
    """Activates ONLY when D2 returns MATCHES_SYLLABUS_REPEAT.
    Scans session memory, aggregates topic coverage, translates the
    prompt into structured intent, and produces a reasoning packet."""

    def __init__(self, model, syllabus_texts=None, syllabus_ids=None,
                 syllabus_embs=None):
        self.model = model
        self.central_conn = sqlite3.connect(CENTRAL_DB)
        self.central_conn.row_factory = sqlite3.Row
        self.session_conn = sqlite3.connect(SESSION_DB)
        self.session_conn.row_factory = sqlite3.Row
        self._syllabus_texts = syllabus_texts or []
        self._syllabus_ids   = syllabus_ids or []
        self._syllabus_embs  = syllabus_embs

    def set_syllabus(self, texts, ids, embs):
        self._syllabus_texts = texts
        self._syllabus_ids   = ids
        self._syllabus_embs  = embs

    # ---- topic hierarchy ----
    def _build_topic_hierarchy(self, topic_id: int) -> dict:
        row = self.central_conn.execute(
            """SELECT s.subject_id, s.subject_name,
                      c.chapter_id, c.chapter_name,
                      t.topic_id, t.topic_name
               FROM topics t
               JOIN chapters c ON t.chapter_id = c.chapter_id
               JOIN subjects s ON c.subject_id = s.subject_id
               WHERE t.topic_id = ?""", (topic_id,)
        ).fetchone()
        if row is None:
            return {}

        subject_name = row["subject_name"]
        chapter_name = row["chapter_name"]

        chapters = self.central_conn.execute(
            """SELECT c.chapter_id, c.chapter_name
               FROM chapters c
               JOIN subjects s ON c.subject_id = s.subject_id
               WHERE s.subject_name = ?
               ORDER BY c.chapter_id""", (subject_name,)
        ).fetchall()

        chaps = []
        for ch in chapters:
            topics = self.central_conn.execute(
                """SELECT topic_id, topic_name
                   FROM topics WHERE chapter_id = ?
                   ORDER BY topic_id""", (ch["chapter_id"],)
            ).fetchall()
            chaps.append({
                "chapter_id":   ch["chapter_id"],
                "chapter_name": ch["chapter_name"],
                "topics": [{"topic_id": t["topic_id"],
                             "topic_name": t["topic_name"]}
                            for t in topics],
            })

        return {
            "subject_id":   row["subject_id"],
            "subject_name": subject_name,
            "triggered_chapter": chapter_name,
            "triggered_topic":   row["topic_name"],
            "chapters": chaps,
        }

    # ---- prompt translation ----
    def _translate_prompt(self, question: str) -> dict:
        q_emb = self.model.encode([question], convert_to_numpy=True,
                                  show_progress_bar=False)[0]
        sims = self._cosine_sim(q_emb.reshape(1, -1), self._syllabus_embs)[0]
        best = int(np.argmax(sims))
        parts = self._syllabus_texts[best].split(" > ")

        q_lower = question.lower()
        if any(w in q_lower for w in ["what", "define", "explain"]):
            intent = "Conceptual explanation"
        elif any(w in q_lower for w in ["how", "solve", "find"]):
            intent = "Problem-solving"
        elif any(w in q_lower for w in ["why", "reason"]):
            intent = "Causal reasoning"
        elif any(w in q_lower for w in ["example", "show"]):
            intent = "Example request"
        else:
            intent = "General query"

        if any(w in q_lower for w in ["derive", "prove", "advanced"]):
            difficulty = "Advanced"
        elif any(w in q_lower for w in ["basic", "simple", "introduction"]):
            difficulty = "Beginner"
        else:
            difficulty = "Intermediate"

        return {
            "domain":     parts[0] if len(parts) > 0 else "Unknown",
            "chapter":    parts[1] if len(parts) > 1 else "Unknown",
            "topic":      parts[2] if len(parts) > 2 else "Unknown",
            "intent":     intent,
            "difficulty": difficulty,
            "confidence": round(float(sims[best]), 4),
        }

    # ---- aggregation ----
    def aggregate(self, student_id: int, d2_result: dict) -> dict:
        question  = d2_result["new_question"]
        topic_id  = d2_result["syllabus_topic_id"]

        topic_hierarchy = self._build_topic_hierarchy(topic_id)
        subject_name    = topic_hierarchy.get("subject_name", "Unknown")
        all_topics      = [t["topic_name"]
                           for ch in topic_hierarchy.get("chapters", [])
                           for t in ch["topics"]]

        # get all past student questions from session memory
        rows = self.session_conn.execute(
            """SELECT content, metadata FROM session_memory
               WHERE session_id = ? AND role = 'student'
               ORDER BY id""", (f"session_{student_id}",)
        ).fetchall()
        past_questions = [r["content"] for r in rows]

        # map each past question to its closest subtopic via embeddings
        if past_questions and all_topics:
            q_embs  = self.model.encode(past_questions, convert_to_numpy=True,
                                        show_progress_bar=False)
            t_embs  = self.model.encode(all_topics, convert_to_numpy=True,
                                        show_progress_bar=False)
            q_sims  = self._cosine_sim(q_embs, t_embs)  # (n_questions, n_topics)

            matched_subtopics = set()
            for qi in range(len(past_questions)):
                best_topic_idx = int(np.argmax(q_sims[qi]))
                if q_sims[qi][best_topic_idx] >= 0.4:
                    matched_subtopics.add(all_topics[best_topic_idx])
        else:
            matched_subtopics = set()

        covered = sorted(matched_subtopics)
        missing = sorted(set(all_topics) - matched_subtopics)
        coverage_pct = (len(covered) / len(all_topics) * 100) if all_topics else 0

        # cross-subject related topics
        cross_subject_related = []
        if past_questions:
            other_rows = self.central_conn.execute(
                """SELECT s.subject_name FROM subjects s
                   WHERE s.subject_name != ?
                   ORDER BY s.subject_name""", (subject_name,)
            ).fetchall()
            other_subjects = [r["subject_name"] for r in other_rows]
            if other_subjects:
                s_embs = self.model.encode(other_subjects, convert_to_numpy=True,
                                           show_progress_bar=False)
                q_embs_all = self.model.encode(past_questions, convert_to_numpy=True,
                                               show_progress_bar=False)
                all_sims = self._cosine_sim(q_embs_all, s_embs)
                for si, sname in enumerate(other_subjects):
                    if float(np.max(all_sims[:, si])) >= 0.35:
                        cross_subject_related.append(sname)

        translation = self._translate_prompt(question)

        return {
            "triggered_by_question": question,
            "d2_classification":     d2_result["classification"],
            "syllabus_topic":        d2_result["syllabus_topic"],
            "syllabus_topic_id":     topic_id,
            "detected_parent_topic": subject_name,
            "subtopic_detected":     topic_hierarchy.get("triggered_topic", ""),
            "previous_context_found": True,
            "repeated_prompt_score": d2_result["repeat_score"],
            "recommended_context_merge": coverage_pct < 80,
            "translation":     translation,
            "coverage":        covered,
            "missing_topics":  missing,
            "coverage_pct":    round(coverage_pct, 1),
            "related_topics":  cross_subject_related,
        }

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_n, b_n.T)

    def close(self):
        self.central_conn.close()
        self.session_conn.close()


# =========================================================================
# 4. Pipeline — orchestrates input → D2 → (D1) → output
# =========================================================================
class Pipeline:
    def __init__(self):
        self.handler    = StudentInputHandler()
        self.classifier = D2Classifier()
        self.aggregator = D1Aggregator(model=self.classifier.model)

    # ---- display ----
    @staticmethod
    def _banner(text: str, char: str = "="):
        width = 60
        print(f"\n{char * width}")
        print(f"  {text}")
        print(f"{char * width}")

    @staticmethod
    def _section(title: str):
        print(f"\n--- {title} {'─' * (50 - len(title))}")

    def show_students(self):
        self._banner("REGISTERED STUDENTS")
        for s in self.handler.list_students():
            print(f"  [{s['student_id']}] {s['name']}  (grade {s['grade']})")

    def show_result(self, intake_info: dict, analysis: dict):
        # ---- Input received ----
        self._banner("INPUT STORED TO SESSION MEMORY DB")
        print(f"  Student  : {intake_info['student_name']} (grade {intake_info['student_grade']})")
        print(f"  Date     : {intake_info['date']}")
        print(f"  Question : \"{intake_info['question']}\"")
        print(f"  Row ID   : {intake_info['memory_row_id']}")

        # ---- D2 analysis ----
        self._banner("D2 CLASSIFICATION RESULT")

        # classification label
        label = analysis["classification"]
        print(f"\n  >>> {label}")

        # Q1: matches syllabus?
        self._section("Q1: Matches existing syllabus data?")
        if analysis["syllabus_match"]:
            print(f"      YES  (similarity={analysis['syllabus_score']:.4f}  "
                  f"threshold={D2Classifier.SIMILARITY_THRESHOLD})")
            print(f"      Best match: {analysis['syllabus_topic']}")
            print(f"      Topic ID  : {analysis['syllabus_topic_id']}")
        else:
            print(f"      NO   (similarity={analysis['syllabus_score']:.4f}  "
                  f"threshold={D2Classifier.SIMILARITY_THRESHOLD})")
            print(f"      Closest syllabus topic: {analysis['syllabus_topic']}")

        # Q2: repeated?
        self._section("Q2: Is the user repeating a previous topic?")
        if analysis["repeat_matched_q"] is None:
            print("      N/A  (no past questions in session history)")
        elif analysis["is_repeat"]:
            print(f"      YES  (similarity={analysis['repeat_score']:.4f}  "
                  f"threshold={D2Classifier.SIMILARITY_THRESHOLD})")
            print(f"      Matched previous question: \"{analysis['repeat_matched_q']}\"")
            print(f"      That question was from:    {analysis['repeat_date']}")
        else:
            print(f"      NO   (similarity={analysis['repeat_score']:.4f}  "
                  f"threshold={D2Classifier.SIMILARITY_THRESHOLD})")

        # Q3: combined verdict
        self._section("Q3: Combined Verdict")
        print(f"      {analysis['detail']}")

    def show_d1_result(self, d1: dict):
        self._banner("D1 TOPIC AGGREGATION (activated: repeated syllabus topic)")

        # ---- Prompt Translation ----
        tr = d1["translation"]
        self._section("Prompt Translation")
        print(f"      Domain     : {tr['domain']}")
        print(f"      Chapter    : {tr['chapter']}")
        print(f"      Topic      : {tr['topic']}")
        print(f"      Intent     : {tr['intent']}")
        print(f"      Difficulty : {tr['difficulty']}")

        # ---- Topic Aggregation ----
        self._section("Topic Aggregation")
        print(f"      Parent Topic  : {d1['detected_parent_topic']}")
        print(f"      Subtopic Hit  : {d1['subtopic_detected']}")
        print(f"      Context Found : {d1['previous_context_found']}")
        print(f"      Repeat Score  : {d1['repeated_prompt_score']}")
        print(f"      Merge Recmd.  : {d1['recommended_context_merge']}")

        # ---- Coverage ----
        bar_len = 30
        filled  = int(bar_len * d1["coverage_pct"] / 100)
        bar     = "█" * filled + "░" * (bar_len - filled)
        self._section(f"Coverage: {d1['coverage_pct']}%  [{bar}]")

        if d1["coverage"]:
            print("      Covered subtopics:")
            for s in d1["coverage"]:
                print(f"        ✓ {s}")

        if d1["missing_topics"]:
            print("      Missing subtopics:")
            for s in d1["missing_topics"]:
                print(f"        ✗ {s}")

        # ---- Related cross-subject ----
        if d1["related_topics"]:
            self._section("Related Cross-Subject Topics")
            for t in d1["related_topics"]:
                print(f"      ↔ {t}")

        # ---- Structured JSON ----
        self._section("Structured Reasoning Packet (JSON)")
        packet = {
            "main_topic":            d1["detected_parent_topic"],
            "subtopic_detected":     d1["subtopic_detected"],
            "coverage":              d1["coverage"],
            "missing_topics":        d1["missing_topics"],
            "coverage_pct":          d1["coverage_pct"],
            "repeated_prompt_score": d1["repeated_prompt_score"],
            "recommended_context_merge": d1["recommended_context_merge"],
        }
        print(json.dumps(packet, indent=6))

    # ---- run one question ----
    def run_question(self, student_id: int, date: str, question: str):
        # 1. load history BEFORE intake so D2 compares against
        #    past questions only — not the one we're about to store
        self.classifier.load_history(student_id)

        # 2. intake → session_memory.db
        intake_info = self.handler.intake(student_id, date, question)

        # 3. D2 classify
        d2_result = self.classifier.classify(question)

        # 4. store D2 result back to session memory
        self.handler.store_d2_result(student_id, d2_result)

        # 5. display D2
        self.show_result(intake_info, d2_result)

        # 6. D1 — activate ONLY when D2 says repeated + in syllabus
        d1_result = None
        if d2_result["classification"] == "MATCHES_SYLLABUS_REPEAT":
            # sync syllabus data to D1 (may have been reloaded)
            self.aggregator.set_syllabus(
                self.classifier._syllabus_texts,
                self.classifier._syllabus_ids,
                self.classifier._syllabus_embs,
            )
            d1_result = self.aggregator.aggregate(student_id, d2_result)
            self.handler.store_d1_result(student_id, d1_result)
            self.show_d1_result(d1_result)

        return {"d2": d2_result, "d1": d1_result}

    # ---- CLI loop ----
    def run(self):
        self._banner("AI AGENT PIPELINE — MOCK ENV")
        self.show_students()

        while True:
            print()
            raw = input("Enter student_id (or 'quit'): ").strip()
            if raw.lower() in ("quit", "exit", "q"):
                break
            try:
                student_id = int(raw)
            except ValueError:
                print("Please enter a valid integer student_id.")
                continue

            student = self.handler.get_student(student_id)
            if student is None:
                print(f"student_id={student_id} not found. Try again.")
                continue

            date = input("Date (YYYY-MM-DD) [Enter=today]: ").strip()
            if not date:
                date = datetime.now().strftime("%Y-%m-%d")

            question = input("Question: ").strip()
            if not question:
                print("Empty question, skipping.")
                continue

            # load syllabus + history for this student
            self.classifier.load_context(student_id)

            # run pipeline
            self.run_question(student_id, date, question)

    def close(self):
        self.handler.close()
        self.classifier.close()
        self.aggregator.close()


# =========================================================================
if __name__ == "__main__":
    pipeline = Pipeline()
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pipeline.close()
