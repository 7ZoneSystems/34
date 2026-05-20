"""
Main Pipeline — Student Input → D2 → D2.1 → D4 / D1 → Mentor Router

Usage:
    python pipeline.py

Pipeline flow:
  1. StudentInputHandler  — validates student, stores question to session_memory.db
  2. D2Classifier         — classifies question (syllabus match? repeated topic?)
  3. D2_1NoveltyHandler   — (if D2 says "No") web-searches topic, re-matches syllabus,
                            or redirects to D4
  4. D4NoveltyRedirect    — (if D2.1 says redirect) curiosity detection, novel query
                            classification, mock web search, guide mode
  5. D1Aggregator         — (if repeated+syllabus) aggregates topic coverage + gaps
  6. MentorRouter         — (if syllabus match) routes to relevant mentor(s) via
                            mock_mentor.py CLI, stores mentor responses
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import requests
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CENTRAL_DB  = os.path.join(BASE_DIR, "central.db")
SESSION_DB  = os.path.join(BASE_DIR, "session_memory.db")
MODEL_NAME  = "all-MiniLM-L6-v2"

# load API keys from api.env
load_dotenv(os.path.join(BASE_DIR, "api.env"))

# ---------------------------------------------------------------------------
# D4 — Groq API key & mock web search data
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROK_MODEL   = os.getenv("GROK_MODEL", "llama-3.3-70b-versatile")

MOCK_WEB_RESULTS = """
Quantum entanglement is a phenomenon in quantum physics where two or more
particles become correlated in such a way that the quantum state of each
particle cannot be described independently. When particles are entangled,
measuring the state of one particle instantly determines the state of the
other, regardless of the distance between them.

Key concepts related to quantum entanglement:
- Bell's Theorem and Bell inequalities - prove entanglement is real
- EPR Paradox - Einstein's objection to "spooky action at a distance"
- Quantum superposition - particles exist in multiple states simultaneously
- Quantum decoherence - how entangled states break down
- Quantum teleportation - using entanglement to transfer quantum information
- Quantum computing applications - entanglement as a computational resource
- No-communication theorem - entanglement cannot transmit information faster than light

Related physics topics:
- Wave-particle duality
- Heisenberg uncertainty principle
- Schrodinger's cat thought experiment
- Quantum field theory
- Quantum cryptography and quantum key distribution
"""


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

    def store_d2_1_result(self, student_id: int, analysis: dict):
        self.session_conn.execute(
            """INSERT INTO session_memory
                   (session_id, role, content, metadata)
               VALUES (?, 'd2_1_analysis', ?, ?)""",
            (
                f"session_{student_id}",
                analysis.get("topic", ""),
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

    def store_d4_result(self, student_id: int, analysis: dict):
        self.session_conn.execute(
            """INSERT INTO session_memory
                   (session_id, role, content, metadata)
               VALUES (?, 'd4_analysis', ?, ?)""",
            (
                f"session_{student_id}",
                analysis.get("topic", ""),
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
# 2.1  D2.1 — Novelty Handler (web search + syllabus re-match)
# =========================================================================
class D2_1NoveltyHandler:
    """Activates when D2 returns NO_MATCH_NEW or NO_MATCH_REPEAT.
    Searches the web for the unmatched topic, extracts subtopics,
    and vector-matches them against the Central DB syllabus.
    If a match is found → returns structured output for the student.
    If no match → calls reach_mentor_endpoint() → D4."""

    SIMILARITY_THRESHOLD = 0.55

    def __init__(self, model):
        self.model = model
        self.central_conn = sqlite3.connect(CENTRAL_DB)
        self.central_conn.row_factory = sqlite3.Row
        self.session_conn = sqlite3.connect(SESSION_DB)
        self.session_conn.row_factory = sqlite3.Row

        self._syllabus_texts: list[str] = []
        self._syllabus_ids: list[int] = []
        self._syllabus_embs: np.ndarray | None = None

    # ---- encoding helpers ----
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_n, b_n.T)

    # ---- web search ----
    def web_search(self, query: str, max_results: int = 8) -> list[dict]:
        """Search the web via DuckDuckGo Instant Answer API."""
        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"[D2.1] Web search failed: {e}")
            return []

        results = []
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", query),
                "snippet": data["AbstractText"],
                "source": data.get("AbstractSource", "DuckDuckGo"),
            })
        for item in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(item, dict) and "Text" in item:
                results.append({
                    "title": item.get("Text", "")[:100],
                    "snippet": item.get("Text", ""),
                    "source": item.get("FirstURL", "DuckDuckGo"),
                })
        return results

    # ---- subtopic extraction ----
    def extract_subtopics(self, search_results: list[dict]) -> list[str]:
        """Parse web search results into a list of sub-topic strings."""
        subtopics = []
        for r in search_results:
            text = r.get("title", "") + " " + r.get("snippet", "")
            parts = re.split(r"[,;|\-\–\—]", text)
            for part in parts:
                clean = part.strip().strip(".")
                if len(clean) > 5 and clean not in subtopics:
                    subtopics.append(clean)
        return subtopics[:20]

    # ---- syllabus loader ----
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
        self._syllabus_ids = [r["topic_id"] for r in rows]
        self._syllabus_texts = [r["full_path"] for r in rows]
        self._syllabus_embs = self.model.encode(
            self._syllabus_texts, convert_to_numpy=True, show_progress_bar=False
        )

    # ---- syllabus matching ----
    def match_to_syllabus(self, topic: str, subtopics: list[str]) -> dict:
        """Vector-match web-discovered subtopics against central syllabus."""
        if self._syllabus_embs is None:
            self.load_syllabus()

        candidates = [topic] + subtopics
        c_embs = self.model.encode(candidates, convert_to_numpy=True, show_progress_bar=False)
        sims = self._cosine(c_embs, self._syllabus_embs)

        best_score = 0.0
        best_syllabus_idx = 0
        for ci in range(len(candidates)):
            row_max_idx = int(np.argmax(sims[ci]))
            row_max_score = float(sims[ci][row_max_idx])
            if row_max_score > best_score:
                best_score = row_max_score
                best_syllabus_idx = row_max_idx

        matched = best_score >= self.SIMILARITY_THRESHOLD

        ranked = []
        for si in range(len(self._syllabus_texts)):
            score = float(np.max(sims[:, si]))
            ranked.append({
                "syllabus_topic": self._syllabus_texts[si],
                "score": round(score, 4),
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)

        return {
            "matched": matched,
            "best_score": round(best_score, 4),
            "best_syllabus_idx": best_syllabus_idx,
            "ranked": ranked[:5],
        }

    # ---- D4 redirect ----
    def reach_mentor_endpoint(self, student_id: int, topic: str,
                              d2_result: dict) -> dict:
        """No syllabus match — route to mentor endpoint (D4)."""
        if self._syllabus_embs is None:
            self.load_syllabus()

        t_emb = self.model.encode([topic], convert_to_numpy=True,
                                  show_progress_bar=False)
        sims = self._cosine(t_emb, self._syllabus_embs)[0]
        closest_idx = int(np.argmax(sims))
        closest_path = self._syllabus_texts[closest_idx]
        closest_subject = closest_path.split(" > ")[0]

        mentors = self.central_conn.execute(
            """SELECT DISTINCT m.mentor_id, m.mentor_name
               FROM mentors m
               JOIN mentor_subjects ms ON m.mentor_id = ms.mentor_id
               JOIN subjects s ON ms.subject_id = s.subject_id
               WHERE s.subject_name = ?
               ORDER BY m.mentor_id""", (closest_subject,)
        ).fetchall()

        mentor_list = [{"id": m["mentor_id"], "name": m["mentor_name"]}
                       for m in mentors]

        self.session_conn.execute(
            """INSERT INTO session_memory
                   (session_id, role, content, metadata)
               VALUES (?, 'd2_1_redirect', ?, ?)""",
            (
                f"session_{student_id}",
                topic,
                json.dumps({
                    "action": "redirect_to_mentor",
                    "closest_subject": closest_subject,
                    "closest_topic": closest_path,
                    "mentors": mentor_list,
                    "original_d2": d2_result["classification"],
                }),
            ),
        )
        self.session_conn.commit()

        return {
            "status": "redirect_to_mentor",
            "topic": topic,
            "reason": "not_in_syllabus",
            "closest_subject": closest_subject,
            "closest_syllabus_topic": closest_path,
            "mentors_available": mentor_list,
        }

    # ---- main entry point ----
    def run_d2_1(self, student_id: int, input_topic: str,
                 d2_result: dict) -> dict:
        """Full D2.1 pipeline: web search → extract → match → output or D4."""
        print(f"\n[D2.1] Novelty handler activated for: \"{input_topic}\"")

        # 1. web search
        print("[D2.1] Searching the web ...")
        search_results = self.web_search(input_topic)
        print(f"[D2.1] Got {len(search_results)} search results.")

        # 2. extract subtopics
        subtopics = self.extract_subtopics(search_results)
        print(f"[D2.1] Extracted {len(subtopics)} candidate subtopics.")

        # 3. vector match against syllabus
        print("[D2.1] Matching against Central DB syllabus ...")
        match_result = self.match_to_syllabus(input_topic, subtopics)
        print(f"[D2.1] Best similarity = {match_result['best_score']:.4f}  "
              f"(threshold={self.SIMILARITY_THRESHOLD})")

        # 4. branch: syllabus match or redirect to mentor (D4)
        if match_result["matched"]:
            best = match_result["ranked"][0]
            confirmed_topics = [
                r["syllabus_topic"] for r in match_result["ranked"]
                if r["score"] >= self.SIMILARITY_THRESHOLD
            ]
            output = {
                "status": "syllabus_match",
                "topic": best["syllabus_topic"],
                "subtopics": confirmed_topics,
                "source": "web_search",
                "confirmed": True,
                "match_score": best["score"],
                "web_subtopics": subtopics,
                "all_ranked": match_result["ranked"],
            }
            print(f"[D2.1] MATCH FOUND -> {best['syllabus_topic']} "
                  f"(score={best['score']:.4f})")
        else:
            output = self.reach_mentor_endpoint(student_id, input_topic,
                                                d2_result)
            print(f"[D2.1] No syllabus match -> redirecting to mentor endpoint (D4)")

        return output

    def close(self):
        self.central_conn.close()
        self.session_conn.close()


# =========================================================================
# 4. D4 — Novelty Redirection & Classification Layer
# =========================================================================
class D4NoveltyRedirect:
    """Activates when D2.1 returns redirect_to_mentor.
    Checks for curiosity pattern, classifies novel queries,
    and triggers mock web search for genuinely novel topics."""

    def __init__(self, session_conn: sqlite3.Connection):
        self.session_conn = session_conn
        if not GROQ_API_KEY or GROQ_API_KEY == "your-groq-api-key-here":
            print("[D4] WARNING: GROQ_API_KEY not set in api.env — "
                  "D4 will use fallback logic (no LLM calls)")
            self.groq_client = None
        else:
            self.groq_client = Groq(api_key=GROQ_API_KEY)
        # tracks questions per student for curiosity pattern detection
        self._history: dict[int, list[str]] = {}

    # ---- curiosity pattern detection ----
    def _load_recent_questions(self, student_id: int, limit: int = 10):
        rows = self.session_conn.execute(
            """SELECT content FROM session_memory
               WHERE session_id = ? AND role = 'student'
               ORDER BY id DESC LIMIT ?""",
            (f"session_{student_id}", limit),
        ).fetchall()
        self._history[student_id] = [r["content"] for r in rows]

    def check_curiosity_pattern(self, student_id: int,
                                current_question: str) -> bool:
        """Use Groq to detect if the student is repeatedly exploring
        a new direction (curiosity pattern)."""
        self._load_recent_questions(student_id)
        past = self._history.get(student_id, [])

        if len(past) < 3:
            return False

        past_block = "\n".join(f"- {q}" for q in past[:10])
        prompt = (
            "You are an educational AI analyzing a student's question history.\n"
            "The student has been asking questions that fall outside the "
            "standard syllabus.\n\n"
            f"Recent questions:\n{past_block}\n\n"
            f"Current question: \"{current_question}\"\n\n"
            "Is the student repeatedly exploring a coherent new direction "
            "or topic out of genuine curiosity? "
            "Answer ONLY 'yes' or 'no'."
        )

        if not self.groq_client:
            # fallback: detect curiosity by keyword overlap with past questions
            print("[D4] Using fallback curiosity detection (no API key)")
            return len(past) >= 3

        try:
            resp = self.groq_client.chat.completions.create(
                model=GROK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=3,
            )
            answer = resp.choices[0].message.content.strip().lower()
            return answer.startswith("yes")
        except Exception as e:
            print(f"[D4] Groq curiosity check failed: {e}")
            return False

    # ---- novel query classification ----
    def classify_novel_query(self, question: str,
                             d2_result: dict) -> str:
        """Use Groq to classify the novel query as genuinely_novel
        or off_topic."""
        topic = d2_result.get("new_question", question)
        classification = d2_result.get("classification", "NO_MATCH_NEW")

        prompt = (
            "You are an educational AI classifier.\n"
            "A student asked a question that does not match the school "
            "syllabus.\n\n"
            f"Question: \"{question}\"\n"
            f"D2 classification: {classification}\n\n"
            "Classify this query:\n"
            "- 'genuinely_novel' if it is a sincere academic or intellectual "
            "question that could enrich the student's learning\n"
            "- 'off_topic' if it is irrelevant, casual, or not constructive\n\n"
            "Answer ONLY with the tag: genuinely_novel or off_topic."
        )

        if not self.groq_client:
            # fallback: assume genuinely novel
            print("[D4] Using fallback classification (no API key)")
            return "genuinely_novel"

        try:
            resp = self.groq_client.chat.completions.create(
                model=GROK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
            )
            tag = resp.choices[0].message.content.strip().lower()
            if "off_topic" in tag:
                return "off_topic"
            return "genuinely_novel"
        except Exception as e:
            print(f"[D4] Groq classification failed: {e}")
            return "genuinely_novel"

    # ---- mock web search ----
    def web_search(self, query: str) -> str:
        """Mock web search — prints loading message, waits, returns
        MOCK_WEB_RESULTS."""
        print("[D4] Searching the web...")
        time.sleep(2)
        print("[D4] Web search complete.")
        return MOCK_WEB_RESULTS.strip()

    # ---- guide mode ----
    def enter_guide_mode(self, student_id: int,
                         question: str) -> dict:
        """Validate ideas and enter guide mode — triggers web search
        and returns structured output."""
        print("[D4] Curiosity pattern detected — entering Guide Mode")
        print("[D4] Validating ideas and searching for enrichment material...")
        search_data = self.web_search(question)

        return {
            "d4_status": "guide_mode",
            "topic": question,
            "classification_tag": "curiosity_pattern",
            "result": (
                "It looks like you're developing a genuine interest in this "
                "area! Here's what I found to help you explore further:\n\n"
                f"{search_data}"
            ),
        }

    # ---- main entry point ----
    def run_d4(self, student_id: int, question: str,
               d2_result: dict) -> dict:
        """Full D4 pipeline: curiosity check → classify → search/redirect."""
        print(f"\n[D4] Novelty redirect activated for: \"{question}\"")

        # 1. curiosity pattern check
        is_curious = self.check_curiosity_pattern(student_id, question)

        if is_curious:
            # 2a. validate ideas → guide mode → web search
            output = self.enter_guide_mode(student_id, question)
        else:
            # 2b. classify novel query
            tag = self.classify_novel_query(question, d2_result)
            print(f"[D4] Classification tag: {tag}")

            if tag == "genuinely_novel":
                # trigger web search
                search_data = self.web_search(question)
                output = {
                    "d4_status": "novel_output",
                    "topic": question,
                    "classification_tag": "genuinely_novel",
                    "result": search_data,
                }
            else:
                # off_topic — polite redirect, stop
                output = {
                    "d4_status": "off_topic_redirect",
                    "topic": question,
                    "classification_tag": "off_topic",
                    "result": (
                        "That's an interesting question, but it seems to be "
                        "outside the scope of your current studies. Let's "
                        "focus on your syllabus topics for now — feel free "
                        "to ask your mentor if you'd like to explore this "
                        "further after your coursework!"
                    ),
                }

        return output

    def close(self):
        pass


# =========================================================================
# 5. Mentor Router — finds mentors for a subject & calls mock_mentor.py
# =========================================================================
class MentorRouter:
    """Given a subject from D2's matched topic, finds available mentors
    in central.db and invokes mock_mentor.py for each one."""

    MENTOR_SCRIPT = os.path.join(BASE_DIR, "mock_mentor.py")

    def __init__(self):
        self.central_conn = sqlite3.connect(CENTRAL_DB)
        self.central_conn.row_factory = sqlite3.Row
        self.session_conn = sqlite3.connect(SESSION_DB)
        self.session_conn.row_factory = sqlite3.Row

    # ---- lookup ----
    def get_mentors_for_subject(self, subject_name: str) -> list[dict]:
        rows = self.central_conn.execute(
            """SELECT DISTINCT m.mentor_id, m.mentor_name, m.bio
               FROM mentors m
               JOIN mentor_subjects ms ON m.mentor_id = ms.mentor_id
               JOIN subjects s ON ms.subject_id = s.subject_id
               WHERE s.subject_name = ?
               ORDER BY m.mentor_id""", (subject_name,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_mentors_for_topic(self, topic_id: int) -> list[dict]:
        rows = self.central_conn.execute(
            """SELECT DISTINCT m.mentor_id, m.mentor_name, m.bio
               FROM mentors m
               JOIN mentor_subjects ms ON m.mentor_id = ms.mentor_id
               JOIN chapters c ON c.subject_id = ms.subject_id
               JOIN topics t ON t.chapter_id = c.chapter_id
               WHERE t.topic_id = ?
               ORDER BY m.mentor_id""", (topic_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- call a single mentor via mock_mentor.py ----
    def call_mentor(self, mentor_id: int, question: str,
                    topic_id: int | None = None) -> dict:
        cmd = [sys.executable, self.MENTOR_SCRIPT,
               "--mentor_id", str(mentor_id),
               "--question", question]
        if topic_id is not None:
            cmd += ["--topic_id", str(topic_id)]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # parse the JSON response from mock_mentor.py stdout
        try:
            json_start = result.stdout.rindex("--- RESPONSE (JSON) ---")
            json_blob = result.stdout[json_start +
                                      len("--- RESPONSE (JSON) ---"):].strip()
            return json.loads(json_blob)
        except (ValueError, json.JSONDecodeError):
            return {
                "mentor_id": mentor_id,
                "error": "Could not parse mentor response",
                "raw_stdout": result.stdout,
                "raw_stderr": result.stderr,
            }

    # ---- call all mentors for a subject ----
    def route_and_call(self, subject_name: str, question: str,
                       topic_id: int | None = None) -> list[dict]:
        mentors = self.get_mentors_for_subject(subject_name)
        if not mentors:
            return [{"error": f"No mentors found for subject '{subject_name}'"}]

        responses = []
        for m in mentors:
            print(f"\n  >>> Routing to mentor: {m['mentor_name']} "
                  f"(id={m['mentor_id']})")
            resp = self.call_mentor(m["mentor_id"], question, topic_id)
            responses.append(resp)
        return responses

    # ---- store mentor responses to session memory ----
    def store_mentor_responses(self, student_id: int,
                               responses: list[dict]):
        for resp in responses:
            self.session_conn.execute(
                """INSERT INTO session_memory
                       (session_id, role, content, metadata)
                   VALUES (?, 'mentor_response', ?, ?)""",
                (
                    f"session_{student_id}",
                    resp.get("question", ""),
                    json.dumps(resp),
                ),
            )
        self.session_conn.commit()

    def close(self):
        self.central_conn.close()
        self.session_conn.close()


# =========================================================================
# 6. Pipeline — orchestrates input → D2 → D2.1 → D4 / D1 → Mentor → output
# =========================================================================
class Pipeline:
    def __init__(self):
        self.handler    = StudentInputHandler()
        self.classifier = D2Classifier()
        self.aggregator = D1Aggregator(model=self.classifier.model)
        self.novelty    = D2_1NoveltyHandler(model=self.classifier.model)
        self.d4         = D4NoveltyRedirect(self.handler.session_conn)
        self.router     = MentorRouter()

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
                print(f"        [+] {s}")

        if d1["missing_topics"]:
            print("      Missing subtopics:")
            for s in d1["missing_topics"]:
                print(f"        [-] {s}")

        # ---- Related cross-subject ----
        if d1["related_topics"]:
            self._section("Related Cross-Subject Topics")
            for t in d1["related_topics"]:
                print(f"      <-> {t}")

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

    def show_mentor_responses(self, subject: str, responses: list[dict]):
        self._banner(f"MENTOR RESPONSES ({subject})")
        for r in responses:
            if "error" in r:
                print(f"  ERROR: {r['error']}")
                continue
            print(f"  Mentor   : {r.get('mentor_name', '?')} "
                  f"(id={r.get('mentor_id', '?')})")
            print(f"  Decision : {r.get('decision', '?').upper()}")
            if r.get("reasoning"):
                print(f"  Reasoning: {r['reasoning']}")
            print()

    def show_d2_1_result(self, result: dict):
        self._banner("D2.1 NOVELTY HANDLER RESULT")

        status = result["status"]
        self._section(f"Status: {status}")

        if status == "syllabus_match":
            print(f"  Matched Topic : {result['topic']}")
            print(f"  Match Score   : {result['match_score']:.4f}")
            print(f"  Source        : {result['source']}")
            print(f"  Confirmed     : {result['confirmed']}")
            print()
            print("  Syllabus subtopics matched:")
            for t in result["subtopics"]:
                print(f"    [+] {t}")
            if result.get("web_subtopics"):
                print()
                print("  Web-discovered subtopics:")
                for t in result["web_subtopics"][:8]:
                    print(f"    -> {t}")

        elif status == "redirect_to_mentor":
            print(f"  Topic         : {result['topic']}")
            print(f"  Reason        : {result['reason']}")
            print(f"  Closest subj  : {result.get('closest_subject', '?')}")
            print(f"  Closest topic : {result.get('closest_syllabus_topic', '?')}")
            if result.get("mentors_available"):
                print("  Available mentors:")
                for m in result["mentors_available"]:
                    print(f"    -> {m['name']} (id={m['id']})")

    def show_d4_result(self, result: dict):
        self._banner("D4 NOVELTY REDIRECT RESULT")

        status = result["d4_status"]
        tag = result["classification_tag"]
        self._section(f"Status: {status}  |  Tag: {tag}")

        print(f"  Topic : {result['topic']}")
        print()
        print("  Result:")
        for line in result["result"].split("\n"):
            print(f"    {line}")

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

        # 6. D2.1 — activate when D2 says "No" (NO_MATCH_NEW / NO_MATCH_REPEAT)
        d2_1_result = None
        if d2_result["classification"] in ("NO_MATCH_NEW", "NO_MATCH_REPEAT"):
            d2_1_result = self.novelty.run_d2_1(
                student_id, question, d2_result
            )
            self.handler.store_d2_1_result(student_id, d2_1_result)
            self.show_d2_1_result(d2_1_result)

        # 6b. D4 — activate when D2.1 says redirect_to_mentor
        d4_result = None
        if d2_1_result and d2_1_result.get("status") == "redirect_to_mentor":
            d4_result = self.d4.run_d4(student_id, question, d2_result)
            self.handler.store_d4_result(student_id, d4_result)
            self.show_d4_result(d4_result)

        # 7. D1 — activate ONLY when D2 says repeated + in syllabus
        d1_result = None
        if d2_result["classification"] == "MATCHES_SYLLABUS_REPEAT":
            self.aggregator.set_syllabus(
                self.classifier._syllabus_texts,
                self.classifier._syllabus_ids,
                self.classifier._syllabus_embs,
            )
            d1_result = self.aggregator.aggregate(student_id, d2_result)
            self.handler.store_d1_result(student_id, d1_result)
            self.show_d1_result(d1_result)

        # 8. Mentor routing — activate when D2 matches a syllabus topic
        mentor_responses = []
        if d2_result["syllabus_match"]:
            # extract subject name from matched syllabus path
            subject = d2_result["syllabus_topic"].split(" > ")[0]
            topic_id = d2_result["syllabus_topic_id"]

            # build context-rich question for the mentor
            mentor_q = question
            if d1_result:
                # include D1 context: what's been covered and what's missing
                mentor_q += (f"\n[D1 Context: Parent topic={d1_result['detected_parent_topic']}, "
                             f"Coverage={d1_result['coverage_pct']}%, "
                             f"Missing={d1_result['missing_topics']}]")

            mentor_responses = self.router.route_and_call(
                subject, mentor_q, topic_id
            )
            self.router.store_mentor_responses(student_id, mentor_responses)
            self.show_mentor_responses(subject, mentor_responses)

        return {"d2": d2_result, "d2_1": d2_1_result, "d4": d4_result,
                "d1": d1_result, "mentors": mentor_responses}

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
        self.novelty.close()
        self.d4.close()
        self.router.close()


# =========================================================================
if __name__ == "__main__":
    pipeline = Pipeline()
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pipeline.close()
