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
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CENTRAL_DB  = os.path.join(BASE_DIR, "central.db")
SESSION_DB  = os.path.join(BASE_DIR, "session_memory.db")
POLICY_DB   = os.path.join(BASE_DIR, "policy.db")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")

# force offline — never hit HuggingFace Hub at runtime
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# load API keys from api.env
load_dotenv(os.path.join(BASE_DIR, "api.env"))

# ---------------------------------------------------------------------------
# xAI (Grok) API — single centralized key for all LLM calls
# ---------------------------------------------------------------------------
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_MODEL   = os.getenv("XAI_MODEL", "grok-3-mini")

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
        print("[D2] Loading embedding model (local) ...")
        self.model = SentenceTransformer(MODEL_DIR)
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
# 2.1  D2.1 — Novelty Handler (xAI web search + syllabus re-match)
# =========================================================================
class D2_1NoveltyHandler:
    """Activates when D2 returns NO_MATCH_NEW or NO_MATCH_REPEAT.

    Workflow:
      1. xAI live web search → gather info about the novel topic
      2. Extract subtopics from web results + memory DB context
      3. Vector-match subtopics against Central DB syllabus
      4. If match found → structured output for student
      5. If no match → queue redirect to mentor (D4 handles it)
    """

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

        # xAI client for live web search
        if XAI_API_KEY and XAI_API_KEY != "your-xai-api-key-here":
            self.xai_client = OpenAI(
                api_key=XAI_API_KEY,
                base_url="https://api.x.ai/v1",
            )
            print("[D2.1] xAI web search client initialized.")
        else:
            self.xai_client = None
            print("[D2.1] WARNING: XAI_API_KEY not set — "
                  "web search will use DuckDuckGo fallback.")

    # ---- encoding helpers ----
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_n, b_n.T)

    # ---- context loaders ----
    def load_syllabus(self):
        """Load all topics from central DB as syllabus."""
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

    def load_memory_context(self, student_id: int) -> list[str]:
        """Load past student questions from session memory DB."""
        rows = self.session_conn.execute(
            """SELECT content FROM session_memory
               WHERE session_id = ? AND role = 'student'
               ORDER BY id""", (f"session_{student_id}",)
        ).fetchall()
        return [r["content"] for r in rows]

    # ---- xAI live web search ----
    def xai_web_search(self, query: str,
                       memory_context: list[str] | None = None) -> tuple[str, list[dict]]:
        """Use xAI live search to get web results for the query.
        Enriches search with memory DB context.
        Returns (full_response_text, citations_list)."""
        if not self.xai_client:
            # fallback to DuckDuckGo
            results = self._ddg_fallback(query)
            combined = " ".join(r["snippet"] for r in results)
            return combined, results

        # build context-aware search prompt
        context_block = ""
        if memory_context:
            recent = memory_context[-5:]
            context_block = (
                "\n\nStudent's recent questions for context:\n"
                + "\n".join(f"- {q}" for q in recent)
            )

        prompt = (
            f"Research the topic: \"{query}\"{context_block}\n\n"
            "Provide a comprehensive overview covering:\n"
            "1. What this topic is about\n"
            "2. Key subtopics and concepts\n"
            "3. How it relates to mathematics, physics, or computer science\n"
            "4. Important sub-areas a student should explore\n"
            "List specific subtopics as a structured breakdown."
        )

        try:
            resp = self.xai_client.chat.completions.create(
                model=XAI_MODEL,
                messages=[
                    {"role": "system",
                     "content": ("You are an educational research assistant. "
                                 "Search the web and provide structured, "
                                 "accurate information about academic topics.")},
                    {"role": "user", "content": prompt},
                ],
                extra_body={
                    "search_parameters": {
                        "mode": "auto",
                        "return_citations": True,
                    },
                },
                temperature=0.3,
                max_tokens=1500,
            )
            response_text = resp.choices[0].message.content.strip()

            # extract citations from response
            citations = []
            if hasattr(resp, 'citations') and resp.citations:
                for c in resp.citations:
                    citations.append({
                        "title": getattr(c, "title", ""),
                        "snippet": getattr(c, "snippet", ""),
                        "source": getattr(c, "url", ""),
                    })

            return response_text, citations

        except Exception as e:
            print(f"[D2.1] xAI search failed: {e}, falling back to DuckDuckGo")
            results = self._ddg_fallback(query)
            combined = " ".join(r["snippet"] for r in results)
            return combined, results

    def _ddg_fallback(self, query: str) -> list[dict]:
        """DuckDuckGo fallback when xAI is unavailable."""
        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json",
                         "no_html": 1, "skip_disambig": 1},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"[D2.1] DuckDuckGo fallback failed: {e}")
            return []

        results = []
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", query),
                "snippet": data["AbstractText"],
                "source": data.get("AbstractSource", "DuckDuckGo"),
            })
        for item in data.get("RelatedTopics", [])[:8]:
            if isinstance(item, dict) and "Text" in item:
                results.append({
                    "title": item.get("Text", "")[:100],
                    "snippet": item.get("Text", ""),
                    "source": item.get("FirstURL", "DuckDuckGo"),
                })
        return results

    # ---- subtopic extraction via xAI ----
    def extract_subtopics_xai(self, query: str,
                              web_response: str) -> list[str]:
        """Use xAI to extract structured subtopics from web search results."""
        if not self.xai_client:
            return self._extract_subtopics_regex(web_response)

        try:
            resp = self.xai_client.chat.completions.create(
                model=XAI_MODEL,
                messages=[
                    {"role": "system",
                     "content": ("Extract a clean list of academic subtopics "
                                 "from the given text. Return ONLY a JSON "
                                 "array of strings, nothing else.")},
                    {"role": "user",
                     "content": (f"Topic: \"{query}\"\n\n"
                                 f"Research text:\n{web_response[:3000]}\n\n"
                                 "Extract the key subtopics as a JSON array "
                                 "of short strings (2-5 words each).")},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            # parse JSON array from response
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                subtopics = json.loads(json_match.group())
                return [s.strip() for s in subtopics if isinstance(s, str)][:20]
        except Exception as e:
            print(f"[D2.1] xAI subtopic extraction failed: {e}")

        return self._extract_subtopics_regex(web_response)

    @staticmethod
    def _extract_subtopics_regex(text: str) -> list[str]:
        """Fallback regex-based subtopic extraction."""
        subtopics = []
        parts = re.split(r"[,;|\-\–\—\n]", text)
        for part in parts:
            clean = part.strip().strip(".")
            if 5 < len(clean) < 80 and clean not in subtopics:
                subtopics.append(clean)
        return subtopics[:20]

    # ---- syllabus matching ----
    def match_to_syllabus(self, topic: str, subtopics: list[str]) -> dict:
        """Vector-match web-discovered subtopics against central DB syllabus."""
        if self._syllabus_embs is None:
            self.load_syllabus()

        candidates = [topic] + subtopics
        c_embs = self.model.encode(candidates, convert_to_numpy=True,
                                   show_progress_bar=False)
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

    # ---- memory-augmented matching ----
    def match_with_memory(self, topic: str, subtopics: list[str],
                          memory_context: list[str]) -> dict:
        """Vector-match against syllabus AND memory DB past questions.
        Returns combined match result."""
        syllabus_match = self.match_to_syllabus(topic, subtopics)

        # also check if any subtopics relate to past questions
        memory_related = []
        if memory_context:
            m_embs = self.model.encode(memory_context, convert_to_numpy=True,
                                       show_progress_bar=False)
            s_embs = self.model.encode(subtopics[:10], convert_to_numpy=True,
                                       show_progress_bar=False)
            sims = self._cosine(s_embs, m_embs)

            for si, sub in enumerate(subtopics[:10]):
                best_mem_idx = int(np.argmax(sims[si]))
                if sims[si][best_mem_idx] >= 0.5:
                    memory_related.append({
                        "subtopic": sub,
                        "related_past_q": memory_context[best_mem_idx],
                        "similarity": round(float(sims[si][best_mem_idx]), 4),
                    })

        syllabus_match["memory_related"] = memory_related
        syllabus_match["memory_context_used"] = len(memory_context)
        return syllabus_match

    # ---- D4 redirect (queue request) ----
    def queue_for_d4(self, student_id: int, topic: str,
                     d2_result: dict, web_response: str,
                     subtopics: list[str]) -> dict:
        """No match found — queue redirect to D4 for mentor handling."""
        if self._syllabus_embs is None:
            self.load_syllabus()

        # find closest subject for mentor routing
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

        # store redirect in session memory (queued for D4)
        self.session_conn.execute(
            """INSERT INTO session_memory
                   (session_id, role, content, metadata)
               VALUES (?, 'd2_1_redirect', ?, ?)""",
            (
                f"session_{student_id}",
                topic,
                json.dumps({
                    "action": "redirect_to_d4",
                    "closest_subject": closest_subject,
                    "closest_topic": closest_path,
                    "mentors": mentor_list,
                    "web_subtopics": subtopics[:10],
                    "web_response_snippet": web_response[:500],
                    "original_d2": d2_result["classification"],
                }),
            ),
        )
        self.session_conn.commit()

        return {
            "status": "redirect_to_mentor",
            "topic": topic,
            "reason": "subtopics_not_in_syllabus",
            "closest_subject": closest_subject,
            "closest_syllabus_topic": closest_path,
            "mentors_available": mentor_list,
            "web_subtopics": subtopics[:10],
            "web_response_snippet": web_response[:500],
        }

    # ---- main entry point ----
    def run_d2_1(self, student_id: int, input_topic: str,
                 d2_result: dict) -> dict:
        """Full D2.1 pipeline:
          1. xAI web search about novel topic
          2. Extract subtopics
          3. Match subtopics against central DB + memory DB
          4. If match → output; if fail → queue redirect to D4
        """
        print(f"\n[D2.1] Novelty handler activated for: \"{input_topic}\"")

        # 1. load context from memory DB
        memory_context = self.load_memory_context(student_id)
        print(f"[D2.1] Loaded {len(memory_context)} past questions from memory DB.")

        # 2. xAI live web search
        print("[D2.1] Searching the web via xAI ...")
        web_response, citations = self.xai_web_search(input_topic,
                                                       memory_context)
        print(f"[D2.1] Got web response ({len(web_response)} chars, "
              f"{len(citations)} citations).")

        # 3. extract subtopics
        print("[D2.1] Extracting subtopics ...")
        subtopics = self.extract_subtopics_xai(input_topic, web_response)
        print(f"[D2.1] Extracted {len(subtopics)} subtopics: "
              f"{subtopics[:5]}{'...' if len(subtopics) > 5 else ''}")

        # 4. vector match against central DB syllabus + memory DB
        print("[D2.1] Matching subtopics against Central DB + Memory DB ...")
        match_result = self.match_with_memory(input_topic, subtopics,
                                               memory_context)
        print(f"[D2.1] Best syllabus similarity = "
              f"{match_result['best_score']:.4f} "
              f"(threshold={self.SIMILARITY_THRESHOLD})")

        # 5. branch: match found or redirect to D4
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
                "web_subtopics": subtopics,
                "match_score": best["score"],
                "citations": citations,
                "memory_related": match_result.get("memory_related", []),
                "web_response_snippet": web_response[:500],
                "all_ranked": match_result["ranked"],
                "source": "xai_web_search",
                "confirmed": True,
            }
            print(f"[D2.1] MATCH FOUND -> {best['syllabus_topic']} "
                  f"(score={best['score']:.4f})")
        else:
            # no match — queue for D4 (mentor redirect)
            output = self.queue_for_d4(student_id, input_topic,
                                        d2_result, web_response, subtopics)
            print(f"[D2.1] No syllabus match -> queued for D4 mentor redirect")

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
        if XAI_API_KEY and XAI_API_KEY != "your-xai-api-key-here":
            self.xai_client = OpenAI(
                api_key=XAI_API_KEY,
                base_url="https://api.x.ai/v1",
            )
        else:
            self.xai_client = None
            print("[D4] WARNING: XAI_API_KEY not set — "
                  "D4 will use fallback logic (no LLM calls)")
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
        """Use xAI to detect if the student is repeatedly exploring
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

        if not self.xai_client:
            # fallback: detect curiosity by keyword overlap with past questions
            print("[D4] Using fallback curiosity detection (no API key)")
            return len(past) >= 3

        try:
            resp = self.xai_client.chat.completions.create(
                model=XAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=3,
            )
            answer = resp.choices[0].message.content.strip().lower()
            return answer.startswith("yes")
        except Exception as e:
            print(f"[D4] xAI curiosity check failed: {e}")
            return False

    # ---- novel query classification ----
    def classify_novel_query(self, question: str,
                             d2_result: dict) -> str:
        """Use xAI to classify the novel query as genuinely_novel
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

        if not self.xai_client:
            # fallback: assume genuinely novel
            print("[D4] Using fallback classification (no API key)")
            return "genuinely_novel"

        try:
            resp = self.xai_client.chat.completions.create(
                model=XAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
            )
            tag = resp.choices[0].message.content.strip().lower()
            if "off_topic" in tag:
                return "off_topic"
            return "genuinely_novel"
        except Exception as e:
            print(f"[D4] xAI classification failed: {e}")
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

        # pick one mentor per query (round-robin by question hash)
        idx = hash(question) % len(mentors)
        m = mentors[idx]
        print(f"\n  >>> Routing to mentor: {m['mentor_name']} "
              f"(id={m['mentor_id']})")
        resp = self.call_mentor(m["mentor_id"], question, topic_id)
        return [resp]

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
# 5. D3 — Output Safety & Policy Enforcement Layer
# =========================================================================
class D3OutputSafety:
    """Checks pipeline output against policy.db before showing to student.
    Up to 2 retries on failure, then queues for mentor review."""

    def __init__(self, central_conn: sqlite3.Connection,
                 session_conn: sqlite3.Connection):
        self.central_conn = central_conn
        self.session_conn = session_conn
        self.policy_conn = sqlite3.connect(POLICY_DB)
        self.policy_conn.row_factory = sqlite3.Row
        self._load_config()

    def _load_config(self):
        rows = self.policy_conn.execute(
            "SELECT config_key, config_value FROM safety_config"
        ).fetchall()
        self.config = {r["config_key"]: r["config_value"] for r in rows}
        self.max_retries = int(self.config.get("max_retries", 2))
        self.min_length = int(self.config.get("min_output_length", 10))
        self.max_length = int(self.config.get("max_output_length", 5000))

    def _get_blocked(self) -> list[dict]:
        return self.policy_conn.execute(
            "SELECT category, pattern, action FROM blocked_patterns"
        ).fetchall()

    def _get_rules(self) -> list[dict]:
        return self.policy_conn.execute(
            "SELECT rule_name, rule_type, pattern, severity FROM content_rules"
        ).fetchall()

    # ---- core check ----
    def check_output(self, output_text: str, question: str,
                     pipeline_result: dict) -> dict:
        """Validate output_text against policy DB.
        Returns {safe: bool, violations: [...], action: str}."""
        text_lower = output_text.lower()
        violations = []

        # 1. length checks
        if len(output_text.strip()) < self.min_length:
            violations.append({
                "rule": "min_length",
                "detail": f"Output too short ({len(output_text)} < {self.min_length})",
                "severity": 2,
            })
        if len(output_text) > self.max_length:
            violations.append({
                "rule": "max_length",
                "detail": f"Output too long ({len(output_text)} > {self.max_length})",
                "severity": 2,
            })

        # 2. blocked patterns
        for bp in self._get_blocked():
            if bp["pattern"] in text_lower:
                violations.append({
                    "rule": f"blocked:{bp['category']}",
                    "detail": f"Blocked pattern found: '{bp['pattern']}'",
                    "severity": 3 if bp["action"] == "block" else 2,
                    "action": bp["action"],
                })

        # 3. content rules
        for rule in self._get_rules():
            if rule["rule_type"] == "block" and rule["pattern"] in text_lower:
                violations.append({
                    "rule": rule["rule_name"],
                    "detail": f"Blocked by rule: {rule['pattern']}",
                    "severity": rule["severity"],
                    "action": "block",
                })
            elif rule["rule_type"] == "flag" and rule["pattern"] in text_lower:
                violations.append({
                    "rule": rule["rule_name"],
                    "detail": f"Flagged by rule: {rule['pattern']}",
                    "severity": rule["severity"],
                    "action": "flag",
                })

        # 4. mentor response validation
        mentors = pipeline_result.get("mentors", [])
        for m in mentors:
            if m.get("decision") == "no" and m.get("reasoning"):
                reasoning_lower = m["reasoning"].lower()
                for bp in self._get_blocked():
                    if bp["pattern"] in reasoning_lower:
                        violations.append({
                            "rule": f"mentor_reasoning:{bp['category']}",
                            "detail": (f"Mentor {m.get('mentor_name', '?')} "
                                       f"reasoning contains: '{bp['pattern']}'"),
                            "severity": 2,
                        })

        # 5. empty / error state check
        if not output_text.strip() or output_text.strip() == "(no output)":
            violations.append({
                "rule": "empty_output",
                "detail": "Pipeline produced no output",
                "severity": 2,
            })

        # determine action
        has_block = any(v.get("action") == "block" or v["severity"] >= 3
                        for v in violations)
        has_flag = any(v.get("action") == "flag" or v["severity"] == 2
                       for v in violations)

        if has_block:
            action = "block"
        elif has_flag:
            action = "flag"
        else:
            action = "pass"

        return {
            "safe": action == "pass" and not violations,
            "action": action,
            "violations": violations,
            "violation_count": len(violations),
        }

    # ---- retry loop ----
    def validate_with_retry(self, student_id: int, question: str,
                            pipeline_result: dict,
                            generate_fn) -> dict:
        """Run output through policy check with up to max_retries.
        generate_fn() returns (output_text, result_dict).

        Returns:
          {status: 'approved'|'mentor_review',
           output: str, check: dict, retry_count: int}
        """
        last_check = None
        last_output = ""

        for attempt in range(self.max_retries + 1):
            output_text, result = generate_fn(attempt)
            last_output = output_text

            check = self.check_output(output_text, question, result)
            last_check = check

            if check["safe"]:
                if attempt > 0:
                    print(f"[D3] Output approved on retry #{attempt}.")
                return {
                    "status": "approved",
                    "output": output_text,
                    "check": check,
                    "retry_count": attempt,
                }

            if check["action"] == "block":
                print(f"[D3] BLOCKED (attempt {attempt + 1}): "
                      f"{check['violations'][0]['detail']}")
            else:
                print(f"[D3] FLAGGED (attempt {attempt + 1}): "
                      f"{check['violations'][0]['detail']}")

        # all retries exhausted — queue for mentor review
        print(f"[D3] Max retries ({self.max_retries}) exhausted. "
              f"Queuing for mentor review.")
        self._queue_mentor_review(student_id, question,
                                  last_output, last_check)

        return {
            "status": "mentor_review",
            "output": None,
            "check": last_check,
            "retry_count": self.max_retries,
        }

    # ---- mentor review queue ----
    def _queue_mentor_review(self, student_id: int, question: str,
                             failed_output: str, check: dict):
        """Insert into mentor_reviews table and update student pending count."""
        fail_reason = "; ".join(v["detail"] for v in check.get("violations", []))

        # find best mentor for the student's question context
        assigned = self._find_review_mentor(question)

        self.central_conn.execute(
            """INSERT INTO mentor_reviews
                   (student_id, question, failed_output, fail_reason,
                    retry_count, assigned_mentor_id, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', datetime('now'))""",
            (student_id, question, failed_output, fail_reason,
             self.max_retries, assigned),
        )

        # increment student's pending_reviews count
        self.central_conn.execute(
            """UPDATE students
               SET pending_reviews = pending_reviews + 1,
                   review_status = 'pending_review'
               WHERE student_id = ?""",
            (student_id,),
        )
        self.central_conn.commit()

        # also log to session memory
        self.session_conn.execute(
            """INSERT INTO session_memory
                   (session_id, role, content, metadata)
               VALUES (?, 'd3_review', ?, ?)""",
            (
                f"session_{student_id}",
                question,
                json.dumps({
                    "action": "queued_for_mentor_review",
                    "fail_reason": fail_reason,
                    "assigned_mentor_id": assigned,
                    "retry_count": self.max_retries,
                }),
            ),
        )
        self.session_conn.commit()

    def _find_review_mentor(self, question: str) -> int | None:
        """Pick the most relevant mentor for review based on question."""
        # use first available mentor as default for mock
        row = self.central_conn.execute(
            "SELECT mentor_id FROM mentors ORDER BY mentor_id LIMIT 1"
        ).fetchone()
        return row["mentor_id"] if row else None

    def get_student_review_status(self, student_id: int) -> dict:
        row = self.central_conn.execute(
            """SELECT review_status, pending_reviews
               FROM students WHERE student_id = ?""",
            (student_id,),
        ).fetchone()
        if row:
            return {
                "review_status": row["review_status"],
                "pending_reviews": row["pending_reviews"],
            }
        return {"review_status": "unknown", "pending_reviews": 0}

    def close(self):
        self.policy_conn.close()


# =========================================================================
# 6. AI Tutor — xAI Grok-powered guided learning
# =========================================================================
class AITutor:
    """Teaches subtopics one by one using xAI Grok.
    Enters guided learning mode after pipeline identifies a topic."""

    def __init__(self, central_conn: sqlite3.Connection):
        self.central_conn = central_conn
        if XAI_API_KEY and XAI_API_KEY != "your-xai-api-key-here":
            self.client = OpenAI(
                api_key=XAI_API_KEY,
                base_url="https://api.x.ai/v1",
            )
        else:
            self.client = None
            print("[AITutor] WARNING: XAI_API_KEY not set — "
                  "guided learning disabled.")

    def get_subtopics(self, topic: str, pipeline_result: dict) -> list[str]:
        """Extract ordered subtopics from pipeline result."""
        d2 = pipeline_result.get("d2", {})
        d2_1 = pipeline_result.get("d2_1")
        d1 = pipeline_result.get("d1")

        # syllabus match — get all chapter topics
        if d2.get("syllabus_match") and d2.get("syllabus_topic_id"):
            topic_id = d2["syllabus_topic_id"]
            row = self.central_conn.execute(
                """SELECT c.chapter_id FROM topics t
                   JOIN chapters c ON t.chapter_id = c.chapter_id
                   WHERE t.topic_id = ?""", (topic_id,)
            ).fetchone()
            if row:
                topics = self.central_conn.execute(
                    """SELECT topic_name FROM topics
                       WHERE chapter_id = ?
                       ORDER BY topic_id""", (row["chapter_id"],)
                ).fetchall()
                subs = [r["topic_name"] for r in topics]
                if subs:
                    return subs

        # D2.1 match — use matched subtopics
        if d2_1 and d2_1.get("status") == "syllabus_match":
            if d2_1.get("subtopics"):
                return [s.split(" > ")[-1] for s in d2_1["subtopics"]
                        if " > " in s][:8]

        # D4 — parse web results into subtopics
        d4 = pipeline_result.get("d4")
        if d4 and d4.get("result"):
            lines = [l.strip().lstrip("- ").strip()
                     for l in d4["result"].split("\n")
                     if l.strip().startswith("- ")]
            if lines:
                return lines[:8]

        # fallback — single topic
        return [topic]

    def teach_subtopic(self, topic: str, subtopic: str,
                       history: list[dict]) -> str:
        """Call xAI to explain one subtopic."""
        if not self.client:
            return (f"Let's learn about {subtopic}. "
                    f"This is an important part of {topic}. "
                    f"(AI tutor unavailable — add XAI_API_KEY to api.env)")

        system = (
            "You are a friendly, patient tutor helping a student learn. "
            "Explain the subtopic clearly in 3-5 short paragraphs. "
            "Use simple language, give one example, and end with a "
            "brief summary. Be encouraging."
        )
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-6:])  # keep recent context
        messages.append({
            "role": "user",
            "content": (
                f"We're learning about {topic}. "
                f"Now explain this subtopic: {subtopic}"
            ),
        })

        try:
            resp = self.client.chat.completions.create(
                model=XAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Could not get explanation for {subtopic}: {e}"

    def close(self):
        pass


# =========================================================================
# 7. Pipeline — orchestrates input → D2 → D2.1 → D4 / D1 → D3 → output
# =========================================================================
class Pipeline:
    def __init__(self):
        self.handler    = StudentInputHandler()
        self.classifier = D2Classifier()
        self.aggregator = D1Aggregator(model=self.classifier.model)
        self.novelty    = D2_1NoveltyHandler(model=self.classifier.model)
        self.d4         = D4NoveltyRedirect(self.handler.session_conn)
        self.router     = MentorRouter()
        self.d3         = D3OutputSafety(self.handler.central_conn,
                                         self.handler.session_conn)
        self.tutor      = AITutor(self.handler.central_conn)

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

    def show_d3_result(self, d3: dict):
        self._banner("D3 OUTPUT SAFETY CHECK")
        status = d3["status"]
        check = d3.get("check", {})
        self._section(f"Status: {status}  |  Retries: {d3.get('retry_count', 0)}")
        if check.get("violations"):
            print("  Violations:")
            for v in check["violations"]:
                print(f"    [{v.get('severity', '?')}] {v['detail']}")
        else:
            print("  No violations — output approved.")
        if status == "mentor_review":
            print()
            print("  >>> Query queued for mentor review.")
            print("  >>> Student will see: 'waiting for mentor review'.")

    # ---- build output text for D3 checking ----
    @staticmethod
    def _build_output_text(d2: dict, d2_1: dict | None, d4: dict | None,
                           d1: dict | None, mentors: list[dict]) -> str:
        """Combine pipeline results into a single text block for D3 validation."""
        parts = []

        # classification
        parts.append(f"Classification: {d2['classification']}")
        if d2.get("syllabus_topic"):
            parts.append(f"Topic: {d2['syllabus_topic']}")

        # D2.1
        if d2_1:
            parts.append(f"D2.1 status: {d2_1.get('status', 'n/a')}")
            if d2_1.get("topic"):
                parts.append(f"D2.1 topic: {d2_1['topic']}")
            if d2_1.get("web_response_snippet"):
                parts.append(f"Web data: {d2_1['web_response_snippet'][:500]}")

        # D4
        if d4:
            parts.append(f"D4 status: {d4.get('d4_status', 'n/a')}")
            parts.append(f"D4 tag: {d4.get('classification_tag', 'n/a')}")
            if d4.get("result"):
                parts.append(f"D4 result: {d4['result'][:500]}")

        # D1
        if d1:
            parts.append(f"D1 coverage: {d1.get('coverage_pct', 0)}%")

        # mentors
        for m in mentors:
            if "error" in m:
                parts.append(f"Mentor error: {m['error']}")
            else:
                parts.append(
                    f"Mentor {m.get('mentor_name', '?')}: "
                    f"{m.get('decision', '?')}"
                )
                if m.get("reasoning"):
                    parts.append(f"  Reasoning: {m['reasoning']}")

        return "\n".join(parts)

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

        # 8. Mentor routing — ONLY when D2 does NOT match syllabus
        mentor_responses = []
        if not d2_result["syllabus_match"]:
            if d2_1_result and d2_1_result.get("status") == "redirect_to_mentor":
                closest_subject = d2_1_result.get("closest_subject", "Mathematics")
                mentor_responses = self.router.route_and_call(
                    closest_subject, question, None
                )
                self.router.store_mentor_responses(student_id, mentor_responses)
                self.show_mentor_responses(closest_subject, mentor_responses)

        # 9. D3 — Output safety check
        output_text = self._build_output_text(
            d2_result, d2_1_result, d4_result, d1_result, mentor_responses
        )
        partial = {"d2": d2_result, "d2_1": d2_1_result, "d4": d4_result,
                   "d1": d1_result, "mentors": mentor_responses}
        d3_result = self.d3.validate_with_retry(
            student_id, question, partial,
            lambda _attempt: (output_text, partial),
        )
        self.show_d3_result(d3_result)

        partial["d3"] = d3_result
        return partial

    # ---- silent run (returns results without printing) ----
    def run_question_silent(self, student_id: int, date: str,
                            question: str) -> dict:
        """Run the full pipeline without any display output.
        Returns the same dict as run_question."""
        self.classifier.load_context(student_id)
        intake_info = self.handler.intake(student_id, date, question)
        d2_result = self.classifier.classify(question)
        self.handler.store_d2_result(student_id, d2_result)

        d2_1_result = None
        if d2_result["classification"] in ("NO_MATCH_NEW", "NO_MATCH_REPEAT"):
            d2_1_result = self.novelty.run_d2_1(
                student_id, question, d2_result
            )
            self.handler.store_d2_1_result(student_id, d2_1_result)

        d4_result = None
        if d2_1_result and d2_1_result.get("status") == "redirect_to_mentor":
            d4_result = self.d4.run_d4(student_id, question, d2_result)
            self.handler.store_d4_result(student_id, d4_result)

        d1_result = None
        if d2_result["classification"] == "MATCHES_SYLLABUS_REPEAT":
            self.aggregator.set_syllabus(
                self.classifier._syllabus_texts,
                self.classifier._syllabus_ids,
                self.classifier._syllabus_embs,
            )
            d1_result = self.aggregator.aggregate(student_id, d2_result)
            self.handler.store_d1_result(student_id, d1_result)

        # 8. D3 — Output safety check
        partial = {"d2": d2_result, "d2_1": d2_1_result, "d4": d4_result,
                   "d1": d1_result, "mentors": []}
        output_text = self._build_output_text(
            d2_result, d2_1_result, d4_result, d1_result, []
        )
        d3_result = self.d3.validate_with_retry(
            student_id, question, partial,
            lambda _attempt: (output_text, partial),
        )

        # 9. Extract topic + subtopics for AI Tutor
        topic = question
        if d2_result.get("syllabus_topic"):
            topic = d2_result["syllabus_topic"]
        elif d2_1_result and d2_1_result.get("topic"):
            topic = d2_1_result["topic"]
        elif d4_result and d4_result.get("topic"):
            topic = d4_result["topic"]

        subtopics = self.tutor.get_subtopics(topic, partial)

        return {
            "intake": intake_info,
            "d2": d2_result, "d2_1": d2_1_result, "d4": d4_result,
            "d1": d1_result, "mentors": [], "d3": d3_result,
            "topic": topic, "subtopics": subtopics,
        }

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
        self.d3.close()
        self.tutor.close()


# =========================================================================
if __name__ == "__main__":
    pipeline = Pipeline()
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pipeline.close()
