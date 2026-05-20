# How to Run & Verify the Pipeline

## Prerequisites

```bash
pip install sentence-transformers requests numpy
```

The embedding model `all-MiniLM-L6-v2` will auto-download on first run (~80MB).

---

## 1. Setup Databases (first time only)

```bash
python setup_central_db.py
python setup_session_db.py
```

- `central.db` — syllabus (3 subjects, 9 chapters, 27 topics), 5 students, 5 mentors
- `session_memory.db` — empty; stores all session interactions

Verify:

```bash
python -c "import sqlite3; conn=sqlite3.connect('central.db'); [print(f'  {t}: {conn.execute(f\"SELECT COUNT(*) FROM {t}\").fetchone()[0]}') for t in ['students','subjects','chapters','topics','mentors']]; conn.close()"
```

Expected output:

```
  students: 5
  subjects: 3
  chapters: 9
  topics: 27
  mentors: 5
```

---

## 2. Run the Pipeline (interactive)

```bash
python pipeline.py
```

You'll see:

```
============================================================
  AI AGENT PIPELINE — MOCK ENV
============================================================
  [1] Aarav Patel  (grade 10)
  [2] Priya Sharma  (grade 12)
  [3] Rohan Gupta  (grade 11)
  [4] Ananya Singh  (grade 10)
  [5] Vikram Reddy  (grade 12)
```

Enter a student ID, date, and a question. The pipeline auto-routes:

| D2 Result | Branch | What Happens |
|-----------|--------|-------------|
| `MATCHES_SYLLABUS` | Mentor routing | Routes to subject mentor(s) |
| `MATCHES_SYLLABUS_REPEAT` | D1 + Mentor | Aggregates coverage, then routes to mentor |
| `NO_MATCH_NEW` | **D2.1** | Web search → syllabus re-match → student output or D4 redirect |
| `NO_MATCH_REPEAT` | **D2.1** | Same as above |

---

## 3. Verify Each Module

### D2 — Topic Classifier

```bash
python -c "
from pipeline import D2Classifier
c = D2Classifier()
c.load_syllabus()
result = c.classify('Explain Newton laws of motion')
print(f'Classification : {result[\"classification\"]}')
print(f'Syllabus match : {result[\"syllabus_match\"]} (score={result[\"syllabus_score\"]})')
print(f'Best topic     : {result[\"syllabus_topic\"]}')
c.close()
"
```

Expected: `MATCHES_SYLLABUS`, score > 0.65, topic = `Physics > Mechanics > Newton's Laws`

### D2.1 — Novelty Handler (syllabus match path)

```bash
python -c "
from pipeline import D2_1NoveltyHandler
from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer('all-MiniLM-L6-v2')
h = D2_1NoveltyHandler(model=model)

d2_no = {
    'classification': 'NO_MATCH_NEW',
    'new_question': 'What is quantum entanglement?',
    'syllabus_match': False, 'syllabus_score': 0.3,
    'syllabus_topic': 'Physics > Optics > Reflection', 'syllabus_topic_id': 1,
}
result = h.run_d2_1(1, 'What is quantum entanglement?', d2_no)
print(json.dumps(result, indent=2))
h.close()
"
```

Expected: `status = "syllabus_match"`, `confirmed = true`, match score > 0.55

### D2.1 — Novelty Handler (redirect to mentor path)

```bash
python -c "
from pipeline import D2_1NoveltyHandler
from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer('all-MiniLM-L6-v2')
h = D2_1NoveltyHandler(model=model)
h.SIMILARITY_THRESHOLD = 0.95  # force no match

d2_no = {
    'classification': 'NO_MATCH_NEW',
    'new_question': 'Best restaurants in Tokyo',
    'syllabus_match': False, 'syllabus_score': 0.1,
    'syllabus_topic': 'Mathematics > Algebra > Linear Equations', 'syllabus_topic_id': 1,
}
result = h.run_d2_1(1, 'Best restaurants in Tokyo', d2_no)
print(json.dumps(result, indent=2))
h.close()
"
```

Expected: `status = "redirect_to_mentor"`, `reason = "not_in_syllabus"`, mentors listed

### D1 — Aggregator (repeated syllabus topic)

```bash
python -c "
from pipeline import D2Classifier, D1Aggregator
c = D2Classifier()
c.load_syllabus()
# D1 only activates on MATCHES_SYLLABUS_REPEAT — need history first
# Run the interactive pipeline twice with the same topic to trigger it
print('D1 activates when a student re-asks a syllabus topic.')
print('Use interactive mode: python pipeline.py')
c.close()
"
```

### Mentor Router

```bash
python mock_caller.py --question "Explain derivatives" --topic_id 5
```

---

## 4. Quick Smoke Test (full pipeline, one question)

```bash
python -c "
from pipeline import Pipeline
import json

p = Pipeline()
p.classifier.load_syllabus()
p.classifier.load_history(1)

# new syllabus topic — routes to mentor
result = p.run_question(1, '2026-05-20', 'What are quadratic equations?')
print()
print('D2  classification:', result['d2']['classification'])
print('D2.1:', result['d2_1'])
print('D1  :', result['d1'])
print('Mentors:', len(result['mentors']), 'response(s)')
p.close()
"
```

---

## 5. File Map

| File | Purpose |
|------|---------|
| `pipeline.py` | Main pipeline — D1, D2, D2.1, MentorRouter, Pipeline orchestrator |
| `setup_central_db.py` | Creates `central.db` (syllabus, students, mentors) |
| `setup_session_db.py` | Creates `session_memory.db` (session storage) |
| `mock_mentor.py` | CLI mentor endpoint (human types yes/no) |
| `mock_caller.py` | Batch-calls mentors for a topic |

---

## 6. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: sentence_transformers` | `pip install sentence-transformers` |
| `central.db not found` | Run `python setup_central_db.py` |
| `UnicodeEncodeError` on Windows | Already fixed in code; if you see it, set `PYTHONIOENCODING=utf-8` |
| Web search returns 0 results | DuckDuckGo API may be rate-limited; try again in a few seconds |
| D1 never activates | D1 requires the student to ask the same syllabus topic twice |
