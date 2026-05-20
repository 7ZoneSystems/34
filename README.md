# AI Agent Pipeline — Mock Environment

## Setup

1. Add your xAI API key to `api.env`:
   ```
   XAI_API_KEY=xai-your_actual_key_here
   XAI_MODEL=grok-3-mini
   ```
   Without the key, D2.1 uses DuckDuckGo fallback and D4 uses heuristic fallback (no LLM calls).

2. Install dependencies:
   ```bash
   .env/bin/pip install sentence-transformers openai python-dotenv requests
   ```

3. Initialize the Policy DB (safety rules for D3):
   ```bash
   .env/bin/python setup_policy_db.py
   ```

---

## How to Run

### Student Mode (clean interface)
```bash
.env/bin/python main.py
```
You act as a student. Pick your ID, ask questions, and see friendly responses.
All internal pipeline logic runs silently in the background.
Type `switch` to change student, `quit` to exit.

### Debug Mode (full pipeline output)
```bash
.env/bin/python pipeline.py
```
Shows every pipeline layer's internal output (D2 classification, D2.1 matching, D4 analysis, D1 aggregation, mentor routing). Use this to test and debug the pipeline.

---

## Registered Students

| ID | Name | Grade |
|----|------|-------|
| 1  | Aarav Patel | 10 |
| 2  | Priya Sharma | 12 |
| 3  | Rohan Gupta | 11 |
| 4  | Ananya Singh | 10 |
| 5  | Vikram Reddy | 12 |

---

## Test Prompts

Use these prompts to trigger each pipeline path. Run in debug mode (`pipeline.py`) to see the full internal flow, or student mode (`main.py`) to see the clean response.

### Path 1: MATCHES_SYLLABUS_NEW
First-time question about a syllabus topic. D2 matches → mentor routing.
```
How to solve linear equations?
Explain the quadratic formula
What is a for loop?
How do convex lenses work?
What is momentum?
Explain Newton's laws of motion
What is a linked list?
How does quicksort work?
What is binary search?
```

### Path 2: MATCHES_SYLLABUS_REPEAT
Ask a similar question to one already asked. D2 matches + detects repeat → D1 aggregation → mentor routing with context.
```
# First:  "What are quadratic equations?"
# Second: "How to solve quadratic equations again?"

# First:  "Explain Newton's laws of motion"
# Second: "Tell me more about inertia and Newton's first law"

# First:  "What is a linked list?"
# Second: "How to detect a cycle in a linked list?"
```

### Path 3: NO_MATCH_NEW → D2.1 syllabus re-match
Off-syllabus question, but D2.1 finds a syllabus connection via web search + vector matching.
```
What is quantum entanglement?
How does GPS work?
What is the photoelectric effect?
Explain the theory of relativity
How do black holes form?
```

### Path 4: NO_MATCH_NEW → D2.1 → D4 (genuinely_novel)
Off-syllabus question that doesn't match any syllabus topic. D4 classifies as genuinely novel → enrichment output.
```
What is machine learning?
How does blockchain work?
What is CRISPR gene editing?
Explain the Turing test
What is dark matter?
```

### Path 5: NO_MATCH_REPEAT → D2.1 → D4 (curiosity pattern)
Ask multiple off-syllabus questions in sequence. After 3+ novel questions, D4 detects curiosity pattern → guide mode.
```
# Ask these in sequence (same session):
What is machine learning?
How do neural networks work?
What is deep learning?
# The third question should trigger D4 curiosity detection
```

### Path 6: Off-topic
Casual or irrelevant questions. D4 classifies as off_topic → polite redirect.
```
What's the best pizza place?
Tell me a joke
What's the weather today?
```

### Path 7: D3 Safety — Blocked content
Questions that trigger policy.db blocked patterns. D3 flags output → retries → queues for mentor review.
To test: add a custom blocked pattern to policy.db, then ask a question whose pipeline output would contain it.
```bash
# Add a test blocked pattern:
.env/bin/python -c "
import sqlite3
conn = sqlite3.connect('policy.db')
conn.execute(\"INSERT INTO blocked_patterns (category, pattern, action) VALUES ('test', 'momentum', 'block')\")
conn.commit(); conn.close()
"
# Now ask: "What is momentum?" — D3 will block the output containing "momentum"
# After 2 retries → "Your query is waiting for mentor review"
```

---

## Pipeline Flow

```
Student Question
       |
       v
  [Memory DB] — stores input
       |
       v
  [D2] Classifier — syllabus match? repeat?
       |
  +----+----+
  |         |
  MATCH    NO MATCH
  |         |
  v         v
 [D1]     [D2.1] Web Search
 Aggreg.    + subtopic extraction
  |         |
  v         Match?
  |       Y → topic found
  |       N → D4 Curiosity Guide
  v         |
 [D3] Output Safety Check
  |
  v
 [AI Tutor] — xAI Grok guided learning
  |
  v
 Teaches subtopics one by one
  |
 Student says "new topic" → back to question
```

## What Each Layer Does

- **D2** — Classifies your question: matches syllabus? repeated topic?
- **D2.1** — If D2 says "no match": searches the web via xAI, extracts subtopics, vector-matches against central DB + memory DB.
- **D4** — If D2.1 can't match: detects curiosity patterns, classifies queries (genuinely_novel vs off_topic).
- **D1** — If D2 says "repeated + in syllabus": aggregates topic coverage, identifies gaps.
- **D3** — Output safety gate. Checks pipeline output against policy.db. Up to 2 retries. If still unsafe → queues for mentor review.
- **AI Tutor** — xAI Grok-powered guided learning. Takes topic + subtopics from pipeline, teaches them one by one. Student stays in learning mode until they say "new topic".

---

## Available Topics (27 total)

### Mathematics (9 topics)

#### Algebra
| ID | Topic | Example Questions |
|----|-------|-------------------|
| 1  | Linear Equations | "How to solve linear equations?", "What is a system of linear equations?" |
| 2  | Quadratic Equations | "Explain the quadratic formula", "How does the discriminant affect roots?" |
| 3  | Polynomials | "What are polynomials?", "How to factor a polynomial?" |

#### Calculus
| ID | Topic | Example Questions |
|----|-------|-------------------|
| 4  | Limits | "What is a limit in calculus?", "How to evaluate limits?" |
| 5  | Derivatives | "Explain derivatives", "How to find the derivative of x^2?" |
| 6  | Integrals | "What is integration?", "How to compute a definite integral?" |

#### Geometry
| ID | Topic | Example Questions |
|----|-------|-------------------|
| 7  | Triangles | "What are the types of triangles?", "How to prove triangle congruence?" |
| 8  | Circles | "What is the equation of a circle?", "How to find arc length?" |
| 9  | Coordinate Geometry | "What is coordinate geometry?", "How to find distance between two points?" |

### Physics (9 topics)

#### Mechanics
| ID | Topic | Example Questions |
|----|-------|-------------------|
| 10 | Newton's Laws | "Explain Newton's laws of motion", "What is inertia?" |
| 11 | Work & Energy | "What is work in physics?", "Explain conservation of energy" |
| 12 | Momentum | "What is momentum?", "Explain elastic vs inelastic collision" |

#### Thermodynamics
| ID | Topic | Example Questions |
|----|-------|-------------------|
| 13 | Heat Transfer | "What are the modes of heat transfer?", "How does conduction work?" |
| 14 | Laws of Thermodynamics | "Explain the first law of thermodynamics", "What is entropy?" |
| 15 | Entropy | "What is entropy?", "How does entropy relate to disorder?" |

#### Optics
| ID | Topic | Example Questions |
|----|-------|-------------------|
| 16 | Reflection | "What is the law of reflection?", "How do mirrors work?" |
| 17 | Refraction | "What is refraction?", "Explain Snell's law" |
| 18 | Lenses | "How do convex lenses work?", "What is focal length?" |

### Computer Science (9 topics)

#### Programming Fundamentals
| ID | Topic | Example Questions |
|----|-------|-------------------|
| 19 | Variables & Data Types | "What are variables?", "Explain data types in Python" |
| 20 | Control Flow | "What is a for loop?", "How do if-else statements work?" |
| 21 | Functions | "What is a function?", "How to write a recursive function?" |

#### Data Structures
| ID | Topic | Example Questions |
|----|-------|-------------------|
| 22 | Arrays | "What is an array?", "How to reverse an array?" |
| 23 | Linked Lists | "What is a linked list?", "How to detect a cycle in a linked list?" |
| 24 | Trees & Graphs | "What is a binary tree?", "Explain BFS vs DFS" |

#### Algorithms
| ID | Topic | Example Questions |
|----|-------|-------------------|
| 25 | Sorting | "How does quicksort work?", "What is merge sort?" |
| 26 | Searching | "What is binary search?", "How does linear search work?" |
| 27 | Dynamic Programming | "What is dynamic programming?", "Explain memoization vs tabulation" |

---

## Available Mentors

| ID | Name | Subjects | Bio |
|----|------|----------|-----|
| 1  | Dr. Meera Iyer | Mathematics | Expert in algebra and calculus with 15 years experience |
| 2  | Prof. Arjun Nair | Physics | Physics researcher specialising in mechanics and thermodynamics |
| 3  | Ms. Kavya Das | Computer Science | Full-stack developer and CS educator |
| 4  | Mr. Rahul Menon | Mathematics | Mathematician with focus on geometry and number theory |
| 5  | Dr. Sneha Kapoor | Physics, Computer Science | Interdisciplinary mentor covering physics and CS |
