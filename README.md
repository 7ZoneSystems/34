# AI Agent Pipeline — Mock Environment

## Setup

1. Add your Groq API key to `api.env`:
   ```
   GROQ_API_KEY=gsk_your_actual_key_here
   GROK_MODEL=llama-3.3-70b-versatile
   ```
   Without the key, D4 runs in fallback mode (no LLM calls — uses heuristics).

2. Install dependencies (already done if you followed setup):
   ```bash
   .env/bin/pip install sentence-transformers groq python-dotenv requests
   ```

## How to Run

```bash
.env/bin/python pipeline.py
```

You act as a **mock student**. The pipeline will:
1. Show registered students
2. Ask you for student_id, date, and a question
3. Run D2 → D2.1 (if off-syllabus) → D4 (if novel redirect) → D1 (if repeated) → Mentor routing
4. When mentors are called, you answer yes/no from the terminal

Type `quit` to exit.

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

## Available Topics (27 total)

These are the topics in the syllabus. Ask questions about them and D2 will match them.

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

---

## Pipeline Flow

```
Student Question
       |
       v
  [D2] Classifier
  Is it in the syllabus? Is it a repeat?
       |
  +---------+---------+---------+
  |         |         |         |
YES+NEW   YES+REPEAT  NO+NEW   NO+REPEAT
  |         |         |         |
  v         v         v         v
Mentor    [D1]      [D2.1]    [D2.1]
Router    Aggreg.   WebSearch  WebSearch
  |         |         |         |
  v         v         v         v
  |       Mentor    Match?    Match?
  |       Router    Y->Mentor  Y->Mentor
  |         |       N->D4      N->D4
  v         v         v         v
 [Session Memory DB — all results stored]
                         |
                         v
                    [D4] Curiosity Guide
                    (Groq LLM powered)
                    - Curiosity pattern?
                    - Novel query classification
                    - Mock web search
                    - Guide mode output
```

## What Each Layer Does

- **D2** — Classifies your question: matches syllabus? repeated topic?
- **D2.1** — If D2 says "no match": searches the web, tries to re-match to syllabus, or redirects to D4
- **D4** — If D2.1 can't match: uses Groq LLM to detect curiosity patterns, validate novel ideas, classify queries (genuinely_novel vs off_topic), and guide the student with enrichment material
- **D1** — If D2 says "repeated + in syllabus": aggregates topic coverage, identifies gaps, generates reasoning packet
- **Mentor Router** — Routes to relevant mentor(s) for the matched subject, calls mock_mentor.py (you answer yes/no)

---

## Questions That Trigger Each Path

### MATCHES_SYLLABUS (D2 says yes, first time)
Ask a question about any topic above for the first time.
- Example: "How to solve linear equations?" → matches "Algebra > Linear Equations"
- Pipeline: D2 -> Mentor Router (2 mentors called for Mathematics)

### MATCHES_SYLLABUS_REPEAT (D2 says yes, repeated)
Ask a similar question to one you already asked.
- Example: First ask "What are quadratic equations?", then ask "How to solve quadratic equations again?"
- Pipeline: D2 -> D1 (coverage analysis) -> Mentor Router (with D1 context)

### NO_MATCH_NEW (D2 says no, brand new)
Ask about something not in the syllabus at all.
- Example: "What is quantum entanglement?"
- Pipeline: D2 -> D2.1 (web search + syllabus re-match)
- If D2.1 can't match: D2.1 -> D4 (curiosity detection, classification, guidance)
  - D4 classifies as `genuinely_novel` -> mock web search + enrichment output
  - D4 classifies as `off_topic` -> polite redirect to focus on syllabus

### NO_MATCH_REPEAT (D2 says no, but similar to past)
Ask about something off-syllabus that's similar to a previous off-syllabus question.
- Pipeline: D2 -> D2.1 (web search + syllabus re-match)
- If D2.1 can't match: D2.1 -> D4 (curiosity pattern detection)
  - If 3+ past off-syllabus questions detected -> D4 enters Guide Mode
  - Guide Mode: validates ideas, runs mock web search, returns enrichment material
