# Redrob AI — Intelligent Candidate Discovery & Ranking

A hybrid AI-powered candidate ranking system built for the Redrob challenge.

---

## Project Structure

## Project Structure

```text
candidate-ranking-ai/
│
├── data/
│   ├── sample_candidates.json
│   └── candidates.jsonl
│
├── outputs/
│   └── submission.csv
│
├── src/
│   ├── main.py
│   ├── preprocess.py
│   └── ranker.py
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your data file
cp candidates.jsonl data/candidates.jsonl   # or .json array

# 3. Run the ranker
python src/main.py --candidates data/candidates.jsonl

# Output is written to outputs/submission.csv
```

---

## System Design

### Why Hybrid?

Pure semantic similarity rewards keyword stuffers.  
Pure rule-based scoring misses engineers who describe work idiomatically.  
The hybrid combination is robust to both failure modes.

### Scoring Formula

```
final_score = 0.50 × semantic_similarity
            + 0.20 × experience_score
            + 0.15 × skill_score
            + 0.15 × behavioral_score
```

All four component scores are **min-max normalised** before combining, so no single component can dominate due to different raw ranges.

---

### Component Details

#### 1. Semantic Similarity (50%)

Model: `sentence-transformers/all-MiniLM-L6-v2`

The candidate profile text is constructed from:
- Headline + current title (recency signal)
- Summary (candidate's own framing)
- Career history titles + descriptions (most signal-dense)
- Skills (proficiency-weighted repetition for embedding boost)
- Education field of study
- Certifications

The JD text is hand-crafted to emphasise **what the role actually does** — retrieval, semantic search, ranking, vector databases, production ML — rather than a list of keywords.

**Key insight**: A Marketing Manager who casually lists "LLMs" will still embed far from the JD because their career descriptions mention brand campaigns, not engineering systems. Semantic similarity catches this naturally.

#### 2. Experience Score (20%)

- YoE fit: Gaussian peak centred at **7 years** (middle of 5–9 sweet spot), σ=3.5.  Candidates above 9 years are gently down-scored (over-qualification for a founding IC role).
- Title quality: positive score for AI/ML/Search/NLP engineer titles; negative score for Marketing, HR, Operations, Content roles.
- Product-company bonus (+0.2) if any role was at a Software/SaaS/Fintech company.

#### 3. Skill Score (15%)

Weighted against a 30-item target skill list (Python, Retrieval, Ranking, Embeddings, Vector DBs, LLMs, NLP, RAG, A/B Testing, …).

Each match is amplified by:
- Proficiency multiplier: advanced=1.0, intermediate=0.75, beginner=0.4
- Duration factor: `min(duration_months / 12, 3) / 3`  (capped at 3 years)

Substring matching catches "vector database" when the target is "vector".

#### 4. Behavioral Score (15%)

Positive signals:
| Signal | Weight |
|---|---|
| profile_completeness_score | 0.20 |
| open_to_work_flag | 0.15 |
| recruiter_response_rate | 0.20 |
| github_activity_score | 0.20 |
| interview_completion_rate | 0.10 |
| saved_by_recruiters_30d (log-scaled) | 0.10 |
| search_appearance_30d (log-scaled) | 0.05 |

Penalties (multiplicative):
| Condition | Penalty |
|---|---|
| last_active > 180 days | ×0.60 |
| last_active 90–180 days | ×0.80 |
| recruiter_response_rate < 30% | ×0.80 |
| notice_period > 180 days | ×0.70 |
| notice_period 90–180 days | ×0.85 |

`github_activity_score == -1` (not connected) is treated as **neutral (0.5)** — engineers who don't share GitHub shouldn't be penalised; only confirmed low activity is penalised.

---

## Edge Cases Handled

| Edge Case | Handling |
|---|---|
| `github_activity_score == -1` | Treated as neutral 0.5 |
| `offer_acceptance_rate == -1` | Field not used in scoring |
| Missing `last_active_date` | No inactivity penalty |
| Candidate with empty career history | Scores low on semantic + experience naturally |
| Keyword stuffer (AI words, non-tech roles) | Semantic similarity filters this out |
| Very senior candidate (15+ yrs) | YoE Gaussian decays, title check applies |
| JSON array vs JSONL input | Auto-detected by file extension |

---

## Output Format

`outputs/submission.csv`:

```
candidate_id,rank,score,reasoning
CAND_0000007,1,0.7843,"Strong semantic alignment... | YoE (7.2) in ideal range..."
...
```

Ties share the same rank. Score is rounded to 6 decimal places.
