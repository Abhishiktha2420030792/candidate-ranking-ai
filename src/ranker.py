"""
ranker.py
---------
Hybrid candidate ranking engine for the Redrob AI challenge.

Scoring architecture (final_score):
    0.50 × semantic_similarity   (embedding cosine similarity)
    0.20 × experience_score      (YoE bucket + role/industry quality)
    0.15 × skill_score           (presence + proficiency of target skills)
    0.15 × behavioral_score      (redrob_signals: activity, responsiveness)

All component scores are normalised to [0, 1] before weighting.

Key design decisions
--------------------
1.  JD text is hand-crafted from the challenge brief to maximise semantic
    relevance without over-fitting to keyword lists.
2.  Experience scoring uses a Gaussian-like peak at 5-9 yrs.  Beyond 9 yrs
    the score decays slowly (over-qualification for a founding-team IC role).
3.  Skill score is computed over a weighted target-skill list; proficiency
    and skill duration amplify the base match signal.
4.  Behavioral score explicitly penalises: last_active > 180 days ago,
    recruiter_response_rate < 0.3, notice_period > 90 days.
5.  Semantic similarity guards against pure keyword stuffers: a Marketing
    Manager who lists "LLM fine-tuning" in skills but whose career descriptions
    talk about brand and campaigns will embed far away from the JD.
6.  github_activity_score == -1 means "not connected" — treated as neutral
    (0.5) not penalised, to avoid punishing engineers who don't use GitHub.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Job-description text (hand-crafted for best embedding signal)
# ---------------------------------------------------------------------------

JD_TEXT = """
Senior AI Engineer founding team.
Build and own retrieval systems, semantic search, ranking systems, and
recommendation engines from scratch.
Design embedding pipelines, vector search infrastructure, and LLM-powered
features.  Ship production ML systems end to end: data ingestion, model
inference, evaluation frameworks, A/B testing, and monitoring.
Strong Python engineer with hands-on experience in NLP, information retrieval,
dense retrieval, approximate nearest neighbour search, vector databases
(Faiss, Milvus, Pinecone, Weaviate), and large language models.
Experience with fine-tuning, prompt engineering, RAG architectures.
Startup experience or product-company background preferred.
5 to 9 years of relevant engineering experience building real systems.
Evaluation-driven development, rigorous experimentation, fast iteration.
"""

# ---------------------------------------------------------------------------
# Target skills with relative importance weights
# ---------------------------------------------------------------------------

TARGET_SKILLS: dict[str, float] = {
    # Core must-haves
    "python":              1.0,
    "retrieval":           1.0,
    "search":              1.0,
    "ranking":             1.0,
    "embeddings":          1.0,
    "vector":              0.9,
    "vector database":     0.9,
    "milvus":              0.9,
    "faiss":               0.9,
    "pinecone":            0.9,
    "weaviate":            0.9,
    "qdrant":              0.9,
    "llm":                 0.95,
    "llms":                0.95,
    "large language model":0.95,
    "nlp":                 0.9,
    "recommendation":      0.85,
    "evaluation":          0.8,
    "a/b testing":         0.75,
    "rag":                 0.9,
    "fine-tuning":         0.85,
    "fine-tuning llms":    0.85,
    "information retrieval":1.0,
    # Good-to-have
    "transformer":         0.7,
    "hugging face":        0.7,
    "pytorch":             0.7,
    "tensorflow":          0.65,
    "mlops":               0.65,
    "experiment tracking": 0.6,
    "weights & biases":    0.6,
    "lora":                0.7,
    "semantic search":     1.0,
    "dense retrieval":     1.0,
}

# Proficiency multipliers for skill scoring
PROFICIENCY_MULT: dict[str, float] = {
    "advanced":     1.0,
    "intermediate": 0.75,
    "beginner":     0.4,
}

# Titles that are strong positive indicators
POSITIVE_TITLE_TOKENS = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "nlp engineer", "search engineer", "ranking engineer",
    "applied scientist", "research engineer", "data scientist",
    "senior ai", "senior ml", "staff ai", "staff ml",
    "retrieval", "recommendation",
}

# Titles that are strong negative indicators (buzzword-stuffers)
NEGATIVE_TITLE_TOKENS = {
    "marketing manager", "hr manager", "operations manager",
    "content writer", "business analyst", "accountant",
    "customer support", "brand manager", "sales manager",
    "finance manager",
}

# Industries that signal product-company experience
PRODUCT_INDUSTRIES = {
    "software", "technology", "saas", "internet", "e-commerce",
    "fintech", "edtech", "healthtech", "product", "ai", "ml",
}


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_MODEL: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print("[ranker] Loading sentence-transformers model …")
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def compute_semantic_score(
    candidate_texts: list[str],
    jd_embedding: np.ndarray,
) -> list[float]:
    """
    Compute cosine similarity between each candidate text and the JD.
    Returns raw cosine similarities (not yet normalised).
    """
    model = _get_model()
    print(f"[ranker] Encoding {len(candidate_texts)} candidate profiles …")
    cand_embeddings = model.encode(
        candidate_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # jd_embedding is already L2-normalised, so dot product == cosine similarity
    similarities = (cand_embeddings @ jd_embedding).tolist()
    return similarities


def compute_jd_embedding() -> np.ndarray:
    model = _get_model()
    emb = model.encode(
        [JD_TEXT],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb[0]


# ---------------------------------------------------------------------------

def _yoe_score(years: float) -> float:
    """
    Gaussian-like peak centred at 7 years (middle of 5-9 sweet spot).
    σ = 3.5 — gentle taper outside the ideal range.
    """
    peak = 7.0
    sigma = 3.5
    return math.exp(-0.5 * ((years - peak) / sigma) ** 2)


def _title_quality(candidate: dict) -> float:
    """
    +0.25 for each positive title token found in career history.
    -0.30 for each negative title token (capped at -0.60).
    Score clamped to [0, 1].
    """
    score = 0.0
    all_titles = []
    current_title = candidate.get("profile", {}).get("current_title", "").lower()
    all_titles.append(current_title)
    for job in candidate.get("career_history", []):
        all_titles.append(job.get("title", "").lower())

    for title in all_titles:
        for pos in POSITIVE_TITLE_TOKENS:
            if pos in title:
                score += 0.25
                break
        for neg in NEGATIVE_TITLE_TOKENS:
            if neg in title:
                score -= 0.30
                break

    return max(0.0, min(1.0, score))


def _product_company_bonus(candidate: dict) -> float:
    """0.2 bonus if the candidate has spent time in product/tech companies."""
    for job in candidate.get("career_history", []):
        industry = (job.get("industry") or "").lower()
        for pi in PRODUCT_INDUSTRIES:
            if pi in industry:
                return 0.2
    return 0.0


def compute_experience_score(candidate: dict) -> float:
    """
    Combines:
    - YoE fit (Gaussian peak at 7 years)        → weight 0.5
    - Title quality (relevant vs irrelevant)     → weight 0.3
    - Product-company bonus                      → weight 0.2
    """
    years = float(candidate.get("profile", {}).get("years_of_experience") or 0)
    yoe   = _yoe_score(years)
    title = _title_quality(candidate)
    prod  = _product_company_bonus(candidate)

    return 0.5 * yoe + 0.3 * title + 0.2 * prod


# ---------------------------------------------------------------------------

def compute_skill_score(candidate: dict) -> float:
    """
    For each target skill found in the candidate's skill list:
      match_weight × proficiency_multiplier × min(duration_months / 12, 3) / 3

    Duration is capped at 3 years so that very long durations don't dominate.
    The result is normalised by the maximum possible score (all skills,
    advanced, 3+ years) so the output is always in [0, 1].
    """
    candidate_skills = {
        s["name"].lower(): s
        for s in candidate.get("skills", [])
        if s.get("name")
    }

    max_possible = sum(TARGET_SKILLS.values())  # if all skills advanced, 3 yrs
    raw_score = 0.0

    for skill_name, skill_weight in TARGET_SKILLS.items():
        # Exact match
        match = candidate_skills.get(skill_name)
        if match is None:
            # Try substring match (e.g. "vector" matches "vector database")
            for csk, cskv in candidate_skills.items():
                if skill_name in csk or csk in skill_name:
                    match = cskv
                    break
        if match is None:
            continue

        proficiency   = match.get("proficiency", "beginner")
        prof_mult     = PROFICIENCY_MULT.get(proficiency, 0.4)
        duration_yrs  = min((match.get("duration_months") or 0) / 12.0, 3.0) / 3.0

        # Duration factor: at least 0.3 even if duration is 0 (skill is listed)
        duration_factor = max(0.3, duration_yrs)
        raw_score += skill_weight * prof_mult * duration_factor

    return min(raw_score / max_possible, 1.0)


# ---------------------------------------------------------------------------

def _days_since(date_str: str | None) -> float:
    """Return number of days between date_str and today. -1 if unparseable."""
    if not date_str:
        return -1
    try:
        dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - dt).days
    except ValueError:
        return -1


def compute_behavioral_score(candidate: dict) -> float:
    """
    Combines positive engagement signals and penalises inactivity.

    Signals (positive):
    - profile_completeness_score / 100      → weight 0.20
    - open_to_work_flag (bool → 0/1)        → weight 0.15
    - recruiter_response_rate               → weight 0.20
    - github_activity_score / 100           → weight 0.20 (-1 → neutral 0.5)
    - interview_completion_rate             → weight 0.10
    - saved_by_recruiters_30d (log-scaled)  → weight 0.10
    - search_appearance_30d (log-scaled)    → weight 0.05

    Penalties (multiplicative, all in [0.5, 1.0]):
    - last_active > 180 days              → ×0.60
    - last_active 90-180 days             → ×0.80
    - recruiter_response_rate < 0.3       → ×0.80
    - notice_period > 90 days             → ×0.85
    - notice_period > 180 days            → ×0.70
    """
    sig = candidate.get("redrob_signals", {})

    # --- positive signals ---
    completeness   = (sig.get("profile_completeness_score") or 0) / 100.0
    open_to_work   = 1.0 if sig.get("open_to_work_flag") else 0.0
    response_rate  = float(sig.get("recruiter_response_rate") or 0)

    github_raw = sig.get("github_activity_score", -1)
    github     = 0.5 if github_raw == -1 else float(github_raw) / 100.0

    interview_rate = float(sig.get("interview_completion_rate") or 0)

    saved_30d       = float(sig.get("saved_by_recruiters_30d") or 0)
    saved_scaled    = math.log1p(saved_30d) / math.log1p(30)   # cap at ~30

    appearances_30d = float(sig.get("search_appearance_30d") or 0)
    appear_scaled   = math.log1p(appearances_30d) / math.log1p(500) # cap ~500

    score = (
        0.20 * completeness
        + 0.15 * open_to_work
        + 0.20 * response_rate
        + 0.20 * github
        + 0.10 * interview_rate
        + 0.10 * saved_scaled
        + 0.05 * appear_scaled
    )

    # --- penalties ---
    days_inactive = _days_since(sig.get("last_active_date"))
    if days_inactive > 180:
        score *= 0.60
    elif days_inactive > 90:
        score *= 0.80

    if response_rate < 0.3:
        score *= 0.80

    notice = int(sig.get("notice_period_days") or 0)
    if notice > 180:
        score *= 0.70
    elif notice > 90:
        score *= 0.85

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _minmax_normalise(values: list[float]) -> list[float]:
    """Min-max normalise a list to [0, 1]. Returns as-is if all equal."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


# ---------------------------------------------------------------------------
# Reasoning generator
# ---------------------------------------------------------------------------

def _build_reasoning(
    candidate: dict,
    semantic: float,
    experience: float,
    skill: float,
    behavioral: float,
    final: float,
) -> str:
    """
    Produce a concise, human-readable justification for the score.
    """
    profile = candidate.get("profile", {})
    name    = profile.get("anonymized_name", "Unknown")
    title   = profile.get("current_title", "")
    yoe     = profile.get("years_of_experience", 0)
    sig     = candidate.get("redrob_signals", {})

    parts = []

    # Semantic signal
    if semantic >= 0.75:
        parts.append("Strong semantic alignment with JD (retrieval/search/ML systems).")
    elif semantic >= 0.50:
        parts.append("Moderate semantic alignment; some relevant engineering background.")
    else:
        parts.append("Low semantic alignment; profile text diverges from JD focus areas.")

    # Experience
    if 5 <= yoe <= 9:
        parts.append(f"YoE ({yoe:.1f}) in ideal 5-9 yr sweet spot.")
    elif yoe < 5:
        parts.append(f"YoE ({yoe:.1f}) below preferred 5-9 yr range.")
    else:
        parts.append(f"YoE ({yoe:.1f}) exceeds preferred range; slight over-qualification signal.")

    # Title check
    title_l = title.lower()
    for neg in NEGATIVE_TITLE_TOKENS:
        if neg in title_l:
            parts.append(f"Current title '{title}' is non-engineering — penalised.")
            break
    for pos in POSITIVE_TITLE_TOKENS:
        if pos in title_l:
            parts.append(f"Current title '{title}' maps to target role family.")
            break

    # Skill signal
    if skill >= 0.6:
        parts.append("High skill alignment: multiple core skills (retrieval/LLMs/vector) present.")
    elif skill >= 0.3:
        parts.append("Partial skill overlap with JD requirements.")
    else:
        parts.append("Low skill overlap; few target skills listed.")

    # Behavioral
    days_inactive = _days_since(sig.get("last_active_date"))
    if days_inactive > 180:
        parts.append(f"Inactive for {days_inactive}d — significant penalty applied.")
    elif days_inactive > 90:
        parts.append(f"Moderately inactive ({days_inactive}d) — mild penalty applied.")
    else:
        parts.append(f"Recently active ({days_inactive}d ago).")

    if sig.get("open_to_work_flag"):
        parts.append("Open-to-work flag set.")

    rr = sig.get("recruiter_response_rate", 0)
    if isinstance(rr, float) and rr < 0.3:
        parts.append(f"Low recruiter response rate ({rr:.0%}) — penalised.")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Top-level ranking function
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates: list[dict[str, Any]],
    profile_texts: list[str],
) -> list[dict[str, Any]]:
    """
    Run the full hybrid ranking pipeline.

    Returns a list of result dicts sorted by final_score descending:
      { candidate_id, rank, score, reasoning }
    """
    print("[ranker] Computing JD embedding …")
    jd_emb = compute_jd_embedding()

    # --- semantic ---
    raw_semantic = compute_semantic_score(profile_texts, jd_emb)

    # --- per-candidate components ---
    exp_scores  = [compute_experience_score(c) for c in candidates]
    skill_scores = [compute_skill_score(c)     for c in candidates]
    beh_scores  = [compute_behavioral_score(c) for c in candidates]

    # --- normalise each component ---
    norm_semantic = _minmax_normalise(raw_semantic)
    norm_exp      = _minmax_normalise(exp_scores)
    norm_skill    = _minmax_normalise(skill_scores)
    norm_beh      = _minmax_normalise(beh_scores)

    # --- weighted combination ---
    final_scores = [
        0.50 * s + 0.20 * e + 0.15 * sk + 0.15 * b
        for s, e, sk, b in zip(norm_semantic, norm_exp, norm_skill, norm_beh)
    ]

    # --- build result rows ---
    results = []
    for i, candidate in enumerate(candidates):
        cid = candidate.get("candidate_id", f"CAND_{i:07d}")
        reasoning = _build_reasoning(
            candidate,
            norm_semantic[i],
            norm_exp[i],
            norm_skill[i],
            norm_beh[i],
            final_scores[i],
        )
        results.append({
            "candidate_id": cid,
            "score":        round(final_scores[i], 6),
            "reasoning":    reasoning,
        })

    # Sort descending by score
    results.sort(key=lambda r: r["score"], reverse=True)

    # Assign ranks (ties share the same rank)
    rank = 1
    for idx, row in enumerate(results):
        if idx > 0 and row["score"] < results[idx - 1]["score"]:
            rank = idx + 1
        row["rank"] = rank

    print(f"[ranker] Ranking complete. Top score: {results[0]['score']:.4f}")
    return results