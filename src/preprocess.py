"""
preprocess.py
-------------
Builds a rich text profile for each candidate, combining all signal-bearing
fields into a single string suitable for embedding.

Design notes:
- Career history descriptions and titles carry the most semantic signal.
- Skills are included with proficiency weight; advanced/intermediate skills
  are repeated for slight TF-IDF-like boosting in embedding space.
- Certifications and education field-of-study contribute but are weighted less.
- We intentionally exclude redrob_signals (behavioral signals) from the text
  profile — those are scored separately and numerically.
"""

import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Field-level text extractors
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Strip excess whitespace from a string."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_skills_text(skills: list[dict]) -> str:
    """
    Weight skills by proficiency:
    - advanced   → repeated 3×  (strong signal)
    - intermediate → repeated 2×
    - beginner   → repeated 1×
    """
    proficiency_repeat = {"advanced": 3, "intermediate": 2, "beginner": 1}
    parts = []
    for skill in skills:
        name = _clean(skill.get("name", ""))
        if not name:
            continue
        repeat = proficiency_repeat.get(skill.get("proficiency", "beginner"), 1)
        parts.extend([name] * repeat)
    return " ".join(parts)


def _extract_career_text(career_history: list[dict]) -> str:
    """
    Combine job titles and descriptions.
    Current roles are repeated once extra to boost recency signal.
    """
    parts = []
    for job in career_history:
        title = _clean(job.get("title", ""))
        desc  = _clean(job.get("description", ""))
        is_current = job.get("is_current", False)

        if title:
            parts.append(title)
            if is_current:
                parts.append(title)     # recency boost
        if desc:
            parts.append(desc)
    return " ".join(parts)


def _extract_education_text(education: list[dict]) -> str:
    """Include degree type and field of study."""
    parts = []
    for edu in education:
        degree = _clean(edu.get("degree", ""))
        field  = _clean(edu.get("field_of_study", ""))
        if field:
            parts.append(f"{degree} {field}".strip())
    return " ".join(parts)


def _extract_certifications_text(certifications: list[dict]) -> str:
    return " ".join(_clean(c.get("name", "")) for c in certifications if c.get("name"))


# ---------------------------------------------------------------------------
# Main profile builder
# ---------------------------------------------------------------------------

def build_candidate_profile_text(candidate: dict[str, Any]) -> str:
    """
    Assemble a single text string from all semantically relevant fields.
    Returns a non-empty string (may be sparse for low-info candidates).
    """
    profile  = candidate.get("profile", {})
    headline = _clean(profile.get("headline", ""))
    summary  = _clean(profile.get("summary", ""))
    title    = _clean(profile.get("current_title", ""))

    skills_text  = _extract_skills_text(candidate.get("skills", []))
    career_text  = _extract_career_text(candidate.get("career_history", []))
    edu_text     = _extract_education_text(candidate.get("education", []))
    cert_text    = _extract_certifications_text(candidate.get("certifications", []))

    parts = [headline, title, summary, career_text, skills_text, edu_text, cert_text]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    """
    Load candidates from a JSON array file (.json) or
    a JSON-Lines file (.jsonl / .ndjson).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")

    if suffix in {".jsonl", ".ndjson"}:
        candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:                                       # assume .json array
        candidates = json.loads(raw)

    print(f"[preprocess] Loaded {len(candidates)} candidates from {path.name}")
    return candidates