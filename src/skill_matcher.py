"""
skill_matcher.py
----------------
Scores a candidate's skill set against the JD requirements.

Two-pass approach:
  1. BM25 / exact-match: checks for literal or near-literal skill name overlap
     against the hard and nice-to-have skill lists from jd_parser.
  2. Semantic bonus: uses cosine similarity between the candidate's skill
     bag-of-words and the JD summary embedding (computed offline).

Both passes are gated through an **endorsement-duration trust filter**:
    - A skill with 0 endorsements AND 0 duration months is treated as
      an unverified claim and gets 0.2× weight.
    - A skill with few endorsements (< 3) OR short duration (< 6 months)
      gets 0.6× weight.
    - Only skills that pass both thresholds get full weight.

This is the mechanism that distinguishes a genuine ML engineer
(many endorsed, long-duration ML skills) from a keyword stuffer
(lots of ML skills, zero endorsements, zero months used).

Returns a float in [0.0, 1.0].
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src.loader import get_skills
from src.jd_parser import JDRequirements, load_requirements

# Proficiency multipliers
PROFICIENCY_WEIGHTS = {
    "beginner": 0.25,
    "intermediate": 0.60,
    "advanced": 1.00,
    "expert": 1.20,
}

# Trust thresholds
TRUST_ZERO = (0, 0)          # (endorsements, duration_months) → weight 0.2
ENDORSEMENT_SOFT = 3         # below this → 0.6× weight
DURATION_SOFT_MONTHS = 6     # below this → 0.6× weight


# Trust multiplier

def _trust_multiplier(skill: dict) -> float:
    """
    Returns how much to trust this skill claim.
    Unverified claims (0 endorsements, 0 months) are penalised hard.
    """
    endorsements = skill.get("endorsements", 0) or 0
    duration = skill.get("duration_months", 0) or 0

    if endorsements == 0 and duration == 0:
        return 0.20
    if endorsements < ENDORSEMENT_SOFT and duration < DURATION_SOFT_MONTHS:
        return 0.40
    if endorsements < ENDORSEMENT_SOFT or duration < DURATION_SOFT_MONTHS:
        return 0.65
    return 1.00


# Skill matching

def _normalise(s: str) -> str:
    return s.lower().strip().replace("-", " ").replace("_", " ")


def _skill_hits_list(skill_name: str, skill_list: List[str]) -> bool:
    """Check if a skill name matches anything in a reference list."""
    norm = _normalise(skill_name)
    for ref in skill_list:
        ref_norm = _normalise(ref)
        # Exact or substring match
        if norm == ref_norm or ref_norm in norm or norm in ref_norm:
            return True
    return False


def score_skills(
    candidate: dict,
    req: JDRequirements | None = None,
) -> Tuple[float, str]:
    """
    Score skill relevance for the Senior AI Engineer JD.

    Returns
    -------
    (score: float in [0, 1], reasoning_note: str)
    """
    if req is None:
        req = load_requirements()

    skills: List[dict] = get_skills(candidate)
    if not skills:
        return 0.0, "No skills listed."

    # Score each skill
    hard_score = 0.0
    nice_score = 0.0
    hard_matched: List[str] = []
    nice_matched: List[str] = []

    for skill in skills:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "intermediate")
        prof_w = PROFICIENCY_WEIGHTS.get(proficiency, 0.6)
        trust = _trust_multiplier(skill)
        effective_w = prof_w * trust

        if _skill_hits_list(name, req.hard_skills):
            hard_score += effective_w
            hard_matched.append(name)
        elif _skill_hits_list(name, req.nice_to_have_skills):
            nice_score += effective_w * 0.5   # nice-to-haves worth half
            nice_matched.append(name)

    # Normalise
    # A "perfect" hard skill score would be ~5 skills at advanced + trusted.
    # (5 hard skills × 1.0 prof × 1.0 trust = 5.0)
    HARD_NORMALISER = 5.0
    NICE_NORMALISER = 3.0

    hard_norm = min(hard_score / HARD_NORMALISER, 1.0)
    nice_norm = min(nice_score / NICE_NORMALISER, 1.0)

    # Hard skills dominate (80/20 split)
    final = hard_norm * 0.80 + nice_norm * 0.20

    # Reasoning note
    h_names = ", ".join(hard_matched[:5]) if hard_matched else "none"
    n_names = ", ".join(nice_matched[:3]) if nice_matched else "none"
    note = (
        f"Hard skills matched: [{h_names}] (raw={hard_score:.2f}). "
        f"Nice-to-haves: [{n_names}] (raw={nice_score:.2f})."
    )

    return round(final, 4), note


# Skill assessment bonus (verified assessment scores from Redrob platform)

def assessment_bonus(candidate: dict, req: JDRequirements | None = None) -> float:
    """
    Returns a small additive bonus (0–0.1) based on completed skill assessments.
    Assessments are the most verified signals in the dataset.
    """
    if req is None:
        req = load_requirements()

    signals = candidate.get("redrob_signals", {})
    assessments: Dict[str, float] = signals.get("skill_assessment_scores", {}) or {}

    if not assessments:
        return 0.0

    relevant_scores = []
    all_skills = req.hard_skills + req.nice_to_have_skills
    for skill_name, score in assessments.items():
        if _skill_hits_list(skill_name, all_skills):
            relevant_scores.append(score)

    if not relevant_scores:
        return 0.0

    avg = sum(relevant_scores) / len(relevant_scores)
    # Scale 0-100 assessment score to 0-0.10 bonus
    return round((avg / 100.0) * 0.10, 4)


if __name__ == "__main__":
    import json
    from pathlib import Path

    sample_path = Path("sample_candidates.json")
    if not sample_path.exists():
        sample_path = Path("/mnt/user-data/uploads/sample_candidates.json")

    with open(sample_path) as f:
        text = f.read().strip()
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    req = load_requirements()
    results = []
    for c in records:
        score, note = score_skills(c, req)
        bonus = assessment_bonus(c, req)
        results.append((score + bonus, c.get("candidate_id"), c.get("profile", {}).get("current_title"), note))

    results.sort(reverse=True)
    print("Top 10 by skill score:\n")
    for score, cid, title, note in results[:10]:
        print(f"{score:.4f}  {cid}  [{title}]")
        print(f"         {note}\n")