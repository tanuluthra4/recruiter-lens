"""
rank_engine.py
--------------
Combines all scoring components into a final composite score.

Scoring weights:
    career_fit      35%   — title + trajectory + consulting penalty
    skill_relevance 25%   — endorsement-weighted BM25 skill match
    experience      15%   — years, company type, tenure stability
    location        10%   — geography, work mode, notice period
    [behavioral signals act as a multiplicative modifier, not a dimension]

Final score = (weighted_sum) × behavioral_modifier

Honeypot-flagged candidates are pushed to score 0.001 to ensure they
never appear in top-100.
"""

from __future__ import annotations

from typing import List, Tuple

from src.loader import get_profile, candidate_id
from src.jd_parser import JDRequirements, load_requirements
from src.career_analyzer import score_career
from src.skill_matcher import score_skills, assessment_bonus
from src.experience_scorer import score_experience
from src.location_scorer import score_location
from src.signals import availability_modifier
from src.honeypot_detector import detect_honeypot

# Component weights (must sum to 1.0)
WEIGHTS = {
    "career":     0.35,
    "skill":      0.25,
    "experience": 0.15,
    "location":   0.10,
    # Note: signals is a multiplier, not a weight here
    # The remaining 0.15 is absorbed by the behavioral multiplier range
}

# We leave 0.15 of "weight" for the behavioral signals to modulate.
# Base score is computed on 0.85 effective scale, then multiplied.
BASE_SCALE = 0.85


def score_candidate(
    candidate: dict,
    req: JDRequirements | None = None,
) -> Tuple[float, dict]:
    """
    Compute the final composite score for a single candidate.

    Returns
    -------
    (final_score: float in [0, 1], breakdown: dict)
    """
    if req is None:
        req = load_requirements()

    # 0. Honeypot check — push to bottom if flagged
    honeypot_risk, hp_reason = detect_honeypot(candidate)
    if candidate.get("_honeypot_risk", False):
        breakdown = {
            "career": 0.0, "skill": 0.0, "experience": 0.0,
            "location": 0.0, "behavioral_modifier": 0.0,
            "honeypot": True, "honeypot_reason": hp_reason,
            "notes": {}
        }
        return 0.001, breakdown

    # 1. Component scores
    career_score,  career_note  = score_career(candidate, req)
    skill_score,   skill_note   = score_skills(candidate, req)
    assess_bonus                = assessment_bonus(candidate, req)
    skill_score                 = min(skill_score + assess_bonus, 1.0)
    exp_score,     exp_note     = score_experience(candidate, req)
    loc_score,     loc_note     = score_location(candidate)
    beh_modifier,  beh_note     = availability_modifier(candidate)

    # 2. Weighted base score
    base = (
        career_score  * WEIGHTS["career"]
        + skill_score * WEIGHTS["skill"]
        + exp_score   * WEIGHTS["experience"]
        + loc_score   * WEIGHTS["location"]
    )

    # 3. Apply behavioral modifier
    final = base * beh_modifier

    # 4. Hard gate: non-technical title → hard cap at 0.20
    #    (career_analyzer already returns 0.10 for these, so the
    #    composite will naturally be low, but we add an explicit ceiling
    #    as a safety net)
    # ------------------------------------------------------------------
    profile = get_profile(candidate)
    current_title = profile.get("current_title", "").lower()
    non_tech_keywords = {
        "marketing", "hr ", "accountant", "graphic designer",
        "content writer", "sales", "civil engineer", "mechanical engineer",
        "operations manager", "business analyst", "customer support",
        "project manager",
    }
    if any(kw in current_title for kw in non_tech_keywords):
        final = min(final, 0.20)

    final = round(min(final, 1.0), 6)

    breakdown = {
        "career":              round(career_score, 4),
        "skill":               round(skill_score, 4),
        "experience":          round(exp_score, 4),
        "location":            round(loc_score, 4),
        "behavioral_modifier": beh_modifier,
        "honeypot_risk":       honeypot_risk,
        "honeypot":            False,
        "notes": {
            "career":     career_note,
            "skill":      skill_note,
            "experience": exp_note,
            "location":   loc_note,
            "behavioral": beh_note,
        }
    }

    return final, breakdown


def rank_candidates(
    candidates: List[dict],
    req: JDRequirements | None = None,
    top_n: int = 100,
    verbose: bool = False,
) -> List[Tuple[float, dict, dict]]:
    """
    Score and rank all candidates, return sorted list.

    Returns
    -------
    List of (score, candidate, breakdown) tuples, sorted best-first.
    Truncated to top_n.
    """
    if req is None:
        req = load_requirements()

    results = []
    for i, c in enumerate(candidates):
        if verbose and i % 5000 == 0:
            print(f"  Scoring {i}/{len(candidates)}...")
        score, breakdown = score_candidate(c, req)
        c["_score_breakdown"] = breakdown
        results.append((score, c, breakdown))

    # Sort by score descending; tie-break by candidate_id ascending (spec requirement)
    results.sort(key=lambda x: (-x[0], x[1].get("candidate_id", "")))

    return results[:top_n]