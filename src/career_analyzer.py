"""
career_analyzer.py
------------------
Scores a candidate's career trajectory for fit with the Senior AI Engineer JD.

This is the MOST IMPORTANT component in the ranker. The JD explicitly warns
that keyword-stuffing is a trap: a Marketing Manager with "embeddings" in
their skills section is not a fit. This module catches that.

Scoring logic
-------------
1. Current title check  — is the current role technical / ML-adjacent?
2. Career trajectory    — what fraction of their career is in technical roles?
3. Product vs consulting — did they spend most of their career at product
                           companies, or services firms?
4. ML depth signal      — do any roles explicitly describe ML/retrieval/AI work?
5. Disqualifier gates   — pure consulting career or non-technical career → hard cap.

Returns a float in [0.0, 1.0].
"""

from __future__ import annotations

import re
from typing import List, Tuple

from src.loader import get_profile, get_career
from src.jd_parser import JDRequirements, load_requirements

# Title classification helpers

# Titles that map cleanly to "technical ML/AI/data/software engineering"
TECHNICAL_TITLE_KEYWORDS = {
    "ml engineer", "machine learning engineer", "ai engineer", "data scientist",
    "nlp engineer", "applied scientist", "research engineer", "applied ml",
    "search engineer", "recommendation", "data engineer", "software engineer",
    "backend engineer", "full stack", "fullstack", "platform engineer",
    "devops", "cloud engineer", "sre", "infrastructure engineer",
    "computer vision", "deep learning engineer",
    # Slightly adjacent but acceptable
    "data analyst", "analytics engineer", "bi engineer",
    "frontend engineer",  # acceptable, not ideal
    "java developer", ".net developer", "mobile developer",
    "qa engineer",        # borderline
}

# Titles that are non-technical and are strong negative signals
NON_TECHNICAL_TITLE_KEYWORDS = {
    "marketing", "hr ", "human resource", "accountant", "accounting",
    "graphic designer", "content writer", "sales", "civil engineer",
    "mechanical engineer", "operations manager", "business analyst",
    "customer support", "project manager", "project management",
    "supply chain", "procurement",
}

# ML-specific role keywords found in job descriptions (career_history.description)
ML_DESCRIPTION_KEYWORDS = [
    r"embed", r"retriev", r"vector", r"nlp", r"machine learning", r"ranking",
    r"recommendation", r"search", r"transformer", r"llm", r"language model",
    r"fine.tun", r"faiss", r"bm25", r"semantic", r"neural", r"deep learning",
    r"pytorch", r"tensorflow", r"sklearn", r"hugging face",
    r"a/b test", r"ndcg", r"mrr", r"evaluation",
    r"feature engineer", r"model deploy", r"mlops", r"pipeline",
]
_ML_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ML_DESCRIPTION_KEYWORDS]


def _classify_title(title: str) -> str:
    """
    Returns 'technical', 'non_technical', or 'ambiguous'.
    """
    t = title.lower()
    if any(kw in t for kw in NON_TECHNICAL_TITLE_KEYWORDS):
        return "non_technical"
    if any(kw in t for kw in TECHNICAL_TITLE_KEYWORDS):
        return "technical"
    return "ambiguous"


def _is_consulting_company(company: str, industry: str, req: JDRequirements) -> bool:
    c = company.lower().strip()
    i = industry.lower().strip()
    return (
        c in req.consulting_companies
        or i in req.consulting_industries
    )


def _ml_depth_in_description(description: str) -> float:
    """
    Returns a 0–1 score for how much ML/retrieval work is described in a role.
    """
    if not description:
        return 0.0
    hits = sum(1 for p in _ML_PATTERNS if p.search(description))
    # Normalise: 5+ keyword hits → 1.0
    return min(hits / 5.0, 1.0)

# Main scorer

def score_career(candidate: dict, req: JDRequirements | None = None) -> Tuple[float, str]:
    """
    Score career fit for the Senior AI Engineer JD.

    Parameters
    ----------
    candidate : dict — a single candidate record (from loader)
    req       : JDRequirements (optional, loaded automatically if None)

    Returns
    -------
    (score: float in [0, 1], reasoning_note: str)
    """
    if req is None:
        req = load_requirements()

    profile = get_profile(candidate)
    career = get_career(candidate)

    current_title = profile.get("current_title", "").lower()
    current_company = profile.get("current_company", "").lower()
    current_industry = profile.get("current_industry", "").lower()

    # 1. Current title classification
    title_class = _classify_title(current_title)

    # Hard gate: if current title is clearly non-technical, cap at 0.15.
    # (A Marketing Manager who recently took an AI course is not this hire.)
    if title_class == "non_technical":
        return 0.10, f"Non-technical current title: '{profile.get('current_title', '')}'"

    # 2. Career trajectory analysis
    total_months = 0
    technical_months = 0
    ml_months = 0
    consulting_months = 0
    product_months = 0

    role_ml_depths: List[float] = []

    for role in career:
        dur = role.get("duration_months", 0) or 0
        title = role.get("title", "").lower()
        company = role.get("company", "").lower()
        industry = role.get("industry", "").lower()
        description = role.get("description", "")

        total_months += dur
        t_class = _classify_title(title)

        if t_class == "technical":
            technical_months += dur
        elif t_class == "ambiguous":
            technical_months += dur * 0.5  # partial credit

        ml_depth = _ml_depth_in_description(description)
        role_ml_depths.append(ml_depth)
        ml_months += dur * ml_depth

        if _is_consulting_company(company, industry, req):
            consulting_months += dur
        else:
            if t_class in ("technical", "ambiguous"):
                product_months += dur

    if total_months == 0:
        return 0.20, "No career history found."

    technical_fraction = technical_months / total_months
    consulting_fraction = consulting_months / total_months
    avg_ml_depth = (sum(role_ml_depths) / len(role_ml_depths)) if role_ml_depths else 0.0
    ml_fraction = ml_months / total_months

    # 3. Consulting-only penalty
    # The JD says: "people who have ONLY worked at consulting firms" → disqualify.
    # We apply a hard cap if consulting is > 80% of career.
    if consulting_fraction > 0.80:
        return 0.15, (
            f"Consulting-heavy career ({consulting_fraction:.0%} at services firms). "
            f"JD explicitly excludes candidates with no product-company experience."
        )

    # 4. Compose score
    # Start from technical fraction (0–0.6 of the score)
    base = technical_fraction * 0.6

    # ML depth bonus (0–0.3)
    base += ml_fraction * 0.3

    # Title bonus/penalty
    if title_class == "technical":
        base += 0.10
    elif title_class == "ambiguous":
        base += 0.04

    # Consulting penalty (even if not career-wide, heavy consulting presence is bad)
    if consulting_fraction > 0.50:
        base *= 0.70
    elif consulting_fraction > 0.30:
        base *= 0.85

    # 5. Build reasoning note
    note = (
        f"Tech trajectory: {technical_fraction:.0%} technical roles, "
        f"{ml_fraction:.0%} ML/AI-related work. "
        f"Consulting exposure: {consulting_fraction:.0%}. "
        f"Avg ML depth in descriptions: {avg_ml_depth:.2f}."
    )

    return min(base, 1.0), note


if __name__ == "__main__":
    import json, sys
    from pathlib import Path

    # Quick smoke test on the sample candidates
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
        score, note = score_career(c, req)
        results.append((score, c.get("candidate_id"), c.get("profile", {}).get("current_title"), note))

    results.sort(reverse=True)
    for score, cid, title, note in results[:10]:
        print(f"{score:.3f}  {cid}  [{title}]")
        print(f"       {note}")