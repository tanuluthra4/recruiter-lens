"""
experience_scorer.py
--------------------
Scores a candidate's experience based on:
  - Total years (ideal: 5–9 per JD)
  - Company type quality (product companies > consulting/services)
  - Career progression (are they growing, stagnant, or declining?)
  - Relevance of industries worked in
"""

from __future__ import annotations

from typing import Tuple

from src.loader import get_profile, get_career
from src.jd_parser import JDRequirements, load_requirements


# High-signal product / tech companies (not exhaustive — used as a quality boost)
PRODUCT_COMPANY_SIGNALS = {
    "google", "meta", "amazon", "microsoft", "apple", "netflix", "uber",
    "flipkart", "zomato", "swiggy", "ola", "cred", "razorpay", "phonepe",
    "meesho", "groww", "zepto", "blinkit", "dunzo", "nykaa",
    "atlassian", "adobe", "salesforce", "stripe", "shopify",
    "mad street den", "sharechat", "slice", "jupiter",
}

PREFERRED_INDUSTRIES = {
    "ai/ml", "fintech", "e-commerce", "food delivery", "transportation",
    "software", "saas",
}


def score_experience(
    candidate: dict,
    req: JDRequirements | None = None,
) -> Tuple[float, str]:
    """
    Returns (score: float in [0, 1], reasoning_note: str).
    """
    if req is None:
        req = load_requirements()

    profile = get_profile(candidate)
    career = get_career(candidate)

    yoe = profile.get("years_of_experience", 0) or 0

    # 1. Years of experience score (bell curve around 5-9)
    if yoe < req.exp_min:
        yoe_score = max(0.0, yoe / req.exp_min) * 0.5   # Under 4y → low
    elif yoe <= req.exp_ideal_max:
        # Linear scale from soft_min to ideal: 0.7 → 1.0
        yoe_score = 0.70 + (yoe - req.exp_soft_min) / (req.exp_ideal_max - req.exp_soft_min) * 0.30
        yoe_score = min(yoe_score, 1.0)
    elif yoe <= req.exp_soft_max:
        # Slight decay above ideal max
        yoe_score = 1.0 - (yoe - req.exp_ideal_max) / (req.exp_soft_max - req.exp_ideal_max) * 0.15
    else:
        # Well above range — harder to place, less ideal
        yoe_score = 0.75

    # Clamp
    yoe_score = max(0.0, min(1.0, yoe_score))

    # 2. Company quality signal
    total_months = sum(r.get("duration_months", 0) or 0 for r in career)
    product_company_months = 0

    for role in career:
        company = role.get("company", "").lower()
        industry = role.get("industry", "").lower()
        dur = role.get("duration_months", 0) or 0

        is_product = (
            company in PRODUCT_COMPANY_SIGNALS
            or any(ind in industry for ind in PREFERRED_INDUSTRIES)
        )
        if is_product:
            product_company_months += dur

    product_fraction = product_company_months / total_months if total_months > 0 else 0.0
    company_quality_score = min(product_fraction * 1.2, 1.0)  # boost slight minority

    # 3. Tenure stability signal
    #    JD wants 3+ year commitment; job-hoppers (avg < 18 months) are penalised
    avg_tenure = total_months / len(career) if career else 0
    if avg_tenure >= 24:
        tenure_score = 1.0
    elif avg_tenure >= 18:
        tenure_score = 0.85
    elif avg_tenure >= 12:
        tenure_score = 0.70
    else:
        tenure_score = 0.50   # Job-hopper signal

    # 4. Compose
    final = (
        yoe_score * 0.50
        + company_quality_score * 0.30
        + tenure_score * 0.20
    )

    note = (
        f"YoE: {yoe:.1f} (score={yoe_score:.2f}). "
        f"Product company months: {product_company_months}/{total_months} "
        f"({product_fraction:.0%}). "
        f"Avg tenure: {avg_tenure:.0f} months."
    )

    return round(final, 4), note


if __name__ == "__main__":
    import json
    from pathlib import Path

    sample_path = Path("sample_candidates.json")
    if not sample_path.exists():
        sample_path = Path("/mnt/user-data/uploads/sample_candidates.json")

    with open(sample_path) as f:
        text = f.read().strip()
    records = json.loads(text) if text.startswith("[") else \
              [json.loads(l) for l in text.splitlines() if l.strip()]

    req = load_requirements()
    results = []
    for c in records:
        s, note = score_experience(c, req)
        results.append((s, c.get("candidate_id"), c.get("profile", {}).get("current_title"), note))
    results.sort(reverse=True)
    for s, cid, title, note in results[:8]:
        print(f"{s:.4f}  {cid}  [{title}]")
        print(f"         {note}\n")