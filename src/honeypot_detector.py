"""
honeypot_detector.py
--------------------
Flags candidate profiles that show signs of being synthetic honeypots.

The dataset contains ~80 honeypot profiles per the spec. Common tells:
  - Skill claimed as 'expert' with 0 endorsements AND 0 duration_months
  - Current company tenure > company's plausible age (hard to detect without
    external data, but impossible-looking durations are a signal)
  - Years of experience far exceed what education timelines allow
  - Expected salary range with min > max (data integrity error)
  - Skills with non-zero proficiency but negative or impossible values

Honeypots that land in top-100 trigger Stage 3 disqualification if rate > 10%.
We detect them conservatively — false positives (flagging real candidates) are
worse than false negatives (missing a honeypot).

Sets candidate['_honeypot_risk'] = True if suspicious.
Returns a risk score [0.0, 1.0]; anything > 0.5 is treated as likely honeypot.
"""

from __future__ import annotations

from typing import Tuple

from src.loader import get_profile, get_career, get_skills, get_signals


def detect_honeypot(candidate: dict) -> Tuple[float, str]:
    """
    Returns (risk_score: float in [0, 1], reason: str).
    risk_score > 0.5  → flag as likely honeypot.
    """
    risk = 0.0
    reasons = []

    profile = get_profile(candidate)
    career = get_career(candidate)
    skills = get_skills(candidate)
    signals = get_signals(candidate)

    # 1. Salary range min > max  (data integrity impossibility)
    sal = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = sal.get("min", 0)
    sal_max = sal.get("max", 0)
    if sal_min > 0 and sal_max > 0 and sal_min > sal_max:
        risk += 0.35
        reasons.append(f"salary min ({sal_min}) > max ({sal_max})")

    # 2. Expert skills with zero endorsements AND zero duration
    expert_zero_count = 0
    for skill in skills:
        if skill.get("proficiency") == "expert":
            if (skill.get("endorsements", 0) == 0 and
                    skill.get("duration_months", 0) == 0):
                expert_zero_count += 1

    if expert_zero_count >= 3:
        risk += 0.40
        reasons.append(f"{expert_zero_count} expert skills with 0 endorsements & 0 duration")
    elif expert_zero_count >= 1:
        risk += 0.15
        reasons.append(f"{expert_zero_count} expert skill(s) with 0 endorsements & 0 duration")

    # 3. Impossible experience: years_of_experience >>
    #    sum of career duration months / 12
    yoe = profile.get("years_of_experience", 0)
    total_career_months = sum(r.get("duration_months", 0) or 0 for r in career)
    career_years = total_career_months / 12.0

    if career and yoe > 0:
        # Allow 20% slack for gaps / rounding
        if career_years > yoe * 1.5 + 3:
            risk += 0.20
            reasons.append(
                f"Career months ({total_career_months}) >> stated YoE ({yoe})"
            )

    # 4. Career overlap: same company listed twice as current
    current_companies = [
        r.get("company", "").lower()
        for r in career
        if r.get("is_current", False)
    ]
    if len(current_companies) > len(set(current_companies)):
        risk += 0.25
        reasons.append("Duplicate current company entries")

    # 5. Implausibly high skill count with all beginner proficiency
    #    (lazy list injection)
    if len(skills) >= 15:
        all_beginner = all(s.get("proficiency") == "beginner" for s in skills)
        if all_beginner:
            risk += 0.20
            reasons.append(f"{len(skills)} skills, all 'beginner'")

    # 6. Profile completeness < 30 with very high YoE (senior ghost)
    completeness = signals.get("profile_completeness_score", 100)
    if completeness < 30 and yoe >= 8:
        risk += 0.10
        reasons.append(f"Senior candidate ({yoe}y) with {completeness:.0f}% profile completeness")

    # 7. Signup date after last_active_date (temporal impossibility)
    signup = signals.get("signup_date", "")
    last_active = signals.get("last_active_date", "")
    if signup and last_active and signup > last_active:
        risk += 0.30
        reasons.append(f"Signup date ({signup}) > last active ({last_active})")

    # Clamp and annotate
    risk = round(min(risk, 1.0), 4)
    candidate["_honeypot_risk"] = risk > 0.50

    reason_str = "; ".join(reasons) if reasons else "No honeypot signals detected."
    return risk, reason_str


def flag_honeypots(candidates: list) -> int:
    """
    Runs detect_honeypot on all candidates in-place.
    Returns count of flagged candidates.
    """
    flagged = 0
    for c in candidates:
        risk, _ = detect_honeypot(c)
        if risk > 0.50:
            flagged += 1
    return flagged


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

    print("Honeypot scan results:\n")
    flagged = 0
    for c in records:
        risk, reason = detect_honeypot(c)
        if risk > 0.20:
            print(f"  {c.get('candidate_id')} risk={risk:.2f} | {reason}")
            if risk > 0.5:
                flagged += 1
    print(f"\nFlagged as likely honeypots: {flagged}/{len(records)}")