"""
signals.py
----------
Scores a candidate's behavioral availability and engagement from Redrob signals.

Acts as a multiplicative modifier on the final composite score, NOT an
additive component. A technically excellent candidate who is completely
unreachable still gets downweighted — but not replaced by someone who is
online 24/7 but has the wrong skills.

Modifier range: [0.5, 1.2]
  - 1.2  → highly engaged, recently active, fast responder, open to work
  - 1.0  → baseline — average engagement
  - 0.7  → inactive, slow responder, never completes interviews
  - 0.5  → effectively unreachable (no login in 180+ days, 0% response rate)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Tuple

from src.loader import get_signals


# Helpers

def _days_since(date_str: str | None, reference: date | None = None) -> int | None:
    """
    Returns days between date_str and reference (default: today).
    Returns None if date_str is missing/invalid.
    """
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        ref = reference or date.today()
        return (ref - d).days
    except ValueError:
        return None


# Main scorer

def availability_modifier(candidate: dict) -> Tuple[float, str]:
    """
    Returns (multiplier: float in [0.5, 1.2], reasoning_note: str).

    This is a MULTIPLIER applied to the composite score, not an
    additive dimension. Weights intentionally kept conservative —
    behavioral signals should not override strong technical fit.
    """
    sig = get_signals(candidate)
    notes = []
    modifier = 1.0

    # 1. Recency — when did they last log in?
    days_inactive = _days_since(sig.get("last_active_date"))
    if days_inactive is not None:
        if days_inactive <= 14:
            modifier += 0.08
            notes.append(f"Active {days_inactive}d ago (+)")
        elif days_inactive <= 30:
            modifier += 0.04
            notes.append(f"Active {days_inactive}d ago (~)")
        elif days_inactive <= 90:
            pass  # neutral
            notes.append(f"Active {days_inactive}d ago (neutral)")
        elif days_inactive <= 180:
            modifier -= 0.10
            notes.append(f"Inactive {days_inactive}d (-)")
        else:
            modifier -= 0.20
            notes.append(f"Inactive {days_inactive}d (--)")

    # 2. Open to work flag
    if sig.get("open_to_work_flag") is True:
        modifier += 0.06
        notes.append("Open to work (+)")
    else:
        modifier -= 0.03
        notes.append("Not marked open to work (-)")

    # 3. Recruiter response rate
    rrr = sig.get("recruiter_response_rate")
    if rrr is not None and rrr >= 0:
        if rrr >= 0.70:
            modifier += 0.06
            notes.append(f"Response rate {rrr:.0%} (+)")
        elif rrr >= 0.40:
            modifier += 0.02
        elif rrr >= 0.20:
            pass  # neutral
        else:
            modifier -= 0.08
            notes.append(f"Low response rate {rrr:.0%} (-)")

    # 4. Average response time
    avg_resp = sig.get("avg_response_time_hours")
    if avg_resp is not None and avg_resp >= 0:
        if avg_resp <= 24:
            modifier += 0.04
        elif avg_resp <= 72:
            pass  # neutral
        elif avg_resp > 168:   # > 1 week
            modifier -= 0.05
            notes.append(f"Slow responder ({avg_resp:.0f}h avg) (-)")

    # 5. Interview completion rate
    icr = sig.get("interview_completion_rate")
    if icr is not None:
        if icr >= 0.80:
            modifier += 0.04
        elif icr >= 0.60:
            pass
        elif icr < 0.40:
            modifier -= 0.06
            notes.append(f"Low interview completion {icr:.0%} (-)")

    # 6. GitHub activity (proxy for active engineering work)
    github = sig.get("github_activity_score", -1)
    if github and github > 0:
        if github >= 30:
            modifier += 0.05
            notes.append(f"GitHub activity {github:.0f} (+)")
        elif github >= 10:
            modifier += 0.02

    # 7. Profile completeness
    completeness = sig.get("profile_completeness_score", 0)
    if completeness >= 80:
        modifier += 0.03
    elif completeness < 35:
        modifier -= 0.04

    # 8. Verification bonus (low weight — just a hygiene check)
    verified = (
        int(sig.get("verified_email", False))
        + int(sig.get("verified_phone", False))
    )
    modifier += verified * 0.01

    # 9. Notice period — JD prefers < 30 days, can buy out 30
    notice = sig.get("notice_period_days", 90)
    if notice <= 30:
        modifier += 0.03
        notes.append(f"Notice {notice}d (+)")
    elif notice <= 60:
        pass
    elif notice > 90:
        modifier -= 0.04
        notes.append(f"Notice {notice}d (-)")

    # Clamp
    modifier = round(max(0.50, min(1.20, modifier)), 4)
    note = "; ".join(notes) if notes else "Average engagement signals."

    return modifier, note


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

    results = []
    for c in records:
        mod, note = availability_modifier(c)
        results.append((mod, c.get("candidate_id"), c.get("profile", {}).get("current_title"), note))
    results.sort(reverse=True)
    print("Top 10 by availability modifier:\n")
    for mod, cid, title, note in results[:10]:
        print(f"{mod:.3f}  {cid}  [{title}]")
        print(f"        {note}\n")