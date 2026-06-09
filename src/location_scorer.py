"""
location_scorer.py
------------------
Scores location and logistics fit for the Senior AI Engineer JD.

JD says:
  - Preferred: Pune, Noida. Also: Hyderabad, Bangalore, Mumbai, Delhi NCR.
  - Open to relocation candidates from Tier-1 Indian cities.
  - Outside India: case-by-case, no visa sponsorship.
  - Notice period: ideally < 30d, can buy out 30d. 30+ days raises bar.
"""

from __future__ import annotations

from typing import Tuple

from src.loader import get_profile, get_signals


# City → score mapping (India-specific; based on JD language)
LOCATION_SCORES = {
    # Explicitly preferred
    "pune":       1.00,
    "noida":      1.00,
    # Also welcome per JD
    "hyderabad":  0.95,
    "bangalore":  0.90,
    "bengaluru":  0.90,
    "mumbai":     0.85,
    "delhi":      0.85,
    "gurgaon":    0.85,
    # Other Indian cities — case-by-case
    "chandigarh": 0.70,
    "chennai":    0.70,
    "kolkata":    0.70,
    "ahmedabad":  0.70,
    "trivandrum": 0.65,
    "kochi":      0.65,
    "coimbatore": 0.65,
    "indore":     0.65,
    "bhubaneswar":0.60,
    "vizag":      0.60,
}

# Country-level fallback
COUNTRY_SCORES = {
    "india": 0.75,    # Indian city not in our map but still India
    "uk":    0.40,    # International, but Tier-1 tech hub
    "usa":   0.35,
    "uae":   0.30,
    "germany": 0.30,
    "australia": 0.25,
    "singapore": 0.40,
    "canada": 0.30,
}


def score_location(candidate: dict) -> Tuple[float, str]:
    """
    Returns (score: float in [0, 1], reasoning_note: str).
    """
    profile = get_profile(candidate)
    signals = get_signals(candidate)

    location_raw = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    # 1. Base location score
    base_loc = 0.0
    matched_city = None
    for city, score in LOCATION_SCORES.items():
        if city in location_raw:
            base_loc = score
            matched_city = city
            break

    if base_loc == 0.0:
        # Fallback to country
        base_loc = COUNTRY_SCORES.get(country, 0.20)

    # 2. Relocation bonus
    reloc_bonus = 0.0
    if willing_to_relocate and base_loc < 0.85:
        reloc_bonus = 0.15  # willing to move to Pune/Noida

    # 3. Work mode fit
    work_mode = signals.get("preferred_work_mode", "flexible")
    # JD says "hybrid — flexible cadence"
    work_mode_score = {
        "hybrid":   1.00,
        "flexible": 0.90,
        "onsite":   0.85,
        "remote":   0.55,   # JD is not remote-first
    }.get(work_mode, 0.80)

    # 4. Notice period
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        notice_score = 1.00
    elif notice <= 60:
        notice_score = 0.80
    elif notice <= 90:
        notice_score = 0.60
    else:
        notice_score = 0.35   # > 90 days is a soft disqualifier

    # 5. Compose
    loc_combined = min(base_loc + reloc_bonus, 1.0)
    final = (
        loc_combined  * 0.50
        + work_mode_score * 0.25
        + notice_score    * 0.25
    )

    note = (
        f"Location: {profile.get('location', 'unknown')} "
        f"(score={loc_combined:.2f}, relocate={willing_to_relocate}). "
        f"Work mode: {work_mode} ({work_mode_score:.2f}). "
        f"Notice: {notice}d ({notice_score:.2f})."
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

    results = [(score_location(c)[0], c.get("candidate_id"),
                c.get("profile", {}).get("location")) for c in records]
    results.sort(reverse=True)
    for s, cid, loc in results[:8]:
        print(f"{s:.4f}  {cid}  [{loc}]")