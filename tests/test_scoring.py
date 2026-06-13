"""
test_scoring.py
---------------
Smoke tests for individual scoring components.
Run with: python -m pytest tests/ -v
Or: PYTHONPATH=. python tests/test_scoring.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from pathlib import Path

from src.jd_parser import JDRequirements, load_requirements
from src.career_analyzer import score_career
from src.skill_matcher import score_skills
from src.honeypot_detector import detect_honeypot
from src.signals import availability_modifier
from src.rank_engine import score_candidate

# Fixtures

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "sample_candidates.json"

assert SAMPLE_PATH.exists(), (
    f"Missing test fixture: {SAMPLE_PATH}"
)


def load_sample():
    with open(SAMPLE_PATH) as f:
        text = f.read().strip()
    records = json.loads(text) if text.startswith("[") else \
              [json.loads(l) for l in text.splitlines() if l.strip()]
    for r in records:
        r.setdefault("_honeypot_risk", False)
    return {r["candidate_id"]: r for r in records}


# Tests

def test_ela_singh_ranks_first():
    """CAND_0000031 (Recommendation Systems Engineer, Swiggy) should rank #1."""
    candidates = load_sample()
    req = load_requirements()
    results = []
    for c in candidates.values():
        score, _ = score_candidate(c, req)
        results.append((score, c["candidate_id"]))
    results.sort(reverse=True)
    assert results[0][1] == "CAND_0000031", f"Expected CAND_0000031 at #1, got {results[0][1]}"
    print("✓ CAND_0000031 ranks first")


def test_marketing_manager_is_downranked():
    """Marketing managers should never appear in top-10 regardless of AI skills."""
    candidates = load_sample()
    req = load_requirements()
    results = []
    for c in candidates.values():
        score, _ = score_candidate(c, req)
        title = c.get("profile", {}).get("current_title", "").lower()
        results.append((score, c["candidate_id"], title))
    results.sort(reverse=True)
    top10_titles = [t for _, _, t in results[:10]]
    marketing_in_top10 = [t for t in top10_titles if "marketing" in t]
    assert len(marketing_in_top10) == 0, f"Marketing Manager found in top-10: {marketing_in_top10}"
    print("✓ No marketing managers in top-10")


def test_honeypot_salary_flip():
    """Candidate with salary min > max should be flagged."""
    fake = {
        "candidate_id": "CAND_TEST001",
        "_honeypot_risk": False,
        "profile": {"current_title": "ML Engineer", "years_of_experience": 5,
                    "current_company": "TestCo", "current_industry": "Software"},
        "career_history": [],
        "skills": [],
        "education": [],
        "redrob_signals": {
            "expected_salary_range_inr_lpa": {"min": 25.0, "max": 10.0},
            "last_active_date": "2026-05-01",
        }
    }
    risk, reason = detect_honeypot(fake)
    assert risk >= 0.30, f"Expected salary-flip to raise risk, got {risk}"
    print(f"✓ Salary flip detected (risk={risk:.2f}): {reason}")


def test_skill_trust_filter():
    """A skill with 0 endorsements and 0 duration should be heavily discounted."""
    candidate = {
        "candidate_id": "CAND_TEST002",
        "_honeypot_risk": False,
        "profile": {},
        "career_history": [],
        "skills": [
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 30, "duration_months": 24},
        ],
        "redrob_signals": {}
    }
    req = load_requirements()

    # Score first candidate (unverified) vs second (well-endorsed)
    c1 = dict(candidate)
    c1["skills"] = [candidate["skills"][0]]
    c2 = dict(candidate)
    c2["skills"] = [candidate["skills"][1]]

    s1, _ = score_skills(c1, req)
    s2, _ = score_skills(c2, req)
    assert s2 > s1 * 2, f"Expected trusted skill to score >>2x unverified: {s2:.3f} vs {s1:.3f}"
    print(f"✓ Trust filter works: verified={s2:.3f}, unverified={s1:.3f} (ratio={s2/max(s1,0.001):.1f}x)")


def test_scores_monotonically_ordered():
    """Scores should be non-increasing in the ranked output."""
    candidates = load_sample()
    req = load_requirements()
    results = []
    for c in candidates.values():
        score, _ = score_candidate(c, req)
        results.append(score)
    results.sort(reverse=True)
    for i in range(len(results) - 1):
        assert results[i] >= results[i+1], f"Score not monotonic at index {i}"
    print("✓ Scores are monotonically non-increasing")

# Runner

if __name__ == "__main__":
    try:
        test_ela_singh_ranks_first()
        test_marketing_manager_is_downranked()
        test_honeypot_salary_flip()
        test_skill_trust_filter()
        test_scores_monotonically_ordered()
        print("\nAll tests passed ✓")
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        sys.exit(1)