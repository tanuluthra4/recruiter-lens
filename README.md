# recruiter-lens

> AI-powered candidate ranking for the Redrob Hackathon — *India Runs*.

Recruiters don't miss the right person because the talent isn't there. They miss them because keyword filters can't see what actually matters. `recruiter-lens` ranks candidates the way a great recruiter would — by understanding career trajectory, skill depth, and genuine availability — not by counting keywords.

---

## Architecture

Five scoring dimensions, each independently computed and combined with explicit weights:

| Dimension | Weight | What it measures |
|---|---|---|
| Career & Title Fit | 35% | Is this person actually an ML/AI engineer at a product company? |
| Skill Relevance | 25% | BM25 skill match with endorsement-duration trust filter |
| Experience Quality | 15% | Years, company type (product vs. consulting), tenure stability |
| Behavioral Availability | multiplier ±20% | Recency, response rate, notice period, open-to-work |
| Location & Logistics | 10% | India cities, relocation willingness, work mode |

**Anti-trap logic** (per the JD's explicit warning):
- Candidates with non-technical career histories (Marketing, HR, Accounting) are hard-capped at 0.20 regardless of skill listings
- Skills with zero endorsements AND zero usage months are treated as unverified (0.2× weight)
- Honeypot profiles (salary min > max, expert proficiency + 0 endorsements/duration, temporal impossibilities) are flagged and pushed to rank last
- Consulting-heavy careers (>80% at IT services firms) are hard-capped regardless of current title

**Compute budget:** Processes 100K candidates in ~2 minutes on an 8-core CPU with 16GB RAM. No GPU and no external API calls during ranking.

---

## Benchmarks

Tested on:
- Windows 11 consumer laptop
- 8 CPU cores
- 16GB RAM
- CPU-only execution

Results on 100K candidates:
- Dataset loading: 13.9s
- Candidate scoring: 60.4s
- End-to-end runtime: 75.3s
- Honeypot profiles detected: 1,470

---

## System Guarantees

- Deterministic rankings
- CPU-only execution
- Offline operation (no network/API calls)
- Explainable component-wise scoring
- Reproducible outputs for identical inputs
- No LLM inference during ranking

---

## Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Rank candidates (produces team_recruiter-lens CSV)
python rank.py --candidates candidates.jsonl --out team_recruiter-lens.csv

# 3. Validate format
python validate_submission.py team_recruiter-lens.csv
```

Full end-to-end from `candidates.jsonl` to `team_recruiter-lens.csv` runs in **≤5 minutes on CPU, 16GB RAM**.

---

## Repository structure

```
recruiter-lens/
├── rank.py                    # CLI entrypoint — produces submission CSV
├── app.py                    # Streamlit app for HuggingFace Space sandbox
├── requirements.txt
├── submission_metadata.yaml
├── validate_submission.py     # Provided by hackathon bundle
├── src/
│   ├── __init__.py
│   ├── jd_parser.py           # Structured JD requirements
│   ├── loader.py              # JSONL / JSONL.GZ candidate loader
│   ├── career_analyzer.py     # Title + career trajectory fit, consulting penalty
│   ├── skill_matcher.py       # Endorsement-weighted skill relevance
│   ├── signals.py             # Behavioral availability scoring (multiplier)
│   ├── experience_scorer.py   # Experience years + company type quality
│   ├── location_scorer.py     # Geography + notice period + relocation
│   ├── honeypot_detector.py   # Flags impossible profiles
│   └── rank_engine.py         # Combines all components → final score
├── artifacts/
│   └── jd_requirements.json   # Pre-parsed JD requirements (auto-generated)
├── data/
│   ├── sample_candidates.json 
│   └── sample_submission.csv  # Format reference only
└── tests/
    └── test_scoring.py
```

---

## Methodology

This is a **rule-guided semantic ranker**. The decisive component is career/title fit — it separates a genuine ML engineer from a keyword-stuffer. Skill scoring uses BM25-style exact matching gated through an endorsement-duration trust filter that halves the weight of unverified skill claims. Behavioral signals act as multiplicative modifiers rather than independent scores—a technically strong but unreachable candidate is downweighted, not replaced by weaker profiles.

See `submission_metadata.yaml` for the full ≤200-word methodology summary.

---

*Built for the India Runs hackathon by Redrob AI × Hack2skill.*