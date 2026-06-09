# recruiter-lens

> AI-powered candidate ranking for the Redrob Hackathon — *India Runs*.

Recruiters don't miss the right person because the talent isn't there. They miss them because keyword filters can't see what actually matters. `recruiter-lens` ranks candidates the way a great recruiter would — by understanding career trajectory, skill depth, and genuine availability — not by counting keywords.

---

## Architecture

Five scoring dimensions, each independently computed and combined with learned weights:

| Dimension | Weight | What it measures |
|---|---|---|
| Career & Title Fit | 35% | Is this person actually an ML/AI engineer at a product company? |
| Skill Relevance | 25% | Semantic + BM25 skill match with endorsement trust filter |
| Experience Quality | 15% | Years, company type (product vs. consulting), trajectory |
| Behavioral Availability | 15% | Recency, response rate, notice period, open-to-work |
| Location & Logistics | 10% | India cities, relocation willingness, work mode |

**Anti-trap logic** (per the JD's explicit warning):
- Candidates with non-technical career histories (Marketing, HR, Accounting) are hard-capped regardless of skill listings
- Skills with zero endorsements and zero usage months are treated as unverified
- Honeypot profiles (impossible timelines, `expert` proficiency + 0 endorsements) are flagged and ranked last
- Inactive candidates (last login >90 days + low response rate) receive a behavioral decay penalty

**Compute budget:** Runs on CPU in ~2–3 minutes for 100K candidates. No GPU. No external API calls during ranking. Uses `all-MiniLM-L6-v2` (80MB) for semantic matching — precomputed JD embedding stored in `artifacts/`.

---

## Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# (One-time) Pre-compute the JD embedding
python src/precompute.py --jd job_description.md --out artifacts/jd_embedding.npy

# Rank candidates
python rank.py --candidates candidates.jsonl --out submission.csv

# Validate format
python validate_submission.py submission.csv
```

Full end-to-end from `candidates.jsonl` to `submission.csv` runs in **≤5 minutes on CPU, 16GB RAM**.

---

## Repository structure

```
recruiter-lens/
├── rank.py                   # CLI entrypoint — produces submission.csv
├── requirements.txt
├── submission_metadata.yaml
├── src/
│   ├── jd_parser.py          # Extracts hard/soft/disqualifier requirements from JD
│   ├── career_analyzer.py    # Title + career trajectory fit, consulting penalty
│   ├── skill_matcher.py      # Endorsement-weighted skill relevance
│   ├── signals.py            # Behavioral availability scoring
│   ├── experience_scorer.py  # Experience years + company type quality
│   ├── location_scorer.py    # Geography + notice period + relocation
│   ├── honeypot_detector.py  # Flags impossible profiles
│   └── rank_engine.py        # Combines all components → final score
├── artifacts/
│   ├── jd_embedding.npy      # Precomputed sentence-transformer embedding of JD
│   └── jd_requirements.json  # Structured requirements parsed from JD
├── data/
│   └── sample_submission.csv # Format reference only
└── tests/
    └── test_scoring.py
```

---

## Methodology

See `submission_metadata.yaml` for the ≤200-word summary submitted with the ranking.

The short version: this is a **rule-guided semantic ranker**. The career/title component is the decisive signal — it's what separates a genuine ML engineer from a keyword-stuffer. Skill scoring uses BM25 for exact matches and sentence-transformer cosine similarity for semantic proximity, but both are gated through an endorsement-duration trust filter that halves the weight of unverified skill claims. Behavioral signals act as a multiplicative modifier, not an additive component — a technically strong candidate who is unreachable gets downweighted, not replaced.

---

*Built for the India Runs hackathon by Redrob AI × Hack2skill.*