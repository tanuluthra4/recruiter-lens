#!/usr/bin/env python3
"""
rank.py
-------
CLI entrypoint for recruiter-lens.

Usage:
    python rank.py --candidates candidates.jsonl --out submission.csv
    python rank.py --candidates candidates.jsonl.gz --out submission.csv
    python rank.py --candidates sample_candidates.json --out submission.csv --top 100

Produces a CSV with columns: candidate_id, rank, score, reasoning
Compatible with validate_submission.py from the hackathon bundle.

Compute budget: CPU only, ≤16GB RAM, ≤5 minutes for 100K candidates.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

# Candidate loading (handles .jsonl, .jsonl.gz, and sample .json arrays)

def load_candidates(path: Path) -> list:
    candidates = []
    if path.suffix == ".gz":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    elif path.suffix in (".jsonl", ".json"):
        opener = lambda: open(path, "r", encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    with opener() as f:
        text = f.read().strip()

    # Handle both JSON arrays (sample) and JSONL (main dataset)
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    for r in records:
        r.setdefault("_honeypot_risk", False)
        r.setdefault("_score_breakdown", {})

    return records


# Reasoning generator

def generate_reasoning(candidate: dict, breakdown: dict, rank: int) -> str:
    """
    Generate a specific, honest 1-2 sentence reasoning string.
    Rules from spec Stage 4:
      - Reference specific facts from the profile
      - Connect to JD requirements
      - Acknowledge concerns where they exist
      - Never hallucinate (only use data from the record)
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "")
    notes = breakdown.get("notes", {})
    signals = candidate.get("redrob_signals", {})

    career_score = breakdown.get("career", 0)
    skill_score = breakdown.get("skill", 0)
    beh = breakdown.get("behavioral_modifier", 1.0)
    notice = signals.get("notice_period_days", 90)
    open_to_work = signals.get("open_to_work_flag", False)
    response_rate = signals.get("recruiter_response_rate", 0)

    # Build parts of the reasoning
    parts = []

    # --- Who they are ---
    parts.append(f"{yoe:.0f}-year {title} at {company}" if company else f"{yoe:.0f}-year {title}")

    # --- Why they fit (or don't) ---
    if career_score >= 0.70:
        career_note = notes.get("career", "")
        ml_hint = ""
        if "ML/AI-related" in career_note:
            # Extract the percentage
            import re
            m = re.search(r"([\d]+)% ML/AI-related", career_note)
            if m:
                ml_hint = f"; {m.group(1)}% of career in ML/AI-related roles"
        parts.append(f"strong technical trajectory{ml_hint}")
    elif career_score >= 0.40:
        parts.append("solid technical background but limited ML-specific role history")
    else:
        parts.append("limited technical career fit for this ML-focused role")

    # --- Key skills ---
    skill_note = notes.get("skill", "")
    import re
    m = re.search(r"Hard skills matched: \[([^\]]+)\]", skill_note)
    if m and m.group(1) != "none":
        skill_list = m.group(1)
        parts.append(f"matched skills: {skill_list}")

    # --- Location ---
    if location:
        parts.append(f"based in {location}")

    # --- Concerns (honest acknowledgment) ---
    concerns = []
    if notice > 90:
        concerns.append(f"notice period {notice}d above JD preference")
    if response_rate < 0.20 and response_rate >= 0:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")
    if beh < 0.80:
        concerns.append("limited recent platform activity")
    if not open_to_work and rank <= 30:
        concerns.append("not currently marked open to work")

    # Build final string
    main_clause = "; ".join(parts[:3])
    if concerns:
        concern_clause = " Concerns: " + ", ".join(concerns) + "."
    else:
        concern_clause = ""

    reasoning = main_clause + "." + concern_clause
    return reasoning[:300]   # Spec says 1-2 sentences; keep it bounded


# CSV writer

def write_csv(results: list, out_path: Path) -> None:
    """
    Write ranked results to CSV matching the submission spec exactly.
    columns: candidate_id, rank, score, reasoning
    """
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_1based, (score, candidate, breakdown) in enumerate(results, start=1):
            cid = candidate.get("candidate_id", "UNKNOWN")
            reasoning = generate_reasoning(candidate, breakdown, rank_1based)
            writer.writerow([cid, rank_1based, f"{score:.6f}", reasoning])

    print(f"[rank] Written {len(results)} rows to {out_path}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="recruiter-lens: rank candidates for the Redrob Senior AI Engineer JD."
    )
    parser.add_argument(
        "--candidates", "-c",
        required=True,
        help="Path to candidates.jsonl, candidates.jsonl.gz, or sample_candidates.json"
    )
    parser.add_argument(
        "--out", "-o",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of top candidates to include (default: 100)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress during scoring"
    )
    args = parser.parse_args()

    t0 = time.time()

    # --- Load ---
    candidates_path = Path(args.candidates)
    print(f"[rank] Loading candidates from {candidates_path}...")
    candidates = load_candidates(candidates_path)
    print(f"[rank] Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # --- Lazy import (keeps startup fast) ---
    from src.jd_parser import load_requirements
    from src.rank_engine import rank_candidates
    from src.honeypot_detector import flag_honeypots

    # --- Pre-flight: flag honeypots ---
    t1 = time.time()
    n_honeypots = flag_honeypots(candidates)
    if n_honeypots > 0:
        print(f"[rank] Flagged {n_honeypots} likely honeypot profiles (will be excluded from top-100)")

    # --- Score ---
    req = load_requirements()
    print(f"[rank] Scoring {len(candidates):,} candidates...")
    t2 = time.time()
    ranked = rank_candidates(candidates, req=req, top_n=args.top, verbose=args.verbose)
    print(f"[rank] Scoring complete in {time.time()-t2:.1f}s")

    # --- Write ---
    out_path = Path(args.out)
    write_csv(ranked, out_path)

    elapsed = time.time() - t0
    print(f"[rank] Total time: {elapsed:.1f}s")
    if elapsed > 300:
        print("[rank] WARNING: exceeded 5-minute compute budget")

    # --- Preview top-5 ---
    print("\nTop 5 candidates:")
    for i, (score, c, bd) in enumerate(ranked[:5], 1):
        p = c.get("profile", {})
        print(f"  #{i} {c.get('candidate_id')} | {p.get('current_title')} at {p.get('current_company')} "
              f"| score={score:.4f} | career={bd.get('career'):.2f} skill={bd.get('skill'):.2f}")


if __name__ == "__main__":
    main()