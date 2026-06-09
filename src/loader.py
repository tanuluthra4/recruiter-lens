"""
loader.py
---------
Loads candidate records from JSONL (plain or gzipped).
Also performs lightweight schema checks and attaches a `_honeypot_risk`
flag that honeypot_detector.py will populate later.

Usage:
    from src.loader import load_candidates
    candidates = load_candidates("candidates.jsonl")      # list of dicts
    candidates = load_candidates("candidates.jsonl.gz")   # auto-detects gzip
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Iterator, List

# Core loader

def _open_jsonl(path: Path) -> Iterator[str]:
    """Yield raw lines from a plain or gzipped JSONL file."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from f
    else:
        with open(path, "r", encoding="utf-8") as f:
            yield from f


def load_candidates(path: str | Path, max_records: int | None = None) -> List[dict]:
    """
    Load candidates from a JSONL or JSONL.GZ file.

    Parameters
    ----------
    path        : Path to the file.
    max_records : If set, stops after loading this many records (useful for
                  testing on the sample).

    Returns
    -------
    List of candidate dicts, each with an extra ``_honeypot_risk`` bool
    field initialised to False.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    candidates: List[dict] = []
    skipped = 0

    for i, line in enumerate(_open_jsonl(path)):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            skipped += 1
            if skipped <= 5:
                print(f"[loader] Skipping malformed line {i+1}: {e}", file=sys.stderr)
            continue

        # Attach runtime-only fields
        record["_honeypot_risk"] = False
        record["_score_breakdown"] = {}

        candidates.append(record)

        if max_records and len(candidates) >= max_records:
            break

    if skipped:
        print(f"[loader] Skipped {skipped} malformed lines.", file=sys.stderr)

    return candidates

# Convenience accessors (avoid KeyError noise in scoring code)

def get_profile(c: dict) -> dict:
    return c.get("profile", {})

def get_career(c: dict) -> list:
    return c.get("career_history", [])

def get_skills(c: dict) -> list:
    return c.get("skills", [])

def get_signals(c: dict) -> dict:
    return c.get("redrob_signals", {})

def get_education(c: dict) -> list:
    return c.get("education", [])

def candidate_id(c: dict) -> str:
    return c.get("candidate_id", "UNKNOWN")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_candidates.json"
    # sample_candidates.json is a JSON array, not JSONL — handle both
    p = Path(path)
    if p.suffix == ".json":
        with open(p) as f:
            text = f.read().strip()
        # Could be an array or newline-delimited objects
        if text.startswith("["):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        for r in records:
            r["_honeypot_risk"] = False
            r["_score_breakdown"] = {}
        print(f"Loaded {len(records)} candidates from {path}")
    else:
        records = load_candidates(path, max_records=5)
        print(f"Loaded {len(records)} candidates (preview)")
        print("First ID:", candidate_id(records[0]))