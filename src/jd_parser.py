"""
jd_parser.py
------------
Parses the job description into structured requirements used by the ranker.

Rather than doing NLP on the raw JD text at runtime (which would be slow and
unreliable), this module encodes domain knowledge about the specific JD into
structured Python objects. The parsed output is also serialised to
artifacts/jd_requirements.json so the rank_engine can load it in < 1ms.

Usage:
    from src.jd_parser import JDRequirements, load_requirements
    req = load_requirements()          # fast path — loads from artifacts/
    req = JDRequirements.from_text()   # re-parses and saves
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Set

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
JD_REQUIREMENTS_PATH = ARTIFACTS_DIR / "jd_requirements.json"

# Hard requirements — a candidate MUST satisfy these to rank in top-20.
# Drawn directly from the "Things you absolutely need" section of the JD.
HARD_REQUIRED_SKILLS: List[str] = [
    # Embeddings / retrieval
    "sentence transformers", "sentence-transformers", "openai embeddings",
    "embeddings", "embedding", "vector search", "dense retrieval",
    "hybrid search", "semantic search",
    # Vector stores
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "chroma",
    # Ranking / IR
    "information retrieval", "learning to rank", "ranking", "reranking",
    "bm25", "ndcg", "mrr", "map", "recall", "retrieval",
    # Core ML
    "python", "nlp", "machine learning", "ml", "deep learning",
    # Evaluation
    "a/b testing", "ab testing", "offline evaluation", "eval framework",
]

# Soft / nice-to-have skills — contribute to score but not gates.
NICE_TO_HAVE_SKILLS: List[str] = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "fine-tuning llms",
    "langchain", "llm", "llms", "gpt", "transformers", "hugging face",
    "huggingface", "pytorch", "tensorflow", "xgboost", "lightgbm",
    "recommendation systems", "mlops", "mlflow", "kubeflow",
    "distributed systems", "kafka", "spark", "airflow",
    "open source", "github",
]

# Disqualifier signals — any of these significantly reduce score.
# Drawn from "Things we explicitly do NOT want" in the JD.

# Titles that are strong signals the candidate is NOT an ML engineer,
# regardless of what skills they list.
DISQUALIFIER_TITLES: Set[str] = {
    "marketing manager", "hr manager", "accountant", "graphic designer",
    "content writer", "sales executive", "civil engineer",
    "mechanical engineer", "operations manager", "business analyst",
    "customer support", "project manager",
}

# Companies that count as pure consulting / services.
# The JD says: "entire career at consulting firms" is a disqualifier;
# but one stint there is fine.
CONSULTING_COMPANIES: Set[str] = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis",
}

# Industries that are consulting/services.
CONSULTING_INDUSTRIES: Set[str] = {
    "it services", "consulting",
}

# Location preferences
PREFERRED_LOCATIONS: Set[str] = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon",
    "bangalore", "bengaluru", "india",
}

# Experience range
EXP_MIN_YEARS: float = 4.0    # below this is nearly disqualifying
EXP_SOFT_MIN: float = 5.0     # ideal lower bound from JD
EXP_IDEAL_MIN: float = 5.0
EXP_IDEAL_MAX: float = 9.0
EXP_SOFT_MAX: float = 12.0    # above 12 years gets modest penalty

# Notice period preference
NOTICE_IDEAL_MAX_DAYS: int = 30
NOTICE_OK_MAX_DAYS: int = 60
NOTICE_HARD_MAX_DAYS: int = 90   # 90+ is a soft negative per JD

# Dataclass for convenient access

@dataclass
class JDRequirements:
    hard_skills: List[str] = field(default_factory=lambda: HARD_REQUIRED_SKILLS)
    nice_to_have_skills: List[str] = field(default_factory=lambda: NICE_TO_HAVE_SKILLS)
    disqualifier_titles: Set[str] = field(default_factory=lambda: DISQUALIFIER_TITLES)
    consulting_companies: Set[str] = field(default_factory=lambda: CONSULTING_COMPANIES)
    consulting_industries: Set[str] = field(default_factory=lambda: CONSULTING_INDUSTRIES)
    preferred_locations: Set[str] = field(default_factory=lambda: PREFERRED_LOCATIONS)
    exp_min: float = EXP_MIN_YEARS
    exp_soft_min: float = EXP_SOFT_MIN
    exp_ideal_min: float = EXP_IDEAL_MIN
    exp_ideal_max: float = EXP_IDEAL_MAX
    exp_soft_max: float = EXP_SOFT_MAX
    notice_ideal_max: int = NOTICE_IDEAL_MAX_DAYS
    notice_ok_max: int = NOTICE_OK_MAX_DAYS
    notice_hard_max: int = NOTICE_HARD_MAX_DAYS

    # JD summary text used for semantic embedding matching
    jd_summary: str = (
        "Senior AI Engineer founding team role requiring production experience with "
        "embeddings-based retrieval systems, vector databases, hybrid search, "
        "Python, NLP, ranking and recommendation systems, evaluation frameworks "
        "for ranking (NDCG, MRR), LLM fine-tuning, sentence-transformers. "
        "Must have shipped end-to-end ML systems at product companies. "
        "5-9 years experience. Located in India. Hybrid work mode. "
        "Not suitable for pure consulting backgrounds or non-technical titles."
    )

    def save(self, path: Path = JD_REQUIREMENTS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = asdict(self)
        # Sets aren't JSON-serialisable — convert to lists
        for key, val in serialisable.items():
            if isinstance(val, set):
                serialisable[key] = list(val)
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)

    @classmethod
    def load(cls, path: Path = JD_REQUIREMENTS_PATH) -> "JDRequirements":
        with open(path) as f:
            data = json.load(f)
        # Convert lists back to sets where appropriate
        for key in ("disqualifier_titles", "consulting_companies",
                    "consulting_industries", "preferred_locations"):
            if key in data:
                data[key] = set(data[key])
        return cls(**data)


def load_requirements() -> JDRequirements:
    """Fast path: loads from artifacts if available, else builds from scratch."""
    if JD_REQUIREMENTS_PATH.exists():
        return JDRequirements.load()
    req = JDRequirements()
    req.save()
    return req


if __name__ == "__main__":
    req = JDRequirements()
    req.save()
    print(f"Saved JD requirements to {JD_REQUIREMENTS_PATH}")
    print(f"  Hard skills: {len(req.hard_skills)}")
    print(f"  Nice-to-have: {len(req.nice_to_have_skills)}")
    print(f"  Disqualifier titles: {len(req.disqualifier_titles)}")