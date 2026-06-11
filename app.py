"""
app.py — Streamlit sandbox for recruiter-lens
Deployable on HuggingFace Spaces (Docker/Streamlit template).
All heavy imports are lazy — loaded only when ranking is triggered.
"""

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd

# Make src/ importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="recruiter-lens",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
.metric-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 recruiter-lens")
st.caption("AI-powered candidate ranking · India Runs Hackathon · Redrob AI × Hack2skill")
st.markdown("Ranks candidates by **career trajectory**, **verified skill depth**, and **genuine availability** — not keywords.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Job Description")
    st.markdown("""
**Role:** Senior AI Engineer (Founding Team)

**Must-have:**
- Embeddings, vector search, FAISS / Pinecone / Weaviate
- Information retrieval, BM25, learning-to-rank
- Python, NLP, production ML systems
- Evaluation: NDCG, MRR

**Nice-to-have:**
- LLM fine-tuning, LangChain, sentence-transformers
- MLOps, MLflow, Kubeflow

**Disqualifiers:**
- Non-technical title (Marketing, HR, Accounting…)
- Entire career at IT services / consulting
- 0-endorsement skill claims
""")
    st.divider()
    st.header("⚖️ Weights")
    st.dataframe(pd.DataFrame({
        "Component": ["Career & Title Fit", "Skill Relevance",
                      "Experience Quality", "Location & Logistics",
                      "Behavioral (multiplier)"],
        "Weight": ["35%", "25%", "15%", "10%", "±20%"]
    }), hide_index=True, use_container_width=True)

# ── Load sample ───────────────────────────────────────────────────────────────
@st.cache_data
def load_builtin_sample():
    for p in ["sample_candidates.json", "data/sample_candidates.json"]:
        if Path(p).exists():
            with open(p) as f:
                text = f.read().strip()
            return json.loads(text) if text.startswith("[") else \
                   [json.loads(l) for l in text.splitlines() if l.strip()]
    return None

builtin = load_builtin_sample()

# ── Input ─────────────────────────────────────────────────────────────────────
st.header("Upload candidates")
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader(
        "Upload sample_candidates.json or any JSONL subset",
        type=["json", "jsonl"]
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    use_builtin = False
    if builtin:
        use_builtin = st.button("▶ Use built-in sample")
    else:
        st.info("No built-in sample found.")

# ── Ranking (all src imports are inside this function) ────────────────────────
def run_ranking(records):
    # Lazy imports — only happen when user triggers ranking
    from src.jd_parser import load_requirements
    from src.rank_engine import score_candidate
    from src.honeypot_detector import detect_honeypot

    req = load_requirements()
    results = []
    for c in records:
        c.setdefault("_honeypot_risk", False)
        c.setdefault("_score_breakdown", {})
        detect_honeypot(c)
        score, breakdown = score_candidate(c, req)
        results.append((score, c, breakdown))
    results.sort(key=lambda x: (-x[0], x[1].get("candidate_id", "")))
    return results

# ── Render results ────────────────────────────────────────────────────────────
def render_results(ranked, elapsed):
    st.success(f"Ranked {len(ranked)} candidates in {elapsed:.1f}s")

    scores = [s for s, _, _ in ranked]
    honeypots = sum(1 for _, c, _ in ranked if c.get("_honeypot_risk"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates", len(ranked))
    c2.metric("Top score", f"{max(scores):.3f}")
    c3.metric("Median score", f"{sorted(scores)[len(scores)//2]:.3f}")
    c4.metric("Honeypots flagged", honeypots)

    st.divider()
    st.subheader("Full ranking")

    rows = []
    for rank, (score, c, bd) in enumerate(ranked, 1):
        p = c.get("profile", {})
        import re
        skill_note = bd.get("notes", {}).get("skill", "")
        m = re.search(r"Hard skills matched: \[([^\]]+)\]", skill_note)
        skills_str = m.group(1) if m and m.group(1) != "none" else "—"
        rows.append({
            "Rank": rank,
            "ID": c.get("candidate_id", ""),
            "Title": p.get("current_title", ""),
            "Company": p.get("current_company", ""),
            "YoE": p.get("years_of_experience", 0),
            "Score": round(score, 4),
            "Career": round(bd.get("career", 0), 2),
            "Skills": round(bd.get("skill", 0), 2),
            "Matched skills": skills_str,
            "⚠️": "honeypot" if c.get("_honeypot_risk") else "",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df, hide_index=True, use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
            "Career": st.column_config.ProgressColumn("Career", min_value=0, max_value=1),
            "Skills": st.column_config.ProgressColumn("Skills", min_value=0, max_value=1),
        }
    )

    st.divider()
    st.subheader("Top 5 — detailed breakdown")
    for rank, (score, c, bd) in enumerate(ranked[:5], 1):
        p = c.get("profile", {})
        sig = c.get("redrob_signals", {})
        with st.expander(
            f"#{rank} · {p.get('current_title')} at {p.get('current_company')} · {score:.4f}",
            expanded=(rank == 1)
        ):
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown(f"**{p.get('current_title')}** at **{p.get('current_company')}**")
                st.caption(f"{p.get('years_of_experience', 0):.0f} yrs · {p.get('location', '')}")
                summary = p.get("summary", "")
                st.markdown(f"*{summary[:200]}{'...' if len(summary) > 200 else ''}*")

                import re
                skill_note = bd.get("notes", {}).get("skill", "")
                m = re.search(r"Hard skills matched: \[([^\]]+)\]", skill_note)
                if m and m.group(1) != "none":
                    st.markdown("**Matched JD skills:** " + m.group(1))

            with col_b:
                for label, val in [
                    ("Career fit", bd.get("career", 0)),
                    ("Skill relevance", bd.get("skill", 0)),
                    ("Experience", bd.get("experience", 0)),
                    ("Location", bd.get("location", 0)),
                ]:
                    st.progress(float(val), text=f"{label}: {val:.2f}")

                st.markdown(f"**Behavioral modifier:** {bd.get('behavioral_modifier', 1.0):.2f}×")

                flags = []
                if sig.get("open_to_work_flag"): flags.append("✅ Open to work")
                if sig.get("notice_period_days", 90) <= 30: flags.append("✅ Quick joiner")
                if sig.get("recruiter_response_rate", 0) >= 0.7: flags.append("✅ Responsive")
                if c.get("_honeypot_risk"): flags.append("⚠️ Honeypot risk")
                for f in flags:
                    st.markdown(f)

    # Download button
    st.divider()
    import re
    csv_rows = []
    for rank, (score, c, bd) in enumerate(ranked, 1):
        p = c.get("profile", {})
        sig = c.get("redrob_signals", {})
        notes = bd.get("notes", {})
        skill_note = notes.get("skill", "")
        m = re.search(r"Hard skills matched: \[([^\]]+)\]", skill_note)
        skills_str = m.group(1) if m and m.group(1) != "none" else "none"
        career_note = notes.get("career", "")
        ml_m = re.search(r"(\d+)% ML/AI-related", career_note)
        ml_hint = f"; {ml_m.group(1)}% ML/AI roles" if ml_m else ""
        career_score = bd.get("career", 0)
        traj = "strong technical trajectory" + ml_hint if career_score >= 0.70 \
               else "solid technical background" if career_score >= 0.40 \
               else "limited ML career fit"
        concerns = []
        notice = sig.get("notice_period_days", 90)
        rr = sig.get("recruiter_response_rate", 0)
        if notice > 90: concerns.append(f"notice {notice}d")
        if rr < 0.20: concerns.append(f"low response rate {rr:.0%}")
        reasoning = (f"{p.get('years_of_experience',0):.0f}-year {p.get('current_title')} "
                     f"at {p.get('current_company')}; {traj}; matched: {skills_str}.")
        if concerns:
            reasoning += " Concerns: " + ", ".join(concerns) + "."
        csv_rows.append({
            "candidate_id": c.get("candidate_id"),
            "rank": rank,
            "score": round(score, 6),
            "reasoning": reasoning[:300]
        })

    csv_str = pd.DataFrame(csv_rows).to_csv(index=False)
    st.download_button("⬇️ Download submission CSV", data=csv_str,
                       file_name="team_recruiter-lens.csv", mime="text/csv")

# ── Trigger ───────────────────────────────────────────────────────────────────
records = None

if uploaded:
    text = uploaded.read().decode("utf-8").strip()
    try:
        records = json.loads(text) if text.startswith("[") else \
                  [json.loads(l) for l in text.splitlines() if l.strip()]
        st.info(f"Loaded {len(records)} candidates from uploaded file.")
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
elif use_builtin and builtin:
    records = builtin
    st.info(f"Using built-in sample: {len(records)} candidates.")

if records:
    with st.spinner("Scoring candidates... (first run may take ~30s to load models)"):
        t0 = time.time()
        try:
            ranked = run_ranking(records)
            elapsed = time.time() - t0
            render_results(ranked, elapsed)
        except Exception as e:
            st.error(f"Ranking failed: {e}")
            st.exception(e)
else:
    st.info("Upload a candidate JSON file or click **Use built-in sample** to start.")
    st.markdown("""
    **How to try it:**
    1. Click **▶ Use built-in sample** (uses the 50-candidate sample bundled with this Space), or
    2. Upload `sample_candidates.json` from the hackathon bundle

    **What you'll see:**
    - Full ranked table with score breakdowns
    - Top-5 detailed cards with matched skills and concerns
    - Downloadable submission CSV
    """)