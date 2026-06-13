"""
app.py — Streamlit sandbox for recruiter-lens
Deployable on HuggingFace Spaces (streamlit SDK).
Lets judges upload a small candidate JSON sample and see the
ranker work end-to-end in the browser — no GPU, no API calls.
"""

import json
import sys
import os
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Make src/ importable
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="recruiter-lens",
    page_icon="🔍",
    layout="wide",
)

# ── Styles 
st.markdown("""
<style>
    .main { background: #0f1117; }
    .score-high  { color: #10b981; font-weight: 700; }
    .score-mid   { color: #f59e0b; font-weight: 700; }
    .score-low   { color: #ef4444; font-weight: 700; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.6rem;
        border: 1px solid #2d3147;
    }
    .tag {
        display: inline-block;
        background: #1d4ed8;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .tag-warn {
        background: #b45309;
    }
    .stButton > button {
        background: #1d4ed8;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Header 
st.title("🔍 recruiter-lens")
st.caption("AI-powered candidate ranking · India Runs Hackathon · Redrob AI × Hack2skill")

st.markdown("""
Ranks candidates the way a great recruiter would - by understanding **career trajectory**,
**verified skill depth**, and **genuine availability** - not by counting keywords.
""")

# ── JD Summary sidebar 
with st.sidebar:
    st.header("📋 Job Description")
    st.markdown("""
**Role:** Senior AI Engineer (Founding Team)
**Must-have skills:**
- Embeddings, vector search, FAISS/Pinecone/Weaviate
- Information retrieval, BM25, learning-to-rank
- Python, NLP, ML (production systems)
- Evaluation: NDCG, MRR, offline eval
**Nice-to-have:**
- LLM fine-tuning (LoRA, PEFT)
- LangChain, sentence-transformers
- MLOps, MLflow, Kubeflow
**Disqualifiers:**
- Non-technical current title (Marketing, HR, etc.)
- Entire career at IT services / consulting
- Keyword-stuffed profiles (0 endorsements, 0 months used)
""")

    st.divider()
    st.header("⚖️ Scoring Weights")
    weights_df = pd.DataFrame({
        "Component": ["Career & Title Fit", "Skill Relevance",
                       "Experience Quality", "Location & Logistics",
                       "Behavioral (multiplier)"],
        "Weight": ["35%", "25%", "15%", "10%", "±20%"]
    })
    st.dataframe(weights_df, hide_index=True, use_container_width=True)

# ── Load sample data
SAMPLE_PATHS = [
    Path("sample_candidates.json"),
    Path("data/sample_candidates.json"),
]

@st.cache_data
def load_builtin_sample():
    for p in SAMPLE_PATHS:
        if p.exists():
            with open(p) as f:
                text = f.read().strip()
            records = json.loads(text) if text.startswith("[") else \
                      [json.loads(l) for l in text.splitlines() if l.strip()]
            return records
    return None

builtin = load_builtin_sample()

# ── Input section
st.header("Upload candidates")
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader(
        "Upload a JSON array or JSONL file of candidate records",
        type=["json", "jsonl"],
        help="Upload sample_candidates.json from the hackathon bundle, or any subset of candidates.jsonl"
    )
with col2:
    use_builtin = False
    if builtin:
        st.markdown("<br>", unsafe_allow_html=True)
        use_builtin = st.button("▶ Use built-in sample (50 candidates)")

def build_reasoning(c, bd):
    """Build full reasoning string for a candidate — no truncation."""
    import re
    p     = c.get("profile", {})
    sig   = c.get("redrob_signals", {})
    notes = bd.get("notes", {})

    yoe     = p.get("years_of_experience", 0)
    title   = p.get("current_title", "")
    company = p.get("current_company", "")

    career_score = bd.get("career", 0)
    ml_m = re.search(r"(\d+)% ML/AI-related", notes.get("career", ""))
    ml_hint = f" ({ml_m.group(1)}% of career in ML/AI roles)" if ml_m else ""

    if career_score >= 0.70:
        traj = f"Strong technical trajectory{ml_hint}."
    elif career_score >= 0.40:
        traj = "Solid technical background but limited ML-specific role history."
    else:
        traj = "Limited technical fit for this ML-focused role."

    m = re.search(r"Hard skills matched: \[([^\]]+)\]", notes.get("skill", ""))
    skills_str = m.group(1) if m and m.group(1) != "none" else "none"

    consulting_m = re.search(r"Consulting exposure: (\d+)%", notes.get("career", ""))
    consulting_line = ""
    if consulting_m and int(consulting_m.group(1)) > 30:
        consulting_line = f" Consulting exposure: {consulting_m.group(1)}% of career."

    concerns, positives = [], []
    notice = sig.get("notice_period_days", 90)
    rr     = sig.get("recruiter_response_rate", 0)
    beh    = bd.get("behavioral_modifier", 1.0)
    github = sig.get("github_activity_score", -1)

    if notice > 90:                     concerns.append(f"notice period {notice}d (JD prefers ≤30)")
    if rr < 0.20 and rr >= 0:           concerns.append(f"low recruiter response rate ({rr:.0%})")
    if beh < 0.80:                      concerns.append("limited recent platform activity")
    if not sig.get("open_to_work_flag"):concerns.append("not currently marked open to work")

    if sig.get("open_to_work_flag"):    positives.append("actively open to work")
    if notice <= 30:                    positives.append(f"short notice period ({notice}d)")
    if rr >= 0.70:                      positives.append(f"highly responsive to recruiters ({rr:.0%})")
    if github and github >= 30:         positives.append(f"active GitHub contributor (score {github:.0f})")

    text = f"{yoe:.0f}-year {title} at {company}. {traj}{consulting_line} Matched JD skills: {skills_str}."
    if positives: text += " ✅ " + "; ".join(positives) + "."
    if concerns:  text += " ⚠️ Concerns: " + "; ".join(concerns) + "."
    return text

# ── Ranking 
def run_ranking(records):
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


def score_color(s):
    if s >= 0.75: return "score-high"
    if s >= 0.45: return "score-mid"
    return "score-low"


def render_results(ranked):
    st.success(f"Ranked {len(ranked)} candidates in {st.session_state.get('elapsed', 0):.1f}s")

    # Summary stats
    scores = [s for s, _, _ in ranked]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates ranked", len(ranked))
    c2.metric("Top score", f"{max(scores):.3f}")
    c3.metric("Median score", f"{sorted(scores)[len(scores)//2]:.3f}")
    honeypots = sum(1 for _, c, _ in ranked if c.get("_honeypot_risk"))
    c4.metric("Honeypots flagged", honeypots)

    st.divider()

    # Table view
    st.subheader("Rankings")
    table_data = []
    for rank, (score, c, bd) in enumerate(ranked, 1):
        p = c.get("profile", {})
        skills = c.get("skills", [])
        top_skills = [s["name"] for s in skills
                      if s.get("endorsements", 0) >= 3 and s.get("duration_months", 0) >= 6][:4]
        table_data.append({
            "Rank": rank,
            "ID": c.get("candidate_id", ""),
            "Title": p.get("current_title", ""),
            "Company": p.get("current_company", ""),
            "YoE": p.get("years_of_experience", 0),
            "Location": p.get("location", ""),
            "Score": round(score, 4),
            "Career": round(bd.get("career", 0), 2),
            "Skills": round(bd.get("skill", 0), 2),
            "Honeypot": "⚠️" if c.get("_honeypot_risk") else "",
        })

    df = pd.DataFrame(table_data)
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
            "Career": st.column_config.ProgressColumn("Career", min_value=0, max_value=1),
            "Skills": st.column_config.ProgressColumn("Skills", min_value=0, max_value=1),
        }
    )

    # Detailed card view
    st.divider()
    st.subheader("Detailed breakdown — Top 10")

    for rank, (score, c, bd) in enumerate(ranked[:10], 1):
        p = c.get("profile", {})
        sig = c.get("redrob_signals", {})
        notes = bd.get("notes", {})

        with st.expander(
            f"#{rank} · {p.get('current_title')} at {p.get('current_company')} · score {score:.4f}",
            expanded=(rank <= 3)
        ):
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown(f"**{p.get('current_title')}** at **{p.get('current_company')}**")
                st.caption(f"{p.get('years_of_experience', 0):.0f} years · {p.get('location', 'Unknown')}")
                summary = p.get("summary", "")
                if summary:
                    st.markdown(
                        f"<p style='color:#c9d1d9; font-size:0.95rem;'><em>{summary}</em></p>",
                        unsafe_allow_html=True
                    )
        
                skill_note = notes.get("skill", "")
                import re
                m = re.search(r"Hard skills matched: \[([^\]]+)\]", skill_note)
                if m and m.group(1) != "none":
                    skills_html = " ".join(
                        f'<span class="tag">{sk}</span>'
                        for sk in m.group(1).split(", ")[:6]
                    )
                    
                    st.markdown(
                        f'<div style="margin-top:6px;color:#8b949e;">Skills: {skills_html}</div>',
                        unsafe_allow_html=True
                    )
                # ── Full reasoning shown plainly ──────────────────────
                st.markdown(f"""
<p style="color:#8b949e; font-size:0.88rem; margin-top:6px;">{build_reasoning(c, bd)}</p>
""", unsafe_allow_html=True)

            with col_b:
                st.markdown("**Score breakdown:**")
                metrics = {
                    "Career fit": bd.get("career", 0),
                    "Skill relevance": bd.get("skill", 0),
                    "Experience": bd.get("experience", 0),
                    "Location": bd.get("location", 0),
                }
                for label, val in metrics.items():
                    st.progress(val, text=f"{label}: {val:.2f}")

                st.markdown(f"**Behavioral modifier:** {bd.get('behavioral_modifier', 1.0):.2f}×")

                flags = []
                if sig.get("open_to_work_flag"):
                    flags.append("✅ Open to work")
                if sig.get("notice_period_days", 90) <= 30:
                    flags.append("✅ Quick joiner")
                if sig.get("recruiter_response_rate", 0) >= 0.7:
                    flags.append("✅ Responsive")
                if c.get("_honeypot_risk"):
                    flags.append("⚠️ Honeypot risk")
                for f in flags:
                    st.markdown(f)

    # Download
    st.divider()
    rows = []
    for rank, (score, c, bd) in enumerate(ranked, 1):
        p = c.get("profile", {})
        sig = c.get("redrob_signals", {})
        notes = bd.get("notes", {})
        yoe = p.get("years_of_experience", 0)
        title = p.get("current_title", "")
        company = p.get("current_company", "")
        career_score = bd.get("career", 0)
        notice = sig.get("notice_period_days", 90)
        response_rate = sig.get("recruiter_response_rate", 0)
        beh = bd.get("behavioral_modifier", 1.0)
        open_to_work = sig.get("open_to_work_flag", False)

        skill_note = notes.get("skill", "")
        import re
        m = re.search(r"Hard skills matched: \[([^\]]+)\]", skill_note)
        skill_str = m.group(1) if m and m.group(1) != "none" else "none"

        career_note = notes.get("career", "")
        ml_hint = ""
        ml_m = re.search(r"(\d+)% ML/AI-related", career_note)
        if ml_m:
            ml_hint = f"; {ml_m.group(1)}% of career in ML/AI roles"

        if career_score >= 0.70:
            traj = f"strong technical trajectory{ml_hint}"
        elif career_score >= 0.40:
            traj = "solid technical background, limited ML-specific history"
        else:
            traj = "limited technical fit for this ML role"

        concerns = []
        if notice > 90:
            concerns.append(f"notice {notice}d")
        if response_rate < 0.20:
            concerns.append(f"low response rate {response_rate:.0%}")
        if beh < 0.80:
            concerns.append("low recent activity")
        if not open_to_work and rank <= 30:
            concerns.append("not open to work")

        main = f"{yoe:.0f}-year {title} at {company}; {traj}; matched: {skill_str}."
        if concerns:
            main += " Concerns: " + ", ".join(concerns) + "."

        rows.append({
            "candidate_id": c.get("candidate_id"),
            "rank": rank,
            "score": round(score, 6),
            "reasoning": main[:300]
        })

    csv_str = pd.DataFrame(rows).to_csv(index=False)
    st.download_button(
        "⬇️ Download team_recruiter-lens CSV",
        data=csv_str,
        file_name="team_recruiter-lens.csv",
        mime="text/csv"
    )


# ── Trigger ranking 
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
    with st.spinner("Scoring candidates..."):
        t0 = time.time()
        try:
            ranked = run_ranking(records)
            st.session_state["elapsed"] = time.time() - t0
            render_results(ranked)
        except Exception as e:
            st.error(f"Ranking failed: {e}")
            st.exception(e)
else:
    st.info("Upload a candidate file or click 'Use built-in sample' to see the ranker in action.")
    st.markdown("""
    **How to try it:**
    1. Upload `sample_candidates.json` from the hackathon bundle, or
    2. Click **Use built-in sample** if the file is bundled with this Space
    **What you'll see:**
    - Full ranked table with score breakdowns
    - Top-10 detailed cards with matched skills and concerns
    - Downloadable team_recruiter-lens CSV
    """)