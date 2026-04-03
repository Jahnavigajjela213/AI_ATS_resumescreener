"""
AI Resume Screening & ATS System - Streamlit Frontend v3.0
All features: Upload, Match, SkillGap, Recommendations, Domain Breakdown,
Batch Screener, Cover Letter, Salary Estimator, Resume Compare, Model Insights.
"""
import sys, os, re, json, io, csv
from pathlib import Path
from datetime import datetime
import streamlit as st

st.set_page_config(
    page_title="AI ATS Resume Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data.data_loader import extract_text_from_bytes, load_jobs
from preprocessing.text_cleaner import extract_years_experience, extract_education_level, extract_sections
from preprocessing.nlp_pipeline import NLPPipeline
from utils.skill_extractor import extract_skills, extract_skills_by_domain
from utils.skill_gap import compute_ats_score, compute_skill_gap
from utils.recommender import recommend_learning_path, suggest_career_path
from utils.cover_letter import generate_cover_letter, TEMPLATES
from utils.salary_estimator import estimate_salary, get_location_options
from models.resume_matcher import ResumeMatcher

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0f1117;}
.hero-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 50%,#f093fb 100%);
  padding:2.5rem 2rem;border-radius:16px;margin-bottom:2rem;text-align:center;
  box-shadow:0 20px 60px rgba(102,126,234,0.4);}
.hero-header h1{color:white;font-size:2.2rem;font-weight:700;margin:0;}
.hero-header p{color:rgba(255,255,255,0.85);font-size:1rem;margin-top:0.5rem;}
.score-card{background:linear-gradient(145deg,#1e2130,#252a3d);
  border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:1.5rem;
  margin:.5rem 0;box-shadow:0 8px 32px rgba(0,0,0,0.3);}
.metric-box{background:linear-gradient(135deg,#1a1f2e,#232940);border-radius:12px;
  padding:1.2rem;border-left:4px solid #667eea;margin-bottom:1rem;}
.metric-box .metric-val{font-size:2rem;font-weight:700;color:#667eea;}
.metric-box .metric-label{font-size:.85rem;color:#8892b0;margin-top:.2rem;}
.grade-excellent{background:linear-gradient(90deg,#00c9a7,#00e4c0);color:#fff;}
.grade-good{background:linear-gradient(90deg,#667eea,#764ba2);color:#fff;}
.grade-fair{background:linear-gradient(90deg,#f7971e,#fd6200);color:#fff;}
.grade-weak{background:linear-gradient(90deg,#eb3349,#f45c43);color:#fff;}
.grade-poor{background:#2d1515;color:#ff6b6b;}
.grade-badge{display:inline-block;padding:.4rem 1.2rem;border-radius:50px;
  font-weight:700;font-size:1rem;letter-spacing:1px;}
.skill-matched{background:rgba(0,201,167,.15);color:#00c9a7;border:1px solid #00c9a7;}
.skill-missing{background:rgba(235,51,73,.15);color:#eb3349;border:1px solid #eb3349;}
.skill-extra{background:rgba(102,126,234,.15);color:#667eea;border:1px solid #667eea;}
.skill-chip{display:inline-block;padding:.25rem .8rem;border-radius:50px;
  font-size:.8rem;font-weight:500;margin:.2rem;}
.suggestion-card{background:rgba(247,151,30,.1);border-left:3px solid #f7971e;
  border-radius:8px;padding:.8rem 1rem;margin:.4rem 0;color:#fcd34d;font-size:.9rem;}
.highlight-matched{background:rgba(0,201,167,.25);border-radius:3px;padding:0 2px;}
.highlight-missing{background:rgba(235,51,73,.25);border-radius:3px;padding:0 2px;}
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0d1117 0%,#161b29 100%) !important;
  border-right:1px solid rgba(255,255,255,.06);}
hr{border-color:rgba(255,255,255,.08);}
.salary-band{background:linear-gradient(135deg,#1a1f2e,#232940);border-radius:16px;
  padding:1.5rem;border:1px solid rgba(102,126,234,.3);margin:.5rem 0;text-align:center;}
.sal-num{font-size:2.2rem;font-weight:700;color:#00c9a7;}
.sal-label{font-size:.8rem;color:#8892b0;}
.compare-better{color:#00c9a7;font-weight:600;}
.compare-worse{color:#eb3349;font-weight:600;}
.kw-pill{display:inline-flex;align-items:center;padding:.2rem .7rem;border-radius:50px;
  font-size:.78rem;margin:.15rem;font-weight:500;}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for k in ["resume_text","ats_result","job_text","gap_result",
          "match_result","predicted_role","score_history","cover_letter"]:
    if k not in st.session_state:
        st.session_state[k] = None
if "score_history" not in st.session_state or st.session_state.score_history is None:
    st.session_state.score_history = []


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_matcher():
    jobs_df = load_jobs()
    m = ResumeMatcher()
    m.fit_jobs(jobs_df)
    return m

@st.cache_resource(show_spinner=False)
def load_nlp():
    return NLPPipeline()


# ── Chart helpers ─────────────────────────────────────────────────────────────
def gauge_chart(score, title="ATS Score"):
    color = "#00c9a7" if score>=70 else "#f7971e" if score>=50 else "#eb3349"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=score,
        domain={"x":[0,1],"y":[0,1]},
        title={"text":title,"font":{"size":18,"color":"white","family":"Inter"}},
        delta={"reference":70,"increasing":{"color":"#00c9a7"},"decreasing":{"color":"#eb3349"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"rgba(255,255,255,0.3)"},
               "bar":{"color":color,"thickness":.3},"bgcolor":"#1a1f2e","borderwidth":0,
               "steps":[{"range":[0,40],"color":"rgba(235,51,73,0.1)"},
                        {"range":[40,70],"color":"rgba(247,151,30,0.1)"},
                        {"range":[70,100],"color":"rgba(0,201,167,0.1)"}],
               "threshold":{"line":{"color":"white","width":3},"thickness":.75,"value":70}},
        number={"font":{"size":44,"color":color,"family":"Inter"},"suffix":"/100"},
    ))
    fig.update_layout(height=300,margin=dict(l=20,r=20,t=60,b=20),
                      paper_bgcolor="rgba(0,0,0,0)",font={"color":"white"})
    return fig

def radar_chart(breakdown):
    labels = list(breakdown.keys())
    vals   = list(breakdown.values())
    disp   = [l.replace("_"," ").title() for l in labels]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=disp+[disp[0]], fill="toself",
        fillcolor="rgba(102,126,234,0.2)", line=dict(color="#667eea",width=2)
    ))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True,range=[0,100],tickfont=dict(color="rgba(255,255,255,0.5)",size=9)),
        angularaxis=dict(tickfont=dict(color="white",size=11))),
        showlegend=False,paper_bgcolor="rgba(0,0,0,0)",height=350,
        margin=dict(l=60,r=60,t=40,b=40),font={"color":"white"})
    return fig

def render_chips(skills, chip_class, max_n=40):
    html = " ".join(f'<span class="skill-chip {chip_class}">{s}</span>' for s in sorted(skills)[:max_n])
    st.markdown(html, unsafe_allow_html=True)

def grade_html(g):
    return f'<span class="grade-badge grade-{g.lower()}">{g.upper()}</span>'


# ── Keyword Highlighter ───────────────────────────────────────────────────────
def highlight_text(text: str, matched: set, missing: set, max_chars=3000) -> str:
    """Return HTML with matched keywords highlighted green, missing ones red."""
    snippet = text[:max_chars]
    # Sort by length desc so compound phrases match first
    all_kw = sorted(matched | missing, key=len, reverse=True)
    for kw in all_kw:
        cls = "highlight-matched" if kw in matched else "highlight-missing"
        pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        snippet = pattern.sub(f'<mark class="{cls}">{kw}</mark>', snippet)
    return snippet.replace("\n", "<br>")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 AI Resume Screener")
    st.markdown("---")
    page = st.radio("Nav", [
        "📄 Upload & Analyze",
        "🎯 Job Matching",
        "🔍 Skill Gap Analysis",
        "📚 Recommendations",
        "✉️ Cover Letter",
        "💰 Salary Estimator",
        "📦 Batch Screener",
        "⚖️ Resume Compare",
        "📊 Domain Breakdown",
        "📈 Score History",
        "📊 Model Insights",
    ], label_visibility="collapsed")
    st.markdown("---")
    if st.session_state.ats_result:
        sc = st.session_state.ats_result["ats_score"]
        gr = st.session_state.ats_result["grade"]
        col = "#00c9a7" if sc>=70 else "#f7971e" if sc>=50 else "#eb3349"
        st.markdown(f"""
        <div style='text-align:center;padding:.8rem;background:rgba(102,126,234,.1);
        border-radius:12px;border:1px solid rgba(102,126,234,.3);'>
          <div style='font-size:.7rem;color:#8892b0;'>Current Score</div>
          <div style='font-size:2rem;font-weight:700;color:{col};'>{sc}</div>
          <div style='font-size:.8rem;color:{col};'>{gr}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("")
    st.caption("v3.0 · AI/ML Portfolio Project")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD & ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📄 Upload & Analyze":
    st.markdown("""<div class="hero-header">
        <h1>🤖 AI ATS Resume Screener</h1>
        <p>Upload · Score · Highlight · Export · Optimize</p></div>""", unsafe_allow_html=True)

    col_up, col_job = st.columns(2, gap="large")
    with col_up:
        st.markdown("#### 📄 Upload Resume")
        uploaded = st.file_uploader("PDF / DOCX / TXT", type=["pdf","docx","doc","txt"])
        if uploaded:
            with st.spinner("Extracting..."):
                try:
                    txt = extract_text_from_bytes(uploaded.read(), uploaded.name)
                    st.session_state.resume_text = txt
                    st.success(f"✅ {len(txt):,} chars from **{uploaded.name}**")
                except Exception as e:
                    st.error(f"❌ {e}")
        st.markdown("#### ✍️ Or Paste Resume")
        pasted = st.text_area("", height=180, placeholder="Paste resume text...", label_visibility="collapsed")
        if pasted.strip():
            st.session_state.resume_text = pasted

    with col_job:
        st.markdown("#### 💼 Job Description")
        job_title = st.text_input("Job Title", placeholder="Senior Data Scientist")
        job_desc  = st.text_area("", height=140, placeholder="Paste JD here...", label_visibility="collapsed")
        req_skills = st.text_input("Required Skills (optional)", placeholder="Python, ML, SQL")
        c1,c2 = st.columns(2)
        min_exp = c1.number_input("Min Exp (yrs)", 0.0, 25.0, 3.0, 0.5)
        edu_req = c2.selectbox("Education", ["Bachelors","Masters","PhD","Associate","High School"])

    st.markdown("---")
    col_btn, col_opt = st.columns([2,1])
    analyze_btn = col_btn.button("🚀 Analyze Resume", type="primary", use_container_width=True)
    show_highlighter = col_opt.checkbox("🔑 Show Keyword Highlighter", value=True)

    if analyze_btn:
        if not st.session_state.resume_text:
            st.error("⚠️ Upload or paste a resume first.")
        elif not job_desc.strip():
            st.error("⚠️ Enter a job description.")
        else:
            with st.spinner("🧠 Running AI analysis..."):
                rt = st.session_state.resume_text
                yrs = extract_years_experience(rt)
                edu = extract_education_level(rt)
                rsk = extract_skills(rt)
                jsk = {s.strip().lower() for s in req_skills.split(",") if s.strip()} if req_skills.strip() else extract_skills(job_desc)

                result = compute_ats_score(rt, job_desc, rsk, jsk, yrs, min_exp, edu, edu_req)
                st.session_state.ats_result  = result
                st.session_state.job_text    = job_desc
                st.session_state.gap_result  = result["skill_gap"]
                st.session_state.predicted_role = job_title or "Software Engineering"

                # Append to history
                st.session_state.score_history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "job": (job_title or job_desc[:40]).strip(),
                    "score": result["ats_score"],
                    "grade": result["grade"],
                })

    if st.session_state.ats_result:
        result = st.session_state.ats_result
        score  = result["ats_score"]
        bd     = result["breakdown"]
        gap    = result["skill_gap"]
        yrs    = extract_years_experience(st.session_state.resume_text or "")
        edu    = extract_education_level(st.session_state.resume_text or "")

        st.markdown("## 📊 ATS Results")
        m1,m2,m3,m4 = st.columns(4)
        for col, val, label, color in [
            (m1, score, "ATS Score", "#667eea"),
            (m2, f"{gap['total_matched']}/{gap['total_required']}", "Skills Matched", "#00c9a7"),
            (m3, f"{yrs:.1f}", "Yrs Experience", "#f7971e"),
            (m4, edu, "Education", "#b264fb"),
        ]:
            col.markdown(f"""<div class="metric-box">
              <div class="metric-val" style="color:{color}">{val}</div>
              <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

        gc, rc = st.columns(2)
        with gc:
            st.plotly_chart(gauge_chart(score), use_container_width=True)
            st.markdown(f"<center>{grade_html(result['grade'])}</center>", unsafe_allow_html=True)
        with rc:
            st.plotly_chart(radar_chart(bd), use_container_width=True)

        # Breakdown table
        weights = ["30%","30%","15%","10%","10%","5%"]
        bdf = pd.DataFrame([{"Component":k.replace("_"," ").title(),"Score":round(v,1),"Weight":w}
                             for (k,v),w in zip(bd.items(),weights)])
        st.dataframe(bdf.style.background_gradient(subset=["Score"],cmap="RdYlGn"),
                     use_container_width=True, hide_index=True)

        # Sections
        secs = result["sections_found"]
        sc_ = st.columns(len(secs))
        for i,(s,f) in enumerate(secs.items()):
            sc_[i].markdown(f"<center><b style='color:{'#00c9a7' if f else '#eb3349'}'>{'✅' if f else '❌'} {s.title()}</b></center>",
                            unsafe_allow_html=True)

        # Suggestions
        if result["suggestions"]:
            st.markdown("#### 💡 AI Suggestions")
            for s in result["suggestions"]:
                st.markdown(f'<div class="suggestion-card">{s}</div>', unsafe_allow_html=True)

        # Keyword highlighter
        if show_highlighter:
            st.markdown("#### 🔑 Keyword Highlighter")
            matched_set = set(gap["matched_skills"])
            missing_set = set(gap["missing_skills"])
            st.markdown("""<small style='color:#8892b0;'>
              <span style='background:rgba(0,201,167,.25);padding:0 4px;border-radius:3px;'>✅ Matched</span> &nbsp;
              <span style='background:rgba(235,51,73,.25);padding:0 4px;border-radius:3px;'>❌ Missing</span>
            </small>""", unsafe_allow_html=True)
            html = highlight_text(st.session_state.resume_text or "", matched_set, missing_set)
            st.markdown(f"<div style='background:#1a1f2e;padding:1rem;border-radius:10px;font-size:.88rem;line-height:1.7;max-height:350px;overflow-y:auto;'>{html}</div>",
                        unsafe_allow_html=True)

        # Skills chips
        sc1,sc2 = st.columns(2)
        with sc1:
            st.markdown("#### ✅ Matched Skills")
            render_chips(gap["matched_skills"], "skill-matched")
        with sc2:
            st.markdown("#### ❌ Missing Skills")
            render_chips(gap["missing_skills"], "skill-missing")

        # ── EXPORTS ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📤 Export Results")
        e1,e2,e3 = st.columns(3)

        # JSON export
        with e1:
            export_data = {
                "ats_score": score, "grade": result["grade"],
                "breakdown": bd, "matched_skills": gap["matched_skills"],
                "missing_skills": gap["missing_skills"], "suggestions": result["suggestions"],
            }
            st.download_button("📋 Download JSON", data=json.dumps(export_data, indent=2),
                               file_name="ats_report.json", mime="application/json",
                               use_container_width=True)

        # CSV export
        with e2:
            rows = [{"metric":"ATS Score","value":score},{"metric":"Grade","value":result["grade"]}]
            rows += [{"metric":k.replace("_"," ").title(),"value":v} for k,v in bd.items()]
            csv_buf = io.StringIO()
            writer = csv.DictWriter(csv_buf, fieldnames=["metric","value"])
            writer.writeheader(); writer.writerows(rows)
            st.download_button("📊 Download CSV", data=csv_buf.getvalue(),
                               file_name="ats_report.csv", mime="text/csv",
                               use_container_width=True)

        # PDF export
        with e3:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                try:
                    from utils.pdf_exporter import generate_pdf_report
                    pdf_bytes = generate_pdf_report(result, st.session_state.resume_text or "",
                                                    st.session_state.predicted_role or "")
                    st.download_button("⬇️ Download PDF", data=pdf_bytes,
                                       file_name="ats_report.pdf", mime="application/pdf",
                                       use_container_width=True)
                except ImportError:
                    st.warning("Install fpdf2: `pip install fpdf2`")

        # ── OPTIMIZER ────────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("🔁 Resume Score Optimizer — How to hit 80+"):
            if score >= 80:
                st.success("🎉 Your resume already scores 80+! Focus on the cover letter now.")
            else:
                gap_to_80 = 80 - score
                st.markdown(f"**You need +{gap_to_80:.1f} points to reach 80.**")
                actions = []
                if bd["keyword_match"] < 70:
                    actions.append(f"**+{min(gap_to_80,12):.0f} pts** — Mirror more phrases from the JD in your resume (tools, methodologies, outcomes)")
                if bd["skill_match"] < 70:
                    top3 = gap["ranked_missing"][:3]
                    skills_str = ", ".join(s["skill"] for s in top3)
                    actions.append(f"**+{min(gap_to_80,10):.0f} pts** — Add or demonstrate these skills: {skills_str}")
                if bd["structure"] < 80:
                    missing_s = [k for k,v in result["sections_found"].items() if not v]
                    if missing_s:
                        actions.append(f"**+{min(gap_to_80,8):.0f} pts** — Add missing sections: {', '.join(missing_s)}")
                if bd["readability"] < 60:
                    actions.append("**+5 pts** — Rewrite bullets as 'Action Verb + Result + Metric' (e.g., 'Built X achieving 30% faster Y')")
                if not actions:
                    actions.append("Tailor your resume more closely to the exact wording of the job description.")
                for a in actions:
                    st.markdown(f"• {a}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — JOB MATCHING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Job Matching":
    st.markdown("""<div class="hero-header"><h1>🎯 Job Matching Engine</h1>
        <p>AI cosine-similarity matching against the job database</p></div>""", unsafe_allow_html=True)

    resume_input = st.text_area("Resume Text", value=st.session_state.resume_text or "", height=200,
                                placeholder="Paste resume or upload on Upload page...")
    if resume_input: st.session_state.resume_text = resume_input
    top_n = st.slider("Top matches", 1, 15, 5)

    if st.button("🔍 Find Matching Jobs", type="primary", use_container_width=True):
        if not resume_input.strip(): st.error("Enter a resume first.")
        else:
            with st.spinner("Scanning..."):
                try:
                    m = load_matcher()
                    st.session_state.match_result = m.match(resume_input, top_n=top_n)
                except Exception as e: st.error(str(e))

    if st.session_state.match_result:
        matches = st.session_state.match_result
        titles = [m["job_title"] for m in matches]
        scores = [m["match_percentage"] for m in matches]
        fig = px.bar(x=scores,y=titles,orientation="h",color=scores,
                     color_continuous_scale=["#eb3349","#f7971e","#00c9a7"],
                     labels={"x":"Match %","y":"Job"}, range_color=[0,100])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,25,40,.8)",
                          font={"color":"white"},height=max(280,len(matches)*55),
                          coloraxis_showscale=False,yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        for m in matches:
            with st.expander(f"#{m['rank']} {m['job_title']} — {m['match_percentage']:.1f}%"):
                c1,c2 = st.columns([2,1])
                c1.markdown(f"**Snippet:** {m['description_snippet']}")
                c1.markdown(f"**Skills:** `{m['required_skills']}`")
                c2.metric("Match",f"{m['match_percentage']:.1f}%")
                c2.metric("Min Exp",f"{m['min_experience']} yrs")
                c2.metric("Salary",f"${m['salary_range']}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SKILL GAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Skill Gap Analysis":
    st.markdown("""<div class="hero-header"><h1>🔍 Skill Gap Detector</h1>
        <p>Identify exactly which skills bridge the gap to your dream role</p></div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2, gap="large")
    resume_in = c1.text_area("Your Resume", value=st.session_state.resume_text or "",
                              height=250, placeholder="Paste resume...")
    job_in    = c2.text_area("Job Description", value=st.session_state.job_text or "",
                              height=250, placeholder="Paste job description...")

    if st.button("🔍 Analyze Skill Gap", type="primary", use_container_width=True):
        if not resume_in.strip() or not job_in.strip():
            st.error("Need both resume and JD.")
        else:
            with st.spinner("Analyzing..."):
                gap = compute_skill_gap(resume_in, job_in)
                st.session_state.gap_result = gap
                st.session_state.resume_text = resume_in
                st.session_state.job_text = job_in

    if st.session_state.gap_result:
        gap = st.session_state.gap_result
        fig = go.Figure(go.Pie(labels=["Matched","Missing"],
            values=[gap["total_matched"], max(gap["total_required"]-gap["total_matched"],0)],
            hole=.65,marker_colors=["#00c9a7","#eb3349"],textfont=dict(color="white",size=12)))
        fig.add_annotation(text=f"{gap['match_score']:.0f}%<br>Match",
                           x=.5,y=.5,showarrow=False,font=dict(size=20,color="white"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=300,
                          margin=dict(l=0,r=0,t=20,b=0),legend=dict(font=dict(color="white")))

        m1,m2,m3 = st.columns(3)
        m1.metric("✅ Matched",f"{gap['total_matched']}")
        m2.metric("❌ Missing",f"{len(gap['missing_skills'])}")
        m3.metric("⚡ Match",f"{gap['match_score']:.1f}%")
        st.plotly_chart(fig, use_container_width=False)

        if gap["ranked_missing"]:
            st.markdown("#### 📋 Missing Skills by Priority")
            df = pd.DataFrame(gap["ranked_missing"])
            def _pri_style(val):
                c={"Critical":"#eb3349","High":"#f7971e","Medium":"#667eea","Low":"#8892b0"}.get(val,"white")
                return f"color:{c};font-weight:600"
            st.dataframe(df[["skill","importance","domain","priority"]].style.applymap(_pri_style,subset=["priority"]),
                         use_container_width=True,hide_index=True)

        r1,r2,r3 = st.columns(3)
        with r1: st.markdown("#### ✅ Matched"); render_chips(gap["matched_skills"],"skill-matched")
        with r2: st.markdown("#### ❌ Missing"); render_chips(gap["missing_skills"],"skill-missing")
        with r3: st.markdown("#### 💎 Extra");  render_chips(gap["extra_skills"],"skill-extra")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Recommendations":
    st.markdown("""<div class="hero-header"><h1>📚 Learning Roadmap</h1>
        <p>Personalized courses, certifications, and career progression</p></div>""", unsafe_allow_html=True)

    if not st.session_state.gap_result:
        st.info("Run Skill Gap Analysis first.")
        ms = st.text_input("Or enter missing skills manually", placeholder="python, mlflow, docker")
        role = st.selectbox("Target Role",["Data Science","AI/ML","Software Engineering","DevOps/Cloud","Data Engineering","Cybersecurity"])
        pct  = st.slider("Skill Match %",0,100,40)
        if st.button("Generate Roadmap",type="primary"):
            sl = [s.strip().lower() for s in ms.split(",") if s.strip()]
            st.session_state.gap_result = {"ranked_missing":[{"skill":s}for s in sl],"match_score":pct}
            st.session_state.predicted_role = role
            st.rerun()
    else:
        gap = st.session_state.gap_result
        missing = [s["skill"] for s in gap.get("ranked_missing",[])]
        pct = gap.get("match_score",50)
        role = st.session_state.predicted_role or "Software Engineering"

        roadmap = recommend_learning_path(missing[:12])
        career  = suggest_career_path(role, pct)
        stages  = career["career_stages"]
        cur_idx = stages.index(career["estimated_current_level"]) if career["estimated_current_level"] in stages else 0

        st.markdown("### 🚀 Career Path")
        cols = st.columns(len(stages))
        for i,(col,s) in enumerate(zip(cols,stages)):
            icon,color = ("✅","#00c9a7") if i<cur_idx else ("📍","#667eea") if i==cur_idx else ("⭕","#8892b0")
            col.markdown(f"<center style='color:{color};font-size:.8rem;font-weight:600'>{icon}<br>{s}</center>",unsafe_allow_html=True)

        st.markdown(f"""<div class="score-card">
          <b>🎯 Next:</b> <span style='color:#667eea'>{career['next_step']}</span><br>
          <b>📍 Now:</b> {career['estimated_current_level']}</div>""", unsafe_allow_html=True)
        for a in career["recommended_actions"]: st.markdown(f"- {a}")

        st.markdown("### 📖 Skill Roadmap")
        for item in roadmap:
            dc = {"Beginner":"#00c9a7","Intermediate":"#f7971e","Advanced":"#eb3349"}.get(item["difficulty"],"white")
            with st.expander(f"📘 {item['skill'].title()} — 🕐 {item['estimated_time']} — **{item['difficulty']}**"):
                if item["courses"]:
                    st.markdown("**📚 Courses:**")
                    for c in item["courses"]: st.markdown(f"  - {c}")
                if item["certifications"]:
                    st.markdown("**🏆 Certifications:**")
                    for c in item["certifications"]: st.markdown(f"  - 🎖️ {c}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — COVER LETTER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "✉️ Cover Letter":
    st.markdown("""<div class="hero-header"><h1>✉️ Cover Letter Generator</h1>
        <p>AI-tailored cover letters in 3 styles based on your resume analysis</p></div>""", unsafe_allow_html=True)

    col1,col2 = st.columns([2,1])
    with col1:
        jt = st.text_input("Job Title *", placeholder="Senior Data Scientist",
                           value=st.session_state.predicted_role or "")
        co = st.text_input("Company Name *", placeholder="Google, Microsoft, OpenAI…")
    with col2:
        style = st.selectbox("Letter Style", list(TEMPLATES.keys()))
        role_sel = st.selectbox("Your Domain",
            ["Data Science","AI/ML","Software Engineering","DevOps/Cloud","Data Engineering","Cybersecurity"])

    gap = st.session_state.gap_result
    result = st.session_state.ats_result
    resume_text = st.session_state.resume_text or ""

    matched = gap["matched_skills"] if gap else []
    missing = gap["missing_skills"] if gap else []
    yrs  = extract_years_experience(resume_text) if resume_text else 0
    edu  = extract_education_level(resume_text) if resume_text else "Unknown"

    if not gap:
        st.info("💡 Run an analysis on the Upload page first for best results — or fill details manually above.")

    if st.button("✉️ Generate Cover Letter", type="primary", use_container_width=True):
        if not jt.strip() or not co.strip():
            st.error("Please fill in job title and company name.")
        else:
            cl = generate_cover_letter(jt, co, yrs, edu, matched, missing, role_sel, style)
            st.session_state.cover_letter = cl

    if st.session_state.cover_letter:
        st.markdown("#### 📝 Your Cover Letter")
        editable = st.text_area("Edit before copying", value=st.session_state.cover_letter, height=500)
        dl1,dl2 = st.columns(2)
        dl1.download_button("⬇️ Download .txt", data=editable,
                            file_name="cover_letter.txt", mime="text/plain", use_container_width=True)
        if dl2.button("🔄 Regenerate", use_container_width=True):
            cl = generate_cover_letter(jt, co, yrs, edu, matched, missing, role_sel, style)
            st.session_state.cover_letter = cl
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SALARY ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Salary Estimator":
    st.markdown("""<div class="hero-header"><h1>💰 Salary Estimator</h1>
        <p>Market-rate salary prediction based on your skills, experience & location</p></div>""", unsafe_allow_html=True)

    result = st.session_state.ats_result
    gap    = st.session_state.gap_result
    resume_text = st.session_state.resume_text or ""

    c1,c2,c3 = st.columns(3)
    role_sel = c1.selectbox("Your Role Domain",
        ["Data Science","AI/ML","Software Engineering","DevOps/Cloud","Data Engineering","Cybersecurity"],
        index=0)
    location = c2.selectbox("Location", get_location_options())
    yrs_override = c3.number_input("Years Experience", 0.0, 25.0,
        float(extract_years_experience(resume_text) if resume_text else 3.0), 0.5)

    matched = gap["matched_skills"] if gap else []
    edu = extract_education_level(resume_text) if resume_text else "Unknown"

    if st.button("💰 Calculate Salary Range", type="primary", use_container_width=True):
        est = estimate_salary(matched, yrs_override, edu, role_sel, location)

        sa,sb,sc_ = st.columns(3)
        sa.markdown(f"""<div class="salary-band">
          <div class="sal-label">Conservative</div>
          <div class="sal-num">${est['low_usd']:,}</div>
          <div class="sal-label">Base estimate</div></div>""", unsafe_allow_html=True)
        sb.markdown(f"""<div class="salary-band" style="border-color:#00c9a7;">
          <div class="sal-label">Market Rate</div>
          <div class="sal-num" style="color:#667eea;">${est['mid_usd']:,}</div>
          <div class="sal-label">You should target this</div></div>""", unsafe_allow_html=True)
        sc_.markdown(f"""<div class="salary-band" style="border-color:#f093fb;">
          <div class="sal-label">Top of Market</div>
          <div class="sal-num" style="color:#f093fb;">${est['high_usd']:,}</div>
          <div class="sal-label">With strong negotiation</div></div>""", unsafe_allow_html=True)

        if location != "United States":
            st.info(f"📍 **{location} adjusted:** ${est['low_local']:,} — ${est['mid_local']:,} — ${est['high_local']:,} (×{est['location_factor']})")

        st.markdown("---")
        d1,d2,d3 = st.columns(3)
        d1.metric("🎓 Education Bonus", f"${est['education_bonus']:+,}")
        d2.metric("⚡ Skill Premium", f"${est['skill_premium']:,}")
        d3.metric("📊 Exp Percentile", est["experience_percentile"])

        if est["top_contributing_skills"]:
            st.markdown("#### 🏆 Your Highest-Value Skills")
            sdf = pd.DataFrame(est["top_contributing_skills"])
            fig_s = px.bar(sdf, x="bonus", y="skill", orientation="h",
                           color="bonus", color_continuous_scale=["#667eea","#00c9a7"],
                           labels={"bonus":"Salary Bonus $","skill":"Skill"})
            fig_s.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,25,40,.8)",
                                font={"color":"white"},height=300,coloraxis_showscale=False,
                                yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("""<small style='color:#8892b0;'>⚠️ Estimates based on 2024-2025 market data (USD).
        Actual offers vary by company, negotiation, and local market conditions.</small>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — BATCH SCREENER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Batch Screener":
    st.markdown("""<div class="hero-header"><h1>📦 Batch Resume Screener</h1>
        <p>Upload multiple resumes and rank them against one job description — recruiter mode</p></div>""", unsafe_allow_html=True)

    jd_text = st.text_area("Job Description *", height=150, placeholder="Paste the JD to rank candidates against...")
    c1,c2 = st.columns(2)
    min_exp_b = c1.number_input("Min Experience (yrs)", 0.0, 20.0, 2.0, 0.5)
    edu_req_b = c2.selectbox("Min Education", ["Bachelors","Masters","PhD","Associate","Any"])

    uploaded_batch = st.file_uploader(
        "Upload multiple resumes (PDF/DOCX/TXT)",
        type=["pdf","docx","doc","txt"],
        accept_multiple_files=True,
    )

    if st.button("🚀 Screen All Resumes", type="primary", use_container_width=True):
        if not jd_text.strip():
            st.error("Enter a job description first.")
        elif not uploaded_batch:
            st.error("Upload at least one resume.")
        else:
            results = []
            prog = st.progress(0)
            total = len(uploaded_batch)
            for i, f in enumerate(uploaded_batch):
                try:
                    txt = extract_text_from_bytes(f.read(), f.name)
                    yrs = extract_years_experience(txt)
                    edu = extract_education_level(txt)
                    rsk = extract_skills(txt)
                    jsk = extract_skills(jd_text)
                    ats = compute_ats_score(txt, jd_text, rsk, jsk, yrs,
                                           min_exp_b, edu,
                                           "Any" if edu_req_b=="Any" else edu_req_b)
                    results.append({
                        "Filename": f.name,
                        "ATS Score": ats["ats_score"],
                        "Grade": ats["grade"],
                        "Skills Matched": ats["skill_gap"]["total_matched"],
                        "Skills Required": ats["skill_gap"]["total_required"],
                        "Yrs Exp": yrs,
                        "Education": edu,
                        "Missing Skills": ", ".join(ats["skill_gap"]["missing_skills"][:5]),
                    })
                except Exception as e:
                    results.append({"Filename":f.name,"ATS Score":0,"Grade":"Error",
                                    "Skills Matched":0,"Skills Required":0,"Yrs Exp":0,
                                    "Education":"?","Missing Skills":str(e)})
                prog.progress((i+1)/total)

            df = pd.DataFrame(results).sort_values("ATS Score", ascending=False).reset_index(drop=True)
            df.insert(0, "Rank", range(1, len(df)+1))

            st.markdown(f"### 🏆 Ranked {len(df)} Candidates")
            st.dataframe(df.style.background_gradient(subset=["ATS Score"], cmap="RdYlGn"),
                         use_container_width=True, hide_index=True)

            csv_b = df.to_csv(index=False)
            st.download_button("📥 Download Rankings CSV", csv_b,
                               file_name="batch_screening.csv", mime="text/csv",
                               use_container_width=True)

            # Bar chart
            fig_b = px.bar(df, x="ATS Score", y="Filename", orientation="h",
                           color="ATS Score", color_continuous_scale=["#eb3349","#f7971e","#00c9a7"],
                           range_color=[0,100], labels={"Filename":"Candidate"})
            fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,25,40,.8)",
                                font={"color":"white"},height=max(250,len(df)*50),
                                coloraxis_showscale=False,yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_b, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — RESUME COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Resume Compare":
    st.markdown("""<div class="hero-header"><h1>⚖️ Resume Comparator</h1>
        <p>Compare two resume versions side-by-side to see which performs better</p></div>""", unsafe_allow_html=True)

    jd_cmp = st.text_area("Job Description (same for both)", height=120,
                           value=st.session_state.job_text or "",
                           placeholder="Paste the target job description...")
    c1,c2 = st.columns(2)
    rv1 = c1.text_area("📄 Resume Version A", height=280, placeholder="Paste Version A…")
    rv2 = c2.text_area("📄 Resume Version B", height=280, placeholder="Paste Version B…")

    c_exp,c_edu = st.columns(2)
    min_e = c_exp.number_input("Min Exp (yrs)", 0.0, 20.0, 2.0, 0.5)
    edu_r = c_edu.selectbox("Min Education", ["Bachelors","Masters","PhD","Associate","High School"])

    if st.button("⚖️ Compare Resumes", type="primary", use_container_width=True):
        if not jd_cmp.strip():
            st.error("Enter a job description.")
        elif not rv1.strip() or not rv2.strip():
            st.error("Paste both resume versions.")
        else:
            def _score(rt, jt, me, er):
                y = extract_years_experience(rt)
                e = extract_education_level(rt)
                rs = extract_skills(rt); js = extract_skills(jt)
                return compute_ats_score(rt, jt, rs, js, y, me, e, er), y, e

            with st.spinner("Comparing..."):
                r1,y1,e1 = _score(rv1, jd_cmp, min_e, edu_r)
                r2,y2,e2 = _score(rv2, jd_cmp, min_e, edu_r)

            s1,s2 = r1["ats_score"], r2["ats_score"]
            winner = "A" if s1>s2 else "B" if s2>s1 else "Tie"

            if winner != "Tie":
                st.success(f"🏆 **Resume {winner}** wins with **{max(s1,s2):.1f}** vs **{min(s1,s2):.1f}** (+{abs(s1-s2):.1f} pts)")
            else:
                st.info("🤝 Both resumes score equally.")

            # Side-by-side metrics
            m1col,m2col = st.columns(2)
            for col, res, label, yrs, edu in [(m1col,r1,"Version A",y1,e1),(m2col,r2,"Version B",y2,e2)]:
                s = res["ats_score"]
                c = "#00c9a7" if s>=70 else "#f7971e" if s>=50 else "#eb3349"
                col.markdown(f"### {label}")
                col.plotly_chart(gauge_chart(s,label), use_container_width=True)
                col.markdown(f"<center>{grade_html(res['grade'])}</center>",unsafe_allow_html=True)
                col.markdown(f"**Matched:** {res['skill_gap']['total_matched']}/{res['skill_gap']['total_required']} skills &nbsp;|&nbsp; **Exp:** {yrs:.1f} yrs &nbsp;|&nbsp; **Edu:** {edu}")

            # Breakdown comparison chart
            st.markdown("#### 📊 Score Breakdown Comparison")
            bd1 = r1["breakdown"]; bd2 = r2["breakdown"]
            cats = [k.replace("_"," ").title() for k in bd1]
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name="Version A",x=cats,y=list(bd1.values()),marker_color="#667eea"))
            fig_cmp.add_trace(go.Bar(name="Version B",x=cats,y=list(bd2.values()),marker_color="#00c9a7"))
            fig_cmp.update_layout(barmode="group",paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(20,25,40,.8)",font={"color":"white"},
                                   height=350,legend=dict(font=dict(color="white")))
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Delta table
            delta_rows = [{"Component":k.replace("_"," ").title(),
                           "Version A":round(v,1),"Version B":round(bd2[k],1),
                           "Δ (B-A)":round(bd2[k]-v,1)} for k,v in bd1.items()]
            ddf = pd.DataFrame(delta_rows)
            def _dstyle(val):
                if isinstance(val,(int,float)):
                    return "color:#00c9a7;font-weight:600" if val>0 else "color:#eb3349;font-weight:600" if val<0 else ""
                return ""
            st.dataframe(ddf.style.applymap(_dstyle,subset=["Δ (B-A)"]),
                         use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — DOMAIN BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Domain Breakdown":
    st.markdown("""<div class="hero-header"><h1>📊 Domain Skills Breakdown</h1>
        <p>See which domains you master — and where to grow</p></div>""", unsafe_allow_html=True)

    rt = st.text_area("Paste Resume", value=st.session_state.resume_text or "",
                       height=180, placeholder="Paste your resume...")
    if rt.strip():
        with st.spinner("Analyzing domains..."):
            ds = extract_skills_by_domain(rt)
        if ds:
            dn = list(ds.keys()); dc = [len(v) for v in ds.values()]
            fig = px.bar(x=dn,y=dc,color=dc,color_continuous_scale=["#667eea","#00c9a7"],
                         labels={"x":"Domain","y":"Skills Found"},title="Skills per Domain")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,25,40,.8)",
                              font={"color":"white"},height=380,coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)
            for domain,skills in ds.items():
                with st.expander(f"**{domain}** — {len(skills)} skills"):
                    render_chips(list(skills),"skill-matched",50)
        else:
            st.warning("No known skills found. Try a more detailed resume.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — SCORE HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Score History":
    st.markdown("""<div class="hero-header"><h1>📈 ATS Score History</h1>
        <p>Track your resume improvement over time</p></div>""", unsafe_allow_html=True)

    history = st.session_state.score_history or []
    if not history:
        st.info("📊 Analyze at least 2 resumes or iterations on the Upload page to see your progress here.")
        st.markdown("""**How to use:**
        1. Paste your resume → Analyze → note the score
        2. Edit your resume based on suggestions
        3. Paste again → Analyze → see the improvement!""")
    else:
        df_h = pd.DataFrame(history)
        df_h["run"] = [f"Run {i+1}" for i in range(len(df_h))]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=df_h["run"], y=df_h["score"],
            mode="lines+markers+text",
            text=[f"{s:.0f}" for s in df_h["score"]],
            textposition="top center",
            line=dict(color="#667eea",width=3),
            marker=dict(size=12,color=df_h["score"],colorscale=[[0,"#eb3349"],[.5,"#f7971e"],[1,"#00c9a7"]],
                        showscale=False),
            name="ATS Score",
        ))
        fig_hist.add_hline(y=70,line_dash="dash",line_color="rgba(0,201,167,.5)",
                           annotation_text="Target: 70",annotation_font=dict(color="#00c9a7"))
        fig_hist.add_hline(y=85,line_dash="dot",line_color="rgba(102,126,234,.4)",
                           annotation_text="Excellent: 85",annotation_font=dict(color="#667eea"))
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,25,40,.8)",
            font={"color":"white"},height=400,yaxis=dict(range=[0,105]),
            xaxis=dict(tickfont=dict(color="white")),
            title="ATS Score Progress Over Time",
        )
        st.plotly_chart(fig_hist,use_container_width=True)

        # Stats
        scores = [h["score"] for h in history]
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("🏁 Latest Score", f"{scores[-1]:.1f}")
        s2.metric("📈 Best Score",   f"{max(scores):.1f}")
        s3.metric("📊 Avg Score",    f"{sum(scores)/len(scores):.1f}")
        s4.metric("⬆️ Improvement",  f"{scores[-1]-scores[0]:+.1f}" if len(scores)>1 else "—")

        st.markdown("#### 📋 History Log")
        st.dataframe(pd.DataFrame(history)[["timestamp","job","score","grade"]],
                     use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear History"):
            st.session_state.score_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 11 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Insights":
    st.markdown("""<div class="hero-header"><h1>📊 Model Insights</h1>
        <p>Feature importance, model comparison, and explainability</p></div>""", unsafe_allow_html=True)

    saved_dir = ROOT / "models" / "saved"
    cmp_file = saved_dir / "classifier_comparison.json"
    if cmp_file.exists():
        with open(cmp_file) as f: cmp = json.load(f)
        df_c = pd.DataFrame(cmp)
        fig_c = px.bar(df_c,x="model",y=["accuracy","f1_weighted","precision_weighted","recall_weighted"],
                       barmode="group",color_discrete_sequence=["#667eea","#00c9a7","#f7971e","#eb3349"])
        fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,25,40,.8)",
                             font={"color":"white"},height=380)
        st.plotly_chart(fig_c,use_container_width=True)
    else:
        st.info("🔧 Train models first: `python models/train_models.py`")

    st.markdown("### 🔍 Hiring Predictor Feature Importance")
    try:
        from models.hiring_predictor import HiringPredictor
        pred = HiringPredictor(); pred.load(str(saved_dir))
        fi = pred.feature_importance(); df_fi = pd.DataFrame(fi)
        fig_fi = px.bar(df_fi,x="importance",y="feature",orientation="h",
                        color="importance",color_continuous_scale=["#667eea","#00c9a7"])
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,25,40,.8)",
                              font={"color":"white"},height=380,coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi,use_container_width=True)
    except Exception:
        st.info("Train the hiring predictor first.")

    st.markdown("### 💡 Feature Explanations")
    for feat,desc in [
        ("ats_score","Overall ATS match — strongest single predictor"),
        ("skill_match_score","% of required skills found in resume"),
        ("keyword_match_score","TF-IDF keyword overlap with JD"),
        ("years_experience","Detected years of professional experience"),
        ("education_level","Encoded: 0=Unknown → 5=PhD"),
        ("skill_count","Total unique skills detected"),
        ("structure_score","Resume section completeness (%),"),
    ]:
        st.markdown(f"- **`{feat}`**: {desc}")
