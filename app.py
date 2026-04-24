"""
app.py — Red Flag Detector
Streamlit frontend. Supports conversation and scenario inputs.
Analyzes narrator + optionally the other person in scenarios.
"""

import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

from utils.preprocessor import (
    parse_input, get_target_turns, get_available_speakers,
    get_conversation_stats, extract_quoted_speech, InputMode
)
from utils.feature_extractor import extract_features, score_summary
from utils.llm_analyzer import analyze_with_llm, get_red_dimensions, get_green_dimensions
from utils.aggregator import aggregate

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Red Flag Detector", page_icon="🚩", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.stApp { background: #0d0d0d; color: #f0ece4; }
section[data-testid="stSidebar"] { background: #141414; border-right: 1px solid #1f1f1f; }
.verdict-box { border-radius: 16px; padding: 28px; text-align: center; margin: 16px 0; font-family: 'Syne', sans-serif; }
.red-verdict   { background: linear-gradient(135deg,#2d0a0a,#1a0505); border:1px solid #ef4444; box-shadow:0 0 40px rgba(239,68,68,.15); }
.green-verdict { background: linear-gradient(135deg,#052d0f,#021a07); border:1px solid #22c55e; box-shadow:0 0 40px rgba(34,197,94,.15); }
.neutral-verdict { background: linear-gradient(135deg,#1a1400,#0d0b00); border:1px solid #f59e0b; box-shadow:0 0 40px rgba(245,158,11,.10); }
.verdict-title { font-size:2.4rem; font-weight:800; margin-bottom:6px; letter-spacing:-1px; }
.verdict-confidence { font-size:.9rem; opacity:.6; font-weight:300; font-family:'DM Sans',sans-serif; }
.flag-item { display:flex; align-items:flex-start; gap:10px; padding:10px 14px; border-radius:8px; margin-bottom:8px; font-size:.88rem; }
.metric-card { background:#141414; border:1px solid #1f1f1f; border-radius:12px; padding:14px 18px; margin-bottom:10px; }
.section-label { font-family:'Syne',sans-serif; font-size:.72rem; text-transform:uppercase; letter-spacing:2px; color:#555; margin-bottom:4px; }
.example-box { background:#141414; border:1px solid #2a2a2a; border-radius:10px; padding:12px 14px; font-size:.8rem; color:#666; font-family:'DM Sans',monospace; line-height:1.8; }
.stTextArea textarea { background:#141414!important; color:#f0ece4!important; border:1px solid #2a2a2a!important; border-radius:10px!important; }
.stButton>button { background:linear-gradient(135deg,#ef4444,#dc2626)!important; color:white!important; border:none!important; border-radius:10px!important; font-family:'Syne',sans-serif!important; font-weight:700!important; padding:12px 30px!important; width:100%!important; }
.stButton>button:hover { transform:translateY(-1px)!important; box-shadow:0 8px 24px rgba(239,68,68,.3)!important; }
hr { border-color:#1f1f1f!important; }
</style>
""", unsafe_allow_html=True)


# ── Verdict image paths — SET YOUR PATHS HERE ─────────────────────────────────
VERDICT_IMAGES = {
    "red_flag":   r"D:\BehavioralAnalysis\images\red_flag.png",
    "green_flag": r"D:\BehavioralAnalysis\images\green_flag.png",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='font-family:Syne;font-size:1.3rem;margin-bottom:4px;'>⚙️ Settings</h2>", unsafe_allow_html=True)
    st.markdown("---")

    use_llm = st.toggle("Enable LLM Analysis", value=True,
                        help="Uses Qwen2.5-7B via HuggingFace for deep behavioral scoring.")

    st.markdown("---")
    st.markdown("<div class='section-label'>Mode 1 · Conversation</div>", unsafe_allow_html=True)
    st.markdown("""<div class='example-box'>
Person A: You never listen to me.<br>
Person B: I do listen, go on.<br>
Person A: If you leave I'll hurt myself.<br>
</div>""", unsafe_allow_html=True)

    st.markdown("<br><div class='section-label'>Mode 2 · Scenario / Reddit post</div>", unsafe_allow_html=True)
    st.markdown("""<div class='example-box'>
So me and this person met online two years ago.<br>
All I could think about was what I wanted from her.<br>
Now she's back and I feel guilty for how I acted...<br>
</div>""", unsafe_allow_html=True)
    st.caption("✨ Mode auto-detected from input.")

    st.markdown("---")
    st.caption("Pipeline: Rule-based NLP + Qwen2.5-7B behavioral scoring via HuggingFace Inference API.")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:36px 0 16px 0;'>
  <h1 style='font-family:Syne;font-size:2.6rem;font-weight:800;margin:0;letter-spacing:-1px;'>🚩 Red Flag Detector</h1>
  <p style='color:#555;font-size:.95rem;margin-top:6px;'>Conversation & scenario analysis · NLP + LLM behavioral pipeline</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
col_in, col_cfg = st.columns([2, 1])

with col_in:
    st.markdown("<div class='section-label'>Paste Conversation or Scenario</div>", unsafe_allow_html=True)
    raw_input = st.text_area(
        label="input", label_visibility="collapsed", height=260,
        placeholder="Paste a chat (Person A: ...) or a first-person story.\nMode is auto-detected.",
    )

with col_cfg:
    st.markdown("<div class='section-label'>Conversation: who to assess?</div>", unsafe_allow_html=True)
    st.caption("Ignored for scenario mode.")
    target_choice = st.selectbox("Target", label_visibility="collapsed",
                                 options=["Person A", "Person B"])

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_other = st.checkbox(
        "Also analyze the other person (scenario only)",
        value=True,
        help="In scenario mode, also runs a separate analysis on the person the narrator describes."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Active layers</div>", unsafe_allow_html=True)
    st.markdown("✅ Rule-based NLP")
    st.markdown("✅ Sentiment (TextBlob)")
    st.markdown("✅ LLM scoring (Qwen2.5-7B)" if use_llm else "⬜ LLM scoring (disabled)")


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_verdict_box(res: dict, label: str):
    vc = {"red_flag": "red-verdict", "green_flag": "green-verdict"}.get(res["verdict_key"], "neutral-verdict")
    summary = res.get("llm_summary", "")
    summary_html = f'<p style="margin-top:10px;font-size:.9rem;opacity:.7;font-style:italic;">"{summary}"</p>' if summary else ""
    st.markdown(f"""
    <div class='verdict-box {vc}'>
      <div style='font-size:.75rem;opacity:.5;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;font-family:Syne;'>{label}</div>
      <div class='verdict-title'>{res['verdict']}</div>
      <div class='verdict-confidence'>Confidence: {res['confidence_pct']}%</div>
      {summary_html}
    </div>""", unsafe_allow_html=True)

    # ── Verdict image ─────────────────────────────────────────────────────────
    verdict_key = res.get("verdict_key", "neutral")
    image_path  = VERDICT_IMAGES.get(verdict_key)

    if image_path and os.path.isfile(image_path):
        col_l, col_img, col_r = st.columns([1, 2, 1])
        with col_img:
            st.image(image_path, use_column_width=True)


def render_signal_bars(res: dict):
    red_pct   = int(res["final_red"]   * 100)
    if red_pct>100:
        red_pct = 100
    green_pct = int(res["final_green"] * 100)
    if green_pct>100:
        green_pct = 100
    st.markdown(f"""
    <div class='metric-card'>
      <div style='display:flex;justify-content:space-between;margin-bottom:5px;'>
        <span style='color:#ef4444;font-weight:500;'>🚩 Red Signal</span>
        <span style='font-weight:700;color:#ef4444;'>{red_pct}%</span>
      </div>
      <div style='background:#1a1a1a;border-radius:5px;height:7px;'>
        <div style='background:#ef4444;width:{red_pct}%;height:7px;border-radius:5px;'></div>
      </div>
    </div>
    <div class='metric-card'>
      <div style='display:flex;justify-content:space-between;margin-bottom:5px;'>
        <span style='color:#22c55e;font-weight:500;'>💚 Green Signal</span>
        <span style='font-weight:700;color:#22c55e;'>{green_pct}%</span>
      </div>
      <div style='background:#1a1a1a;border-radius:5px;height:7px;'>
        <div style='background:#22c55e;width:{green_pct}%;height:7px;border-radius:5px;'></div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_radar(res: dict, color: str = "#ef4444"):
    dims = res.get("dimensions", {})
    if not dims:
        return
    labels = list(dims.keys())
    values = [dims[d].get("score", 0) for d in labels]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=labels + [labels[0]],
        fill='toself', fillcolor=f'rgba({",".join(str(int(c*255)) for c in _hex_to_rgb(color))},0.1)',
        line=dict(color=color, width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#141414',
            radialaxis=dict(visible=True, range=[0,1], color='#444', gridcolor='#222'),
            angularaxis=dict(color='#888', gridcolor='#1f1f1f'),
        ),
        paper_bgcolor='#0d0d0d', plot_bgcolor='#0d0d0d',
        font=dict(color='#888', family='DM Sans'),
        showlegend=False, margin=dict(l=40,r=40,t=30,b=30), height=280,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_dimensions(res: dict):
    dims = res.get("dimensions", {})
    red_dims = get_red_dimensions()
    for dim, val in dims.items():
        score  = val.get("score", 0)
        reason = val.get("reason", "")
        is_red = dim in red_dims
        pct    = int(score * 100)
        bar_c  = "#ef4444" if is_red else "#22c55e"
        lbl_c  = "#ef4444" if (is_red and score > 0.4) else ("#22c55e" if (not is_red and score > 0.4) else "#555")
        st.markdown(f"""
        <div style='margin-bottom:11px;'>
          <div style='display:flex;justify-content:space-between;margin-bottom:2px;'>
            <span style='font-size:.83rem;color:{lbl_c};text-transform:capitalize;'>{dim.replace("_"," ")}</span>
            <span style='font-size:.78rem;color:#444;'>{pct}%</span>
          </div>
          <div style='background:#1a1a1a;border-radius:4px;height:4px;'>
            <div style='background:{bar_c};width:{pct}%;height:4px;border-radius:4px;'></div>
          </div>
          <div style='font-size:.73rem;color:#3a3a3a;margin-top:2px;font-style:italic;'>{reason}</div>
        </div>""", unsafe_allow_html=True)


def render_key_behaviors(res: dict):
    behaviors = res.get("key_behaviors", [])
    if not behaviors:
        return
    st.markdown("<div class='section-label'>Key Behaviors Identified</div>", unsafe_allow_html=True)
    cols = st.columns(len(behaviors))
    for i, b in enumerate(behaviors):
        with cols[i]:
            st.markdown(f"""
            <div style='background:#141414;border:1px solid #2a2a2a;border-radius:10px;
                        padding:12px;text-align:center;font-size:.82rem;color:#888;'>
              {b}
            </div>""", unsafe_allow_html=True)


def render_flags(flags: list):
    if not flags:
        st.caption("No strong rule-based signals detected.")
        return
    for emoji, reason in flags:
        bg  = "#1a0505" if emoji == "🚩" else "#051a0a"
        bdr = "#ef4444" if emoji == "🚩" else "#22c55e"
        st.markdown(f"""
        <div class='flag-item' style='background:{bg};border-left:3px solid {bdr};'>
          <span>{emoji}</span><span>{reason}</span>
        </div>""", unsafe_allow_html=True)


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255 for i in (0,2,4))


def render_person_analysis(res: dict, label: str, color: str, features: dict = None, show_flags: bool = True):
    """Render a full analysis block for one person."""
    render_verdict_box(res, label)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-label'>Signal Scores</div>", unsafe_allow_html=True)
        render_signal_bars(res)
    with c2:
        st.markdown("<div class='section-label'>Behavioral Radar</div>", unsafe_allow_html=True)
        render_radar(res, color)

    c3, c4 = st.columns(2)
    with c3:
        if show_flags and features:
            st.markdown("<div class='section-label'>Rule-based Signals</div>", unsafe_allow_html=True)
            render_flags(res.get("flags", []))
    with c4:
        if res.get("dimensions"):
            st.markdown("<div class='section-label'>LLM Dimension Breakdown</div>", unsafe_allow_html=True)
            render_dimensions(res)

    render_key_behaviors(res)


# ── Main analysis ─────────────────────────────────────────────────────────────
if analyze_btn:
    if not raw_input.strip():
        st.error("Please paste a conversation or scenario.")
        st.stop()

    with st.spinner("Detecting input type..."):
        turns, speaker_map, mode = parse_input(raw_input)

    if not turns:
        st.error("Could not parse input.")
        st.stop()

    # Mode info banner
    if mode == InputMode.SCENARIO:
        st.info("📖 **Scenario mode** — analyzing behavior from first-person narrative.")
        target_label = "Narrator"
    else:
        st.info("💬 **Conversation mode** — analyzing speaker behavior from chat.")
        target_label = target_choice

    target_turns = get_target_turns(turns, target_label)
    if not target_turns:
        st.error(f"No turns for '{target_label}'. Found: {get_available_speakers(turns)}")
        st.stop()

    # Rule-based features
    with st.spinner("Extracting linguistic features..."):
        features = extract_features(target_turns, turns, target_label)
        rule_scores = score_summary(features)

    quoted = extract_quoted_speech(raw_input) if mode == InputMode.SCENARIO else []

    # LLM analysis
    llm_result = None
    if use_llm:
        do_other = (mode == InputMode.SCENARIO and analyze_other)
        spinner_msg = "Analyzing narrator + other person..." if do_other else "Running behavioral analysis..."
        with st.spinner(spinner_msg):
            llm_result = analyze_with_llm(turns, target_label, mode, analyze_other=do_other)
            narrator_ok = llm_result.get("narrator", {}).get("success")
            if not narrator_ok:
                err = llm_result.get("narrator", {}).get("error", "unknown")
                st.warning(f"LLM failed: {err}. Showing rule-based results only.")
                llm_result = None

    # Aggregate
    with st.spinner("Aggregating..."):
        result = aggregate(rule_scores, llm_result, features)
        ILLEGAL_SIGNALS = [
            "without consent", "non-consensual", "while sleeping", "secretly recorded",
            "hidden camera", "stole", "illegal", "assault", "stalking", "underage", "minor",
        ]
        if any(s in raw_input.lower() for s in ILLEGAL_SIGNALS):
            result["primary"]["verdict"]     = "🚩 Red Flag"
            result["primary"]["verdict_key"] = "red_flag"
            result["primary"]["verdict_color"] = "#ef4444"

    primary   = result["primary"]
    secondary = result["secondary"]

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")

    if mode == InputMode.SCENARIO and secondary:
        st.markdown("<h2 style='font-family:Syne;font-size:1.4rem;'>Comparison Analysis</h2>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📖 Narrator (You)", "🧑 The Other Person"])

        with tab1:
            render_person_analysis(primary, "Narrator", "#ef4444", features, show_flags=True)
        with tab2:
            render_person_analysis(secondary, "Other Person (as described)", "#818cf8", show_flags=False)

    else:
        display_label = "Narrator" if mode == InputMode.SCENARIO else target_label
        st.markdown(f"<h2 style='font-family:Syne;font-size:1.4rem;'>Analysis: <span style='color:#666;font-weight:400;'>{display_label}</span></h2>", unsafe_allow_html=True)
        render_person_analysis(primary, display_label, "#ef4444", features, show_flags=True)

    # Quoted speech (scenario)
    if mode == InputMode.SCENARIO and quoted:
        with st.expander(f"💬 Direct quotes from narrative ({len(quoted)} found)"):
            st.caption("Actual words the narrator reports saying or hearing.")
            for q in quoted:
                st.markdown(f"""
                <div style='background:#111;border-left:3px solid #333;border-radius:0 8px 8px 0;
                            padding:8px 14px;margin-bottom:6px;font-size:.86rem;color:#aaa;font-style:italic;'>
                  "{q}"
                </div>""", unsafe_allow_html=True)

    # Parsed input viewer
    label_exp = "📜 View Parsed Narrative Chunks" if mode == InputMode.SCENARIO else "📜 View Parsed Conversation"
    with st.expander(label_exp):
        for turn in turns:
            is_target = turn.speaker == target_label
            bg  = "#1a0d0d" if is_target else "#111"
            bdr = "#3a1a1a" if is_target else "#1f1f1f"
            chunk_label = f"Chunk {turn.turn_index+1}" if mode == InputMode.SCENARIO else turn.speaker
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {bdr};border-radius:8px;
                        padding:10px 14px;margin-bottom:5px;font-size:.86rem;'>
              <span style='color:#555;font-size:.72rem;text-transform:uppercase;
                           letter-spacing:1px;font-family:Syne;'>{chunk_label}</span><br>
              <span style='color:#ccc;'>{turn.text}</span>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center;color:#2a2a2a;font-size:.78rem;padding:16px 0;'>NLP Course Project · Rule-based features + Qwen2.5-7B via HuggingFace Inference API</div>", unsafe_allow_html=True)