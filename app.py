"""
╔═══════════════════════════════════════════════════════════════════════╗
║        Applied Data Science — Virtual Lab Platform                    ║
║        Unified entry-point for 9 experiments                         ║
╚═══════════════════════════════════════════════════════════════════════╝

Run with:  streamlit run app.py
"""

import streamlit as st
import importlib

# ── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Applied Data Science Virtual Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Disable set_page_config for all sub-experiments so they don't crash
st.set_page_config = lambda **kw: None

# ── Theme ───────────────────────────────────────────────────────────────────
from theme import (
    apply_theme,
    experiment_header,
    EXPERIMENT_NAMES,
    EXPERIMENT_DESCRIPTIONS,
    EXPERIMENT_ICONS,
    EXPERIMENT_COLORS,
)

apply_theme()

# ── Experiment registry ─────────────────────────────────────────────────────
EXPERIMENTS = {
    1: ("experiments.exp1_statistics",     "exp1"),
    2: ("experiments.exp4_data_cleaning",  "exp4"),   # Exp 2 in menu = Data Cleaning
    3: ("experiments.exp3_visualization",  "exp3"),
    4: ("experiments.exp2_model_evaluation", "exp2"), # Exp 4 in menu = Model Evaluation
    5: ("experiments.exp5_smote",          "exp5"),
    6: ("experiments.exp6_outlier",        "exp6"),
    7: ("experiments.exp7_timeseries",     "exp7"),
    8: ("experiments.exp8_lifecycle",      "exp8"),
    9: ("experiments.exp9_automl",         "exp9"),
}

# ── Session state ───────────────────────────────────────────────────────────
if "current_experiment" not in st.session_state:
    st.session_state.current_experiment = 0  # 0 = landing page


# ── Sidebar navigation ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.6rem 0 0.3rem;">
        <div style="font-size:1.6rem;">🔬</div>
        <div style="font-size:1rem; font-weight:800;
                    background: linear-gradient(135deg, #00d4ff, #7c3aed);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;">
            ADS Virtual Lab
        </div>
        <div style="font-size:0.65rem; color:#64748b; margin-top:2px;
                    letter-spacing:0.08em; text-transform:uppercase;">
            Applied Data Science
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Home button
    if st.button("🏠  Home", use_container_width=True, key="nav_home"):
        st.session_state.current_experiment = 0
        st.rerun()

    st.markdown(
        "<p style='font-size:0.7rem; color:#64748b; letter-spacing:0.1em;"
        " text-transform:uppercase; margin:0.8rem 0 0.3rem 0.2rem;'>"
        "EXPERIMENTS</p>",
        unsafe_allow_html=True,
    )

    for i in range(1, 10):
        icon = EXPERIMENT_ICONS[i - 1]
        name = EXPERIMENT_NAMES[i - 1]
        is_active = st.session_state.current_experiment == i
        label = f"{icon}  {i}. {name}"
        if st.button(
            label,
            use_container_width=True,
            key=f"nav_{i}",
            type="primary" if is_active else "secondary",
        ):
            st.session_state.current_experiment = i
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; font-size:0.68rem; color:#475569;"
        " line-height:1.6;'>"
        "<b>Department of CSE (DS)</b><br>"
        "Applied Data Science Lab<br>"
        "© 2025"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.current_experiment == 0:

    # Hero section
    st.markdown("""
    <div class="page-transition" style="text-align:center; padding: 2rem 0 0.5rem;">
        <div style="font-size:3.2rem; margin-bottom:0.6rem;">🔬</div>
        <div class="hero-title">Applied Data Science<br>Virtual Lab</div>
        <div class="hero-sub">
            An interactive platform for mastering Data Science through
            hands-on experiments — from statistics to AutoML.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Experiment grid — 3 columns × 3 rows
    for row_start in range(0, 9, 3):
        cols = st.columns(3, gap="medium")
        for col_idx, exp_idx in enumerate(range(row_start, min(row_start + 3, 9))):
            num = exp_idx + 1
            icon = EXPERIMENT_ICONS[exp_idx]
            name = EXPERIMENT_NAMES[exp_idx]
            desc = EXPERIMENT_DESCRIPTIONS[exp_idx]
            accent = EXPERIMENT_COLORS[exp_idx]

            with cols[col_idx]:
                st.markdown(f"""
                <div class="exp-card" style="--card-accent: {accent};">
                    <div class="exp-num">Experiment {num}</div>
                    <div class="exp-title">{name}</div>
                    <div class="exp-desc">{desc}</div>
                    <div class="exp-icon">{icon}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(
                    f"Open Experiment {num}",
                    key=f"card_{num}",
                    use_container_width=True,
                ):
                    st.session_state.current_experiment = num
                    st.rerun()

    # Bottom stats
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Experiments", "9")
    m2.metric("Topics", "Statistics → AutoML")
    m3.metric("Framework", "Streamlit")
    m4.metric("Datasets", "Built-in + Upload")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT PAGES
# ═══════════════════════════════════════════════════════════════════════════
else:
    exp_num = st.session_state.current_experiment
    exp_name = EXPERIMENT_NAMES[exp_num - 1]
    exp_desc = EXPERIMENT_DESCRIPTIONS[exp_num - 1]

    # Header banner
    experiment_header(exp_num, exp_name, exp_desc)

    # Dynamic import and run
    module_path, _prefix = EXPERIMENTS[exp_num]
    try:
        mod = importlib.import_module(module_path)
        mod.run()
    except Exception as e:
        st.error(f"⚠️ Error loading Experiment {exp_num}: {e}")
        st.exception(e)
