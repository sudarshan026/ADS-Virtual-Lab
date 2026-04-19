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

def render_experiments_sidebar():
    with st.sidebar:
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
    render_experiments_sidebar()

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

    # ── What You Can Do ──
    st.markdown("""
    <div class="section-title">What You Can Do</div>
    <div class="section-sub">Explore core data science workflows interactively.</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Analyze Data</div>
            <div class="card-desc">
            Perform statistical analysis and uncover patterns.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">⚙️ Build Models</div>
            <div class="card-desc">
            Train and evaluate machine learning models.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">📈 Visualize Insights</div>
            <div class="card-desc">
            Generate charts and interpret results visually.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Learning Flow ──
    st.markdown("""
    <br>
    <div class="section-title">Learning Flow</div>
    <div class="section-sub">Follow a structured path from basics to advanced topics.</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card" style="font-weight: 600; color: #ea580c; word-spacing: 0.15em; text-align: center;">
        1️⃣ Data Understanding ➔ 
        2️⃣ Cleaning ➔ 
        3️⃣ Visualization ➔ 
        4️⃣ Modeling ➔ 
        5️⃣ Evaluation ➔ 
        6️⃣ Deployment Concepts
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

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

    # ── Footer ──
    st.markdown("""
    <hr>
    <div style="text-align:center; color:#6b7280; font-size:13px;">
    Applied Data Science Virtual Lab • Department of CSE (DS)<br>
    <b>Developed By:</b> Ishan Jadhav • Sudarshan Gopal • Ryan Dsouza • Dhruwal Panchal • Harshit Sachdev<br><br>
    © 2026 All Rights Reserved
    </div>
    """, unsafe_allow_html=True)

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
        
        # ── Experiment Footer ──
        authors = {
            1: "Shravani Bhosale • Vedika Dhamale • Akash Jadhav ",
            2: "Aanchal Gupta • Harshavardhan Khamkar",
            3: "Aditya Upasani • Raziq Sarwar Irfan Mukadam • Rushikesh Shembade",
            4: "Isha Prakash Palkar • Akul Patre • Soham Patil • Veda Patki",
            5: "Aadi Singh Chauhan • Jai Desar • Yash Mahajan • Vedant Mhatre",
            6: "Akshhad Ahuja • Pranjal Ahuja • Moneet Nitin Bhiwandkar",
            7: "Riddhi Narendra Jangale • Bhoomika Makhija • Hanishka Vinay Kataria",
            8: "Riddhi Motwani • Ushma Sukhwani • Rithik Chawla • Ayush Parwani",
            9: "Pradnya Prabhakar Patil • Sonal Rajendra Patil • Diksha Vinit Patkar • Purva Deepak Mhatre"
        }
        
        if exp_num in authors:
            st.markdown(f"""
            <br><br>
            <hr style="margin-top: 2rem;">
            <div style="text-align:center; color:#6b7280; font-size:13px;">
            Applied Data Science Virtual Lab • Department of CSE (DS)<br>
            <b>Developed By:</b> {authors[exp_num]}<br><br>
            © 2026 All Rights Reserved
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error loading Experiment {exp_num}: {e}")
        st.exception(e)
    
    render_experiments_sidebar()
