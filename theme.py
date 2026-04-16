"""
theme.py — Unified "Applied Data Science Virtual Lab" dark‑neon theme.

Provides:
    apply_theme()          → inject master CSS / fonts once per session
    experiment_header()    → render a consistent title banner for any experiment
    glass_card(html)       → render content inside a glassmorphism card
    loading_spinner()      → animated loading indicator
"""

import streamlit as st

# ─── colour tokens ──────────────────────────────────────────────────────────
BG_PRIMARY      = "#0a0a1a"
BG_SECONDARY    = "#0f0f2d"
BG_CARD         = "rgba(15, 15, 50, 0.65)"
NEON_BLUE       = "#00d4ff"
NEON_PURPLE     = "#7c3aed"
NEON_CYAN       = "#06b6d4"
NEON_PINK       = "#ec4899"
TEXT_PRIMARY     = "#e2e8f0"
TEXT_SECONDARY   = "#94a3b8"
BORDER_GLASS     = "rgba(255, 255, 255, 0.08)"

EXPERIMENT_COLORS = [
    "#00d4ff", "#7c3aed", "#06b6d4", "#ec4899", "#f59e0b",
    "#10b981", "#ef4444", "#8b5cf6", "#14b8a6",
]

EXPERIMENT_ICONS = ["📊", "🧹", "📈", "🎯", "⚖️", "🔍", "📉", "🧬", "🤖"]

EXPERIMENT_NAMES = [
    "Descriptive & Inferential Statistics",
    "Data Cleaning & Imputation",
    "Data Visualization",
    "Model Evaluation Metrics",
    "SMOTE Technique",
    "Outlier Detection",
    "Time Series Forecasting",
    "Data Science Lifecycle",
    "AutoML Techniques",
]

EXPERIMENT_DESCRIPTIONS = [
    "Explore mean, median, mode, variance, correlation and hypothesis testing on real datasets.",
    "Handle missing values, remove duplicates, and compare imputation strategies.",
    "Build histograms, scatter plots, heatmaps and more with interactive Plotly charts.",
    "Train classifiers, compare ROC curves, confusion matrices and cross‑validation scores.",
    "Tackle class imbalance with Synthetic Minority Over‑sampling and evaluate the impact.",
    "Detect anomalies using Z‑Score, IQR, Isolation Forest and k‑NN methods.",
    "Decompose, model and forecast temporal data with Moving Average and ARIMA.",
    "Multimodal fusion lab — combine image, text, audio, sensor and video features.",
    "Run FLAML, AutoGluon and H2O AutoML engines and compare results automatically.",
]


# ─── master CSS ─────────────────────────────────────────────────────────────
_MASTER_CSS = """
<style>
/* ── Google Inter font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, %(bg1)s 0%%, %(bg2)s 50%%, #0d0d30 100%%);
    color: %(text)s;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080820 0%%, #0c0c28 100%%);
    border-right: 1px solid rgba(0,212,255,0.12);
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}

/* ── Landing page cards ── */
.exp-card {
    background: rgba(15,15,50,0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.35s cubic-bezier(.4,0,.2,1);
    cursor: pointer;
    min-height: 180px;
    position: relative;
    overflow: hidden;
}
.exp-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--card-accent);
    border-radius: 16px 16px 0 0;
    opacity: 0.7;
    transition: opacity 0.3s;
}
.exp-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 0 28px rgba(0,212,255,0.18), 0 8px 32px rgba(0,0,0,0.4);
    border-color: rgba(0,212,255,0.25);
}
.exp-card:hover::before { opacity: 1; }
.exp-card .exp-num {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--card-accent);
    margin-bottom: 0.4rem;
}
.exp-card .exp-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
    line-height: 1.3;
}
.exp-card .exp-desc {
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.55;
}
.exp-card .exp-icon {
    font-size: 1.8rem;
    position: absolute;
    top: 1.2rem;
    right: 1.2rem;
    opacity: 0.25;
    transition: opacity 0.3s;
}
.exp-card:hover .exp-icon { opacity: 0.5; }

/* ── Hero title ── */
@keyframes title-glow {
    0%%,100%% { text-shadow: 0 0 20px rgba(0,212,255,0.3), 0 0 40px rgba(124,58,237,0.2); }
    50%% { text-shadow: 0 0 30px rgba(0,212,255,0.5), 0 0 60px rgba(124,58,237,0.35); }
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff 0%%, #7c3aed 50%%, #ec4899 100%%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: title-glow 3s ease-in-out infinite;
    text-align: center;
    margin-bottom: 0.4rem;
    line-height: 1.15;
}
.hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 2.5rem;
    letter-spacing: 0.02em;
}

/* ── Experiment page header ── */
.exp-header-banner {
    background: linear-gradient(135deg, rgba(0,212,255,0.08) 0%%, rgba(124,58,237,0.08) 100%%);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(8px);
}
.exp-header-banner h2 {
    color: #f1f5f9 !important;
    margin: 0 0 0.3rem 0;
    font-weight: 700;
    font-size: 1.5rem;
}
.exp-header-banner p {
    color: #94a3b8 !important;
    margin: 0;
    font-size: 0.9rem;
}

/* ── Glass card utility ── */
.glass-card {
    background: rgba(15,15,50,0.5);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

/* ── Page transition ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.page-transition {
    animation: fadeSlideIn 0.4s ease-out;
}

/* ── Loading dots ── */
@keyframes pulse-dot {
    0%%,80%%,100%% { transform: scale(0.6); opacity: 0.4; }
    40%% { transform: scale(1); opacity: 1; }
}
.loading-dots span {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%%;
    margin: 0 4px;
    animation: pulse-dot 1.4s ease-in-out infinite;
}
.loading-dots span:nth-child(1) { background: #00d4ff; animation-delay: 0s; }
.loading-dots span:nth-child(2) { background: #7c3aed; animation-delay: 0.2s; }
.loading-dots span:nth-child(3) { background: #ec4899; animation-delay: 0.4s; }

/* ── Buttons inside experiments ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.25s !important;
}

/* ── Sidebar active indicator ── */
.sidebar-nav-item {
    padding: 0.55rem 1rem;
    border-radius: 8px;
    margin-bottom: 2px;
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
    display: block;
    color: #94a3b8 !important;
    text-decoration: none !important;
}
.sidebar-nav-item:hover {
    background: rgba(0,212,255,0.08);
}
.sidebar-nav-item.active {
    background: rgba(0,212,255,0.12);
    color: #00d4ff !important;
    font-weight: 700;
    border-left: 3px solid #00d4ff;
}
</style>
""" % {"bg1": BG_PRIMARY, "bg2": BG_SECONDARY, "text": TEXT_PRIMARY}


def apply_theme():
    """Inject the master CSS once per Streamlit rerun."""
    st.markdown(_MASTER_CSS, unsafe_allow_html=True)


def experiment_header(number: int, title: str, description: str = ""):
    """Render a styled header banner for an experiment page."""
    icon = EXPERIMENT_ICONS[number - 1] if 1 <= number <= 9 else "🔬"
    desc_html = f"<p>{description}</p>" if description else ""
    st.markdown(f"""
    <div class="page-transition">
    <div class="exp-header-banner">
        <h2>{icon} Experiment {number}: {title}</h2>
        {desc_html}
    </div>
    </div>
    """, unsafe_allow_html=True)


def glass_card(content_html: str):
    """Render content inside a glassmorphism card."""
    st.markdown(f'<div class="glass-card">{content_html}</div>',
                unsafe_allow_html=True)


def loading_spinner():
    """Show animated loading dots."""
    st.markdown("""
    <div style="text-align:center; padding:2rem;">
        <div class="loading-dots">
            <span></span><span></span><span></span>
        </div>
        <p style="color:#94a3b8; margin-top:0.8rem; font-size:0.9rem;">Loading experiment…</p>
    </div>
    """, unsafe_allow_html=True)
