"""
theme.py — Unified "Applied Data Science Virtual Lab" warm-light theme.

Provides:
    apply_theme()          → inject master CSS / fonts once per session
    experiment_header()    → render a consistent title banner for any experiment
    glass_card(html)       → render content inside a glassmorphism card
    loading_spinner()      → animated loading indicator
"""

import streamlit as st

# ─── colour tokens ──────────────────────────────────────────────────────────
BG_PRIMARY      = "#fffaf5"
BG_SECONDARY    = "#fff3ea"
BG_CARD         = "rgba(255, 248, 240, 0.75)"
ACCENT_1        = "#ea580c" # orange
ACCENT_2        = "#dc2626" # red
ACCENT_3        = "#d97706" # amber
ACCENT_4        = "#e11d48" # rose
TEXT_PRIMARY    = "#1e293b"
TEXT_SECONDARY  = "#475569"
BORDER_GLASS    = "rgba(0, 0, 0, 0.08)"

EXPERIMENT_COLORS = [
    "#ea580c", "#dc2626", "#d97706", "#e11d48", "#b45309",
    "#c2410c", "#9f1239", "#b91c1c", "#d97706",
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
    "Train classifiers, compare ROC curves, confusion matrices and cross-validation scores.",
    "Tackle class imbalance with Synthetic Minority Over-sampling and evaluate the impact.",
    "Detect anomalies using Z-Score, IQR, Isolation Forest and k-NN methods.",
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
    background: linear-gradient(135deg, %(bg1)s 0%%, %(bg2)s 50%%, #ffebdb 100%%);
    color: %(text)s;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fffaf6 0%%, #fff0e5 100%%);
    border-right: 1px solid rgba(234, 88, 12, 0.15);
}
section[data-testid="stSidebar"] * {
    color: #475569 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #0f172a !important;
}

/* ── Landing page cards ── */
.exp-card {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.35s cubic-bezier(.4,0,.2,1);
    cursor: pointer;
    min-height: 180px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
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
    box-shadow: 0 12px 24px rgba(234, 88, 12, 0.12), 0 8px 16px rgba(0,0,0,0.06);
    border-color: rgba(234, 88, 12, 0.25);
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
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 0.5rem;
    line-height: 1.3;
}
.exp-card .exp-desc {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.55;
}
.exp-card .exp-icon {
    font-size: 1.8rem;
    position: absolute;
    top: 1.2rem;
    right: 1.2rem;
    opacity: 0.15;
    transition: opacity 0.3s;
}
.exp-card:hover .exp-icon { opacity: 0.35; }

/* ── Hero title ── */
@keyframes title-glow {
    0%,100% { text-shadow: 0 0 15px rgba(234,88,12,0.15), 0 0 30px rgba(220,38,38,0.1); }
    50% { text-shadow: 0 0 25px rgba(234,88,12,0.25), 0 0 50px rgba(220,38,38,0.2); }
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ea580c 0%, #dc2626 50%, #e11d48 100%);
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
    color: #475569;
    font-size: 1.05rem;
    font-weight: 500;
    margin-bottom: 2.5rem;
    letter-spacing: 0.02em;
}

/* ── Experiment page header ── */
.exp-header-banner {
    background: linear-gradient(135deg, rgba(234,88,12,0.06) 0%, rgba(220,38,38,0.07) 100%);
    border: 1px solid rgba(234,88,12,0.15);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(8px);
}
.exp-header-banner h2 {
    color: #0f172a !important;
    margin: 0 0 0.3rem 0;
    font-weight: 800;
    font-size: 1.5rem;
}
.exp-header-banner p {
    color: #475569 !important;
    margin: 0;
    font-size: 0.95rem;
}

/* ── Glass card utility ── */
.glass-card {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(0,0,0,0.05);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
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
    0%,80%,100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}
.loading-dots span {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin: 0 4px;
    animation: pulse-dot 1.4s ease-in-out infinite;
}
.loading-dots span:nth-child(1) { background: #ea580c; animation-delay: 0s; }
.loading-dots span:nth-child(2) { background: #dc2626; animation-delay: 0.2s; }
.loading-dots span:nth-child(3) { background: #e11d48; animation-delay: 0.4s; }

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
    color: #475569 !important;
    text-decoration: none !important;
}
.sidebar-nav-item:hover {
    background: rgba(234,88,12,0.08);
}
.sidebar-nav-item.active {
    background: rgba(234,88,12,0.12);
    color: #ea580c !important;
    font-weight: 700;
    border-left: 3px solid #ea580c;
}
/* ── Additional Sections ── */
.section-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1e293b;
    margin-top: 3.5rem;
    margin-bottom: 0.4rem;
    text-align: center;
}
.section-sub {
    text-align: center;
    color: #475569;
    font-size: 1.05rem;
    margin-bottom: 2rem;
    font-weight: 500;
}
.card {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    text-align: center;
    height: 100%;
}
.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.6rem;
}
.card-desc {
    font-size: 0.9rem;
    color: #475569;
    line-height: 1.5;
}
</style>
""".replace('%(bg1)s', BG_PRIMARY).replace('%(bg2)s', BG_SECONDARY).replace('%(text)s', TEXT_PRIMARY)

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
    st.markdown(f'<div class="glass-card">{content_html}</div>', unsafe_allow_html=True)

def loading_spinner():
    """Show animated loading dots."""
    st.markdown("""
    <div style="text-align:center; padding:2rem;">
        <div class="loading-dots">
            <span></span><span></span><span></span>
        </div>
        <p style="color:#64748b; margin-top:0.8rem; font-size:0.9rem;">Loading experiment…</p>
    </div>
    """, unsafe_allow_html=True)