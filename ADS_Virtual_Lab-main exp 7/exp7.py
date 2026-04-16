import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Virtual Labs - Time Series Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  —  Academic Light Theme
# Palette:
#   Primary blue  : #1A4A8A  (deep navy, readable on white)
#   Accent teal   : #1A7A6E  (muted teal for success / green roles)
#   Warm amber    : #B05A00  (amber/ochre for warnings / gold roles)
#   Error red     : #9B2335  (deep crimson for errors)
#   Surface       : #FFFFFF / #F5F7FA / #EEF1F5
#   Border        : #C8D0DA / #B0BAC6
#   Text          : #1C2B3A (near-black) / #5A6A7A (muted)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&family=Source+Sans+3:wght@300;400;600&display=swap');

:root {
    --cream:   #F8F5EF;
    --ink:     #1A1A1A;
    --muted:   #5A5A5A;
    --border:  #D4CFC6;
    --accent:  #8B2020;
    --accent2: #B5451B;
    --light-bg:#EFEBE3;
    --code-bg: #F0EDE6;
    --step-done: #2E6B3E;
    --step-active: #8B2020;
}

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: var(--cream);
    color: var(--ink);
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1C1612 !important;
    border-right: 2px solid #3A2E26;
}
[data-testid="stSidebar"] * {
    color: #E8DDD0 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F0E6D3 !important;
    font-family: 'EB Garamond', serif !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #C8BDB0 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] {
    margin-left: 1.05rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
    gap: 0.05rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-size: 0.72rem !important;
    padding: 0.16rem 0 !important;
    line-height: 1.2 !important;
}

/* ── Top banner ── */
.lab-header,
.lab-banner {
    background: linear-gradient(135deg, #8B2020 0%, #5C1212 60%, #3A0A0A 100%);
    padding: 1.4rem 2rem 1.2rem;
    border-radius: 4px;
    margin-bottom: 1.5rem;
    border-left: 5px solid #D4A843;
}
.lab-header h1,
.lab-banner h1 {
    font-family: 'EB Garamond', serif;
    font-size: 1.75rem;
    color: #F5ECD7;
    margin: 0 0 0.2rem 0;
    letter-spacing: 0.02em;
}
.lab-header p,
.lab-banner p {
    color: #C8A87A;
    font-size: 0.85rem;
    margin: 0;
    font-family: 'Source Sans 3', sans-serif;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Section headings ── */
.sec-title,
.step-heading {
    font-family: 'EB Garamond', serif;
    font-size: 1.55rem;
    color: var(--accent);
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}
.sec-sub,
.step-sub {
    font-size: 0.92rem;
    color: var(--muted);
    margin-top: -0.6rem;
    margin-bottom: 1.2rem;
}

/* ── Info / Theory boxes ── */
.card,
.theory-box,
.objective-box,
.note-box {
    background: var(--light-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0 1.2rem;
    font-size: 0.93rem;
    line-height: 1.65;
}
.card { border-radius: 4px; }
.card-accent,
.theory-box,
.objective-box { border-left: 4px solid var(--accent); }
.card-green { border-left: 4px solid #2E6B3E; background: #EEF4EE; }
.card-gold { border-left: 4px solid #D4A843; background: #FBF6EC; }
.card-red { border-left: 4px solid #8B2020; background: #FDF0EE; }
.objective-box { background: #EEF4EE; }
.note-box,
.insight {
    background: #FBF6EC;
    border: 1px dashed #C8A050;
    color: #6B5020;
    border-radius: 4px;
    margin: 0.8rem 0;
}
.insight b { color: var(--accent); }

/* ── Nav steps ── */
.nav-item,
.nav-step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.45rem 0.2rem;
    border-bottom: 1px solid #3A2E26;
    font-size: 0.88rem;
    cursor: default;
    margin-bottom: 0;
    border-radius: 0;
    border: none;
}
.nav-item.active,
.nav-step.active { color: #F5C842 !important; font-weight: 700; background: transparent; }
.nav-item.done,
.nav-step.done   { color: #7EC995 !important; }
.nav-item.upcoming,
.nav-step.upcoming { color: #7A6E64 !important; }

/* ── Comparison table ── */
.cmp-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.cmp-table th {
    background: #EDE6DB;
    color: #6B5A48;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.55rem 0.8rem;
    border-bottom: 2px solid var(--border);
    text-align: left;
}
.cmp-table td {
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid var(--border);
    color: var(--ink);
}
.cmp-table tr:hover td { background: #F4EEE5; }
.better { color: #2E6B3E; font-weight: 700; }
.worse  { color: #8B2020; }

/* ── Score banner ── */
.score-banner,
.score-box {
    background: linear-gradient(135deg, #2E6B3E, #1A4A28);
    color: white;
    padding: 1.2rem 2rem;
    border-radius: 4px;
    text-align: center;
    font-family: 'EB Garamond', serif;
    font-size: 1.5rem;
}

/* ── Quiz question card ── */
.quiz-q,
.quiz-card {
    background: var(--light-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.quiz-result-title {
    font-size: 1rem;
    font-weight: 700;
    line-height: 1.45;
    color: var(--ink);
    margin-bottom: 0.25rem;
}
.quiz-result-answer {
    font-size: 0.96rem;
    font-weight: 600;
    color: #2E3A46;
    margin-bottom: 0.2rem;
}
.quiz-result-correct {
    font-size: 0.95rem;
    color: #1A5E63;
    margin-bottom: 0.25rem;
}
.quiz-result-explanation {
    font-size: 0.93rem;
    color: #4B5968;
    line-height: 1.6;
    font-style: normal;
}

/* ── Buttons ── */
.stButton > button {
    background-color: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.4rem 1.4rem !important;
    transition: background 0.2s;
}
.stButton > button:hover {
    background-color: var(--accent2) !important;
}

/* Sidebar nav buttons — keep on one line */
[data-testid="stSidebar"] .stButton > button {
    white-space: nowrap !important;
    font-size: 0.72rem !important;
    padding: 0.38rem 0.7rem !important;
    background: #8B2020 !important;
    color: #FFFFFF !important;
    border: 1px solid #5C1212 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #5C1212 !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
    background: #8D7F74 !important;
    border-color: #8D7F74 !important;
    color: #F8F5EF !important;
}

/* ── Widget labels ── */
.stSelectbox > label, .stSlider > label,
.stRadio > label, .stNumberInput > label {
    color: #1A1A1A !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em !important;
    text-transform: none !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Streamlit metric container ── */
[data-testid="metric-container"] {
    background: var(--light-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 0.6rem 1rem !important;
    box-shadow: none !important;
}
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

/* ── Code inline ── */
code {
    font-family: 'JetBrains Mono', monospace;
    background: var(--code-bg);
    padding: 0.1em 0.35em;
    border-radius: 3px;
    font-size: 0.82em;
    color: var(--accent);
}

hr { border-color: var(--border); margin: 1rem 0; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px;
    overflow: hidden;
}

/* ── Inputs / selects ── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    border-radius: 4px !important;
    border-color: var(--border) !important;
    background-color: var(--cream) !important;
}
[data-baseweb="slider"] [role="slider"] {
    box-shadow: 0 0 0 4px rgba(139,32,32,0.12) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    border-radius: 5px 5px 0 0 !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

/* ── Plotly ── */
.js-plotly-plot {
    border: 1px solid var(--border);
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STEPS
# ─────────────────────────────────────────────
STEPS = [
    (1, "Aim", ""),
    (2, "Theory", ""),
    (3, "Dataset Selection", ""),
    (4, "Visualisation", ""),
    (5, "Decomposition", ""),
    (6, "Model Configuration", ""),
    (7, "Train and Simulate", ""),
    (8, "Results and Analysis", ""),
    (9, "Assessment Quiz", ""),
    (10, "References", ""),
]

QUIZ_LEVELS = ["Beginner", "Intermediate", "Advanced"]

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
def init():
    defs = dict(
        step=1, df=None, dataset_name=None, value_col=None,
        train_pct=80, ma_window=12,
        arima_p=1, arima_d=1, arima_q=1,
        ma_train=None, ma_test=None, ma_forecast=None,
        ar_train=None, ar_test=None, ar_forecast=None,
        trained=False,
        quiz_submitted=False, quiz_answers={}, quiz_score=0,
        quiz_level="Beginner",
        quiz_submitted_levels={},
        quiz_answers_levels={},
        quiz_scores={},
    )
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
init()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

# Plotly layout tuned for academic light theme
PLOTLY_LAYOUT = dict(
    plot_bgcolor="#F8F5EF",
    paper_bgcolor="#F8F5EF",
    font=dict(family="Source Sans 3, sans-serif", color="#1A1A1A", size=12),
    xaxis=dict(gridcolor="#DDD8CE", showgrid=True, zeroline=False,
               linecolor="#C8C1B6", linewidth=1),
    yaxis=dict(gridcolor="#DDD8CE", showgrid=True, zeroline=False,
               linecolor="#C8C1B6", linewidth=1),
    margin=dict(t=50, b=40, l=55, r=20),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(0,0,0,0)",
                bordercolor="#D4CFC6", borderwidth=1,
                font=dict(size=11)),
)

# Chart colour palette (reference theme)
C_ACTUAL  = "#8B2020"   # maroon — actual / train line
C_MA      = "#1C5C8A"   # blue    — Moving Average forecast
C_ARIMA   = "#2E6B3E"   # green   — ARIMA forecast
C_TRAIN   = "#5A5A5A"   # muted   — training history
C_RESID   = "#7A5C2A"   # brown   — residuals

def make_fig(**kw):
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT, **kw)
    return fig

def preprocess(df):
    date_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["date","time","month","year","week"])),
        None
    )
    if date_col is None:
        raise ValueError("No date/time column detected. Ensure your CSV has a 'Date' or 'Time' column.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    return df

def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def infer_period(df):
    freq = pd.infer_freq(df.index)
    if freq is None: return 12
    f = str(freq).upper()
    if "M" in f: return 12
    if "Q" in f: return 4
    if "W" in f: return 52
    if "D" in f: return 7
    return 12

def run_models(df, col, train_pct, ma_w, p, d, q):
    series = df[col].dropna()
    split  = int(len(series) * train_pct / 100)
    train  = series.iloc[:split]
    test   = series.iloc[split:]

    # Moving Average
    ma_forecast = series.rolling(window=ma_w).mean().iloc[split:split + len(test)]
    ma_forecast.index = test.index

    # ARIMA
    ar_model   = ARIMA(train, order=(p, d, q)).fit()
    ar_fc      = ar_model.forecast(steps=len(test))
    ar_forecast = pd.Series(ar_fc.values, index=test.index)

    return train, test, ma_forecast, ar_forecast

def metrics(test, forecast):
    aligned = pd.concat([test, forecast], axis=1).dropna()
    aligned.columns = ["a", "f"]
    mae  = mean_absolute_error(aligned.a, aligned.f)
    mse  = mean_squared_error(aligned.a, aligned.f)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((aligned.a - aligned.f) / aligned.a.replace(0, np.nan))) * 100
    return mae, mse, rmse, mape

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 0.8rem 0 0.4rem;'>
        <div style='font-family: EB Garamond, serif; font-size:1.3rem; color:#F5ECD7; font-weight:600;'>Virtual Labs</div>
    </div>
    <hr style='border-color:#3A2E26; margin:0.6rem 0;'>
    <div style='font-size:0.75rem; color:#A09080; letter-spacing:0.06em; text-transform:uppercase; padding: 0 0 0.5rem;'>
        Experiment Navigation
    </div>
    """, unsafe_allow_html=True)

    cur = st.session_state.step
    nav_html = ""
    for num, label, icon in STEPS:
        if num < cur:   cls, ind = "done",     "✔"
        elif num == cur: cls, ind = "active",  "➜"
        else:            cls, ind = "upcoming", str(num)
        nav_html += (
            f'<div class="nav-item {cls}">'
            f'<span style="min-width:18px;text-align:center;font-size:0.8rem;">{ind}</span>'
            f'<span>{label}</span></div>'
        )
    st.markdown(nav_html, unsafe_allow_html=True)

    if cur == 9:
        st.markdown("<div style='font-size:0.75rem; color:#A09080; letter-spacing:0.06em; text-transform:uppercase; padding: 0.45rem 0 0.25rem;'>Quiz Section</div>", unsafe_allow_html=True)
        st.radio("Quiz Level", QUIZ_LEVELS, key="quiz_level", label_visibility="collapsed")

    st.markdown("<hr style='border-color:#3A2E26; margin:0.55rem 0;'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Prev", disabled=(cur == 1)):
            st.session_state.step = cur - 1; st.rerun()
    with c2:
        if st.button("Next", disabled=(cur == len(STEPS))):
            st.session_state.step = cur + 1; st.rerun()

    st.markdown("""
    <div style='margin-top:1.5rem; font-size:0.78rem; color:#7A6E64; line-height:1.6;'>
        <b style='color:#C8BDB0;'>Experiment:</b><br>
        Time Series Forecasting<br><br>
        <b style='color:#C8BDB0;'>Discipline:</b><br>
        Data Science & Analytics<br><br>
        <b style='color:#C8BDB0;'>Duration:</b><br>
        ~45 minutes
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="lab-banner">
    <h1>Time Series Forecasting</h1>
    <p>Data Science &amp; Analytics Virtual Laboratory</p>
</div>
""", unsafe_allow_html=True)

step = st.session_state.step

# ═══════════════════════════════════════════════════
# STEP 1 — AIM
# ═══════════════════════════════════════════════════
if step == 1:
    st.markdown('<div class="sec-title">Aim &amp; Objectives</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Understand the purpose and expected outcomes of this experiment.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card card-accent">
    <b style="color:#1A4A8A;">AIM</b><br><br>
    To analyse and forecast time-series data using classical statistical techniques —
    <b>Moving Average</b> and <b>ARIMA</b> — and to evaluate model performance using
    standard regression error metrics on real-world datasets.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card card-accent">
        <b style="color:#1A4A8A;">Learning Outcomes</b><br><br>
        After completing this lab, you will be able to:<br><br>
        ① Load and preprocess a time-series dataset<br>
        ② Visualise temporal patterns and spot anomalies<br>
        ③ Decompose a series into Trend + Seasonal + Residual<br>
        ④ Configure and train Moving Average and ARIMA models<br>
        ⑤ Evaluate accuracy using MAE, MSE, RMSE and MAPE<br>
        ⑥ Compare both models and justify which performs better
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card card-green">
        <b style="color:#1A7A6E;">Pre-requisites</b><br><br>
        • Basic descriptive statistics (mean, variance)<br>
        • Concept of regression and prediction<br>
        • Familiarity with Python (optional)<br><br>
        <b style="color:#B05A00;">Lab Instructions</b><br><br>
        • Follow steps <b>in order</b> using the sidebar<br>
        • Load a dataset <i>before</i> proceeding to Step 4+<br>
        • Train models in Step 7 before viewing results<br>
        • Complete the quiz at the end for self-assessment
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight">
    <b>Problem Statement:</b> Time-series forecasting is widely applied in finance, healthcare, and supply-chain
    management. This experiment focuses on identifying temporal structure (trend, seasonality) and building
    predictive models that can generalise to unseen future data.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# STEP 2 — THEORY
# ═══════════════════════════════════════════════════
elif step == 2:
    st.markdown('<div class="sec-title">Theory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Read this background before running the experiment.</div>', unsafe_allow_html=True)

    st.markdown("#### What is a Time Series?")
    st.markdown("""
    <div class="card card-accent">
    A <b>time series</b> is an ordered sequence of observations measured at successive,
    typically equally-spaced, points in time. The fundamental assumption is that
    <i>the past contains information useful for predicting the future</i>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Components of a Time Series")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card card-accent">
        <b>Trend</b><br>
        Long-run increase or decrease in the series. Can be linear (constant slope) or
        non-linear (exponential, logistic). Smoothing and differencing help isolate it.
        </div>
        <div class="card card-gold">
        <b>Seasonality</b><br>
        Repeating patterns of fixed known frequency — e.g., higher ice-cream sales every
        summer, higher retail sales in December. Period (m) must be specified.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="border-left:4px solid #5A7A9A; background:#EBF0F6;">
        <b>Cyclicity</b><br>
        Irregular oscillations of variable length, often linked to economic cycles.
        Unlike seasonality, the period is not fixed and can span years.
        </div>
        <div class="card card-red">
        <b>Residual (Noise)</b><br>
        What remains after extracting Trend and Seasonality. Ideally white noise
        (zero mean, constant variance, no autocorrelation). Patterns here indicate a
        mis-specified model.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Forecasting Models")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card card-accent">
        <b>Moving Average (MA)</b><br><br>
        Smooths the series by averaging the last <i>w</i> observations:
        <br><br><code>Ŷ(t) = [ Y(t-1) + Y(t-2) + … + Y(t-w) ] / w</code><br><br>
        <b>Strengths:</b> Simple, robust to noise, no distributional assumptions.<br>
        <b>Limitations:</b> Lags behind sharp trend changes; cannot capture seasonality.<br>
        <b>Best for:</b> Stationary or slowly-drifting series.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card card-green">
        <b>ARIMA(p, d, q)</b><br><br>
        Combines three mechanisms:<br>
        &nbsp;&nbsp;<b>AR(p)</b> — regress on past <i>p</i> values<br>
        &nbsp;&nbsp;<b>I(d)</b>  — difference <i>d</i> times for stationarity<br>
        &nbsp;&nbsp;<b>MA(q)</b> — use past <i>q</i> forecast errors<br><br>
        <b>Strengths:</b> Handles trends, autocorrelation; theoretically grounded.<br>
        <b>Limitations:</b> Assumes linearity; sensitive to parameter choice.<br>
        <b>Best for:</b> Non-stationary series with autocorrelation structure.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Evaluation Metrics")
    metrics_data = {
        "Metric": ["MAE", "MSE", "RMSE", "MAPE"],
        "Formula": [
            "Mean |actual − forecast|",
            "Mean (actual − forecast)²",
            "√MSE",
            "Mean |actual − forecast| / |actual| × 100",
        ],
        "Unit": ["Same as data", "Squared", "Same as data", "Percentage"],
        "Key Property": [
            "Robust to outliers; linear penalty",
            "Penalises large errors heavily",
            "Interpretable; same scale as data",
            "Scale-free; easy to communicate",
        ],
    }
    st.dataframe(pd.DataFrame(metrics_data), width="stretch", hide_index=True)

    st.markdown("""
    <div class="insight">
    <b>Additive vs Multiplicative Decomposition:</b>
    Use <b>additive</b> when seasonal swings are roughly constant over time.
    Use <b>multiplicative</b> when they grow proportionally with the trend.
    This lab uses <b>additive decomposition</b>.<br><br>
    <b>Reference:</b> Box, G.E.P. &amp; Jenkins, G.M. (1976). <i>Time Series Analysis: Forecasting and Control.</i>
    <a href="https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021" target="_blank">Book link</a>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# STEP 3 — DATASET SELECTION
# ═══════════════════════════════════════════════════
elif step == 3:
    st.markdown('<div class="sec-title">Dataset Selection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Choose a built-in dataset or upload your own CSV.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card card-gold">
    <b>Built-in datasets</b> are synthetic but statistically realistic, designed to
    demonstrate different time series patterns. Upload your own CSV for real-world analysis —
    ensure it has a date/time column and at least one numeric column.
    </div>
    """, unsafe_allow_html=True)

    dataset_choice = st.radio(
        "Select Dataset:",
        ["Apple Stock — 5 Years (Daily)", "Chocolate Sales — 6 Years (Monthly)", "Upload your own CSV"],
        index=0,
    )

    uploaded_file = None
    if "Upload" in dataset_choice:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        st.markdown("""
        <div class="insight">
        Your CSV must have: (1) a <b>Date</b> or <b>Time</b> column parseable by pandas,
        and (2) at least one numeric column.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Load Dataset"):
        try:
            if "Apple" in dataset_choice:
                dates = pd.date_range("2019-01-01", "2023-12-31", freq="B")
                np.random.seed(42)
                price = 150.0; prices = []
                for _ in dates:
                    price *= (1 + np.random.normal(0.00035, 0.013))
                    prices.append(round(price, 2))
                df_raw = pd.DataFrame({
                    "Date":   dates,
                    "Close":  prices,
                    "Open":   [p * np.random.uniform(0.995, 1.005) for p in prices],
                    "Volume": np.random.randint(50_000_000, 150_000_000, len(dates)),
                })
                name = "Apple Stock (5-Year Daily)"

            elif "Chocolate" in dataset_choice:
                dates    = pd.date_range("2018-01-01", periods=72, freq="MS")
                np.random.seed(7)
                trend    = np.linspace(22000, 40000, 72)
                seasonal = 6000 * np.sin(np.linspace(0, 6 * np.pi, 72))
                noise    = np.random.normal(0, 1500, 72)
                sales    = (trend + seasonal + noise).clip(min=0)
                df_raw   = pd.DataFrame({
                    "Date":  dates,
                    "Sales": np.round(sales, 2),
                    "Units": np.round(sales / 25, 0).astype(int),
                })
                name = "Chocolate Sales (6-Year Monthly)"

            else:
                if uploaded_file is None:
                    st.error("Please upload a CSV file first."); st.stop()
                df_raw = pd.read_csv(uploaded_file)
                name   = uploaded_file.name

            df = preprocess(df_raw)
            st.session_state.df           = df
            st.session_state.dataset_name = name
            st.session_state.trained      = False

            st.success(f"{name} loaded - {len(df):,} rows x {len(df.columns)} columns")
            st.markdown("**Preview (first 10 rows):**")
            st.dataframe(df.head(10), width="stretch")

            nc = numeric_cols(df)
            st.markdown("**Numeric columns available:** " + " · ".join(f"`{c}`" for c in nc))

        except Exception as e:
            st.error(f"Error: {e}")

    elif st.session_state.df is not None:
        st.info(f"**{st.session_state.dataset_name}** is already loaded. Re-load or continue to Step 4.")
        st.dataframe(st.session_state.df.head(8), width="stretch")

# ═══════════════════════════════════════════════════
# STEP 4 — VISUALISE
# ═══════════════════════════════════════════════════
elif step == 4:
    st.markdown('<div class="sec-title">Visualise &amp; Explore</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Inspect the raw time series before any modelling.</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Load a dataset in Step 3 first."); st.stop()

    df  = st.session_state.df
    nc  = numeric_cols(df)
    col_sel = st.selectbox(
        "Column to visualise:",
        nc,
        index=nc.index(st.session_state.value_col) if st.session_state.value_col in nc else 0,
    )
    st.session_state.value_col = col_sel

    st.markdown("""
    <div class="card card-accent">
    Look for: <b>trend direction</b> (upward/downward/flat), <b>seasonal cycles</b> (repeating patterns),
    <b>volatility changes</b>, and <b>outliers</b> (sudden spikes/drops).
    </div>
    """, unsafe_allow_html=True)

    # Full series line plot
    fig = make_fig(title=f"<b>{col_sel}</b> — Full Time Series", height=380)
    fig.add_trace(go.Scatter(
        x=df.index, y=df[col_sel],
        mode="lines", name=col_sel,
        line=dict(color=C_ACTUAL, width=1.8),
        fill="tozeroy", fillcolor="rgba(26,74,138,0.07)",
    ))
    fig.update_layout(xaxis_title="Date", yaxis_title=col_sel)
    st.plotly_chart(fig, width="stretch")

    # Descriptive stats + rolling overlay
    c1, c2 = st.columns([1, 1.8])
    with c1:
        st.markdown("#### Descriptive Statistics")
        s = df[[col_sel]].describe().T.round(3)
        st.dataframe(s, width="stretch")
        st.markdown(f"""
        <div class="insight">
        <b>Range:</b> {df[col_sel].min():.2f} → {df[col_sel].max():.2f}<br>
        <b>Std Dev:</b> {df[col_sel].std():.2f}<br>
        <b>Observations:</b> {len(df[col_sel].dropna()):,}
        </div>
        """, unsafe_allow_html=True)
    with c2:
        roll_w = st.slider("Rolling average window (for overlay):", 5, 60, 20)
        fig2   = make_fig(title=f"<b>{col_sel}</b> with {roll_w}-period Rolling Mean", height=320)
        fig2.add_trace(go.Scatter(
            x=df.index, y=df[col_sel],
            mode="lines", name="Raw",
            line=dict(color=C_ACTUAL, width=1.1, dash="dot"), opacity=0.5,
        ))
        fig2.add_trace(go.Scatter(
            x=df.index, y=df[col_sel].rolling(roll_w).mean(),
            mode="lines", name=f"{roll_w}-period MA",
            line=dict(color=C_MA, width=2.1),
        ))
        fig2.update_layout(xaxis_title="Date", yaxis_title=col_sel)
        st.plotly_chart(fig2, width="stretch")

# ═══════════════════════════════════════════════════
# STEP 5 — DECOMPOSITION
# ═══════════════════════════════════════════════════
elif step == 5:
    st.markdown('<div class="sec-title">Time Series Decomposition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Break the series into Trend, Seasonality, and Residual components.</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Load a dataset in Step 3 first."); st.stop()

    df  = st.session_state.df
    nc  = numeric_cols(df)
    col_sel = st.selectbox(
        "Column to decompose:",
        nc,
        index=nc.index(st.session_state.value_col) if st.session_state.value_col in nc else 0,
    )
    st.session_state.value_col = col_sel

    st.markdown("""
    <div class="card card-accent">
    <b>Additive Model:</b> &nbsp; <code>Y(t) = Trend(t) + Seasonal(t) + Residual(t)</code><br>
    Assumes seasonal amplitude is constant over time. Suitable when fluctuations do not grow with the level.
    </div>
    """, unsafe_allow_html=True)

    try:
        period = infer_period(df)
        series = df[col_sel].dropna()
        result = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")

        components = [series, result.trend, result.seasonal, result.resid]
        names  = ["Original", "Trend", "Seasonality", "Residual"]
        colors = [C_ACTUAL, C_MA, C_ARIMA, C_RESID]

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=[f"<b>{n}</b>" for n in names],
            vertical_spacing=0.07,
        )
        for i, (comp, name, color) in enumerate(zip(components, names, colors), 1):
            fig.add_trace(go.Scatter(
                x=comp.index, y=comp.values,
                mode="lines", name=name,
                line=dict(color=color, width=1.5),
            ), row=i, col=1)
            if name == "Residual":
                fig.add_hline(y=0, line_dash="dash", line_color="#8A9AAA", line_width=1, row=i, col=1)

        fig.update_layout(
            height=650, showlegend=False,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
        )
        for i in range(1, 5):
            fig.update_xaxes(gridcolor="#DDE3EB", linecolor="#B0BAC6", row=i, col=1)
            fig.update_yaxes(gridcolor="#DDE3EB", linecolor="#B0BAC6", row=i, col=1)
        st.plotly_chart(fig, width="stretch")

        st.markdown(f"**Inferred period:** `{period}` | **Series length:** `{len(series)}`")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="card card-gold">
            <b>Trend Panel</b><br>
            Smooth long-run direction after eliminating short-term noise.
            Rising = upward drift; flat = no consistent direction.
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="card card-green">
            <b>Seasonality Panel</b><br>
            Repeating periodic pattern. Height of peaks = seasonal effect size.
            Flat line → no detectable seasonality.
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="card card-red">
            <b>Residual Panel</b><br>
            Should look like white noise (random scatter around zero).
            Clear patterns → model is missing structure.
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Decomposition failed: {e}")
        st.markdown("""
        <div class="insight">
        This can happen if the series is too short relative to the period,
        or has many missing values. Try another column or dataset.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# STEP 6 — MODEL CONFIGURATION
# ═══════════════════════════════════════════════════
elif step == 6:
    st.markdown('<div class="sec-title">Model Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Set parameters for both models — defaults are balanced starting points.</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Load a dataset in Step 3 first."); st.stop()

    df  = st.session_state.df
    nc  = numeric_cols(df)
    col_sel = st.selectbox(
        "Target column:",
        nc,
        index=nc.index(st.session_state.value_col) if st.session_state.value_col in nc else 0,
    )
    st.session_state.value_col = col_sel

    st.markdown("---")
    st.markdown("#### 1. Train-Test Split")
    train_pct = st.slider("Training data %:", 70, 90, st.session_state.train_pct, step=5)
    st.session_state.train_pct = train_pct
    n    = len(df[col_sel].dropna())
    n_tr = int(n * train_pct / 100)
    n_te = n - n_tr
    c1, c2, c3 = st.columns(3)
    c1.metric("Total observations", f"{n:,}")
    c2.metric("Train samples",      f"{n_tr:,}")
    c3.metric("Test samples",       f"{n_te:,}")

    st.markdown("---")
    col_ma, col_ar = st.columns(2)

    with col_ma:
        st.markdown("""
        <div class="card card-accent">
        <b>Moving Average</b><br>
        Averages the last <i>w</i> values. Larger <i>w</i> = smoother but slower response.
        Good starting point: set <i>w</i> ≈ half a seasonal period.
        </div>
        """, unsafe_allow_html=True)
        period    = infer_period(df)
        default_w = max(3, period // 2)
        ma_w = st.slider("Window size w:", 2, min(120, n_tr // 3),
                         min(st.session_state.ma_window, n_tr // 3))
        st.session_state.ma_window = ma_w
        st.markdown(
            f"<div class='insight'>Recommended for this series: <b>w = {default_w}</b> "
            f"(half the inferred period of {period})</div>",
            unsafe_allow_html=True,
        )

    with col_ar:
        st.markdown("""
        <div class="card card-green">
        <b>ARIMA(p, d, q)</b><br>
        Default <code>(1,1,1)</code> is a robust baseline for most non-stationary series.
        Increase <i>p</i> if PACF cuts off after lag p; increase <i>q</i> if ACF cuts off after lag q.
        </div>
        """, unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)
        p = cc1.number_input("AR  p:", 0, 5, st.session_state.arima_p)
        d = cc2.number_input("I   d:", 0, 2, st.session_state.arima_d)
        q = cc3.number_input("MA  q:", 0, 5, st.session_state.arima_q)
        st.session_state.arima_p = int(p)
        st.session_state.arima_d = int(d)
        st.session_state.arima_q = int(q)
        st.markdown(
            f"<div class='insight'>Current specification: <b>ARIMA({int(p)}, {int(d)}, {int(q)})</b></div>",
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div class="insight">
    <b>Tip:</b> You can change parameters and re-train as many times as you like in Step 7.
    Experiment with different values to see how they affect forecast accuracy.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# STEP 7 — TRAIN & SIMULATE
# ═══════════════════════════════════════════════════
elif step == 7:
    st.markdown('<div class="sec-title">Train &amp; Simulate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Run both models simultaneously and generate forecasts.</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Load a dataset in Step 3 first."); st.stop()

    df        = st.session_state.df
    col_sel   = st.session_state.value_col or numeric_cols(df)[0]
    train_pct = st.session_state.train_pct
    ma_w      = st.session_state.ma_window
    p, d, q   = st.session_state.arima_p, st.session_state.arima_d, st.session_state.arima_q

    st.markdown(f"""
    <div class="card card-gold">
    <b>Configuration Summary</b><br>
    Column: <code>{col_sel}</code> &nbsp;|&nbsp;
    Train split: <code>{train_pct}%</code> &nbsp;|&nbsp;
    Moving Average window: <code>w = {ma_w}</code> &nbsp;|&nbsp;
    ARIMA: <code>({p},{d},{q})</code>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Train Both Models"):
        with st.spinner("Training Moving Average & ARIMA - please wait..."):
            try:
                train, test, ma_fc, ar_fc = run_models(df, col_sel, train_pct, ma_w, p, d, q)
                st.session_state.ma_train    = train
                st.session_state.ma_test     = test
                st.session_state.ma_forecast = ma_fc
                st.session_state.ar_forecast = ar_fc
                st.session_state.trained     = True
                st.success("Both models trained successfully. Proceed to Step 8 for full results.")

                # Quick preview chart
                fig = make_fig(title=f"<b>{col_sel}</b> — Forecast Preview", height=400)
                fig.add_trace(go.Scatter(x=train.index, y=train.values,
                    mode="lines", name="Train data",
                    line=dict(color=C_TRAIN, width=1.2)))
                fig.add_trace(go.Scatter(x=test.index, y=test.values,
                    mode="lines", name="Actual (Test)",
                    line=dict(color=C_ACTUAL, width=2.2)))
                fig.add_trace(go.Scatter(x=ma_fc.index, y=ma_fc.values,
                    mode="lines", name=f"Moving Average (w={ma_w})",
                    line=dict(color=C_MA, width=2.1, dash="dot")))
                fig.add_trace(go.Scatter(x=ar_fc.index, y=ar_fc.values,
                    mode="lines", name=f"ARIMA({p},{d},{q})",
                    line=dict(color=C_ARIMA, width=2.1, dash="dash")))
                fig.add_vrect(x0=test.index[0], x1=test.index[-1],
                    fillcolor="rgba(26,74,138,0.05)", layer="below", line_width=0)
                fig.update_layout(xaxis_title="Date", yaxis_title=col_sel)
                st.plotly_chart(fig, width="stretch")

            except Exception as e:
                st.error(f"Training failed: {e}")

    elif st.session_state.trained:
        st.info("Models are already trained. Click above to re-train with updated settings, or proceed to Step 8.")

# ═══════════════════════════════════════════════════
# STEP 8 — RESULTS
# ═══════════════════════════════════════════════════
elif step == 8:
    st.markdown('<div class="sec-title">Results &amp; Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Comprehensive evaluation of both forecasting models.</div>', unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("No models trained yet. Train models in Step 7 first."); st.stop()

    train   = st.session_state.ma_train
    test    = st.session_state.ma_test
    ma_fc   = st.session_state.ma_forecast
    ar_fc   = st.session_state.ar_forecast
    col_sel = st.session_state.value_col
    ma_w    = st.session_state.ma_window
    p, d, q = st.session_state.arima_p, st.session_state.arima_d, st.session_state.arima_q

    ma_mae, ma_mse, ma_rmse, ma_mape = metrics(test, ma_fc)
    ar_mae, ar_mse, ar_rmse, ar_mape = metrics(test, ar_fc)

    # ── A: Forecast Plot ─────────────────────────
    st.markdown("### A. Forecast vs Actual")
    fig = make_fig(title=f"<b>{col_sel}</b> — Train / Test / Forecasts", height=450)
    fig.add_trace(go.Scatter(x=train.index, y=train.values,
        mode="lines", name="Train data",
        line=dict(color=C_TRAIN, width=1.3)))
    fig.add_trace(go.Scatter(x=test.index, y=test.values,
        mode="lines", name="Actual (Test)",
        line=dict(color=C_ACTUAL, width=2.3)))
    fig.add_trace(go.Scatter(x=ma_fc.index, y=ma_fc.values,
        mode="lines", name=f"Moving Average (w={ma_w})",
        line=dict(color=C_MA, width=2.1, dash="dot")))
    fig.add_trace(go.Scatter(x=ar_fc.index, y=ar_fc.values,
        mode="lines", name=f"ARIMA({p},{d},{q})",
        line=dict(color=C_ARIMA, width=2.1, dash="dash")))
    fig.add_vrect(x0=test.index[0], x1=test.index[-1],
        fillcolor="rgba(26,74,138,0.05)", layer="below", line_width=0)
    fig.add_annotation(
        x=test.index[len(test) // 2], y=test.max(),
        text="← Test period →", showarrow=False,
        font=dict(color="#8A9AAA", size=11),
    )
    fig.update_layout(xaxis_title="Date", yaxis_title=col_sel)
    st.plotly_chart(fig, width="stretch")

    # ── B: Metric Cards ───────────────────────────
    st.markdown("### B. Evaluation Metrics")

    def tag(a, b):
        if a < b: return "Better", ""
        if b < a: return "",       "Better"
        return "Equal", "Equal"

    ma_mae_tag,  ar_mae_tag  = tag(ma_mae,  ar_mae)
    ma_mse_tag,  ar_mse_tag  = tag(ma_mse,  ar_mse)
    ma_rmse_tag, ar_rmse_tag = tag(ma_rmse, ar_rmse)
    ma_mape_tag, ar_mape_tag = tag(ma_mape, ar_mape)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### Moving Average (w={ma_w})")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MAE",  f"{ma_mae:,.2f}",  delta=ma_mae_tag,  delta_color="off")
        mc2.metric("MSE",  f"{ma_mse:,.2f}",  delta=ma_mse_tag,  delta_color="off")
        mc3.metric("RMSE", f"{ma_rmse:,.2f}", delta=ma_rmse_tag, delta_color="off")
        mc4.metric("MAPE", f"{ma_mape:.2f}%", delta=ma_mape_tag, delta_color="off")
    with col2:
        st.markdown(f"#### ARIMA({p},{d},{q})")
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("MAE",  f"{ar_mae:,.2f}",  delta=ar_mae_tag,  delta_color="off")
        ac2.metric("MSE",  f"{ar_mse:,.2f}",  delta=ar_mse_tag,  delta_color="off")
        ac3.metric("RMSE", f"{ar_rmse:,.2f}", delta=ar_rmse_tag, delta_color="off")
        ac4.metric("MAPE", f"{ar_mape:.2f}%", delta=ar_mape_tag, delta_color="off")

    # ── C: Comparison Table ───────────────────────
    st.markdown("### C. Side-by-Side Comparison Table")

    ma_wins = sum([ma_mae < ar_mae, ma_mse < ar_mse, ma_rmse < ar_rmse, ma_mape < ar_mape])
    ar_wins = sum([ar_mae < ma_mae, ar_mse < ma_mse, ar_rmse < ma_rmse, ar_mape < ma_mape])

    def bc(a, b):
        return "better" if a < b else ("worse" if a > b else "")

    table_html = f"""
    <table class="cmp-table">
    <thead>
        <tr>
            <th>Metric</th>
            <th>Moving Average (w={ma_w})</th>
            <th>ARIMA({p},{d},{q})</th>
            <th>Winner</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>MAE</b></td>
            <td class="{bc(ma_mae,ar_mae)}">{ma_mae:,.4f}</td>
            <td class="{bc(ar_mae,ma_mae)}">{ar_mae:,.4f}</td>
            <td>{"Moving Average" if ma_mae < ar_mae else ("ARIMA" if ar_mae < ma_mae else "Tie")}</td>
        </tr>
        <tr>
            <td><b>MSE</b></td>
            <td class="{bc(ma_mse,ar_mse)}">{ma_mse:,.4f}</td>
            <td class="{bc(ar_mse,ma_mse)}">{ar_mse:,.4f}</td>
            <td>{"Moving Average" if ma_mse < ar_mse else ("ARIMA" if ar_mse < ma_mse else "Tie")}</td>
        </tr>
        <tr>
            <td><b>RMSE</b></td>
            <td class="{bc(ma_rmse,ar_rmse)}">{ma_rmse:,.4f}</td>
            <td class="{bc(ar_rmse,ma_rmse)}">{ar_rmse:,.4f}</td>
            <td>{"Moving Average" if ma_rmse < ar_rmse else ("ARIMA" if ar_rmse < ma_rmse else "Tie")}</td>
        </tr>
        <tr>
            <td><b>MAPE</b></td>
            <td class="{bc(ma_mape,ar_mape)}">{ma_mape:.4f}%</td>
            <td class="{bc(ar_mape,ma_mape)}">{ar_mape:.4f}%</td>
            <td>{"Moving Average" if ma_mape < ar_mape else ("ARIMA" if ar_mape < ma_mape else "Tie")}</td>
        </tr>
    </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # ── D: Bar Chart ──────────────────────────────
    st.markdown("### D. Metric Comparison Chart")
    metric_labels = ["MAE", "MSE", "RMSE", "MAPE (%)"]
    ma_vals = [ma_mae, ma_mse, ma_rmse, ma_mape]
    ar_vals = [ar_mae, ar_mse, ar_rmse, ar_mape]

    fig_bar = make_subplots(rows=1, cols=4, subplot_titles=metric_labels)
    for i, (mv, av, lbl) in enumerate(zip(ma_vals, ar_vals, metric_labels), 1):
        fig_bar.add_trace(go.Bar(
            x=[f"MA (w={ma_w})", f"ARIMA({p},{d},{q})"],
            y=[mv, av],
            marker_color=[C_MA, C_ARIMA],
            showlegend=False,
            name=lbl,
        ), row=1, col=i)
    fig_bar.update_layout(
        height=300, showlegend=False,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
    )
    for i in range(1, 5):
        fig_bar.update_xaxes(gridcolor="#DDE3EB", linecolor="#B0BAC6", row=1, col=i)
        fig_bar.update_yaxes(gridcolor="#DDE3EB", linecolor="#B0BAC6", row=1, col=i)
    st.plotly_chart(fig_bar, width="stretch")

    # ── E: Residuals ──────────────────────────────
    st.markdown("### E. Residual Analysis")
    tab1, tab2 = st.tabs([f"Moving Average (w={ma_w})", f"ARIMA({p},{d},{q})"])

    def residual_panel(fc, label, color):
        aligned = pd.concat([test, fc], axis=1).dropna()
        aligned.columns = ["Actual", "Forecast"]
        resid = aligned["Actual"] - aligned["Forecast"]

        fig_r = make_subplots(rows=1, cols=2,
            subplot_titles=["<b>Residuals over time</b>", "<b>Residual distribution</b>"])
        fig_r.add_trace(go.Scatter(
            x=resid.index, y=resid.values,
            mode="lines+markers", name="Residual",
            line=dict(color=color, width=1.3), marker=dict(size=3),
        ), row=1, col=1)
        fig_r.add_hline(y=0, line_dash="dash", line_color="#8A9AAA", line_width=1.2, row=1, col=1)
        fig_r.add_trace(go.Histogram(
            x=resid.values, nbinsx=20,
            marker_color=color, opacity=0.65, name="Dist",
        ), row=1, col=2)
        fig_r.update_layout(
            height=280, showlegend=False,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
        )
        for i in range(1, 3):
            fig_r.update_xaxes(gridcolor="#DDE3EB", linecolor="#B0BAC6", row=1, col=i)
            fig_r.update_yaxes(gridcolor="#DDE3EB", linecolor="#B0BAC6", row=1, col=i)
        st.plotly_chart(fig_r, width="stretch")
        st.markdown(f"""
        <div class="insight">
        Residual mean: <b>{resid.mean():.3f}</b> (ideally ≈ 0) &nbsp;|&nbsp;
        Std dev: <b>{resid.std():.3f}</b><br>
        {'Residuals appear roughly centered around zero — no strong systematic bias.' if abs(resid.mean()) < resid.std() * 0.3
         else 'Non-zero mean residuals suggest systematic bias; consider adjusting model parameters.'}
        </div>
        """, unsafe_allow_html=True)

    with tab1: residual_panel(ma_fc, f"MA w={ma_w}", C_MA)
    with tab2: residual_panel(ar_fc, f"ARIMA({p},{d},{q})", C_ARIMA)

    # ── F: Plain Language Summary ─────────────────
    st.markdown("### F. Plain Language Interpretation")
    winner       = "Moving Average" if ma_wins >= ar_wins else "ARIMA"
    winner_color = C_MA if winner == "Moving Average" else C_ARIMA
    better_mae   = min(ma_mae, ar_mae)
    better_mape  = min(ma_mape, ar_mape)

    st.markdown(f"""
    <div class="insight" style="border-left:4px solid {winner_color};">
    <b style="color:{winner_color};">Overall winner: {winner}</b><br><br>
    Based on all four metrics, the <b>{winner}</b> model performed better on {max(ma_wins, ar_wins)} out of 4 metrics.<br><br>
    • The best <b>MAE</b> was <b>{better_mae:,.2f}</b>, meaning on average, the forecast was off by that much
      in the original unit ({col_sel}).<br>
    • The best <b>MAPE</b> was <b>{better_mape:.2f}%</b>, meaning the model's predictions were typically
      within {better_mape:.1f}% of the actual values.<br>
    • A <b>RMSE significantly larger than MAE</b> indicates occasional large forecast errors —
      review the residual plot above for those spikes.<br><br>
    <b>Moving Average</b> is simpler and works well when the series has a slow-moving trend with little autocorrelation.<br>
    <b>ARIMA</b> leverages autocorrelation structure and is typically more accurate when the series shows patterns in its lags.<br><br>
    {'<i>Note: Both models show similar performance — the series may be difficult to forecast due to high noise or structural breaks.</i>' if abs(ma_rmse - ar_rmse) / max(ma_rmse, ar_rmse) < 0.05 else ''}
    </div>
    """, unsafe_allow_html=True)

    # ── G: Forecast Table ─────────────────────────
    st.markdown("### G. Forecast Values Table")
    aligned_all = pd.concat([test, ma_fc, ar_fc], axis=1).dropna()
    aligned_all.columns = ["Actual", f"MA (w={ma_w})", f"ARIMA({p},{d},{q})"]
    aligned_all[f"MA Error"]    = (aligned_all["Actual"] - aligned_all[f"MA (w={ma_w})"]).round(4)
    aligned_all[f"ARIMA Error"] = (aligned_all["Actual"] - aligned_all[f"ARIMA({p},{d},{q})"]).round(4)
    st.dataframe(aligned_all.round(4), width="stretch")

# ═══════════════════════════════════════════════════
# STEP 9 — QUIZ
# ═══════════════════════════════════════════════════
elif step == 9:
    st.markdown('<div class="sec-title">Self-Assessment Quiz</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Complete Beginner, Intermediate, and Advanced sections with 10 questions each.</div>', unsafe_allow_html=True)

    QUIZ_BANK = {
        "Beginner": [
            {
                "q": "Which component shows long-term upward or downward movement in a time series?",
                "options": ["Seasonality", "Trend", "Residual", "Noise"],
                "answer": "Trend",
                "explanation": "Trend captures the persistent long-run direction of the series.",
            },
            {
                "q": "Which component represents repeating fixed-interval patterns such as monthly peaks?",
                "options": ["Residual", "Seasonality", "Trend", "Random walk"],
                "answer": "Seasonality",
                "explanation": "Seasonality is a regular pattern repeating at a known frequency.",
            },
            {
                "q": "What does Moving Average primarily do to a noisy time series?",
                "options": ["Adds noise", "Smooths fluctuations", "Removes all trend", "Creates outliers"],
                "answer": "Smooths fluctuations",
                "explanation": "A moving average reduces short-term noise by averaging neighboring observations.",
            },
            {
                "q": "In ARIMA(p, d, q), which term indicates differencing order?",
                "options": ["p", "q", "d", "None"],
                "answer": "d",
                "explanation": "The I component in ARIMA is differencing and its order is d.",
            },
            {
                "q": "Which metric is measured in the same unit as the target variable?",
                "options": ["MSE", "MAE", "Both MAE and RMSE", "MAPE"],
                "answer": "Both MAE and RMSE",
                "explanation": "MAE and RMSE are interpretable in the original data unit.",
            },
            {
                "q": "MAPE expresses forecast error in which form?",
                "options": ["Absolute value", "Squared value", "Percentage", "Log scale"],
                "answer": "Percentage",
                "explanation": "MAPE reports the average absolute percentage error.",
            },
            {
                "q": "For model evaluation, why do we keep a test split?",
                "options": ["To train faster", "To estimate performance on unseen data", "To remove noise", "To increase p"],
                "answer": "To estimate performance on unseen data",
                "explanation": "Test data simulates future unseen observations.",
            },
            {
                "q": "If residuals are ideal, they should look like:",
                "options": ["Strong trend", "Clear seasonality", "White noise around zero", "Monotonic increase"],
                "answer": "White noise around zero",
                "explanation": "Good residuals contain no obvious structure and are centered near zero.",
            },
            {
                "q": "A larger moving-average window generally gives:",
                "options": ["More jagged forecasts", "Smoother but slower response", "No effect", "Higher seasonality"],
                "answer": "Smoother but slower response",
                "explanation": "More averaging reduces volatility but increases lag.",
            },
            {
                "q": "Which model in this lab is typically better at using lag relationships?",
                "options": ["Simple average", "ARIMA", "Histogram", "Standard scaler"],
                "answer": "ARIMA",
                "explanation": "ARIMA explicitly models autocorrelation and error dynamics.",
            },
        ],
        "Intermediate": [
            {
                "q": "Why is differencing applied before fitting many ARIMA models?",
                "options": ["To make data binary", "To achieve stationarity", "To remove all seasonality", "To scale to [0,1]"],
                "answer": "To achieve stationarity",
                "explanation": "Differencing removes trend-like non-stationarity to stabilize statistical properties.",
            },
            {
                "q": "If RMSE is much larger than MAE, it often indicates:",
                "options": ["No errors", "Many tiny errors only", "Presence of large error spikes", "Perfect stationarity"],
                "answer": "Presence of large error spikes",
                "explanation": "RMSE penalizes larger errors more heavily because of squaring.",
            },
            {
                "q": "Which decomposition form is suitable when seasonal amplitude is roughly constant?",
                "options": ["Multiplicative", "Additive", "Logarithmic", "Exponential smoothing"],
                "answer": "Additive",
                "explanation": "Additive decomposition assumes constant seasonal swings.",
            },
            {
                "q": "In a walk-forward setting, what is a key drawback of simple moving average forecasts?",
                "options": ["Cannot use numeric data", "Lag during rapid shifts", "Requires deep learning", "Needs differencing always"],
                "answer": "Lag during rapid shifts",
                "explanation": "Moving averages react slowly to sudden regime changes.",
            },
            {
                "q": "For fair model comparison, both models should be evaluated on:",
                "options": ["Different test windows", "The same test period", "Only training data", "Random subsets per model"],
                "answer": "The same test period",
                "explanation": "Using the same horizon ensures a like-for-like comparison.",
            },
            {
                "q": "When actual values contain zeros, which metric can become unstable or undefined?",
                "options": ["MAE", "RMSE", "MSE", "MAPE"],
                "answer": "MAPE",
                "explanation": "MAPE divides by actual values, so zeros cause division issues.",
            },
            {
                "q": "A positive mean residual suggests the forecast is often:",
                "options": ["Too high", "Too low", "Unbiased", "Seasonal"],
                "answer": "Too low",
                "explanation": "Residual = Actual - Forecast. Positive average residual means under-forecasting.",
            },
            {
                "q": "If period is monthly, a common seasonal period value is:",
                "options": ["4", "7", "12", "52"],
                "answer": "12",
                "explanation": "Monthly seasonal cycles typically repeat every 12 periods.",
            },
            {
                "q": "What does a test split of 80% train imply for forecasting validation?",
                "options": ["No test data", "20% held out for evaluation", "80% test and 20% train", "Only one-step ahead"],
                "answer": "20% held out for evaluation",
                "explanation": "The remaining 20% is used to assess generalization.",
            },
            {
                "q": "In ARIMA notation, q corresponds to:",
                "options": ["Autoregressive lags", "Differencing order", "Moving-average error terms", "Season length"],
                "answer": "Moving-average error terms",
                "explanation": "q controls how many lagged forecast errors are modeled.",
            },
        ],
        "Advanced": [
            {
                "q": "If residuals still show autocorrelation, the current model is likely:",
                "options": ["Well specified", "Under-specified", "Over-normalized", "Non-numeric"],
                "answer": "Under-specified",
                "explanation": "Residual autocorrelation means the model has not captured all temporal structure.",
            },
            {
                "q": "A model with lower MAE but higher RMSE than another model usually has:",
                "options": ["Smaller extreme misses but larger average error", "Fewer large outliers but slightly more medium errors", "Perfect fit", "No variance"],
                "answer": "Fewer large outliers but slightly more medium errors",
                "explanation": "Lower RMSE favors fewer large misses; lower MAE favors lower average absolute error.",
            },
            {
                "q": "When trend and seasonal magnitude both grow with level, preferred decomposition is:",
                "options": ["Additive", "Multiplicative", "Linear", "Piecewise constant"],
                "answer": "Multiplicative",
                "explanation": "Multiplicative form models seasonality proportional to the series level.",
            },
            {
                "q": "For non-seasonal ARIMA, increasing d too much can lead to:",
                "options": ["Under-differencing", "Over-differencing and information loss", "Automatic seasonality capture", "Lower variance always"],
                "answer": "Over-differencing and information loss",
                "explanation": "Excess differencing can introduce extra noise and reduce predictive signal.",
            },
            {
                "q": "Which statement about MSE is true?",
                "options": ["It is robust to outliers", "It treats all errors linearly", "It amplifies large errors quadratically", "It is unit-free"],
                "answer": "It amplifies large errors quadratically",
                "explanation": "Squaring errors gives large misses disproportionately high weight.",
            },
            {
                "q": "If forecast errors shift from centered to consistently positive after a date, this may indicate:",
                "options": ["Structural break", "Perfect calibration", "Lower variance", "Data leakage"],
                "answer": "Structural break",
                "explanation": "A persistent bias change often points to a regime shift in the data.",
            },
            {
                "q": "Compared with MAE, RMSE is more sensitive to:",
                "options": ["Small random noise", "Rare large forecast misses", "Scale changes only", "Zero inflation"],
                "answer": "Rare large forecast misses",
                "explanation": "The square term makes RMSE react strongly to outliers.",
            },
            {
                "q": "What is the best interpretation when MAE and MAPE are both low?",
                "options": ["Only absolute scale accuracy is good", "Both absolute and relative errors are small", "Model is overfitted", "Residuals must be seasonal"],
                "answer": "Both absolute and relative errors are small",
                "explanation": "Low MAE means small raw error; low MAPE means small percent error.",
            },
            {
                "q": "For model diagnostics, residual distribution centered away from zero indicates:",
                "options": ["No issue", "Systematic bias", "Perfect differencing", "Seasonal period mismatch only"],
                "answer": "Systematic bias",
                "explanation": "A non-zero mean residual implies persistent over- or under-forecasting.",
            },
            {
                "q": "In practical forecasting, choosing between MA and ARIMA should prioritize:",
                "options": ["Model complexity alone", "Metric performance and residual diagnostics", "Number of parameters only", "UI preference"],
                "answer": "Metric performance and residual diagnostics",
                "explanation": "Selection should balance accuracy metrics with whether residuals look pattern-free.",
            },
        ],
    }

    section = st.session_state.quiz_level if st.session_state.quiz_level in QUIZ_LEVELS else QUIZ_LEVELS[0]

    questions = QUIZ_BANK[section]
    submitted_levels = st.session_state.quiz_submitted_levels
    answers_levels = st.session_state.quiz_answers_levels
    scores = st.session_state.quiz_scores

    st.markdown(f"""
    <div class="card card-accent">
    <b>{section} Section:</b> Answer all <b>{len(questions)} questions</b> and submit.
    Each correct answer equals 1 mark.
    </div>
    """, unsafe_allow_html=True)

    if not submitted_levels.get(section, False):
        with st.form(f"quiz_form_{section.lower()}"):
            answers = {}
            for i, item in enumerate(questions):
                st.markdown(f'<div class="quiz-q"><b>Q{i + 1}. {item["q"]}</b></div>', unsafe_allow_html=True)
                answers[i] = st.radio(
                    f"{section} Question {i + 1}",
                    item["options"],
                    key=f"{section}_q{i}",
                    index=None,
                    label_visibility="collapsed",
                )
                st.write("")

            submitted = st.form_submit_button(f"Submit {section} Section")
            if submitted:
                score = sum(1 for i, item in enumerate(questions) if answers.get(i) == item["answer"])
                submitted_levels[section] = True
                answers_levels[section] = answers
                scores[section] = score
                st.session_state.quiz_submitted_levels = submitted_levels
                st.session_state.quiz_answers_levels = answers_levels
                st.session_state.quiz_scores = scores
                st.rerun()
    else:
        score = scores.get(section, 0)
        n_q = len(questions)
        pct = int(score / n_q * 100)
        color = C_ARIMA if pct >= 60 else C_RESID
        grade = ("Excellent" if pct >= 90 else "Good" if pct >= 70
                 else "Satisfactory" if pct >= 50 else "Needs Review")

        st.markdown(f"""
        <div class="score-banner" style="background:linear-gradient(135deg,rgba(26,74,138,0.07),#FFFFFF);
             border:1px solid {color}; color:{color};">
            {section}: {score} / {n_q} &nbsp;·&nbsp; {pct}% &nbsp;·&nbsp; {grade}
        </div>
        """, unsafe_allow_html=True)

        section_answers = answers_levels.get(section, {})
        for i, item in enumerate(questions):
            ua = section_answers.get(i)
            correct = item["answer"]
            right = (ua == correct)
            bg = "#EAF4EE" if right else "#FAEAEC"
            bc_c = C_ARIMA if right else C_RESID
            icon = "Correct" if right else "Incorrect"
            correct_line = "" if right else f"<div class='quiz-result-correct'>Correct answer: <i>{correct}</i></div>"

            st.markdown(f"""<div style="background:{bg}; border-left:4px solid {bc_c}; padding:0.85rem 1.1rem; border-radius:0 6px 6px 0; margin-bottom:0.8rem;">
<div class='quiz-result-title' style="color:{bc_c};">{icon} - Q{i + 1}. {item['q']}</div>
<div class='quiz-result-answer'>Your answer: <i>{ua or 'Not answered'}</i></div>
{correct_line}
<div class='quiz-result-explanation'>{item["explanation"]}</div>
</div>""", unsafe_allow_html=True)

        if st.button(f"Retake {section} Section", key=f"retake_{section}"):
            st.session_state.quiz_submitted_levels[section] = False
            st.session_state.quiz_answers_levels[section] = {}
            st.session_state.quiz_scores[section] = 0
            st.rerun()

    st.markdown("""
    <div class="insight" style="margin-top:1.5rem;">
    Complete all levels for comprehensive self-assessment. Continue to <b>Section 10 — References</b>
    for further learning resources.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# STEP 10 — REFERENCES
# ═══════════════════════════════════════════════════
elif step == 10:
    st.markdown('<div class="sec-title">References</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Curated resources to extend your learning beyond this lab.</div>', unsafe_allow_html=True)

    st.markdown("### A. Books and Research Papers")
    st.markdown("""
    <div class="card card-accent">
    1. <b>Box, G. E. P., Jenkins, G. M., Reinsel, G. C., &amp; Ljung, G. M.</b><br>
    <i>Time Series Analysis: Forecasting and Control</i> (5th Edition), Wiley.
    <a href="https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021" target="_blank">Book link</a><br><br>

    2. <b>Hyndman, R. J., &amp; Athanasopoulos, G.</b><br>
    <i>Forecasting: Principles and Practice</i> (Free online text):
    <a href="https://otexts.com/fpp3/" target="_blank">https://otexts.com/fpp3/</a><br><br>

    3. <b>Makridakis, S., Spiliotis, E., &amp; Assimakopoulos, V. (2018).</b><br>
    The M4 Competition: Results, findings, conclusion and way forward,
    <i>International Journal of Forecasting</i>.<br>
    <a href="https://www.sciencedirect.com/science/article/pii/S0169207018300785" target="_blank">Read paper</a><br><br>

    4. <b>Hyndman, R. J., &amp; Khandakar, Y. (2008).</b><br>
    Automatic time series forecasting: the forecast package for R,
    <i>Journal of Statistical Software</i>.<br>
    <a href="https://www.jstatsoft.org/article/view/v027i03" target="_blank">Read paper</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### B. Documentation for Practice")
    st.markdown("""
    <div class="card card-green">
    • Statsmodels ARIMA docs:
    <a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html" target="_blank">ARIMA API</a><br>
    • Seasonal decomposition docs:
    <a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html" target="_blank">seasonal_decompose API</a><br>
    • Pandas time-series guide:
    <a href="https://pandas.pydata.org/docs/user_guide/timeseries.html" target="_blank">Pandas Time Series</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### C. YouTube Videos")
    st.markdown("""
    <div class="card card-gold">
    • StatQuest — ARIMA explained:
    <a href="https://www.youtube.com/watch?v=-aCF0_wfVwY" target="_blank">Watch</a><br>
    • freeCodeCamp — Time series forecasting full tutorial:
    <a href="https://www.youtube.com/watch?v=0E_31WqVzCY" target="_blank">Watch</a><br>
    • Krish Naik — ARIMA in Python practical:
    <a href="https://www.youtube.com/watch?v=8FCDpFhd1zk" target="_blank">Watch</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight" style="margin-top:1.3rem;">
    <b>Experiment Complete.</b> You have now completed all 10 sections of the virtual lab.
    Revisit earlier sections to test additional datasets and parameter settings.
    </div>
    """, unsafe_allow_html=True)