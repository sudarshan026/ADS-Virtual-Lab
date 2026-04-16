import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Multimodal Fusion Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODALITY_META = [
    {
        "key": "image",
        "label": "🖼️ Image",
        "badge_class": "badge-image",
        "features": 50,
        "color": "#f472b6",
        "summary": "Visual patterns",
        "importance": "High discriminative power",
    },
    {
        "key": "text",
        "label": "📝 Text",
        "badge_class": "badge-text",
        "features": 40,
        "color": "#38bdf8",
        "summary": "Semantic features",
        "importance": "Medium importance",
    },
    {
        "key": "audio",
        "label": "🎵 Audio",
        "badge_class": "badge-audio",
        "features": 30,
        "color": "#34d399",
        "summary": "Signal features",
        "importance": "Medium-low importance",
    },
    {
        "key": "sensor",
        "label": "📡 Sensor",
        "badge_class": "badge-sensor",
        "features": 20,
        "color": "#fb7185",
        "summary": "Statistical features",
        "importance": "Lower importance",
    },
    {
        "key": "video",
        "label": "🎬 Video/Social",
        "badge_class": "badge-video",
        "features": 25,
        "color": "#f59e0b",
        "summary": "Temporal engagement features",
        "importance": "Contextual support signal",
    },
]

FEATURE_SLICES = {}
_feature_start = 0
for modality in MODALITY_META:
    FEATURE_SLICES[modality["key"]] = slice(
        _feature_start, _feature_start + modality["features"]
    )
    _feature_start += modality["features"]
TOTAL_FEATURES = _feature_start

st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 55%, #ea580c 100%);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 16px 40px rgba(37, 99, 235, 0.22);
    }

    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .info-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin: 0.8rem 0;
    }

    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 14px;
        text-align: center;
        border-top: 4px solid #2563eb;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }

    .modality-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        margin-bottom: 0.4rem;
        color: white;
    }

    .badge-image { background: linear-gradient(135deg, #ec4899 0%, #f97316 100%); }
    .badge-text { background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%); }
    .badge-audio { background: linear-gradient(135deg, #10b981 0%, #22c55e 100%); }
    .badge-sensor { background: linear-gradient(135deg, #ef4444 0%, #f59e0b 100%); }
    .badge-video { background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); }

    .section-header {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 800;
        margin: 1.7rem 0 0.9rem 0;
        padding-bottom: 0.45rem;
        border-bottom: 3px solid #2563eb;
    }

    .highlight-note {
        background: linear-gradient(135deg, #eff6ff 0%, #fff7ed 100%);
        border-left: 4px solid #2563eb;
        padding: 0.9rem 1rem;
        border-radius: 10px;
        margin-top: 0.8rem;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 10px;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        font-weight: 700;
        padding: 0.75rem 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False


def repeat_to_width(values, width):
    repeats = int(np.ceil(width / values.shape[1]))
    return np.tile(values, (1, repeats))[:, :width]


def add_label_noise(y, ratio, rng):
    y_noisy = y.copy()
    n_noisy = int(len(y_noisy) * ratio)
    if n_noisy <= 0:
        return y_noisy

    classes = np.unique(y_noisy)
    noisy_idx = rng.choice(len(y_noisy), size=n_noisy, replace=False)

    for idx in noisy_idx:
        alternatives = classes[classes != y_noisy[idx]]
        y_noisy[idx] = rng.choice(alternatives)

    return y_noisy


def add_feature_noise(feature_map, rng, noise_scale):
    noisy_map = {}
    for key, values in feature_map.items():
        sample_drift = rng.normal(0, noise_scale * 0.35, size=(values.shape[0], 1))
        noisy_map[key] = (
            values
            + rng.normal(0, noise_scale, size=values.shape)
            + sample_drift
        )
    return noisy_map


def standardize_modalities(feature_map):
    return {
        key: StandardScaler().fit_transform(values)
        for key, values in feature_map.items()
    }


def build_video_features(base_features, rng):
    centered = base_features - base_features.mean(axis=1, keepdims=True)
    head = centered[:, : min(10, centered.shape[1])]
    tail = centered[:, -min(10, centered.shape[1]) :]
    motion_window = base_features[:, : min(14, base_features.shape[1])]

    handcrafted = np.column_stack(
        [
            base_features.mean(axis=1),
            base_features.std(axis=1),
            np.percentile(base_features, 15, axis=1),
            np.percentile(base_features, 85, axis=1),
            np.max(base_features, axis=1) - np.min(base_features, axis=1),
            np.mean(np.square(centered), axis=1),
            np.mean(np.abs(np.diff(motion_window, axis=1)), axis=1),
            np.mean(np.abs(head), axis=1),
            np.mean(np.abs(tail), axis=1),
            head.mean(axis=1) - tail.mean(axis=1),
        ]
    )

    interaction = np.column_stack(
        [
            handcrafted[:, 0] * handcrafted[:, 1],
            handcrafted[:, 2] + handcrafted[:, 3],
            handcrafted[:, 4] * 0.5 + handcrafted[:, 6],
            handcrafted[:, 5] - handcrafted[:, 8],
            handcrafted[:, 7] + handcrafted[:, 9],
        ]
    )

    combined = np.column_stack([handcrafted, interaction])
    combined = repeat_to_width(
        combined, FEATURE_SLICES["video"].stop - FEATURE_SLICES["video"].start
    )
    return combined + rng.normal(0, 0.12, size=combined.shape)


def make_modality_dataframe(feature_map):
    rows = []
    for modality in MODALITY_META:
        values = feature_map[modality["key"]]
        rows.append(
            {
                "Modality": modality["label"],
                "Features": modality["features"],
                "Mean": float(values.mean()),
                "Std Dev": float(values.std()),
                "Max": float(values.max()),
                "Min": float(values.min()),
            }
        )
    return pd.DataFrame(rows)


def split_modalities(concatenated):
    return [concatenated[:, FEATURE_SLICES[m["key"]]] for m in MODALITY_META]


def build_mnist_bundle():
    rng = np.random.default_rng(42)
    data = load_digits()
    X_raw = data.data[:1200]
    y_raw = data.target[:1200]

    mask = y_raw < 5
    X_raw = X_raw[mask]
    y_raw = y_raw[mask]

    image = X_raw[:, :50] * 1.35

    text_base = np.column_stack(
        [
            X_raw.mean(axis=1),
            X_raw.std(axis=1),
            X_raw.max(axis=1),
            X_raw.min(axis=1),
            np.percentile(X_raw, 25, axis=1),
            np.percentile(X_raw, 75, axis=1),
            (X_raw > X_raw.mean(axis=1, keepdims=True)).sum(axis=1),
            (X_raw > 8).sum(axis=1),
        ]
    )
    text = repeat_to_width(text_base, 40) * 1.05

    audio = repeat_to_width(np.diff(X_raw[:, :31], axis=1), 30) * 0.92

    sensor_base = np.column_stack(
        [
            X_raw.var(axis=1),
            np.percentile(X_raw, 10, axis=1),
            np.percentile(X_raw, 90, axis=1),
            (X_raw[:, :32].sum(axis=1) - X_raw[:, 32:].sum(axis=1)),
            np.mean(np.abs(np.diff(X_raw[:, :16], axis=1)), axis=1),
        ]
    )
    sensor = repeat_to_width(sensor_base, 20) * 0.85

    video = build_video_features(X_raw, rng) * 0.9

    features = {
        "image": image,
        "text": text,
        "audio": audio,
        "sensor": sensor,
        "video": video,
    }
    features = add_feature_noise(features, rng, 0.28)
    y = add_label_noise(y_raw, 0.08, rng)

    return {
        "name": "MNIST Handwritten Digits",
        "description": "Digits 0-4 with added feature noise and 8% label corruption for more realistic evaluation.",
        "feature_map": standardize_modalities(features),
        "y": y,
        "n_classes": len(np.unique(y)),
        "original_images": X_raw.reshape(-1, 8, 8),
        "label_noise_ratio": 0.08,
    }


def build_iris_bundle():
    rng = np.random.default_rng(77)
    data = load_iris()
    X_raw = np.tile(data.data, (5, 1)) + rng.normal(
        0, 0.22, size=(len(data.data) * 5, 4)
    )
    y_raw = np.tile(data.target, 5)

    pairwise = np.column_stack(
        [
            X_raw[:, 0],
            X_raw[:, 1],
            X_raw[:, 2],
            X_raw[:, 3],
            X_raw[:, 0] * X_raw[:, 2],
            X_raw[:, 1] * X_raw[:, 3],
            X_raw[:, 0] - X_raw[:, 1],
            X_raw[:, 2] - X_raw[:, 3],
        ]
    )

    image = repeat_to_width(pairwise, 50) * 1.15
    text = repeat_to_width(
        np.column_stack(
            [
                X_raw.mean(axis=1),
                X_raw.std(axis=1),
                X_raw.max(axis=1),
                np.percentile(X_raw, 20, axis=1),
                np.percentile(X_raw, 80, axis=1),
            ]
        ),
        40,
    ) * 1.0
    audio = repeat_to_width(
        np.column_stack([np.diff(X_raw, axis=1), X_raw[:, :1]]), 30
    ) * 0.9
    sensor = repeat_to_width(
        np.column_stack(
            [
                X_raw[:, 0] + X_raw[:, 1],
                X_raw[:, 2] + X_raw[:, 3],
                X_raw[:, 0] / (np.abs(X_raw[:, 1]) + 1e-3),
                X_raw[:, 2] / (np.abs(X_raw[:, 3]) + 1e-3),
            ]
        ),
        20,
    ) * 0.82
    video = build_video_features(repeat_to_width(X_raw, 16), rng) * 0.88

    features = {
        "image": image,
        "text": text,
        "audio": audio,
        "sensor": sensor,
        "video": video,
    }
    features = add_feature_noise(features, rng, 0.26)
    y = add_label_noise(y_raw, 0.07, rng)

    return {
        "name": "Iris Flowers",
        "description": "Real flower measurements expanded with noise so the classes overlap instead of separating perfectly.",
        "feature_map": standardize_modalities(features),
        "y": y,
        "n_classes": len(np.unique(y)),
        "original_images": None,
        "label_noise_ratio": 0.07,
    }


def build_wine_bundle():
    rng = np.random.default_rng(105)
    data = load_wine()
    X_base = np.tile(data.data, (4, 1)) + rng.normal(
        0, 0.35, size=(len(data.data) * 4, data.data.shape[1])
    )
    y_raw = np.tile(data.target, 4)

    image = repeat_to_width(X_base[:, :10], 50) * 1.1
    text = repeat_to_width(
        np.column_stack(
            [
                X_base.mean(axis=1),
                X_base.std(axis=1),
                X_base[:, 0] * X_base[:, 6],
                X_base[:, 9] - X_base[:, 12],
                np.percentile(X_base, 30, axis=1),
                np.percentile(X_base, 70, axis=1),
            ]
        ),
        40,
    ) * 1.0
    audio = repeat_to_width(np.diff(X_base[:, :11], axis=1), 30) * 0.88
    sensor = repeat_to_width(
        np.column_stack(
            [
                X_base[:, 0] / (X_base[:, 1] + 1e-3),
                X_base[:, 3] / (X_base[:, 4] + 1e-3),
                X_base[:, 5] + X_base[:, 6],
                X_base[:, 8] - X_base[:, 10],
            ]
        ),
        20,
    ) * 0.84
    video = build_video_features(X_base[:, :12], rng) * 0.86

    features = {
        "image": image,
        "text": text,
        "audio": audio,
        "sensor": sensor,
        "video": video,
    }
    features = add_feature_noise(features, rng, 0.30)
    y = add_label_noise(y_raw, 0.10, rng)

    return {
        "name": "Wine Recognition",
        "description": "Uses the real sklearn wine dataset instead of perfectly separated synthetic clusters.",
        "feature_map": standardize_modalities(features),
        "y": y,
        "n_classes": len(np.unique(y)),
        "original_images": None,
        "label_noise_ratio": 0.10,
    }


def build_social_bundle():
    rng = np.random.default_rng(222)
    n_samples = 650
    n_classes = 4
    y_raw = rng.integers(0, n_classes, size=n_samples)

    trend = rng.normal(loc=y_raw * 0.9, scale=0.9)
    sentiment = rng.normal(loc=(y_raw % 2) * 1.2 - 0.3, scale=1.0)
    creator_strength = rng.normal(loc=y_raw * 0.6, scale=0.7)
    audience_match = rng.normal(loc=(3 - y_raw) * 0.4, scale=0.9)

    image = repeat_to_width(
        np.column_stack([trend, creator_strength, sentiment, audience_match]), 50
    ) * 1.1
    text = repeat_to_width(
        np.column_stack(
            [sentiment, trend, creator_strength, audience_match, trend * sentiment]
        ),
        40,
    ) * 1.03
    audio = repeat_to_width(
        np.column_stack(
            [
                trend - sentiment,
                creator_strength + audience_match,
                np.sin(trend),
                np.cos(sentiment),
            ]
        ),
        30,
    ) * 0.9
    sensor = repeat_to_width(
        np.column_stack(
            [np.abs(trend), np.abs(sentiment), creator_strength, audience_match]
        ),
        20,
    ) * 0.84
    video = repeat_to_width(
        np.column_stack(
            [
                0.7 * trend + 0.5 * creator_strength,
                0.6 * sentiment + 0.4 * audience_match,
                np.abs(trend - sentiment),
                trend * audience_match,
                creator_strength * sentiment,
            ]
        ),
        25,
    ) * 1.0

    features = {
        "image": image,
        "text": text,
        "audio": audio,
        "sensor": sensor,
        "video": video,
    }
    features = add_feature_noise(features, rng, 0.42)
    y = add_label_noise(y_raw, 0.12, rng)

    return {
        "name": "Social Media Engagement",
        "description": "Synthetic short-video/social dataset with overlapping creator, audience, sentiment, and trend signals.",
        "feature_map": standardize_modalities(features),
        "y": y,
        "n_classes": len(np.unique(y)),
        "original_images": None,
        "label_noise_ratio": 0.12,
    }


def load_dataset_bundle(choice):
    if "MNIST" in choice:
        return build_mnist_bundle()
    if "Iris" in choice:
        return build_iris_bundle()
    if "Wine" in choice:
        return build_wine_bundle()
    return build_social_bundle()


def store_dataset(bundle, dataset_choice):
    st.session_state.feature_map = bundle["feature_map"]
    for key, values in bundle["feature_map"].items():
        st.session_state[f"X_{key}"] = values

    st.session_state.y = bundle["y"]
    st.session_state.n_classes = bundle["n_classes"]
    st.session_state.dataset_name = bundle["name"]
    st.session_state.dataset_choice = dataset_choice
    st.session_state.dataset_description = bundle["description"]
    st.session_state.original_images = bundle["original_images"]
    st.session_state.label_noise_ratio = bundle["label_noise_ratio"]
    st.session_state.data_loaded = True
    st.session_state.model_trained = False


st.markdown(
    """
<div class="main-header">
    <h1>🧬 Multimodal Fusion Virtual Lab</h1>
    <p>Attention-based modality importance with realistic overlap, noisy labels, and interpretable analytics</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/000000/artificial-intelligence.png",
        width=80,
    )
    st.markdown("### 🎯 Navigation")
    page = st.radio(
        "",
        [
            "📚 Introduction",
            "🔬 Load Dataset",
            "👁️ Visualize Data",
            "⚡ Train Model",
            "📊 Results",
            "🎓 Theory",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
<div style='text-align: center; padding: 1rem; border-radius: 12px; color: white; background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);'>
    <small>Built for explainable multimodal data science</small>
</div>
""",
        unsafe_allow_html=True,
    )

if page == "📚 Introduction":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            '<p class="section-header">What Is Multimodal Fusion?</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
<div class="info-card">
    <h3>🎯 The Big Idea</h3>
    <p style='line-height: 1.75;'>
        A strong decision system rarely trusts only one source of information. In healthcare, media analysis,
        robotics, and smart labs, we combine multiple views of the same sample to make a better prediction.
    </p>
    <ul style='line-height: 1.8;'>
        <li>Visual cues from images or frames</li>
        <li>Semantic cues from text-like summaries</li>
        <li>Temporal cues from audio-style signal changes</li>
        <li>Numerical cues from sensor statistics</li>
        <li>Context cues from video or social engagement patterns</li>
    </ul>
    <p style='line-height: 1.75;'>
        This virtual lab trains a lightweight attention-style fusion system so you can see which modality matters most.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="section-header">Why This Version Is More Realistic</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
<div class="info-card">
    <h3>⚖️ We Reduced the Too-Perfect Effect</h3>
    <p style='line-height: 1.75;'>
        Earlier versions could hit unrealistic 100% accuracy because the classes were too cleanly separated.
        This version intentionally introduces overlapping distributions, feature noise, and controlled label noise.
    </p>
    <div class="highlight-note">
        <strong>Viva-ready explanation:</strong> The first setup had highly separable features, so even simple models could classify perfectly.
        We reduced class separation and added controlled noise to better simulate real-world uncertainty.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<p class="section-header">Our 5 Modalities</p>',
            unsafe_allow_html=True,
        )
        for idx, modality in enumerate(MODALITY_META):
            st.markdown(
                f'<span class="modality-badge {modality["badge_class"]}">{modality["label"]}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'{modality["summary"]} ({modality["features"]} features)'
            )
            st.markdown(f'*{modality["importance"]}*')
            if idx < len(MODALITY_META) - 1:
                st.markdown("---")

    st.markdown(
        '<p class="section-header">Why This Fits Data Science</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    messages = [
        (
            "Interpretable",
            "You can inspect every modality weight and compare it against single-modality baselines.",
        ),
        (
            "Fast",
            "Training is quick because each branch uses compact classical models instead of heavy deep networks.",
        ),
        (
            "Explainable",
            "Feature engineering, metrics, and attention weights are visible end-to-end.",
        ),
    ]
    colors = ["#2563eb", "#ec4899", "#059669"]

    for col, (title, body), color in zip(cols, messages, colors):
        with col:
            st.markdown(
                f"""
<div class="metric-card">
    <h4 style='color: {color};'>{title}</h4>
    <p style='color: #475569;'>{body}</p>
</div>
""",
                unsafe_allow_html=True,
            )

elif page == "🔬 Load Dataset":
    st.markdown(
        '<p class="section-header">Load Dataset</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="info-card">
    <h4>📊 Available Dataset Options</h4>
    <p>Each option is converted into five modalities and then lightly corrupted so the model behaves more like a real system.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    dataset_choice = st.selectbox(
        "Select Dataset",
        [
            "MNIST Handwritten Digits (Best for Demo)",
            "Iris Flowers",
            "Wine Recognition",
            "Social Media Engagement (Synthetic)",
        ],
        help="MNIST is best for showing image dominance. Social Media is best for demonstrating the new Video/Social modality.",
    )

    if st.button("📥 Load Dataset", use_container_width=True):
        with st.spinner("Preparing multimodal dataset..."):
            progress_bar = st.progress(0)
            try:
                progress_bar.progress(15)
                bundle = load_dataset_bundle(dataset_choice)
                progress_bar.progress(70)
                store_dataset(bundle, dataset_choice)
                progress_bar.progress(100)
                st.success(
                    f"Loaded {len(st.session_state.y)} samples, {st.session_state.n_classes} classes, "
                    f"{len(MODALITY_META)} modalities, and {int(st.session_state.label_noise_ratio * 100)}% label noise."
                )
                st.balloons()
            except Exception as exc:
                st.error(f"Error while loading dataset: {exc}")

    if st.session_state.data_loaded:
        st.markdown(
            '<p class="section-header">Dataset Summary</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div class="info-card">
    <h4>{st.session_state.dataset_name}</h4>
    <p>{st.session_state.dataset_description}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        cols = st.columns(5)
        cols[0].metric("Samples", len(st.session_state.y))
        cols[1].metric("Features", TOTAL_FEATURES)
        cols[2].metric("Classes", st.session_state.n_classes)
        cols[3].metric("Modalities", len(MODALITY_META))
        cols[4].metric(
            "Label Noise", f"{int(st.session_state.label_noise_ratio * 100)}%"
        )

        fig = px.histogram(
            pd.DataFrame({"Class": st.session_state.y}),
            x="Class",
            color_discrete_sequence=["#2563eb"],
            title=f"Class Distribution - {st.session_state.dataset_name}",
        )
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            make_modality_dataframe(st.session_state.feature_map),
            use_container_width=True,
        )

elif page == "👁️ Visualize Data":
    st.markdown(
        '<p class="section-header">Visualize Multimodal Features</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.data_loaded:
        st.warning("Please load a dataset first.")
    else:
        st.markdown(
            """
<div class="info-card">
    <h4>👁️ Compare What Each Modality Sees</h4>
    <p>These feature views come from the same sample, but each modality emphasizes a different signal.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        if st.session_state.original_images is not None:
            st.markdown("### 🖼️ Original MNIST Samples")
            cols = st.columns(10)
            for i in range(min(10, len(st.session_state.original_images))):
                with cols[i]:
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=st.session_state.original_images[i],
                            colorscale="Greys",
                            showscale=False,
                        )
                    )
                    fig.update_layout(
                        width=80,
                        height=90,
                        margin=dict(l=0, r=0, t=22, b=0),
                        title=dict(
                            text=f"Label {st.session_state.y[i]}",
                            font=dict(size=10),
                        ),
                        xaxis=dict(showticklabels=False),
                        yaxis=dict(showticklabels=False),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        sample_idx = st.slider(
            "Select sample", 0, len(st.session_state.y) - 1, 0
        )
        tabs = st.tabs([m["label"] for m in MODALITY_META])

        for tab, modality in zip(tabs, MODALITY_META):
            with tab:
                values = st.session_state.feature_map[modality["key"]][sample_idx]
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=list(range(1, min(20, len(values)) + 1)),
                        y=values[:20],
                        marker_color=modality["color"],
                    )
                )
                fig.update_layout(
                    title=f"{modality['label']} features for sample {sample_idx} (class {st.session_state.y[sample_idx]})",
                    xaxis_title="Feature Index",
                    yaxis_title="Feature Value",
                    height=320,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📈 Modality Statistics Comparison")
        stats_df = make_modality_dataframe(st.session_state.feature_map)
        st.dataframe(stats_df, use_container_width=True)

        comparison_fig = px.bar(
            stats_df,
            x="Modality",
            y="Std Dev",
            color="Modality",
            color_discrete_sequence=[m["color"] for m in MODALITY_META],
            title="Relative Variability by Modality",
        )
        comparison_fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(comparison_fig, use_container_width=True)

elif page == "⚡ Train Model":
    st.markdown(
        '<p class="section-header">Train Attention-Based Fusion Model</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.data_loaded:
        st.warning("Please load a dataset first.")
    else:
        st.markdown(
            """
<div class="info-card">
    <h4>🧠 Training Flow</h4>
    <p><strong>Step 1:</strong> Train one classifier per modality.</p>
    <p><strong>Step 2:</strong> Learn attention weights for all modalities.</p>
    <p><strong>Step 3:</strong> Fuse probability outputs into a final prediction.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            with st.form("train_form"):
                st.markdown("#### Training Configuration")
                c1, c2 = st.columns(2)

                with c1:
                    learning_rate = st.select_slider(
                        "Learning rate",
                        [0.001, 0.01, 0.03, 0.05, 0.1],
                        value=0.03,
                    )
                    epochs = st.slider("Epochs", 20, 120, 60, 10)

                with c2:
                    test_size = st.slider("Test split %", 10, 40, 20, 5) / 100
                    reg_strength = st.slider(
                        "Weight smoothing", 0.0, 0.3, 0.08, 0.02
                    )

                train_btn = st.form_submit_button(
                    "🚀 Start Training", use_container_width=True
                )

        with col2:
            st.markdown(
                f"""
<div class="metric-card">
    <h4>📐 Fusion Formula</h4>
    <div style='background: #f8fafc; padding: 0.9rem; border-radius: 10px; margin: 1rem 0;'>
        <code>f = Σ αᵢ · hᵢ</code><br>
        <code>Σ αᵢ = 1</code>
    </div>
    <p style='color: #475569;'>Current dataset: {st.session_state.dataset_name}</p>
    <p style='color: #475569;'>Features: {TOTAL_FEATURES}</p>
</div>
""",
                unsafe_allow_html=True,
            )

        if train_btn:
            with st.spinner("Training fusion model..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                X_all = np.concatenate(
                    [st.session_state.feature_map[m["key"]] for m in MODALITY_META],
                    axis=1,
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    X_all,
                    st.session_state.y,
                    test_size=test_size,
                    random_state=42,
                    stratify=st.session_state.y,
                )

                progress_bar.progress(10)
                status_text.text("Training per-modality classifiers...")

                train_mods = split_modalities(X_train)
                test_mods = split_modalities(X_test)

                clfs = []
                train_probas = []
                test_probas = []
                single_modality_accs = []

                for idx, (train_mod, test_mod) in enumerate(
                    zip(train_mods, test_mods)
                ):
                    clf = LogisticRegression(
                        max_iter=1400,
                        random_state=42,
                        C=0.8,
                        solver="lbfgs",
                    )
                    clf.fit(train_mod, y_train)
                    clfs.append(clf)

                    train_probas.append(clf.predict_proba(train_mod))
                    test_prob = clf.predict_proba(test_mod)
                    test_probas.append(test_prob)
                    single_modality_accs.append(
                        accuracy_score(y_test, np.argmax(test_prob, axis=1))
                    )

                    progress_bar.progress(
                        12 + int(((idx + 1) / len(MODALITY_META)) * 18)
                    )
                    time.sleep(0.05)

                attention = np.full(len(MODALITY_META), 1.0 / len(MODALITY_META))
                train_losses = []
                train_accs = []
                val_accs = []
                attention_hist = []

                status_text.text("Learning modality attention weights...")
                eps = 1e-10

                for epoch in range(epochs):
                    train_fused = sum(
                        weight * proba
                        for weight, proba in zip(attention, train_probas)
                    )
                    test_fused = sum(
                        weight * proba
                        for weight, proba in zip(attention, test_probas)
                    )

                    train_pred = np.argmax(train_fused, axis=1)
                    test_pred = np.argmax(test_fused, axis=1)

                    train_acc = accuracy_score(y_train, train_pred)
                    test_acc = accuracy_score(y_test, test_pred)
                    train_loss = -np.mean(
                        np.log(train_fused[np.arange(len(y_train)), y_train] + eps)
                    )

                    grads = np.zeros_like(attention)
                    delta = 1e-5

                    for i in range(len(attention)):
                        plus = attention.copy()
                        plus[i] += delta
                        plus = np.maximum(plus, 1e-6)
                        plus = plus / plus.sum()

                        fused_plus = sum(
                            weight * proba
                            for weight, proba in zip(plus, train_probas)
                        )
                        loss_plus = -np.mean(
                            np.log(
                                fused_plus[np.arange(len(y_train)), y_train] + eps
                            )
                        )
                        grads[i] = (loss_plus - train_loss) / delta

                    attention = attention - learning_rate * grads
                    attention = attention + reg_strength * (
                        (1.0 / len(attention)) - attention
                    )
                    attention = np.maximum(attention, 0.02)
                    attention = attention / attention.sum()

                    if epoch > 0:
                        jitter = np.random.default_rng(epoch + 5).uniform(
                            -0.003, 0.003, len(attention)
                        )
                        attention = np.maximum(attention + jitter, 0.02)
                        attention = attention / attention.sum()

                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    val_accs.append(test_acc)
                    attention_hist.append(attention.copy())

                    progress_bar.progress(30 + int(((epoch + 1) / epochs) * 65))
                    if epoch % 5 == 0 or epoch == epochs - 1:
                        status_text.text(
                            f"Epoch {epoch + 1}/{epochs} | loss={train_loss:.4f} | "
                            f"train acc={train_acc:.3f} | val acc={test_acc:.3f}"
                        )

                final_fused = sum(
                    weight * proba for weight, proba in zip(attention, test_probas)
                )
                y_pred = np.argmax(final_fused, axis=1)
                final_acc = accuracy_score(y_test, y_pred)

                progress_bar.progress(100)
                status_text.empty()

                st.session_state.base_classifiers = clfs
                st.session_state.attention_weights = attention
                st.session_state.attention_hist = np.array(attention_hist)
                st.session_state.train_losses = train_losses
                st.session_state.train_accs = train_accs
                st.session_state.val_accs = val_accs
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.final_acc = final_acc
                st.session_state.model_trained = True
                st.session_state.single_modality_accs = single_modality_accs

                st.success(
                    f"Training complete. Final fused accuracy: {final_acc * 100:.2f}%"
                )
                st.balloons()

        if st.session_state.model_trained:
            st.markdown("---")
            st.markdown("### 📈 Training Progress")

            top_idx = int(np.argmax(st.session_state.attention_weights))
            top_name = MODALITY_META[top_idx]["label"]

            summary_cols = st.columns(3)
            summary_cols[0].metric(
                "Final Accuracy", f"{st.session_state.final_acc * 100:.1f}%"
            )
            summary_cols[1].metric("Best Weighted Modality", top_name)
            summary_cols[2].metric(
                "Label Noise", f"{int(st.session_state.label_noise_ratio * 100)}%"
            )

            metric_cols = st.columns(len(MODALITY_META))
            for col, modality, weight in zip(
                metric_cols, MODALITY_META, st.session_state.attention_weights
            ):
                col.metric(modality["label"], f"{weight:.3f}")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        y=st.session_state.train_losses,
                        mode="lines+markers",
                        line=dict(color="#2563eb", width=3),
                        marker=dict(size=4),
                        name="Training Loss",
                    )
                )
                fig.update_layout(
                    title="Loss Curve",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=320,
                )
                st.plotly_chart(fig, use_container_width=True)

            with chart_col2:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        y=st.session_state.val_accs,
                        mode="lines+markers",
                        line=dict(color="#10b981", width=3),
                        marker=dict(size=4),
                        name="Validation Accuracy",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        y=st.session_state.train_accs,
                        mode="lines",
                        line=dict(color="#38bdf8", width=2, dash="dash"),
                        name="Training Accuracy",
                    )
                )
                fig.update_layout(
                    title="Accuracy Curve",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy",
                    height=320,
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Results":
    st.markdown(
        '<p class="section-header">Analysis & Insights</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.model_trained:
        st.warning("Please train a model first.")
    else:
        modality_names = [m["label"] for m in MODALITY_META]
        colors = [m["color"] for m in MODALITY_META]

        st.markdown("### 🎯 Learned Modality Importance")
        weight_fig = go.Figure()
        weight_fig.add_trace(
            go.Bar(
                x=modality_names,
                y=st.session_state.attention_weights,
                marker=dict(color=colors, line=dict(color="white", width=1.5)),
                text=[f"{w:.3f}" for w in st.session_state.attention_weights],
                textposition="outside",
            )
        )
        weight_fig.update_layout(
            yaxis_title="Attention Weight",
            showlegend=False,
            height=380,
            yaxis=dict(
                range=[0, max(st.session_state.attention_weights) * 1.25]
            ),
        )
        st.plotly_chart(weight_fig, use_container_width=True)

        best_idx = int(np.argmax(st.session_state.attention_weights))
        weakest_idx = int(np.argmin(st.session_state.attention_weights))

        st.success(
            f"Most important modality: {modality_names[best_idx]} "
            f"({st.session_state.attention_weights[best_idx]:.3f})"
        )

        st.markdown("### 📉 Single Modality vs Fused Model")
        baseline_df = pd.DataFrame(
            {
                "Model": modality_names + ["✨ Fused Model"],
                "Accuracy": st.session_state.single_modality_accs
                + [st.session_state.final_acc],
            }
        )
        baseline_fig = px.bar(
            baseline_df,
            x="Model",
            y="Accuracy",
            color="Model",
            color_discrete_sequence=colors + ["#1d4ed8"],
            title="Fusion should outperform most individual modalities",
        )
        baseline_fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(baseline_fig, use_container_width=True)

        st.markdown("### 📈 Attention Evolution During Training")
        evolution_fig = go.Figure()
        for i, modality in enumerate(MODALITY_META):
            evolution_fig.add_trace(
                go.Scatter(
                    y=st.session_state.attention_hist[:, i],
                    mode="lines",
                    line=dict(color=modality["color"], width=3),
                    name=modality["label"],
                )
            )
        evolution_fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Attention Weight",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(evolution_fig, use_container_width=True)

        st.markdown("### 🎯 Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        cm_fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Class {i}" for i in range(st.session_state.n_classes)],
                y=[f"Class {i}" for i in range(st.session_state.n_classes)],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
            )
        )
        cm_fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        st.markdown("### 📋 Classification Report")
        report_df = pd.DataFrame(
            classification_report(
                st.session_state.y_test,
                st.session_state.y_pred,
                output_dict=True,
            )
        ).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        st.markdown("### 💡 Key Insights")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
<div class="info-card">
    <h4>🔍 What Changed</h4>
    <ul style='line-height: 1.8;'>
        <li>Classes are now intentionally more overlapped</li>
        <li>Label noise makes evaluation less artificial</li>
        <li>The new Video/Social branch adds contextual variety</li>
        <li>Fusion remains interpretable through learned weights</li>
    </ul>
</div>
""",
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
<div class="info-card">
    <h4>📊 Model Summary</h4>
    <ul style='line-height: 1.8;'>
        <li><strong>Final Accuracy:</strong> {st.session_state.final_acc * 100:.2f}%</li>
        <li><strong>Most Important:</strong> {modality_names[best_idx]}</li>
        <li><strong>Least Important:</strong> {modality_names[weakest_idx]}</li>
        <li><strong>Label Noise:</strong> {int(st.session_state.label_noise_ratio * 100)}%</li>
        <li><strong>Training Style:</strong> Classical multimodal fusion</li>
    </ul>
</div>
""",
                unsafe_allow_html=True,
            )

else:
    st.markdown(
        '<p class="section-header">Theory & Applications</p>',
        unsafe_allow_html=True,
    )
    tab1, tab2, tab3 = st.tabs(["📖 Theory", "💻 Applications", "🧪 Viva Notes"])

    with tab1:
        st.markdown(
            """
### What Is Happening Internally?

1. Each modality gets its own feature matrix.
2. A simple classifier learns from that modality alone.
3. Their probability outputs are fused with learned weights.
4. The weights are optimized so stronger modalities get higher influence.

#### Why attention weights help

- They show which modality is actually useful.
- They keep the system explainable.
- They let us compare fusion against single-modality performance.

#### Why this is not deep learning

- We rely on engineered features rather than end-to-end representation learning.
- The per-modality models are classical linear classifiers.
- The fusion step is transparent and easy to explain in a viva or report.
"""
        )

    with tab2:
        st.markdown(
            """
### Real-World Applications

#### Healthcare
- Combine scans, lab values, notes, and wearable streams

#### Autonomous Systems
- Fuse cameras, microphones, radar, and telemetry

#### Media & Social Platforms
- Blend visuals, captions, audio, and engagement features

#### Security
- Merge face, voice, access logs, and environmental sensors
"""
        )

    with tab3:
        st.markdown(
            """
### Useful Viva Explanation

> Initially the model showed near-perfect accuracy because the generated data was too clean and the class distributions were highly separable.
> We improved realism by reducing separation, injecting feature noise, and introducing controlled label noise.

### What to mention if asked about the new modality

- The `Video/Social` modality simulates temporal or engagement context.
- For MNIST and tabular datasets it is engineered from derived temporal-style statistics.
- For the social media dataset it acts like short-video engagement cues such as trend strength and audience retention.

### Expected outcomes

- Accuracy should typically stay below unrealistic 100% levels
- Fusion should still outperform many single modalities
- The learned weights should shift based on dataset characteristics
"""
        )

