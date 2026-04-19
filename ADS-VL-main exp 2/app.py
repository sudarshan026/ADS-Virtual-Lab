##############################################################################
# DATA VISUALIZATION VIRTUAL LAB — Applied Data Science 2026
# Complete Streamlit Application (Single File)
##############################################################################

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings

warnings.filterwarnings("ignore")

# sklearn
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Visualization Virtual Lab | ADS 2026",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)


# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""

""", unsafe_allow_html=True)


# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────
def render_header(title: str, subtitle: str = ""):
    """Render a styled header bar."""
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="header-bar"><h1>{title}</h1>{sub_html}</div>', unsafe_allow_html=True)


def render_footer():
    """Render the standard footer."""
    st.markdown('<div class="footer">Data Visualization Virtual Lab | Applied Data Science | 2026</div>', unsafe_allow_html=True)


def render_step_indicator(steps: list, current: int):
    """Render a simple step indicator bar."""
    cols = st.columns(len(steps))
    for i, (col, step_name) in enumerate(zip(cols, steps)):
        if i < current:
            col.markdown(f"<div style='text-align:center;padding:8px;background:#2ecc71;border-radius:8px;color:#fff;font-weight:600;font-size:0.82rem;'>✅ {step_name}</div>", unsafe_allow_html=True)
        elif i == current:
            col.markdown(f"<div style='text-align:center;padding:8px;background:#3498db;border-radius:8px;color:#fff;font-weight:600;font-size:0.82rem;'>🔵 {step_name}</div>", unsafe_allow_html=True)
        else:
            col.markdown(f"<div style='text-align:center;padding:8px;background:#34495e;border-radius:8px;color:#aaa;font-size:0.82rem;'>⬜ {step_name}</div>", unsafe_allow_html=True)


@st.cache_data
def load_default_dataset(name: str) -> pd.DataFrame:
    """Load a default sklearn dataset and return as DataFrame with named target."""
    if name == "Iris":
        data = load_iris()
    elif name == "Wine":
        data = load_wine()
    elif name == "Breast Cancer":
        data = load_breast_cancer()
    else:
        data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = pd.Categorical([data.target_names[t] for t in data.target])
    return df


def get_dataset_info():
    """Return dataset info string for the sidebar."""
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        name = st.session_state.get("dataset_name", "Custom")
        target = st.session_state.get("target_col", "N/A")
        n_classes = df[target].nunique() if target in df.columns else "?"
        return f"**{name}** | {df.shape[0]}×{df.shape[1]} | {n_classes} classes"
    return None


# ─── SIDEBAR NAVIGATION ────────────────────────────────────────────────────
st.sidebar.title("🧭 Navigation")
PAGES = [
    "🏠 Homepage (Theory)",
    "📊 Dataset Explorer & EDA",
    "🤖 ML Classification Lab",
    "📈 Model Results & Evaluation",
    "🌍 Carbon Footprint Tracker",
    "✅ Quiz",
    "📚 References",
]
page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")

# Show dataset info in sidebar
ds_info = get_dataset_info()
if ds_info:
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 📁 Current Dataset")
    st.sidebar.info(ds_info)


##############################################################################
# PAGE 1 — HOMEPAGE (THEORY)
##############################################################################
if page == PAGES[0]:
    render_header("Welcome to the Data Visualization Virtual Lab 📊",
                  "Applied Data Science — Hands-on Experimentation & Learning")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### About This Virtual Lab
        This virtual lab provides **production-quality, interactive tools** for exploring
        data visualization, machine learning classification, and model evaluation concepts.
        Every visualization is rendered with **Plotly** for full interactivity — hover,
        zoom, pan, and export any chart.

        ### Why Data Visualization Matters
        Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

        **Key Principles of Data Visualization:**
        - **Clarity:** Ensure the primary message is communicated clearly and efficiently.
        - **Accuracy:** Prevent misrepresenting the underlying data context or scale.
        - **Efficiency:** Allow viewers to grasp key insights quickly.
        - **Aesthetics:** Use thoughtful design to engage the viewer and highlight importance.

        **Core Use Cases:**
        - 🔍 **Identify Trends** — Spot patterns in complex, high-dimensional datasets
        - 📣 **Communicate Insights** — Convey findings to stakeholders with clarity
        - 🐛 **Diagnose Model Issues** — Visualize residuals, loss curves, and decision boundaries
        - 🎛️ **Interactive Exploration** — Drill into specifics with dynamic dashboards

        ### What You'll Learn
        | Module | Topics Covered |
        |--------|---------------|
        | **Dataset Explorer & EDA** | Distributions, correlations, 3D plots, dimensionality reduction |
        | **ML Classification Lab** | Train 5 classifiers, tune hyperparameters, cross-validation |
        | **Model Results & Evaluation** | ROC curves, confusion matrices, radar charts, learning curves |
        | **Carbon Footprint Tracker** | Estimate CO₂ emissions, eco-efficiency, lifecycle analysis |

        ---
        > **Get started** by navigating to **📊 Dataset Explorer & EDA** in the sidebar!
        """)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=600",
                 caption="Data-driven decision making", use_container_width=True)
        st.markdown("""
        #### 🧪 Lab Features
        - 6 mandatory chart types
        - 5 ML classifiers
        - Real-time carbon tracking
        - Interactive quiz module
        """)

    render_footer()


##############################################################################
# PAGE 2 — DATASET EXPLORER & EDA
##############################################################################
elif page == PAGES[1]:
    render_header("Dataset Explorer & EDA 📊", "Load, explore, and visualize your data")

    # ── Dataset Selection ───────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 📂 Dataset Source")
    data_source = st.sidebar.radio("Choose source:", ["Use Default Dataset", "Upload My Own CSV"], key="ds_source")

    df = None
    target_col = None
    dataset_name = "Custom"

    if data_source == "Use Default Dataset":
        dataset_name = st.sidebar.selectbox("Select Dataset", ["Iris", "Wine", "Breast Cancer"])
        df = load_default_dataset(dataset_name)
        target_col = "target"
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="eda_upload")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            dataset_name = uploaded.name.replace(".csv", "")
            target_col = st.sidebar.selectbox("Select Target Column", df.columns.tolist())
        else:
            st.info("⬆️ Upload a CSV file from the sidebar to get started, or switch to a default dataset.")
            render_footer()
            st.stop()

    # Store in session state
    st.session_state.df = df
    st.session_state.target_col = target_col
    st.session_state.dataset_name = dataset_name

    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns.tolist() if c != target_col]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Dataset Overview ────────────────────────────────────────────────────
    st.markdown("### 📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric Features", len(numeric_cols))
    c4.metric("Classes", df[target_col].nunique())

    with st.expander("🔎 Data Preview & Info", expanded=False):
        st.dataframe(df.head(15), width="stretch")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Data Types**")
            dtype_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values})
            st.dataframe(dtype_df, width="stretch", hide_index=True)
        with col_b:
            st.markdown("**Missing Values**")
            missing = df.isnull().sum().reset_index()
            missing.columns = ["Column", "Missing Count"]
            missing["% Missing"] = (missing["Missing Count"] / len(df) * 100).round(2)
            st.dataframe(missing.style.background_gradient(subset=["Missing Count"], cmap="YlOrRd"),
                         width="stretch", hide_index=True)

    # Class distribution
    class_dist = df[target_col].value_counts().reset_index()
    class_dist.columns = ["Class", "Count"]
    fig_class = px.bar(class_dist, x="Class", y="Count", color="Class",
                       title="Class Distribution", template="plotly_white",
                       color_discrete_sequence=px.colors.qualitative.Set2)
    fig_class.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_class, width="stretch")

    # ── EDA Tabs ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Feature Overview",
        "🔗 Correlation & Relationships",
        "🧊 3D & Multidimensional",
        "🔬 Dimensionality Reduction",
        "🌲 Feature Importance"
    ])

    # ──────────────────────── TAB 1: Feature Overview ───────────────────────
    with tab1:
        st.markdown("### Descriptive Statistics")
        st.dataframe(df.describe().T.style.format("{:.3f}").background_gradient(cmap="Blues"),
                     width="stretch")

        # Sparkline histograms
        st.markdown("### Feature Sparklines")
        if numeric_cols:
            spark_cols = st.columns(min(4, len(numeric_cols)))
            for i, col_name in enumerate(numeric_cols[:12]):
                with spark_cols[i % len(spark_cols)]:
                    fig_spark = px.histogram(df, x=col_name, nbins=25, height=180,
                                            template="plotly_white",
                                            color_discrete_sequence=["#3498db"])
                    fig_spark.update_layout(
                        showlegend=False, margin=dict(l=5, r=5, t=30, b=5),
                        title=dict(text=col_name, font=dict(size=11)),
                        xaxis=dict(visible=False), yaxis=dict(visible=False)
                    )
                    st.plotly_chart(fig_spark, width="stretch")

        # Full distribution
        st.markdown("### Feature Distribution (Histogram + KDE Overlay)")
        selected_features = st.multiselect("Select features to plot:", numeric_cols, default=numeric_cols[:2], key="feat_dist")
        for feat in selected_features:
            fig_dist = go.Figure()
            for cls in df[target_col].unique():
                subset = df[df[target_col] == cls][feat].dropna()
                fig_dist.add_trace(go.Histogram(
                    x=subset, name=str(cls), opacity=0.6, nbinsx=30
                ))
            fig_dist.update_layout(
                barmode="overlay", template="plotly_white",
                title=f"Distribution of {feat} by Class",
                xaxis_title=feat, yaxis_title="Count", height=380,
            )
            st.plotly_chart(fig_dist, width="stretch")

    # ──────────────────── TAB 2: Correlation & Relationships ────────────────
    with tab2:
        # [MANDATORY] HEATMAP — Correlation Heatmap
        st.markdown("### 🔥 Correlation Heatmap")
        corr_method = "spearman" if st.checkbox("Use Spearman (default: Pearson)", key="corr_toggle") else "pearson"
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr(method=corr_method)
            fig_heat = px.imshow(
                corr_matrix, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r",
                title=f"{corr_method.title()} Correlation Heatmap",
                template="plotly_white",
            )
            fig_heat.update_layout(height=550)
            st.plotly_chart(fig_heat, width="stretch")
        else:
            st.warning("Need at least 2 numeric columns for a heatmap.")

        st.markdown("---")

        # [MANDATORY] BUBBLE CHART
        st.markdown("### 🫧 Bubble Chart")
        bc1, bc2, bc3, bc4 = st.columns(4)
        bx = bc1.selectbox("X-axis", numeric_cols, index=0, key="bub_x")
        by = bc2.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="bub_y")
        bsize = bc3.selectbox("Bubble Size", numeric_cols, index=min(2, len(numeric_cols) - 1), key="bub_size")
        bcolor = bc4.selectbox("Color", [target_col] + numeric_cols, index=0, key="bub_color")
        try:
            # Deduplicate columns to avoid narwhals DuplicateError
            bubble_cols = list(dict.fromkeys([bx, by, bsize, bcolor, target_col]))
            bubble_df = df[bubble_cols].dropna(subset=[bx, by, bsize])
            fig_bubble = px.scatter(
                bubble_df, x=bx, y=by, size=bsize, color=bcolor,
                hover_data=[target_col],
                title=f"Bubble Chart: {by} vs {bx} (size={bsize}, color={bcolor})",
                template="plotly_white", size_max=45,
            )
            fig_bubble.update_layout(height=500)
            st.plotly_chart(fig_bubble, width="stretch")
        except Exception as e:
            st.error(f"Could not render bubble chart: {e}")

        st.markdown("---")

        # [MANDATORY] BOX PLOT — by target class
        st.markdown("### 📦 Box Plot by Class")
        box_feat = st.selectbox("Select Numeric Feature", numeric_cols, key="box_feat_eda")
        try:
            box_df = df[[target_col, box_feat]].dropna()
            fig_box = px.box(
                box_df, x=target_col, y=box_feat, color=target_col, points="all",
                title=f"Box Plot of {box_feat} by {target_col}",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig_box.update_layout(height=450)
            st.plotly_chart(fig_box, width="stretch")
        except Exception as e:
            st.error(f"Could not render box plot: {e}")

        st.markdown("---")

        # Scatter Matrix
        st.markdown("### 🔢 Scatter Matrix")
        sm_features = st.multiselect("Select up to 5 features:", numeric_cols,
                                     default=numeric_cols[:min(4, len(numeric_cols))], key="scatter_mat")
        if len(sm_features) >= 2:
            try:
                fig_sm = px.scatter_matrix(
                    df.dropna(subset=sm_features[:5]), dimensions=sm_features[:5], color=target_col,
                    title="Scatter Matrix", template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_sm.update_layout(height=700)
                st.plotly_chart(fig_sm, width="stretch")
            except Exception as e:
                st.error(f"Could not render scatter matrix: {e}")

        st.markdown("---")

        # Violin + Box plot
        st.markdown("### 🎻 Violin Plot (Feature vs Target)")
        viol_feat = st.selectbox("Feature for Violin", numeric_cols, key="violin_feat")
        try:
            viol_df = df[[target_col, viol_feat]].dropna()
            fig_violin = px.violin(viol_df, x=target_col, y=viol_feat, color=target_col,
                                   box=True, points="all",
                                   title=f"Violin + Box Plot: {viol_feat} by {target_col}",
                                   template="plotly_white")
            fig_violin.update_layout(height=450)
            st.plotly_chart(fig_violin, width="stretch")
        except Exception as e:
            st.error(f"Could not render violin plot: {e}")

    # ──────────────── TAB 3: 3D & Multidimensional ──────────────────────────
    with tab3:
        # [MANDATORY] PARALLEL COORDINATES
        st.markdown("### 🌈 Parallel Coordinates Plot")
        pc_features = st.multiselect("Select features for axes:", numeric_cols,
                                     default=numeric_cols[:min(6, len(numeric_cols))], key="pc_feat")
        if len(pc_features) >= 2:
            # encode target numerically
            target_codes = pd.Categorical(df[target_col]).codes
            pc_df = df[pc_features].copy()
            pc_df["_target_code"] = target_codes
            fig_pc = px.parallel_coordinates(
                pc_df, dimensions=pc_features, color="_target_code",
                color_continuous_scale="Viridis",
                title="Parallel Coordinates (colored by class)",
                template="plotly_white",
            )
            fig_pc.update_layout(height=500)
            st.plotly_chart(fig_pc, width="stretch")

        st.markdown("---")

        # 3D Scatter Plot
        st.markdown("### 🧊 3D Scatter Plot")
        if len(numeric_cols) >= 3:
            sc1, sc2, sc3 = st.columns(3)
            x3d = sc1.selectbox("X", numeric_cols, index=0, key="3dx")
            y3d = sc2.selectbox("Y", numeric_cols, index=1, key="3dy")
            z3d = sc3.selectbox("Z", numeric_cols, index=2, key="3dz")
            fig_3d = px.scatter_3d(df, x=x3d, y=y3d, z=z3d, color=target_col,
                                   title=f"3D Scatter: {x3d} × {y3d} × {z3d}",
                                   template="plotly_white",
                                   color_discrete_sequence=px.colors.qualitative.Bold)
            fig_3d.update_layout(height=600)
            st.plotly_chart(fig_3d, width="stretch")
        else:
            st.warning("Need at least 3 numeric columns for 3D scatter.")

        st.markdown("---")

        # 3D Surface (PCA + KNN meshgrid)
        st.markdown("### 🗺️ 3D Decision Surface (PCA + KNN)")
        if len(numeric_cols) >= 2:
            with st.spinner("Computing PCA + KNN decision surface..."):
                try:
                    surf_df = df.dropna(subset=numeric_cols + [target_col])
                    if surf_df.empty:
                        raise ValueError("No complete samples available after dropping missing values.")
                    from sklearn.neighbors import KNeighborsClassifier as KNC_Surface
                    X_surf = surf_df[numeric_cols].values
                    y_surf = pd.Categorical(surf_df[target_col]).codes
                    scaler_surf = StandardScaler()
                    X_scaled = scaler_surf.fit_transform(X_surf)
                    pca_surf = PCA(n_components=2)
                    X_pca = pca_surf.fit_transform(X_scaled)

                    knn_surf = KNC_Surface(n_neighbors=5)
                    knn_surf.fit(X_pca, y_surf)

                    h = 0.3
                    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
                    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                         np.arange(y_min, y_max, h))
                    Z = knn_surf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

                    fig_surface = go.Figure(data=[
                        go.Surface(x=xx, y=yy, z=Z.astype(float), opacity=0.7,
                                   colorscale="Viridis", showscale=False),
                        go.Scatter3d(x=X_pca[:, 0], y=X_pca[:, 1], z=y_surf.astype(float),
                                     mode="markers", marker=dict(size=4, color=y_surf, colorscale="Viridis"),
                                     name="Data Points")
                    ])
                    fig_surface.update_layout(
                        title="KNN Decision Surface (PCA 2D → 3D)",
                        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="Class"),
                        template="plotly_white", height=550,
                    )
                    st.plotly_chart(fig_surface, width="stretch")
                except Exception as e:
                    st.error(f"Could not compute 3D decision surface: {e}")

    # ──────────────── TAB 4: Dimensionality Reduction ───────────────────────
    with tab4:
        dr_df = df.dropna(subset=numeric_cols + [target_col])
        if dr_df.empty:
            st.warning("⚠️ No complete samples available after dropping missing values.")
            st.stop()
        X_dr = dr_df[numeric_cols].values
        y_dr = dr_df[target_col].values
        y_dr_codes = pd.Categorical(dr_df[target_col]).codes

        scaler_dr = StandardScaler()
        X_dr_scaled = scaler_dr.fit_transform(X_dr)

        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric features for dimensionality reduction.")
            st.stop()

        st.markdown("### PCA 2D & 3D")
        max_comp = min(len(numeric_cols), 10)
        n_components_slider = st.slider("Number of PCA components for variance chart", 2, max(2, max_comp), min(max_comp, 5), key="pca_comp")

        with st.spinner("Computing PCA..."):
            pca = PCA(n_components=min(n_components_slider, X_dr_scaled.shape[1]))
            X_pca_full = pca.fit_transform(X_dr_scaled)

            # Explained variance
            fig_var = px.bar(
                x=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                y=pca.explained_variance_ratio_,
                title="PCA Explained Variance Ratio",
                template="plotly_white", labels={"x": "Component", "y": "Variance Ratio"},
                color_discrete_sequence=["#3498db"],
            )
            fig_var.update_layout(height=350)
            st.plotly_chart(fig_var, width="stretch")

        # PCA 2D
        pca2 = PCA(n_components=2).fit_transform(X_dr_scaled)
        pca_df2 = pd.DataFrame({"PC1": pca2[:, 0], "PC2": pca2[:, 1], "Class": y_dr})
        fig_pca2d = px.scatter(pca_df2, x="PC1", y="PC2", color="Class",
                               title="PCA 2D Projection",
                               template="plotly_white",
                               color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pca2d.update_layout(height=420)
        st.plotly_chart(fig_pca2d, width="stretch")

        # PCA 3D
        if X_dr_scaled.shape[1] >= 3:
            pca3 = PCA(n_components=3).fit_transform(X_dr_scaled)
            pca_df3 = pd.DataFrame({"PC1": pca3[:, 0], "PC2": pca3[:, 1], "PC3": pca3[:, 2], "Class": y_dr})
            fig_pca3d = px.scatter_3d(pca_df3, x="PC1", y="PC2", z="PC3", color="Class",
                                      title="PCA 3D Projection",
                                      template="plotly_white",
                                      color_discrete_sequence=px.colors.qualitative.Bold)
            fig_pca3d.update_layout(height=520)
            st.plotly_chart(fig_pca3d, width="stretch")

        st.markdown("---")

        # t-SNE
        st.markdown("### t-SNE 2D Projection")
        perplexity = st.slider("Perplexity", 5, 50, 30, key="tsne_perp")
        with st.spinner("Computing t-SNE (may take a moment)..."):
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=800)
            X_tsne = tsne.fit_transform(X_dr_scaled)
            tsne_df = pd.DataFrame({"Dim1": X_tsne[:, 0], "Dim2": X_tsne[:, 1], "Class": y_dr})
            fig_tsne = px.scatter(tsne_df, x="Dim1", y="Dim2", color="Class",
                                  title=f"t-SNE 2D (perplexity={perplexity})",
                                  template="plotly_white",
                                  color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_tsne.update_layout(height=450)
            st.plotly_chart(fig_tsne, width="stretch")

        st.markdown("---")

        # UMAP
        st.markdown("### UMAP 2D Projection")
        if HAS_UMAP:
            n_neighbors_umap = st.slider("n_neighbors", 5, 50, 15, key="umap_nn")
            with st.spinner("Computing UMAP..."):
                reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors_umap, random_state=42)
                X_umap = reducer.fit_transform(X_dr_scaled)
                umap_df = pd.DataFrame({"Dim1": X_umap[:, 0], "Dim2": X_umap[:, 1], "Class": y_dr})
                fig_umap = px.scatter(umap_df, x="Dim1", y="Dim2", color="Class",
                                      title=f"UMAP 2D (n_neighbors={n_neighbors_umap})",
                                      template="plotly_white",
                                      color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_umap.update_layout(height=450)
                st.plotly_chart(fig_umap, width="stretch")
        else:
            st.warning("⚠️ UMAP is not installed. Install via `pip install umap-learn` to enable.")

    # ──────────────── TAB 5: Feature Importance ─────────────────────────────
    with tab5:
        st.markdown("### 🌲 Random Forest Feature Importance")
        with st.spinner("Training a quick Random Forest for feature importances..."):
            try:
                fi_df_clean = df.dropna(subset=numeric_cols + [target_col])
                if fi_df_clean.empty:
                    raise ValueError("No complete samples available after dropping missing values.")
                X_fi = fi_df_clean[numeric_cols].values
                y_fi = pd.Categorical(fi_df_clean[target_col]).codes
                rf_fi = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf_fi.fit(X_fi, y_fi)
                importances = rf_fi.feature_importances_
                fi_df = pd.DataFrame({"Feature": numeric_cols, "Importance": importances}).sort_values("Importance", ascending=True)

                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                title="Random Forest Feature Importances",
                                template="plotly_white",
                                color="Importance", color_continuous_scale="Blues")
                fig_fi.update_layout(height=max(350, len(numeric_cols) * 30))
                st.plotly_chart(fig_fi, width="stretch")
            except Exception as e:
                st.error(f"Could not compute Random Forest importances: {e}")

        st.markdown("### 📐 Mutual Information Scores")
        with st.spinner("Computing mutual information scores..."):
            try:
                if 'X_fi' not in locals():
                    fi_df_clean = df.dropna(subset=numeric_cols + [target_col])
                    X_fi = fi_df_clean[numeric_cols].values
                    y_fi = pd.Categorical(fi_df_clean[target_col]).codes
                mi_scores = mutual_info_classif(X_fi, y_fi, random_state=42)
                mi_df = pd.DataFrame({"Feature": numeric_cols, "MI Score": mi_scores}).sort_values("MI Score", ascending=True)

                fig_mi = px.bar(mi_df, x="MI Score", y="Feature", orientation="h",
                                title="Mutual Information Scores",
                                template="plotly_white",
                                color="MI Score", color_continuous_scale="Greens")
                fig_mi.update_layout(height=max(350, len(numeric_cols) * 30))
                st.plotly_chart(fig_mi, width="stretch")
            except Exception as e:
                st.error(f"Could not compute Mutual Information Scores: {e}")

    render_footer()


##############################################################################
# PAGE 3 — ML CLASSIFICATION LAB
##############################################################################
elif page == PAGES[2]:
    render_header("ML Classification Lab 🤖", "Train, compare, and tune classifiers")

    # Check dataset
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please go to **📊 Dataset Explorer & EDA** first to load a dataset.")
        if st.button("Go to Dataset Explorer"):
            st.session_state["nav"] = PAGES[1]
        render_footer()
        st.stop()

    df = st.session_state.df
    target_col = st.session_state.target_col
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns.tolist() if c != target_col]

    # Step indicator
    render_step_indicator(["⚙️ Configure", "🧠 Select Models", "🚀 Train", "🔧 Tune"], 0)
    st.markdown("")

    # ── Step 1: Configuration ──────────────────────────────────────────────
    st.markdown("### ⚙️ Step 1 — Configuration")
    with st.expander("Training Configuration", expanded=True):
        cfg1, cfg2, cfg3, cfg4 = st.columns(4)
        test_size = cfg1.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key="test_size")
        random_state = cfg2.number_input("Random State", 0, 9999, 42, key="rand_state")
        cv_folds = cfg3.slider("CV Folds", 2, 10, 5, key="cv_folds")
        apply_scaler = cfg4.checkbox("Apply StandardScaler", value=True, key="apply_scaler")

        # Target column selection
        all_columns = df.columns.tolist()
        default_target_idx = all_columns.index(target_col) if target_col in all_columns else 0
        target_col = cfg4.selectbox("Target Column", all_columns, index=default_target_idx, key="ml_target_col")
        st.session_state.target_col = target_col

        # Recompute numeric features excluding the selected target
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns.tolist() if c != target_col]
        if not numeric_cols:
            st.error("⚠️ No numeric features available (all columns are the target or non-numeric).")
            st.stop()

        # Feature selection
        st.markdown("**Select Features for Classification:**")
        selected_features = st.multiselect(
            "Choose which numeric features to use:",
            numeric_cols, default=numeric_cols, key="selected_features"
        )
        if not selected_features:
            st.error("⚠️ Please select at least one feature.")
            st.stop()

        st.markdown(f"""
        <div class="info-box">
        <b>Configuration Summary:</b> Target=<code>{target_col}</code> | Test={test_size*100:.0f}% | Seed={random_state} | CV={cv_folds}-fold | Scaler={'✅ On' if apply_scaler else '❌ Off'} | Features={len(selected_features)}/{len(numeric_cols)}
        </div>
        """, unsafe_allow_html=True)

    # ── Step 2: Model Selection ────────────────────────────────────────────
    st.markdown("### 🧠 Step 2 — Model Selection")

    m1, m2, m3, m4, m5 = st.columns(5)
    use_lr = m1.checkbox("Logistic Regression", True, key="use_lr")
    use_knn = m2.checkbox("K-Nearest Neighbors", True, key="use_knn")
    use_svm = m3.checkbox("Support Vector Machine", True, key="use_svm")
    use_rf = m4.checkbox("Random Forest", True, key="use_rf")
    use_xgb = m5.checkbox("XGBoost", True, key="use_xgb")

    # Hyperparams
    hp1, hp2, hp3 = st.columns(3)
    knn_k = hp1.slider("KNN — K neighbors", 1, 20, 5, key="knn_k") if use_knn else 5
    svm_kernel = hp2.selectbox("SVM — Kernel", ["linear", "rbf", "poly"], index=1, key="svm_k") if use_svm else "rbf"
    rf_n = hp3.slider("RF — n_estimators", 10, 200, 100, 10, key="rf_n") if use_rf else 100

    # ── Step 3: Training ───────────────────────────────────────────────────
    st.markdown("### 🚀 Step 3 — Train All Selected Models")

    if st.button("🚀 Train All Selected Models", type="primary", width="stretch"):
        # Prepare data — use only user-selected features and drop rows with missing values
        clean_df = df.dropna(subset=selected_features + [target_col])
        
        # Filter out rare classes (< 2 samples) since they break stratified splitting and models like XGBoost
        counts = clean_df[target_col].value_counts()
        valid_classes = counts[counts >= 2].index
        clean_df = clean_df[clean_df[target_col].isin(valid_classes)]

        if clean_df.empty or len(valid_classes) < 2:
            st.error("⚠️ Not enough valid samples or classes. Require at least 2 distinct classes with >= 2 samples each. Please adjust your dataset or features.")
            st.stop()
            
        X = clean_df[selected_features].values
        st.session_state.numeric_cols = selected_features
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(clean_df[target_col].astype(str))
        class_names = le.classes_.tolist()

        # Use stratified split (guaranteed to work since all classes have >= 2 samples)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state), stratify=y
            )
        except Exception:
            # Absolute ultimate fallback just in case
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state)
            )

        # To be completely safe with XGBoost, force re-encoding of y_train just in case 
        # the fallback occurred and skipped a class
        train_le = LabelEncoder()
        y_train = train_le.fit_transform(y_train)
        # map y_test to same domain, replacing unseen classes with 0 safely to avoid crash
        y_test = np.array([np.where(train_le.classes_ == val)[0][0] if val in train_le.classes_ else 0 for val in y_test])
        
        # update class names to only what train_le saw
        class_names = [class_names[c] for c in train_le.classes_]

        # Ensure we have at least 2 classes in the training set
        if len(np.unique(y_train)) < 2:
            st.error("⚠️ The resulting training set contains fewer than 2 classes. Supervised classification requires at least 2 distinct classes. Please use a larger or more balanced dataset.")
            st.stop()

        if apply_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            st.session_state.scaler = scaler

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.class_names = class_names
        st.session_state.numeric_cols = numeric_cols

        # Build model dict
        models = {}
        if use_lr:
            models["Logistic Regression"] = LogisticRegression(max_iter=2000, random_state=int(random_state))
        if use_knn:
            models["K-Nearest Neighbors"] = KNeighborsClassifier(n_neighbors=knn_k)
        if use_svm:
            models["SVM"] = SVC(kernel=svm_kernel, probability=True, random_state=int(random_state))
        if use_rf:
            models["Random Forest"] = RandomForestClassifier(n_estimators=rf_n, random_state=int(random_state), n_jobs=-1)
        if use_xgb:
            if HAS_XGBOOST:
                xgb_eval_metric = "logloss" if len(np.unique(y_train)) == 2 else "mlogloss"
                models["XGBoost"] = xgb.XGBClassifier(
                    n_estimators=rf_n, random_state=int(random_state),
                    use_label_encoder=False, eval_metric=xgb_eval_metric,
                    verbosity=0,
                )
            else:
                st.warning("⚠️ XGBoost is not installed. Skipping.")

        if not models:
            st.error("Select at least one model!")
            st.stop()

        results = {}
        progress = st.progress(0, text="Training models...")
        n_models = len(models)

        for idx, (name, model) in enumerate(models.items()):
            with st.spinner(f"Training {name}..."):
                t0 = time.time()

                # XGBoost with eval_result
                if name == "XGBoost" and HAS_XGBOOST:
                    eval_set = [(X_train, y_train), (X_test, y_test)]
                    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                    xgb_evals = model.evals_result()
                    st.session_state.xgb_evals = xgb_evals
                else:
                    model.fit(X_train, y_train)

                train_time = time.time() - t0

                # Predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                train_acc = accuracy_score(y_train, model.predict(X_train))
                test_acc = accuracy_score(y_test, y_pred)

                # CV scores (per-fold) — fall back to regular KFold if stratified fails
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
                except ValueError:
                    from sklearn.model_selection import KFold
                    effective_folds = min(cv_folds, len(y_train))
                    kf = KFold(n_splits=effective_folds, shuffle=True, random_state=int(random_state))
                    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")

                # Classification report dict — only use labels present in test set
                present_labels = sorted(set(np.concatenate([np.unique(y_test), np.unique(y_pred)])))
                present_names = [str(class_names[i]) if i < len(class_names) else str(i) for i in present_labels]
                cr = classification_report(y_test, y_pred, labels=present_labels, target_names=present_names, output_dict=True)

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=present_labels)

                # ROC data — use only classes the model was trained on
                trained_classes = model.classes_ if hasattr(model, "classes_") else np.unique(y_train)
                n_trained = len(trained_classes)
                roc_data = {}
                if y_prob is not None:
                    if n_trained == 2:
                        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                        roc_auc = auc(fpr, tpr)
                        roc_data["binary"] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
                    else:
                        y_test_bin = label_binarize(y_test, classes=list(trained_classes))
                        # y_test_bin may have fewer columns if some test labels weren't in training
                        n_cols = min(y_test_bin.shape[1], y_prob.shape[1])
                        for i_cl in range(n_cols):
                            try:
                                fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i_cl], y_prob[:, i_cl])
                                roc_auc_i = auc(fpr_i, tpr_i)
                                roc_data[f"class_{i_cl}"] = {"fpr": fpr_i, "tpr": tpr_i, "auc": roc_auc_i}
                            except Exception:
                                pass
                        # micro average
                        try:
                            micro_bin = y_test_bin[:, :n_cols]
                            micro_prob = y_prob[:, :n_cols]
                            fpr_micro, tpr_micro, _ = roc_curve(micro_bin.ravel(), micro_prob.ravel())
                            roc_auc_micro = auc(fpr_micro, tpr_micro)
                            roc_data["micro"] = {"fpr": fpr_micro, "tpr": tpr_micro, "auc": roc_auc_micro}
                        except Exception:
                            pass

                results[name] = {
                    "model": model,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "cv_scores": cv_scores,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "train_time": train_time,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                    "classification_report": cr,
                    "confusion_matrix": cm,
                    "roc_data": roc_data,
                }

            progress.progress((idx + 1) / n_models, text=f"Trained {name} ✅")

        st.session_state.trained_models = results
        progress.empty()

        st.success(f"✅ Successfully trained {len(results)} models!")

        # Summary table
        summary_rows = []
        for name, res in results.items():
            summary_rows.append({
                "Model": name,
                "Train Acc": f"{res['train_acc']:.4f}",
                "Test Acc": f"{res['test_acc']:.4f}",
                "CV Mean ± Std": f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}",
                "Training Time (s)": f"{res['train_time']:.3f}",
            })
        summary_df = pd.DataFrame(summary_rows)

        # Highlight best
        best_idx = summary_df["Test Acc"].astype(float).idxmax()

        def highlight_best(row):
            if row.name == best_idx:
                return ["background-color: #0d3320; color: #2ecc71; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(summary_df.style.apply(highlight_best, axis=1), width="stretch", hide_index=True)

    # ── Step 4: Hyperparameter Tuning ──────────────────────────────────────
    if "trained_models" in st.session_state and st.session_state.trained_models:
        st.markdown("### 🔧 Step 4 — Hyperparameter Tuning (Optional)")
        with st.expander("GridSearchCV on Best Model"):
            results = st.session_state.trained_models
            best_name = max(results, key=lambda k: results[k]["test_acc"])
            st.info(f"Best model by test accuracy: **{best_name}**")

            param_grids = {
                "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
                "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]},
                "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
                "Random Forest": {"n_estimators": [50, 100, 150], "max_depth": [None, 5, 10]},
                "XGBoost": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.3]},
            }

            if st.button("🔍 Run GridSearchCV", type="secondary"):
                grid = param_grids.get(best_name, {})
                if grid:
                    with st.spinner(f"Running GridSearchCV on {best_name}..."):
                        base_model = results[best_name]["model"]
                        gs = GridSearchCV(base_model, grid, cv=cv_folds, scoring="accuracy", n_jobs=-1)
                        gs.fit(st.session_state.X_train, st.session_state.y_train)
                        st.success(f"**Best Params:** {gs.best_params_}")
                        st.metric("Best CV Score", f"{gs.best_score_:.4f}")
                else:
                    st.warning("No grid defined for this model type.")

    render_footer()


##############################################################################
# PAGE 4 — MODEL RESULTS & EVALUATION
##############################################################################
elif page == PAGES[3]:
    render_header("Model Results & Evaluation 📈", "Deep-dive into trained models")

    if "trained_models" not in st.session_state or not st.session_state.trained_models:
        st.warning("⚠️ No trained models found. Please go to **🤖 ML Classification Lab** to train models first.")
        render_footer()
        st.stop()

    results = st.session_state.trained_models
    model_names = list(results.keys())
    class_names = st.session_state.class_names
    y_test = st.session_state.y_test
    X_test = st.session_state.X_test
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    n_classes = len(class_names)

    # ── Section 1: Model Selector ──────────────────────────────────────────
    st.markdown("### 🔎 Section 1 — Model Selector")
    selected_model = st.selectbox("Select model for detailed inspection:", model_names, key="model_select")
    sel_res = results[selected_model]

    # ── Section 2: Classification Metrics ──────────────────────────────────
    st.markdown("### 📊 Section 2 — Classification Metrics")

    col_roc, col_cm = st.columns(2)

    with col_roc:
        # [MANDATORY] ROC CURVES
        st.markdown("#### ROC Curves")
        fig_roc = go.Figure()

        # Diagonal baseline
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="gray", width=1),
            name="Random Baseline", showlegend=True
        ))

        roc_data = sel_res["roc_data"]
        if "binary" in roc_data:
            rd = roc_data["binary"]
            fig_roc.add_trace(go.Scatter(
                x=rd["fpr"], y=rd["tpr"], mode="lines",
                name=f"{selected_model} (AUC = {rd['auc']:.3f})",
                line=dict(width=2)
            ))
        else:
            colors = px.colors.qualitative.Set2
            for key, rd in roc_data.items():
                if key == "micro":
                    fig_roc.add_trace(go.Scatter(
                        x=rd["fpr"], y=rd["tpr"], mode="lines",
                        name=f"Micro-Avg (AUC = {rd['auc']:.3f})",
                        line=dict(width=2, dash="dot", color="#e74c3c")
                    ))
                else:
                    cl_idx = int(key.split("_")[1])
                    cl_name = str(class_names[cl_idx]) if cl_idx < len(class_names) else key
                    fig_roc.add_trace(go.Scatter(
                        x=rd["fpr"], y=rd["tpr"], mode="lines",
                        name=f"{cl_name} (AUC = {rd['auc']:.3f})",
                        line=dict(width=2, color=colors[cl_idx % len(colors)])
                    ))

        fig_roc.update_layout(
            title="ROC Curve (One-vs-Rest)", template="plotly_white",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=450, legend=dict(font=dict(size=10))
        )
        st.plotly_chart(fig_roc, width="stretch")

    with col_cm:
        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        normalize_cm = st.checkbox("Normalize", key="norm_cm")
        cm = sel_res["confusion_matrix"]
        if normalize_cm:
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_display = np.nan_to_num(cm_display)
            fmt = ".2f"
        else:
            cm_display = cm
            fmt = "d"

        fig_cm = px.imshow(
            cm_display, text_auto=True if not normalize_cm else ".2f",
            x=[str(c) for c in class_names], y=[str(c) for c in class_names],
            color_continuous_scale="Blues",
            title=f"Confusion Matrix — {selected_model}",
            template="plotly_white",
            labels={"x": "Predicted", "y": "Actual"}
        )
        # Add annotations for normalized
        if normalize_cm:
            fig_cm.update_traces(text=[[f"{v:.2f}" for v in row] for row in cm_display], texttemplate="%{text}")
        fig_cm.update_layout(height=450)
        st.plotly_chart(fig_cm, width="stretch")

    # Classification Report
    st.markdown("#### Classification Report")
    cr = sel_res["classification_report"]
    cr_df = pd.DataFrame(cr).T
    st.dataframe(cr_df.style.format("{:.3f}", subset=cr_df.columns[:4] if len(cr_df.columns) >= 4 else cr_df.columns).background_gradient(cmap="Greens"),
                 width="stretch")

    # [MANDATORY] BOX PLOT — CV Scores (all models)
    st.markdown("#### Cross-Validation Score Distribution")
    fig_cv_box = go.Figure()
    colors_cv = px.colors.qualitative.Bold
    for i, (mname, mres) in enumerate(results.items()):
        fig_cv_box.add_trace(go.Box(
            y=mres["cv_scores"], name=mname,
            marker_color=colors_cv[i % len(colors_cv)],
            boxpoints="all", jitter=0.3, pointpos=-1.5,
        ))
    fig_cv_box.update_layout(
        title="Cross-Validation Fold Scores (All Models)",
        yaxis_title="Accuracy", template="plotly_white", height=420,
    )
    st.plotly_chart(fig_cv_box, width="stretch")

    # Precision-Recall Curve
    st.markdown("#### Precision-Recall Curve")
    if sel_res["y_prob"] is not None:
        fig_pr = go.Figure()
        y_prob = sel_res["y_prob"]
        if n_classes == 2:
            prec, rec, _ = precision_recall_curve(y_test, y_prob[:, 1])
            ap = average_precision_score(y_test, y_prob[:, 1])
            fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                         name=f"{selected_model} (AP = {ap:.3f})"))
        else:
            y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
            colors_pr = px.colors.qualitative.Set2
            for cl_i in range(n_classes):
                prec_i, rec_i, _ = precision_recall_curve(y_test_bin[:, cl_i], y_prob[:, cl_i])
                ap_i = average_precision_score(y_test_bin[:, cl_i], y_prob[:, cl_i])
                fig_pr.add_trace(go.Scatter(x=rec_i, y=prec_i, mode="lines",
                                              name=f"{class_names[cl_i]} (AP = {ap_i:.3f})",
                                              line=dict(color=colors_pr[cl_i % len(colors_pr)])))
        fig_pr.update_layout(
            title="Precision-Recall Curve", template="plotly_white",
            xaxis_title="Recall", yaxis_title="Precision", height=420,
        )
        st.plotly_chart(fig_pr, width="stretch")
    else:
        st.info("No probability data available for this model.")

    # ── Section 3: Training Dynamics ───────────────────────────────────────
    st.markdown("### 📈 Section 3 — Training Dynamics")

    # XGBoost eval curves
    if "xgb_evals" in st.session_state and selected_model == "XGBoost":
        st.markdown("#### XGBoost Training Loss Curve")
        xgb_evals = st.session_state.xgb_evals
        metric_key = list(xgb_evals["validation_0"].keys())[0]
        fig_xgb_loss = go.Figure()
        fig_xgb_loss.add_trace(go.Scatter(
            y=xgb_evals["validation_0"][metric_key], mode="lines", name="Train"
        ))
        fig_xgb_loss.add_trace(go.Scatter(
            y=xgb_evals["validation_1"][metric_key], mode="lines", name="Test"
        ))
        fig_xgb_loss.update_layout(
            title=f"XGBoost — {metric_key} over Iterations",
            xaxis_title="Iteration", yaxis_title=metric_key,
            template="plotly_white", height=380,
        )
        st.plotly_chart(fig_xgb_loss, width="stretch")

    # Learning Curve
    st.markdown("#### Learning Curve")
    with st.spinner("Computing learning curve..."):
        try:
            model_lc = sel_res["model"]
            train_sizes, train_scores, cv_scores_lc = learning_curve(
                model_lc, X_train, y_train, cv=3,
                train_sizes=np.linspace(0.1, 1.0, 8),
                scoring="accuracy", n_jobs=-1,
            )
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            cv_mean = cv_scores_lc.mean(axis=1)
            cv_std = cv_scores_lc.std(axis=1)

            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(
                x=train_sizes, y=train_mean + train_std, mode="lines",
                line=dict(width=0), showlegend=False
            ))
            fig_lc.add_trace(go.Scatter(
                x=train_sizes, y=train_mean - train_std, mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor="rgba(52,152,219,0.15)",
                showlegend=False
            ))
            fig_lc.add_trace(go.Scatter(
                x=train_sizes, y=train_mean, mode="lines+markers",
                name="Training Score", line=dict(color="#3498db", width=2)
            ))
            fig_lc.add_trace(go.Scatter(
                x=train_sizes, y=cv_mean + cv_std, mode="lines",
                line=dict(width=0), showlegend=False
            ))
            fig_lc.add_trace(go.Scatter(
                x=train_sizes, y=cv_mean - cv_std, mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor="rgba(46,204,113,0.15)",
                showlegend=False
            ))
            fig_lc.add_trace(go.Scatter(
                x=train_sizes, y=cv_mean, mode="lines+markers",
                name="CV Score", line=dict(color="#2ecc71", width=2)
            ))
            fig_lc.update_layout(
                title=f"Learning Curve — {selected_model}",
                xaxis_title="Training Set Size", yaxis_title="Accuracy",
                template="plotly_white", height=420,
            )
            st.plotly_chart(fig_lc, width="stretch")
        except Exception as e:
            st.warning(f"Could not compute learning curve: {e}")

    # Validation Curve
    st.markdown("#### Validation Curve")
    val_param_map = {
        "Logistic Regression": ("C", [0.001, 0.01, 0.1, 1, 10, 100]),
        "K-Nearest Neighbors": ("n_neighbors", list(range(1, 16))),
        "SVM": ("C", [0.01, 0.1, 1, 10, 100]),
        "Random Forest": ("n_estimators", [10, 30, 50, 80, 100, 150, 200]),
        "XGBoost": ("n_estimators", [10, 30, 50, 80, 100, 150, 200]),
    }
    if selected_model in val_param_map:
        vp_name, vp_range = val_param_map[selected_model]
        with st.spinner(f"Computing validation curve for '{vp_name}'..."):
            try:
                train_vc, test_vc = validation_curve(
                    sel_res["model"], X_train, y_train,
                    param_name=vp_name, param_range=vp_range,
                    cv=3, scoring="accuracy", n_jobs=-1,
                )
                fig_vc = go.Figure()
                fig_vc.add_trace(go.Scatter(
                    x=[str(v) for v in vp_range], y=train_vc.mean(axis=1),
                    mode="lines+markers", name="Training", line=dict(color="#3498db")
                ))
                fig_vc.add_trace(go.Scatter(
                    x=[str(v) for v in vp_range], y=test_vc.mean(axis=1),
                    mode="lines+markers", name="CV", line=dict(color="#e74c3c")
                ))
                fig_vc.update_layout(
                    title=f"Validation Curve: {selected_model} — {vp_name}",
                    xaxis_title=vp_name, yaxis_title="Accuracy",
                    template="plotly_white", height=380,
                )
                st.plotly_chart(fig_vc, width="stretch")
            except Exception as e:
                st.warning(f"Could not compute validation curve: {e}")

    # ── Section 4: Comparative Visualizations ──────────────────────────────
    st.markdown("### 🔀 Section 4 — Comparative Visualizations")

    # Prepare comparative data
    comp_data = []
    for mname, mres in results.items():
        cr_dict = mres["classification_report"]
        weighted = cr_dict.get("weighted avg", cr_dict.get("macro avg", {}))
        precision = weighted.get("precision", 0)
        recall = weighted.get("recall", 0)
        f1 = weighted.get("f1-score", 0)
        # AUC
        roc_d = mres["roc_data"]
        if "binary" in roc_d:
            auc_val = roc_d["binary"]["auc"]
        elif "micro" in roc_d:
            auc_val = roc_d["micro"]["auc"]
        else:
            auc_val = 0

        comp_data.append({
            "Model": mname,
            "Accuracy": mres["test_acc"],
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC": auc_val,
            "Train Time": mres["train_time"],
            "CV Mean": mres["cv_mean"],
        })
    comp_df = pd.DataFrame(comp_data)

    # [MANDATORY] RADAR CHART
    st.markdown("#### 🕸️ Radar Chart — Model Comparison")
    fig_radar = go.Figure()
    radar_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "Training Speed", "Eco-Efficiency"]

    # Normalize training speed: fastest = 1, slowest = 0
    max_time = comp_df["Train Time"].max()
    min_time = comp_df["Train Time"].min()

    radar_colors = px.colors.qualitative.Bold
    for i, row in comp_df.iterrows():
        # Normalized training speed (inverse of time, so faster = higher)
        if max_time > min_time:
            speed_norm = 1.0 - (row["Train Time"] - min_time) / (max_time - min_time)
        else:
            speed_norm = 1.0
        # Eco-efficiency: accuracy / (train_time + 0.001)
        eco_raw = row["Accuracy"] / (row["Train Time"] + 0.001)
        eco_vals = [r["Accuracy"] / (r["Train Time"] + 0.001) for r in comp_data]
        eco_max = max(eco_vals)
        eco_min = min(eco_vals)
        eco_norm = (eco_raw - eco_min) / (eco_max - eco_min) if eco_max > eco_min else 1.0

        values = [row["Accuracy"], row["Precision"], row["Recall"], row["F1-Score"], speed_norm, eco_norm]

        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # close the polygon
            theta=radar_metrics + [radar_metrics[0]],
            name=row["Model"],
            fill="toself",
            opacity=0.6,
            line=dict(color=radar_colors[i % len(radar_colors)]),
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Radar Chart: Model Performance Comparison",
        template="plotly_white", height=520,
        legend=dict(font=dict(size=11))
    )
    st.plotly_chart(fig_radar, width="stretch")

    # [MANDATORY] PARALLEL COORDINATES — Models
    st.markdown("#### 🌈 Parallel Coordinates — Model Metrics")
    pc_model_df = comp_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]].copy()
    pc_model_df["_model_code"] = range(len(pc_model_df))
    fig_pc_model = go.Figure(data=go.Parcoords(
        line=dict(color=pc_model_df["_model_code"], colorscale="Viridis", showscale=True),
        dimensions=[
            dict(label="Accuracy", values=pc_model_df["Accuracy"]),
            dict(label="Precision", values=pc_model_df["Precision"]),
            dict(label="Recall", values=pc_model_df["Recall"]),
            dict(label="F1-Score", values=pc_model_df["F1-Score"]),
            dict(label="AUC", values=pc_model_df["AUC"]),
        ],
        customdata=pc_model_df["Model"].values,
    ))
    fig_pc_model.update_layout(
        title="Parallel Coordinates: Model Metric Profiles",
        template="plotly_white", height=450,
    )
    st.plotly_chart(fig_pc_model, width="stretch")

    # Grouped Bar Chart
    st.markdown("#### 📊 Grouped Bar — Train / Test / CV Accuracy")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Train Acc", x=comp_df["Model"],
                             y=[results[m]["train_acc"] for m in comp_df["Model"]],
                             marker_color="#3498db"))
    fig_bar.add_trace(go.Bar(name="Test Acc", x=comp_df["Model"],
                             y=comp_df["Accuracy"], marker_color="#2ecc71"))
    fig_bar.add_trace(go.Bar(name="CV Mean", x=comp_df["Model"],
                             y=comp_df["CV Mean"], marker_color="#e67e22"))
    fig_bar.update_layout(barmode="group", title="Model Accuracy Comparison",
                          template="plotly_white", height=400, yaxis_title="Accuracy")
    st.plotly_chart(fig_bar, width="stretch")

    # [MANDATORY] BOX PLOT — CV Scores (Section 4 repeat for comparative)
    st.markdown("#### 📦 Cross-Validation Scores — All Models")
    fig_cv_box2 = go.Figure()
    for i, (mname, mres) in enumerate(results.items()):
        fig_cv_box2.add_trace(go.Box(
            y=mres["cv_scores"], name=mname,
            marker_color=radar_colors[i % len(radar_colors)],
            boxpoints="all", jitter=0.3, pointpos=-1.5
        ))
    fig_cv_box2.update_layout(
        title="CV Fold Scores (Comparative)", yaxis_title="Accuracy",
        template="plotly_white", height=420,
    )
    st.plotly_chart(fig_cv_box2, width="stretch")

    # ── Section 5: Prediction Explorer ─────────────────────────────────────
    st.markdown("### 🔮 Section 5 — Prediction Explorer")
    max_idx = len(y_test) - 1
    row_idx = st.number_input("Select test sample index:", 0, max_idx, 0, key="pred_idx")

    numeric_cols_pred = st.session_state.get("numeric_cols", [])
    sample = X_test[row_idx]
    pred_cls = sel_res["y_pred"][row_idx]
    actual_cls = y_test[row_idx]

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(f"**Actual Class:** `{class_names[actual_cls]}` — **Predicted:** `{class_names[pred_cls]}`")
        feat_labels = numeric_cols_pred if len(numeric_cols_pred) == len(sample) else [f"F{i}" for i in range(len(sample))]
        fig_feat = px.bar(x=feat_labels, y=sample, title="Feature Values",
                          template="plotly_white", labels={"x": "Feature", "y": "Value"},
                          color_discrete_sequence=["#3498db"])
        fig_feat.update_layout(height=350)
        st.plotly_chart(fig_feat, width="stretch")

    with pc2:
        if sel_res["y_prob"] is not None:
            probs = sel_res["y_prob"][row_idx]
            fig_prob = px.bar(x=[str(c) for c in class_names], y=probs,
                              title="Predicted Probabilities",
                              template="plotly_white",
                              labels={"x": "Class", "y": "Probability"},
                              color_discrete_sequence=["#2ecc71"])
            fig_prob.update_layout(height=350)
            st.plotly_chart(fig_prob, width="stretch")
        else:
            st.info("Probabilities not available for this model.")

    render_footer()


##############################################################################
# PAGE 5 — CARBON FOOTPRINT TRACKER
##############################################################################
elif page == PAGES[4]:
    render_header("Carbon Footprint Tracker 🌍", "Estimate and compare ML carbon emissions")

    # ── Section 1: Manual Carbon Calculator ────────────────────────────────
    st.markdown("### 🧮 Section 1 — Carbon Footprint Calculator (Manual Input)")

    tab_e, tab_g, tab_h, tab_m, tab_t = st.tabs([
        "⚡ Energy", "🌍 Grid & Location", "🖥️ Hardware",
        "🧠 Model Architecture", "⏱️ Training Process"
    ])

    with tab_e:
        ce1, ce2, ce3, ce4 = st.columns(4)
        tdp = ce1.number_input("Hardware TDP (Watts)", 1, 1000, 250, key="c_tdp")
        train_hours = ce2.number_input("Training duration (hours)", 0.01, 10000.0, 1.0, key="c_hours")
        n_runs = ce3.number_input("Number of training runs", 1, 1000, 1, key="c_runs")
        pue = ce4.slider("PUE (Power Usage Effectiveness)", 1.0, 2.0, 1.1, 0.05, key="c_pue")
        energy_kwh = (tdp * train_hours * n_runs * pue) / 1000.0
        st.metric("⚡ Total Energy (kWh)", f"{energy_kwh:.4f}")

    with tab_g:
        presets = {
            "Global Average": 475,
            "India": 713,
            "USA": 386,
            "EU Average": 276,
            "France (nuclear)": 85,
            "Iceland (renewable)": 28,
            "Custom": None,
        }
        region = st.selectbox("Carbon Intensity Preset", list(presets.keys()), key="c_region")
        if region == "Custom":
            carbon_intensity = st.number_input("Custom gCO₂eq/kWh", 1, 2000, 475, key="c_custom_ci")
        else:
            carbon_intensity = presets[region]
            st.info(f"Carbon intensity for {region}: **{carbon_intensity} gCO₂eq/kWh**")

        renewable_pct = st.slider("% Renewable Energy", 0, 100, 0, key="c_renew")

        # Reference bar chart
        ref_df = pd.DataFrame({"Region": [k for k in presets if k != "Custom"],
                                "gCO₂eq/kWh": [v for k, v in presets.items() if k != "Custom"]})
        fig_ref = px.bar(ref_df, x="Region", y="gCO₂eq/kWh", color="gCO₂eq/kWh",
                         color_continuous_scale="RdYlGn_r",
                         title="Carbon Intensity by Region", template="plotly_white")
        fig_ref.update_layout(height=350)
        st.plotly_chart(fig_ref, width="stretch")

    with tab_h:
        hw_type = st.selectbox("Hardware Type", ["CPU", "GPU - Consumer", "GPU - Data Center", "TPU v3", "TPU v4"], key="c_hw")
        n_accel = st.number_input("Number of Accelerators", 1, 128, 1, key="c_naccel")
        peak_mem = st.number_input("Peak Memory Usage (GB)", 0.0, 1000.0, 8.0, key="c_mem")
        embodied_carbon = st.number_input("Embodied Carbon per Chip (kg CO₂e)", 0.0, 5000.0, 143.0, key="c_embodied")

    with tab_m:
        n_params = st.number_input("Number of Parameters (millions)", 0.001, 1000000.0, 1.0, key="c_params")
        sparsity = st.slider("Model Sparsity (%)", 0, 100, 0, key="c_sparsity")
        n_epochs = st.number_input("Number of Epochs", 1, 100000, 10, key="c_epochs")
        batch_size = st.number_input("Batch Size", 1, 10000, 32, key="c_batch")
        n_layers = st.number_input("Number of Layers", 1, 1000, 5, key="c_layers")
        total_flops = st.number_input("Total FLOPs (GFLOPs)", 0.0, 1e12, 1.0, key="c_flops")

    with tab_t:
        inf_time_ms = st.number_input("Inference time per sample (ms)", 0.01, 100000.0, 1.0, key="c_inf_ms")
        n_inf_samples = st.number_input("Number of inference samples", 0, 10000000, 1000, key="c_inf_n")
        hp_trials = st.number_input("Hyperparameter search trials", 1, 10000, 1, key="c_hp_trials")
        data_workers = st.number_input("Data loading workers", 1, 64, 4, key="c_workers")
        ckpt_freq = st.number_input("Checkpointing freq (every N epochs)", 1, 10000, 5, key="c_ckpt")
        cv_folds_carbon = st.number_input("Cross-validation folds", 1, 50, 5, key="c_cv_folds")

    # ── Section 2: Output Metrics Dashboard ────────────────────────────────
    st.markdown("### 📋 Section 2 — Output Metrics Dashboard")

    if st.button("🌱 Calculate Carbon Footprint", type="primary", width="stretch"):
        # Adjust for renewable
        effective_ci = carbon_intensity * (1 - renewable_pct / 100.0)

        co2_train_kg = energy_kwh * effective_ci / 1000.0
        co2_inf_kg = (inf_time_ms / 1000.0) * n_inf_samples * (tdp / 1000.0) * effective_ci / 1000.0 / 3600.0
        total_co2 = co2_train_kg + co2_inf_kg + (embodied_carbon * n_accel / 1000.0)
        eco_efficiency = 0.5 / (co2_train_kg + 0.0001)  # placeholder accuracy=0.5
        energy_per_param = (energy_kwh * 3.6e6) / (n_params * 1e6) if n_params > 0 else 0  # mJ / param
        km_driven = total_co2 / 0.21
        netflix_hrs = total_co2 / 0.036

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("🏭 CO₂e Training (kg)", f"{co2_train_kg:.6f}")
        mc2.metric("🔮 CO₂e Inference (kg)", f"{co2_inf_kg:.8f}")
        mc3.metric("📦 Total Lifecycle CO₂e (kg)", f"{total_co2:.4f}")

        mc4, mc5, mc6 = st.columns(3)
        mc4.metric("⚡ Energy per Parameter (mJ/param)", f"{energy_per_param:.4f}")
        mc5.metric("🚗 Equivalent km Driven", f"{km_driven:.3f}")
        mc6.metric("📺 Equivalent Netflix Hours", f"{netflix_hrs:.3f}")

        # Eco-Efficiency Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(eco_efficiency, 10000),
            title={"text": "Eco-Efficiency Score (Accuracy / CO₂e)"},
            gauge=dict(
                axis=dict(range=[0, 10000]),
                bar=dict(color="#2ecc71"),
                steps=[
                    dict(range=[0, 1000], color="#e74c3c"),
                    dict(range=[1000, 5000], color="#f39c12"),
                    dict(range=[5000, 10000], color="#2ecc71"),
                ],
            ),
        ))
        fig_gauge.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig_gauge, width="stretch")

    # ── Section 3: Auto Carbon Comparison of Trained Models ────────────────
    st.markdown("### 🔍 Section 3 — Auto Carbon Comparison of Trained Models")

    if "trained_models" in st.session_state and st.session_state.trained_models:
        results = st.session_state.trained_models
        class_names_c = st.session_state.get("class_names", [])

        # Default assumptions
        st.info("**Assumptions:** CPU 65W TDP | India 713 gCO₂eq/kWh | PUE 1.2")
        default_tdp = 65  # Watts
        default_ci = 713  # gCO2/kWh
        default_pue = 1.2

        carbon_rows = []
        for mname, mres in results.items():
            t = mres["train_time"]
            energy = (default_tdp * (t / 3600.0) * default_pue) / 1000.0  # kWh
            co2 = energy * default_ci / 1000.0  # kg

            cr_dict = mres["classification_report"]
            w_avg = cr_dict.get("weighted avg", cr_dict.get("macro avg", {}))

            eco_eff = mres["test_acc"] / (co2 + 1e-9)
            carbon_rows.append({
                "Model": mname,
                "CO₂e (kg)": co2,
                "Eco-Efficiency": eco_eff,
                "Accuracy": mres["test_acc"],
                "F1-Score": w_avg.get("f1-score", 0),
                "Train Time (s)": t,
            })
        carbon_df = pd.DataFrame(carbon_rows)

        # [MANDATORY] BUBBLE CHART — Carbon Comparison
        st.markdown("#### 🫧 Carbon Bubble Chart")
        fig_carbon_bubble = px.scatter(
            carbon_df, x="Model", y="CO₂e (kg)",
            size="Train Time (s)", color="Accuracy",
            hover_data=["F1-Score", "Eco-Efficiency"],
            title="Carbon Emissions vs Models (size=Train Time, color=Accuracy)",
            template="plotly_white", size_max=55,
            color_continuous_scale="Viridis",
        )
        fig_carbon_bubble.update_layout(height=450)
        st.plotly_chart(fig_carbon_bubble, width="stretch")

        # [MANDATORY] HEATMAP — Model × Metric
        st.markdown("#### 🔥 Model × Metric Heatmap")
        hm_cols = ["CO₂e (kg)", "Eco-Efficiency", "Accuracy", "F1-Score", "Train Time (s)"]
        hm_df = carbon_df.set_index("Model")[hm_cols]
        # Normalize for heatmap display
        hm_norm = hm_df.copy()
        for col in hm_norm.columns:
            cmin, cmax = hm_norm[col].min(), hm_norm[col].max()
            if cmax > cmin:
                hm_norm[col] = (hm_norm[col] - cmin) / (cmax - cmin)
            else:
                hm_norm[col] = 0.5

        fig_hm = px.imshow(
            hm_norm.values, text_auto=False,
            x=hm_cols, y=hm_df.index.tolist(),
            color_continuous_scale="RdBu_r",
            title="Model × Metric Heatmap (normalized 0–1)",
            template="plotly_white",
        )
        # Overlay raw values as annotations
        for i, model in enumerate(hm_df.index):
            for j, col in enumerate(hm_cols):
                val = hm_df.iloc[i, j]
                fig_hm.add_annotation(
                    x=col, y=model, text=f"{val:.4f}",
                    showarrow=False, font=dict(size=10, color="white")
                )
        fig_hm.update_layout(height=350)
        st.plotly_chart(fig_hm, width="stretch")

        # Summary table
        st.markdown("#### 📋 Carbon Comparison Summary")
        best_eco = carbon_df.loc[carbon_df["Eco-Efficiency"].idxmax(), "Model"]

        def highlight_eco(row):
            if row["Model"] == best_eco:
                return ["background-color: #0d3320; color: #2ecc71; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(carbon_df.style.apply(highlight_eco, axis=1).format({
            "CO₂e (kg)": "{:.8f}", "Eco-Efficiency": "{:.2f}",
            "Accuracy": "{:.4f}", "F1-Score": "{:.4f}", "Train Time (s)": "{:.4f}"
        }), width="stretch", hide_index=True)
        st.success(f"🏆 Most Eco-Efficient Model: **{best_eco}**")
    else:
        st.warning("⚠️ No trained models found. Train models in the **🤖 ML Classification Lab** first to see auto carbon comparison.")

    # ── Section 4: References ──────────────────────────────────────────────
    st.markdown("### 📚 Section 4 — Carbon Footprint References")
    st.markdown("""
    1. **Patterson et al. (2022):** "The Carbon Footprint of Machine Learning Training Will Plateau, Then Shrink"
       — *IEEE Micro*. DOI: [10.1109/MM.2022.3163226](https://doi.org/10.1109/MM.2022.3163226)
    2. **Lacoste et al. (2019):** "Quantifying the Carbon Emissions of Machine Learning"
       — arXiv:1910.09700. [https://arxiv.org/abs/1910.09700](https://arxiv.org/abs/1910.09700)
    3. **ML Emissions Calculator:** [https://mlco2.github.io/impact/](https://mlco2.github.io/impact/)
    """)

    render_footer()


##############################################################################
# PAGE 6 — QUIZ
##############################################################################
elif page == PAGES[5]:
    render_header("Knowledge Check ✅", "Test your understanding of data visualization & ML concepts")

    questions = [
        {"q": "Which plot is best suited for depicting the distribution of a single continuous variable?",
         "options": ["Scatter Plot", "Histogram", "Pie Chart", "Line Graph"], "ans": "Histogram"},
        {"q": "What is the primary purpose of a Correlation Heatmap?",
         "options": ["Show 3D data", "Show frequency of categorical data",
                     "Visualize strength of relationships between numeric variables",
                     "Display training loss"],
         "ans": "Visualize strength of relationships between numeric variables"},
        {"q": "When visualizing regression performance, a Residual Plot helps identify:",
         "options": ["Accuracy percentage", "Whether errors are randomly distributed",
                     "The learning rate", "Class imbalance"],
         "ans": "Whether errors are randomly distributed"},
        {"q": "Which visualization adds a third spatial dimension?",
         "options": ["Violin Plot", "Box Plot", "3D Scatter Plot", "Radar Chart"],
         "ans": "3D Scatter Plot"},
        {"q": "What does PCA (Principal Component Analysis) do?",
         "options": ["Increases dimensionality", "Reduces dimensions while preserving variance",
                     "Classifies data", "Clusters data"],
         "ans": "Reduces dimensions while preserving variance"},
        {"q": "In a ROC curve, what does AUC represent?",
         "options": ["Average Unique Classes", "Area Under the Curve",
                     "Accuracy Under Constraints", "Adjusted Unbiased Coefficient"],
         "ans": "Area Under the Curve"},
        {"q": "What metric is best for imbalanced classification problems?",
         "options": ["Accuracy", "F1-Score", "Training Time", "Number of Features"],
         "ans": "F1-Score"},
        {"q": "What does PUE (Power Usage Effectiveness) measure in carbon footprint analysis?",
         "options": ["Prediction accuracy", "Total energy vs compute energy ratio",
                     "Number of parameters", "Model speed"],
         "ans": "Total energy vs compute energy ratio"},
    ]

    score = 0
    with st.form("quiz_form"):
        user_answers = []
        for i, q in enumerate(questions):
            st.markdown(f'<div class="quiz-card"><b>Question {i+1}:</b> {q["q"]}</div>', unsafe_allow_html=True)
            ans = st.radio(f"Select answer for Q{i+1}:", q["options"], key=f"quiz_q{i}", label_visibility="collapsed")
            user_answers.append(ans)
            st.markdown("")

        submitted = st.form_submit_button("📝 Submit Quiz", type="primary", width="stretch")

    if submitted:
        for i, q in enumerate(questions):
            if user_answers[i] == q["ans"]:
                score += 1
                st.success(f"**Q{i+1}: Correct!** ✅ — {q['ans']}")
            else:
                st.error(f"**Q{i+1}: Incorrect.** ❌ — Correct answer: {q['ans']}")

        pct = score / len(questions) * 100
        st.markdown(f"""
        <div class="{'success-box' if pct >= 70 else 'warning-box' if pct >= 50 else 'error-box'}">
        <h3>Your Total Score: {score} / {len(questions)} ({pct:.0f}%)</h3>
        {'🎉 Excellent work!' if pct >= 70 else '📖 Review the concepts and try again!' if pct >= 50 else '⚠️ Consider revisiting the theory section.'}
        </div>
        """, unsafe_allow_html=True)

    render_footer()


##############################################################################
# PAGE 7 — REFERENCES
##############################################################################
elif page == PAGES[6]:
    render_header("References & Further Reading 📚", "Resources for deeper exploration")

    st.markdown("""
    ### 📊 Data Visualization

    | Resource | Link |
    |----------|------|
    | **Plotly Express Documentation** | [plotly.com/python/plotly-express](https://plotly.com/python/plotly-express/) |
    | **Seaborn Documentation** | [seaborn.pydata.org](https://seaborn.pydata.org/) |
    | **Streamlit Documentation** | [docs.streamlit.io](https://docs.streamlit.io/) |
    | **"Show Me the Numbers"** — Stephen Few | Visualization theory & practice |
    | **Penn State — Residual Plots** | [online.stat.psu.edu/stat462/node/117/](https://online.stat.psu.edu/stat462/node/117/) |

    ---

    ### 🤖 Machine Learning

    | Resource | Link |
    |----------|------|
    | **scikit-learn** | [scikit-learn.org](https://scikit-learn.org/) |
    | **XGBoost** | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |

    ---

    ### 🌍 Carbon Footprint & Sustainability

    | Resource | Details |
    |----------|---------|
    | **Patterson et al. (2022)** | "The Carbon Footprint of ML Training Will Plateau, Then Shrink" — *IEEE Micro*. DOI: [10.1109/MM.2022.3163226](https://doi.org/10.1109/MM.2022.3163226) |
    | **Lacoste et al. (2019)** | "Quantifying the Carbon Emissions of Machine Learning" — [arXiv:1910.09700](https://arxiv.org/abs/1910.09700) |
    | **ML Emissions Calculator** | [mlco2.github.io/impact](https://mlco2.github.io/impact/) |

    ---

    ### 📖 Textbooks & Courses
    - *Python for Data Analysis* — Wes McKinney
    - *Hands-On Machine Learning* — Aurélien Géron
    - *Data Visualization: A Practical Introduction* — Kieran Healy
    """)

    st.markdown("---")
    render_footer()
