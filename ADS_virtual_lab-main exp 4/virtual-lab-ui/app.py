import os
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import requests
import streamlit as st

DEFAULT_API_BASE = os.getenv("ADS_API_BASE_URL", "http://localhost:5000/api")


st.set_page_config(
    page_title="ADS Virtual Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Space+Mono:wght@400;700&display=swap');

        .stApp {
            background:
                radial-gradient(circle at 8% 10%, rgba(255, 209, 102, 0.28), transparent 25%),
                radial-gradient(circle at 92% 0%, rgba(81, 207, 102, 0.22), transparent 28%),
                linear-gradient(180deg, #f8f6ef 0%, #efe9dc 100%);
            color: #151515;
        }

        h1, h2, h3 {
            font-family: 'Fraunces', serif !important;
            letter-spacing: -0.02em;
        }

        p, div, span, label {
            font-family: 'Space Mono', monospace !important;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }

        .ads-card {
            border: 2px solid #151515;
            background: #fffdf8;
            box-shadow: 6px 6px 0 #151515;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }

        .ads-kicker {
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            color: #3f3f3f;
        }

        .ads-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-top: 0.15rem;
        }

        .ads-value {
            font-size: 1.55rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }

        .stButton > button {
            border: 2px solid #151515;
            border-radius: 0;
            background: #ffdd57;
            color: #151515;
            box-shadow: 3px 3px 0 #151515;
            font-weight: 700;
            letter-spacing: 0.03em;
        }

        .stButton > button:hover {
            transform: translate(-1px, -1px);
            box-shadow: 4px 4px 0 #151515;
        }

        [data-testid="stSidebar"] {
            background: #fff6d8;
            border-right: 2px solid #151515;
        }

        [data-testid="stMetric"] {
            border: 2px solid #151515;
            background: #fffdf8;
            padding: 0.55rem;
            box-shadow: 4px 4px 0 #151515;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "api_base_url": DEFAULT_API_BASE,
        "health": None,
        "data_stats": None,
        "data_sample": None,
        "cleaning_result": None,
        "missing_summary": None,
        "impute_single": None,
        "impute_compare": None,
        "preprocess_result": None,
        "model_single": None,
        "model_compare": None,
        "cluster_analysis": None,
        "cluster_pca": None,
        "cluster_elbow": None,
        "fusion_compare": None,
        "fusion_single": None,
        "pipeline_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_card(label: str, title: str, value: Any) -> None:
    st.markdown(
        f"""
        <div class='ads-card'>
            <div class='ads-kicker'>{label}</div>
            <div class='ads-title'>{title}</div>
            <div class='ads-value'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def api_request(
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 300,
) -> Optional[Dict[str, Any]]:
    base = st.session_state.get("api_base_url", DEFAULT_API_BASE).rstrip("/")
    url = f"{base}{endpoint}"

    try:
        response = requests.request(method=method, url=url, params=params, json=payload, timeout=timeout)

        if not response.ok:
            try:
                err_payload = response.json()
                err_msg = err_payload.get("error") or str(err_payload)
            except ValueError:
                err_msg = response.text
            st.error(f"API call failed ({response.status_code}): {err_msg}")
            return None

        if response.content:
            return response.json()

        return {}
    except requests.RequestException as exc:
        st.error(f"Could not connect to backend at {base}. Details: {exc}")
        return None


def render_overview() -> None:
    st.title("ADS Virtual Lab")
    st.caption("Streamlit interface for the full Flask ML pipeline")

    c1, c2, c3 = st.columns(3)
    with c1:
        show_card("PIPELINE", "Data science steps", 10)
    with c2:
        show_card("MODULES", "Interactive modules", 8)
    with c3:
        show_card("DATASET", "UCI Adult rows", "~48K")

    st.markdown("### Workflow")
    st.write(
        "1. Load data and inspect basic statistics\n"
        "2. Run cleaning and compare imputation methods\n"
        "3. Preprocess and train models\n"
        "4. Analyze clustering and PCA\n"
        "5. Compare fusion strategies and run complete pipeline"
    )

    st.markdown("### Backend connection")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Check health", key="overview_health"):
            with st.spinner("Checking backend health..."):
                st.session_state["health"] = api_request("/health")
    with col_b:
        if st.session_state.get("health"):
            st.success("Backend reachable")

    if st.session_state.get("health"):
        st.json(st.session_state["health"])


def render_data_loader() -> None:
    st.header("Data Loader")
    top_col1, top_col2 = st.columns([1, 1])

    with top_col1:
        if st.button("Load dataset statistics", key="load_data_stats"):
            with st.spinner("Loading dataset stats..."):
                st.session_state["data_stats"] = api_request("/data/load")

    with top_col2:
        sample_n = st.slider("Sample rows", min_value=3, max_value=25, value=5, step=1, key="sample_n_slider")
        if st.button("Fetch sample", key="load_data_sample"):
            with st.spinner("Fetching sample rows..."):
                data = api_request("/data/sample", params={"n": sample_n})
                if data:
                    st.session_state["data_sample"] = data.get("sample", [])

    stats = st.session_state.get("data_stats")
    if stats:
        missing_total = sum((stats.get("missing_values") or {}).values())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{stats.get('rows', 0):,}")
        m2.metric("Columns", stats.get("columns", 0))
        m3.metric("Missing values", f"{missing_total:,}")
        m4.metric("Target", "income")

        features = stats.get("features", [])
        data_types = stats.get("data_types", {})
        missing_map = stats.get("missing_values", {})

        if features:
            feature_df = pd.DataFrame(
                {
                    "feature": features,
                    "data_type": [data_types.get(f, "") for f in features],
                    "missing": [missing_map.get(f, 0) for f in features],
                }
            )
            st.markdown("#### Feature schema")
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

        class_dist = stats.get("class_distribution") or {}
        if class_dist:
            class_df = pd.DataFrame(
                {"class": list(class_dist.keys()), "count": list(class_dist.values())}
            )
            st.markdown("#### Class distribution")
            st.bar_chart(class_df.set_index("class")["count"])

    sample_rows = st.session_state.get("data_sample")
    if sample_rows:
        st.markdown("#### Data sample")
        st.dataframe(pd.DataFrame(sample_rows), use_container_width=True, hide_index=True)


def render_cleaning() -> None:
    st.header("Data Cleaning")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Run cleaning pipeline", key="run_cleaning"):
            with st.spinner("Running data cleaning..."):
                st.session_state["cleaning_result"] = api_request("/cleaning/run", method="POST", payload={})

    with c2:
        if st.button("Get missing value summary", key="missing_summary"):
            with st.spinner("Loading missing values..."):
                data = api_request("/cleaning/missing-values")
                if data:
                    st.session_state["missing_summary"] = data.get("missing_values", {})

    cleaning = st.session_state.get("cleaning_result")
    if cleaning:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Original rows", cleaning.get("original_rows", 0))
        m2.metric("Cleaned rows", cleaning.get("cleaned_rows", 0))
        m3.metric("Duplicates removed", cleaning.get("duplicates_removed", 0))
        m4.metric("Outliers detected", cleaning.get("outliers_detected", 0))
        st.info(cleaning.get("summary", "Cleaning completed."))

    missing_summary = st.session_state.get("missing_summary")
    if missing_summary is not None:
        if missing_summary:
            missing_df = pd.DataFrame(
                {
                    "column": list(missing_summary.keys()),
                    "missing": list(missing_summary.values()),
                }
            )
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.success("No missing values found.")


def render_imputation() -> None:
    st.header("Data Imputation")
    method = st.selectbox("Method", ["mode", "knn", "mice"], key="impute_method")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Run selected imputation", key="run_impute_single"):
            with st.spinner(f"Running {method} imputation..."):
                st.session_state["impute_single"] = api_request(
                    "/cleaning/impute",
                    method="POST",
                    payload={"method": method},
                )

    with col_b:
        if st.button("Compare all imputation methods", key="run_impute_compare"):
            with st.spinner("Comparing imputation methods..."):
                st.session_state["impute_compare"] = api_request(
                    "/cleaning/impute/compare",
                    method="POST",
                    payload={},
                    timeout=360,
                )

    single = st.session_state.get("impute_single")
    if single:
        st.markdown("#### Selected method result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Method", single.get("method", "-"))
        m2.metric("Execution time (ms)", single.get("execution_time_ms", 0))
        m3.metric("Missing values remaining", single.get("missing_values_remaining", 0))

    compare = st.session_state.get("impute_compare")
    if compare:
        st.markdown("#### Method comparison")
        rows = compare.get("results", [])
        if rows:
            comp_df = pd.DataFrame(rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            if "execution_time_ms" in comp_df.columns:
                st.bar_chart(comp_df.set_index("method")["execution_time_ms"])


def render_preprocessing() -> None:
    st.header("Preprocessing")
    imputation_method = st.selectbox(
        "Imputation method used before preprocessing",
        ["mode", "knn", "mice"],
        key="prep_imputation_method",
    )
    test_size = st.slider(
        "Test size ratio",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
        key="prep_test_size",
    )

    if st.button("Run preprocessing", key="run_preprocessing"):
        with st.spinner("Preprocessing data..."):
            st.session_state["preprocess_result"] = api_request(
                "/preprocessing/run",
                method="POST",
                payload={"imputation_method": imputation_method, "test_size": test_size},
                timeout=240,
            )

    result = st.session_state.get("preprocess_result")
    if result:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Original features", result.get("original_features", 0))
        m2.metric("Processed features", result.get("processed_features", 0))
        m3.metric("Train samples", result.get("train_samples", 0))
        m4.metric("Test samples", result.get("test_samples", 0))

        detail_df = pd.DataFrame(
            {
                "metric": [
                    "Total samples",
                    "Test size ratio",
                    "Categorical encoded",
                    "Numerical scaled",
                ],
                "value": [
                    result.get("total_samples", 0),
                    result.get("test_size_ratio", 0),
                    result.get("categorical_features_encoded", 0),
                    result.get("numerical_features_scaled", 0),
                ],
            }
        )
        st.dataframe(detail_df, use_container_width=True, hide_index=True)


def render_models() -> None:
    st.header("Model Training")

    selected_model = st.selectbox(
        "Train a single model",
        ["logistic_regression", "random_forest", "xgboost"],
        key="single_model",
    )

    top_a, top_b = st.columns(2)
    with top_a:
        if st.button("Train selected model", key="train_single_model"):
            with st.spinner(f"Training {selected_model}..."):
                st.session_state["model_single"] = api_request(
                    "/models/train",
                    method="POST",
                    payload={"model": selected_model},
                    timeout=360,
                )

    with top_b:
        if st.button("Train and compare all models", key="compare_all_models"):
            with st.spinner("Training all models..."):
                st.session_state["model_compare"] = api_request(
                    "/models/compare",
                    method="POST",
                    payload={},
                    timeout=480,
                )

    single_result = st.session_state.get("model_single")
    if single_result:
        st.markdown("#### Single model result")
        metrics = single_result.get("metrics", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", metrics.get("accuracy", 0))
        m2.metric("Precision", metrics.get("precision", 0))
        m3.metric("Recall", metrics.get("recall", 0))
        m4.metric("F1 score", metrics.get("f1_score", 0))
        m5.metric("ROC AUC", metrics.get("roc_auc", 0))

        st.write(f"Training time (ms): {single_result.get('training_time_ms', 0)}")

        cm = single_result.get("confusion_matrix") or []
        if cm and len(cm) == 2 and len(cm[0]) == 2:
            cm_df = pd.DataFrame(
                cm,
                index=["Actual class 0", "Actual class 1"],
                columns=["Pred class 0", "Pred class 1"],
            )
            st.markdown("#### Confusion matrix")
            st.dataframe(cm_df, use_container_width=True)

        feature_importance = single_result.get("feature_importance") or {}
        if feature_importance:
            importance_df = pd.DataFrame(
                {
                    "feature": list(feature_importance.keys()),
                    "importance": list(feature_importance.values()),
                }
            )
            st.markdown("#### Feature importance")
            st.dataframe(importance_df, use_container_width=True, hide_index=True)

    compare_payload = st.session_state.get("model_compare")
    if compare_payload:
        st.markdown("#### Model comparison")
        rows = compare_payload.get("models", []) if isinstance(compare_payload, dict) else []
        result_rows = [row for row in rows if isinstance(row, dict) and row.get("model")]
        best_row = next((row for row in rows if isinstance(row, dict) and row.get("best_model")), None)

        if result_rows:
            compare_df = pd.DataFrame(result_rows)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            for metric in ["accuracy", "f1_score", "roc_auc"]:
                if metric in compare_df.columns:
                    st.markdown(f"{metric.replace('_', ' ').title()} comparison")
                    st.bar_chart(compare_df.set_index("model")[metric])

        if best_row:
            st.success(
                f"Best model: {best_row.get('best_model')} | Accuracy: {best_row.get('best_accuracy')}"
            )


def render_clustering() -> None:
    st.header("Clustering")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Run clustering analysis", key="cluster_analyze"):
            with st.spinner("Running K-Means analysis..."):
                st.session_state["cluster_analysis"] = api_request(
                    "/clustering/analyze",
                    method="POST",
                    payload={},
                    timeout=360,
                )

    with c2:
        pca_k = st.number_input("PCA cluster count (k)", min_value=2, max_value=8, value=3, step=1, key="pca_k")
        if st.button("Get PCA visualization", key="cluster_pca"):
            with st.spinner("Computing PCA and clusters..."):
                st.session_state["cluster_pca"] = api_request(
                    "/clustering/pca",
                    method="POST",
                    payload={"k": int(pca_k)},
                    timeout=360,
                )

    with c3:
        if st.button("Get elbow curve", key="cluster_elbow"):
            with st.spinner("Loading elbow curve..."):
                st.session_state["cluster_elbow"] = api_request("/clustering/elbow")

    analysis = st.session_state.get("cluster_analysis")
    if analysis:
        st.markdown("#### K-Means metrics by k")
        metrics_df = pd.DataFrame(
            {
                "k": analysis.get("k_values", []),
                "silhouette": analysis.get("silhouette_scores", []),
                "davies_bouldin": analysis.get("davies_bouldin_scores", []),
                "inertia": analysis.get("inertias", []),
            }
        )
        if not metrics_df.empty:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            st.line_chart(metrics_df.set_index("k")[["silhouette", "davies_bouldin"]])
            st.line_chart(metrics_df.set_index("k")[["inertia"]])
            st.success(f"Suggested optimal k: {analysis.get('optimal_k')}")

    pca_payload = st.session_state.get("cluster_pca")
    if pca_payload:
        st.markdown("#### PCA scatter")
        points = pca_payload.get("points", [])
        points_df = pd.DataFrame(points)
        if not points_df.empty:
            scatter_chart = (
                alt.Chart(points_df)
                .mark_circle(size=40, opacity=0.75)
                .encode(
                    x=alt.X("x:Q", title="PC1"),
                    y=alt.Y("y:Q", title="PC2"),
                    color=alt.Color("cluster:N", title="Cluster"),
                    tooltip=["x", "y", "cluster"],
                )
                .properties(height=420)
                .interactive()
            )
            st.altair_chart(scatter_chart, use_container_width=True)

        variance = pca_payload.get("explained_variance", [])
        if variance:
            st.write(f"Explained variance: {variance}")
            st.write(f"Cumulative variance: {pca_payload.get('cumulative_variance')}")

    elbow = st.session_state.get("cluster_elbow")
    if elbow:
        st.markdown("#### Elbow method")
        elbow_points = elbow.get("elbow_method", [])
        if elbow_points:
            elbow_df = pd.DataFrame(elbow_points, columns=["k", "inertia"])
            st.dataframe(elbow_df, use_container_width=True, hide_index=True)
            st.line_chart(elbow_df.set_index("k")["inertia"])
            st.info(f"Optimal k (from analysis): {elbow.get('optimal_k')}")


def render_fusion() -> None:
    st.header("Data Fusion")

    technique_map = {
        "early": "/fusion/early",
        "late": "/fusion/late",
        "weighted": "/fusion/weighted",
        "hybrid": "/fusion/hybrid",
    }

    selected = st.selectbox(
        "Run one fusion technique",
        list(technique_map.keys()),
        key="single_fusion_select",
    )

    f1, f2 = st.columns(2)
    with f1:
        if st.button("Compare all fusion techniques", key="fusion_compare_btn"):
            with st.spinner("Running complete fusion benchmark..."):
                payload = api_request(
                    "/fusion/compare",
                    method="POST",
                    payload={},
                    timeout=600,
                )
                if payload:
                    results = payload.get("fusion_results") if isinstance(payload, dict) else None
                    st.session_state["fusion_compare"] = results or []

    with f2:
        if st.button("Run selected fusion technique", key="fusion_single_btn"):
            with st.spinner(f"Running {selected} fusion..."):
                st.session_state["fusion_single"] = api_request(
                    technique_map[selected],
                    method="POST",
                    payload={},
                    timeout=420,
                )

    single = st.session_state.get("fusion_single")
    if single:
        st.markdown("#### Selected technique output")
        st.json(single)

    fusion_results: List[Dict[str, Any]] = st.session_state.get("fusion_compare") or []
    if fusion_results:
        st.markdown("#### Fusion comparison")
        fusion_df = pd.DataFrame(fusion_results)

        display_cols = [
            col
            for col in [
                "rank",
                "technique",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
            if col in fusion_df.columns
        ]
        st.dataframe(fusion_df[display_cols], use_container_width=True, hide_index=True)

        if "accuracy" in fusion_df.columns:
            st.bar_chart(fusion_df.set_index("technique")["accuracy"])
        if "roc_auc" in fusion_df.columns and fusion_df["roc_auc"].notna().any():
            st.bar_chart(fusion_df.set_index("technique")["roc_auc"])

        ranked = fusion_df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)
        best = ranked.iloc[0] if not ranked.empty else None
        baseline_rows = ranked[
            ranked["technique"].str.contains("baseline", case=False, na=False)
        ] if "technique" in ranked.columns else pd.DataFrame()

        colx, coly, colz = st.columns(3)
        with colx:
            if best is not None:
                st.metric("Best technique", best.get("technique", "-"), f"acc {best.get('accuracy', 0)}")

        with coly:
            avg_acc = float(ranked["accuracy"].mean()) if "accuracy" in ranked.columns else 0.0
            st.metric("Average accuracy", f"{avg_acc:.4f}")

        with colz:
            if best is not None and not baseline_rows.empty:
                gain = float(best.get("accuracy", 0)) - float(baseline_rows.iloc[0].get("accuracy", 0))
                st.metric("Gain vs baseline", f"{gain:+.4f}")
            else:
                st.metric("Gain vs baseline", "N/A")

        st.markdown("#### Technique details")
        for _, row in ranked.iterrows():
            technique = row.get("technique", "Unknown")
            with st.expander(str(technique), expanded=False):
                detail_cols = [
                    "concept",
                    "best_for",
                    "student_task",
                    "weights",
                    "average_attention",
                    "blend_weights",
                    "note",
                ]
                detail_payload = {key: row.get(key) for key in detail_cols if key in row and pd.notna(row.get(key))}
                if detail_payload:
                    st.json(detail_payload)


def render_pipeline() -> None:
    st.header("Run Complete Pipeline")

    if st.button("Execute all pipeline steps", key="pipeline_run_all"):
        with st.spinner("Running the full ADS pipeline..."):
            st.session_state["pipeline_result"] = api_request(
                "/pipeline/run-all",
                method="POST",
                payload={},
                timeout=900,
            )

    result = st.session_state.get("pipeline_result")
    if result:
        if result.get("status") == "success":
            st.success(result.get("message", "Pipeline completed."))
        else:
            st.warning("Pipeline returned without success status.")

        steps = result.get("steps", [])
        if steps:
            steps_df = pd.DataFrame(steps)
            st.dataframe(steps_df, use_container_width=True, hide_index=True)


def main() -> None:
    inject_styles()
    init_session_state()

    st.sidebar.title("ADS Lab Navigator")
    api_base = st.sidebar.text_input(
        "Backend API base URL",
        value=st.session_state.get("api_base_url", DEFAULT_API_BASE),
        help="Use the Flask API base path, for example http://localhost:5000/api",
    )
    st.session_state["api_base_url"] = api_base

    if st.sidebar.button("Health check", key="sidebar_health_check"):
        with st.spinner("Checking API health..."):
            st.session_state["health"] = api_request("/health")

    if st.session_state.get("health"):
        status = st.session_state["health"].get("status", "unknown")
        cache_size = st.session_state["health"].get("cache_size", "?")
        st.sidebar.success(f"API status: {status} | cache: {cache_size}")

    module = st.sidebar.radio(
        "Select module",
        [
            "Overview",
            "Data Loader",
            "Data Cleaning",
            "Imputation",
            "Preprocessing",
            "Model Training",
            "Clustering",
            "Data Fusion",
            "Pipeline Runner",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Run backend first: python api/app.py")
    st.sidebar.caption("Run this UI: streamlit run virtual-lab-ui/app.py")

    if module == "Overview":
        render_overview()
    elif module == "Data Loader":
        render_data_loader()
    elif module == "Data Cleaning":
        render_cleaning()
    elif module == "Imputation":
        render_imputation()
    elif module == "Preprocessing":
        render_preprocessing()
    elif module == "Model Training":
        render_models()
    elif module == "Clustering":
        render_clustering()
    elif module == "Data Fusion":
        render_fusion()
    elif module == "Pipeline Runner":
        render_pipeline()


if __name__ == "__main__":
    main()
