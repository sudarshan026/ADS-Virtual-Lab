import io
import random
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


st.set_page_config(
    page_title="Data Visualization Virtual Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


CHART_LEARNING_CONTENT = {
    "Histogram": {
        "when": "Use it to understand the distribution of a numerical variable.",
        "shows": "It shows how frequently values fall into different bins or ranges.",
        "real_world": "Useful for exam scores, customer ages, salaries, or delivery times.",
    },
    "Scatter": {
        "when": "Use it to study the relationship between two numerical variables.",
        "shows": "It reveals trends, clusters, and possible correlations.",
        "real_world": "Helpful for ad spend vs sales or height vs weight.",
    },
    "Bar": {
        "when": "Use it to compare values across categories.",
        "shows": "It highlights category-wise counts, sums, or averages.",
        "real_world": "Great for sales by product or students by department.",
    },
    "Line": {
        "when": "Use it for ordered or time-based data to track change over sequence.",
        "shows": "It shows trends, growth, decline, and fluctuations.",
        "real_world": "Useful for stock prices, monthly sales, or website visits.",
    },
    "Boxplot": {
        "when": "Use it to compare distributions and spot outliers.",
        "shows": "It summarizes median, quartiles, spread, and unusual values.",
        "real_world": "Helpful for comparing test scores across classes.",
    },
    "Heatmap": {
        "when": "Use it to inspect relationships in a matrix, especially correlations.",
        "shows": "It uses color intensity to show stronger or weaker values.",
        "real_world": "Useful for correlation analysis or activity frequency tables.",
    },
    "Pie": {
        "when": "Use it to show parts of a whole with a small number of categories.",
        "shows": "It displays each category's proportion of the total.",
        "real_world": "Helpful for market share or budget allocation.",
    },
}


def initialize_session_state() -> None:
    defaults = {
        "raw_df": None,
        "cleaned_df": None,
        "uploaded_file_name": None,
        "uploaded_signature": None,
        "last_chart_type": None,
        "quiz_questions": [],
        "quiz_answers": {},
        "quiz_feedback": {},
        "quiz_submitted": set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def data_loader(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            return None, "Unsupported file type. Please upload a CSV, XLSX, or JSON file."

        if df.empty:
            return None, "The uploaded file contains no rows."

        return df, None
    except ValueError as exc:
        return None, f"Unable to parse the file: {exc}"
    except Exception as exc:
        return None, f"Unexpected error while loading the file: {exc}"


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numerical_cols + datetime_cols]
    return {
        "Numerical": numerical_cols,
        "Categorical": categorical_cols,
        "Datetime": datetime_cols,
    }


def is_identifier_column(df: pd.DataFrame, col: str) -> bool:
    lowered = col.strip().lower()
    identifier_tokens = ["id", "transaction", "customer_id"]
    name_looks_like_id = any(token in lowered for token in identifier_tokens)
    unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
    return name_looks_like_id or unique_ratio > 0.9


def get_meaningful_numerical_columns(df: pd.DataFrame) -> List[str]:
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    meaningful_cols = []
    for col in numerical_cols:
        if is_identifier_column(df, col):
            continue
        if df[col].dropna().nunique() <= 1:
            continue
        if float(df[col].dropna().var()) == 0:
            continue
        meaningful_cols.append(col)
    return meaningful_cols


def get_correlation_pairs(df: pd.DataFrame, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
    meaningful_cols = get_meaningful_numerical_columns(df)
    if len(meaningful_cols) < 2:
        return []

    corr = df[meaningful_cols].corr(numeric_only=True)
    pairs: List[Tuple[str, str, float]] = []
    for i, col_a in enumerate(corr.columns):
        for j, col_b in enumerate(corr.columns):
            if j <= i:
                continue
            corr_value = float(corr.iloc[i, j])
            if abs(corr_value) > threshold:
                pairs.append((col_a, col_b, corr_value))

    return sorted(pairs, key=lambda item: abs(item[2]), reverse=True)


def safe_to_datetime(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce", cache=True)


def detect_and_convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    candidate_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in candidate_columns:
        sample = df[col].dropna().astype(str).str.strip().head(20)
        if sample.empty:
            continue

        date_like_ratio = sample.str.contains(r"\d", regex=True).mean()
        if date_like_ratio < 0.6:
            continue

        converted_sample = safe_to_datetime(sample)
        valid_ratio = converted_sample.notna().sum() / max(len(sample), 1)
        if valid_ratio >= 0.6:
            df[col] = safe_to_datetime(df[col])
    return df


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    types = detect_column_types(df)
    rows = []
    for col in df.columns:
        detected_type = (
            "Numerical"
            if col in types["Numerical"]
            else "Datetime"
            if col in types["Datetime"]
            else "Categorical"
        )
        rows.append(
            {
                "Column": col,
                "Pandas dtype": str(df[col].dtype),
                "Detected type": detected_type,
                "Missing values": int(df[col].isna().sum()),
                "Unique values": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def render_dataset_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset Snapshot")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{df.shape[0]:,}")
    metric_cols[1].metric("Columns", f"{df.shape[1]:,}")
    metric_cols[2].metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    metric_cols[3].metric("Duplicates", f"{int(df.duplicated().sum()):,}")

    preview_tab, columns_tab, types_tab = st.tabs(["Preview", "Columns", "Type Summary"])
    with preview_tab:
        st.dataframe(df.head(10), use_container_width=True)
    with columns_tab:
        st.write(df.columns.tolist())
    with types_tab:
        st.dataframe(column_summary(df), use_container_width=True)


def build_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Interactive Filters")
    filtered_df = df.copy()
    detected = detect_column_types(df)

    with st.sidebar.expander("Apply dataset filters", expanded=False):
        for col in detected["Categorical"]:
            unique_vals = sorted(df[col].dropna().astype(str).unique().tolist())
            if 1 < len(unique_vals) <= 100:
                selected = st.multiselect(
                    f"{col}",
                    options=unique_vals,
                    default=unique_vals,
                    key=f"filter_cat_{col}",
                )
                if selected:
                    filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

        for col in detected["Numerical"]:
            series = df[col].dropna()
            if series.empty:
                continue
            min_val = float(series.min())
            max_val = float(series.max())
            if min_val == max_val:
                continue
            selected_range = st.slider(
                f"{col}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=(float(min_val), float(max_val)),
                key=f"filter_num_{col}",
            )
            filtered_df = filtered_df[
                filtered_df[col].fillna(min_val).between(selected_range[0], selected_range[1])
            ]

    return filtered_df


def make_serializable_download(df: pd.DataFrame, file_format: str = "csv") -> Tuple[bytes, str, str]:
    if file_format == "csv":
        data = df.to_csv(index=False).encode("utf-8")
        return data, "cleaned_dataset.csv", "text/csv"

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="CleanedData")
    return (
        output.getvalue(),
        "cleaned_dataset.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def cleaning_module(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Data Cleaning Lab")
    original_df = df.copy()
    working_df = df.copy()

    with st.form("cleaning_form"):
        st.markdown("Use the controls below to transform the dataset safely.")
        col1, col2 = st.columns(2)

        with col1:
            remove_duplicates = st.checkbox("Remove duplicate rows", value=False)
            missing_strategy = st.selectbox(
                "Missing value strategy",
                options=["No change", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"],
            )

        with col2:
            convert_column = st.selectbox("Column to convert", options=["None"] + df.columns.tolist())
            convert_target = st.selectbox(
                "Convert selected column to",
                options=["No conversion", "string", "integer", "float", "datetime"],
            )

        apply_changes = st.form_submit_button("Apply Cleaning")

    if apply_changes:
        if remove_duplicates:
            working_df = working_df.drop_duplicates()

        if missing_strategy == "Drop rows":
            working_df = working_df.dropna()
        elif missing_strategy in {"Fill with mean", "Fill with median", "Fill with mode"}:
            for col in working_df.columns:
                if working_df[col].isna().sum() == 0:
                    continue
                try:
                    if missing_strategy == "Fill with mean" and pd.api.types.is_numeric_dtype(working_df[col]):
                        working_df[col] = working_df[col].fillna(working_df[col].mean())
                    elif missing_strategy == "Fill with median" and pd.api.types.is_numeric_dtype(working_df[col]):
                        working_df[col] = working_df[col].fillna(working_df[col].median())
                    else:
                        mode_series = working_df[col].mode(dropna=True)
                        if not mode_series.empty:
                            working_df[col] = working_df[col].fillna(mode_series.iloc[0])
                except Exception:
                    continue

        if convert_column != "None" and convert_target != "No conversion":
            try:
                if convert_target == "string":
                    working_df[convert_column] = working_df[convert_column].astype("string")
                elif convert_target == "integer":
                    working_df[convert_column] = pd.to_numeric(
                        working_df[convert_column], errors="coerce"
                    ).astype("Int64")
                elif convert_target == "float":
                    working_df[convert_column] = pd.to_numeric(working_df[convert_column], errors="coerce")
                elif convert_target == "datetime":
                    working_df[convert_column] = safe_to_datetime(working_df[convert_column])
            except Exception as exc:
                st.error(f"Column conversion failed: {exc}")

        st.session_state["cleaned_df"] = working_df
        st.success("Cleaning steps applied successfully.")

    current_df = st.session_state.get("cleaned_df") if st.session_state.get("cleaned_df") is not None else original_df

    before_tab, after_tab, compare_tab = st.tabs(["Before", "After", "Comparison"])
    with before_tab:
        st.dataframe(original_df.head(10), use_container_width=True)
        st.caption(f"Shape: {original_df.shape[0]} rows x {original_df.shape[1]} columns")

    with after_tab:
        st.dataframe(current_df.head(10), use_container_width=True)
        st.caption(f"Shape: {current_df.shape[0]} rows x {current_df.shape[1]} columns")

    with compare_tab:
        comparison_df = pd.DataFrame(
            {
                "Metric": ["Rows", "Columns", "Missing Cells", "Duplicates"],
                "Before": [
                    original_df.shape[0],
                    original_df.shape[1],
                    int(original_df.isna().sum().sum()),
                    int(original_df.duplicated().sum()),
                ],
                "After": [
                    current_df.shape[0],
                    current_df.shape[1],
                    int(current_df.isna().sum().sum()),
                    int(current_df.duplicated().sum()),
                ],
            }
        )
        st.dataframe(comparison_df, use_container_width=True)

    return current_df


def create_histogram(df: pd.DataFrame, x: str) -> go.Figure:
    return px.histogram(df, x=x, marginal="box", nbins=30, title=f"Histogram of {x}")


def create_boxplot(df: pd.DataFrame, x: Optional[str], y: str) -> go.Figure:
    title = f"Boxplot of {y}" if not x else f"Boxplot of {y} by {x}"
    return px.box(df, x=x, y=y, points="outliers", title=title)


def get_cardinality_profile(df: pd.DataFrame, col: str) -> Dict[str, object]:
    total_count = max(len(df), 1)
    unique_count = int(df[col].nunique(dropna=True))
    unique_ratio = unique_count / total_count
    is_high_cardinality = unique_ratio > 0.5 or unique_count > 50
    is_low_cardinality = unique_count <= 20 and not is_high_cardinality
    return {
        "unique_count": unique_count,
        "total_count": total_count,
        "unique_ratio": unique_ratio,
        "is_high_cardinality": is_high_cardinality,
        "is_low_cardinality": is_low_cardinality,
        "pie_allowed": unique_count <= 10 and not is_high_cardinality,
    }


def is_numeric_like_series(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    numeric_like_ratio = pd.to_numeric(non_null.astype(str), errors="coerce").notna().mean()
    return numeric_like_ratio >= 0.8


def get_top_category_counts(df: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    counts = df[col].astype(str).value_counts().nlargest(top_n).reset_index()
    counts.columns = [col, "Count"]
    return counts


def create_bar(df: pd.DataFrame, x: str, y: Optional[str], top_n: Optional[int] = None) -> go.Figure:
    if y and pd.api.types.is_numeric_dtype(df[y]):
        grouped = df.groupby(x, dropna=False)[y].mean().reset_index()
        return px.bar(grouped, x=x, y=y, title=f"Average {y} by {x}")
    counts = get_top_category_counts(df, x, top_n or 10)
    top_label = f"top {min(top_n or 10, len(counts))} categories"
    return px.bar(counts, x=x, y="Count", title=f"Count of {top_label} in {x}")


def create_pie(df: pd.DataFrame, x: str, top_n: int = 10) -> go.Figure:
    counts = get_top_category_counts(df, x, top_n)
    if len(counts) <= top_n:
        title = f"Share of categories in {x}"
    else:
        title = f"Share of top {top_n} categories in {x}"
    return px.pie(counts, names=x, values="Count", title=title)


def create_scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str]) -> go.Figure:
    return px.scatter(df, x=x, y=y, color=color, title=f"{x} vs {y}")


def create_line(df: pd.DataFrame, x: str, y: str, color: Optional[str]) -> go.Figure:
    plot_df = df.sort_values(by=x).copy()
    return px.line(plot_df, x=x, y=y, color=color, markers=True, title=f"{y} across {x}")


def create_heatmap(df: pd.DataFrame) -> go.Figure:
    meaningful_cols = get_meaningful_numerical_columns(df)
    corr = df[meaningful_cols].corr(numeric_only=True)
    return px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap")


def render_learning_content(chart_type: str) -> None:
    content = CHART_LEARNING_CONTENT.get(chart_type)
    if not content:
        return
    st.markdown("### Learning Mode")
    col1, col2, col3 = st.columns(3)
    col1.info(f"📌 When to use\n\n{content['when']}")
    col2.info(f"🔍 What it shows\n\n{content['shows']}")
    col3.info(f"🌍 Real-world use case\n\n{content['real_world']}")


def suggest_visualizations(df: pd.DataFrame) -> List[Dict[str, object]]:
    detected = detect_column_types(df)
    numerical = get_meaningful_numerical_columns(df)
    categorical = detected["Categorical"]
    suggestions: List[Dict[str, object]] = []

    if numerical:
        suggestions.append(
            {
                "chart": "Histogram",
                "x": numerical[0],
                "y": None,
                "note": "Distribution of a numerical column",
                "title": f"Histogram of {numerical[0]}",
            }
        )
        suggestions.append(
            {
                "chart": "Boxplot",
                "x": None,
                "y": numerical[0],
                "note": "Spread and outliers for a numerical column",
                "title": f"Boxplot of {numerical[0]}",
            }
        )
    elif detected["Numerical"]:
        suggestions.append(
            {
                "chart": "Skip",
                "x": None,
                "y": None,
                "note": "Numerical columns were excluded because they look like identifiers or constants",
                "warning": "Identifier-like numerical columns were skipped to avoid meaningless charts.",
            }
        )

    for col in categorical:
        profile = get_cardinality_profile(df, col)
        top_n_effective = min(10, int(profile["unique_count"]))
        if profile["is_low_cardinality"]:
            suggestions.append(
                {
                    "chart": "Bar",
                    "x": col,
                    "y": "Count",
                    "note": "Top 10 categories only" if profile["unique_count"] > 10 else "All categories shown",
                    "title": f"Count of top {top_n_effective} categories in {col}",
                    "top_n": 10,
                    "warning": "Too many unique values. Showing top 10 categories only."
                    if profile["unique_count"] > 10
                    else None,
                }
            )
            if profile["pie_allowed"]:
                suggestions.append(
                    {
                        "chart": "Pie",
                        "x": col,
                        "y": "Count",
                        "note": "All categories shown",
                        "title": f"Share of categories in {col}",
                        "top_n": 10,
                    }
                )
        elif is_numeric_like_series(df[col]):
            suggestions.append(
                {
                    "chart": "Histogram",
                    "x": col,
                    "y": None,
                    "note": "Column is stored as text but behaves like numeric data",
                    "title": f"Histogram of numeric-like values in {col}",
                    "transform": "numeric_like",
                    "warning": f"Column '{col}' has high cardinality. Using a histogram because the values are numeric-like.",
                }
            )
        else:
            suggestions.append(
                {
                    "chart": "Skip",
                    "x": col,
                    "y": None,
                    "note": "High-cardinality categorical column",
                    "warning": f"Column '{col}' has too many unique values for meaningful categorical visualization.",
                }
            )

    correlation_pairs = get_correlation_pairs(df, threshold=0.3)
    if correlation_pairs:
        best_x, best_y, best_corr = correlation_pairs[0]
        suggestions.append(
            {
                "chart": "Scatter",
                "x": best_x,
                "y": best_y,
                "note": f"Top correlated numerical pair selected automatically (correlation {best_corr:.2f})",
                "title": f"{best_x} vs {best_y}",
                "correlation": best_corr,
            }
        )
    elif len(numerical) >= 2:
        suggestions.append(
            {
                "chart": "Skip",
                "x": numerical[0],
                "y": numerical[1],
                "note": "No strong numerical relationship available for scatter plotting",
                "warning": "No meaningful relationship found between selected columns.",
                "title": "Scatter plot skipped: No meaningful relationship detected",
            }
        )

    if len(numerical) >= 2:
        suggestions.append(
            {
                "chart": "Heatmap",
                "x": None,
                "y": None,
                "note": "Correlation summary across non-identifier numerical columns",
                "title": "Correlation Heatmap",
            }
        )
    return suggestions


def plot_visualization(df: pd.DataFrame, suggestion: Dict[str, object]) -> Optional[go.Figure]:
    chart_type = suggestion.get("chart")
    x_axis = suggestion.get("x")
    y_axis = suggestion.get("y")
    top_n = int(suggestion.get("top_n", 10))
    warning_message = suggestion.get("warning")

    if warning_message:
        st.warning(str(warning_message))

    if chart_type == "Skip":
        if suggestion.get("title"):
            st.info(str(suggestion["title"]))
        return None

    if chart_type == "Histogram" and x_axis:
        plot_df = df.copy()
        if suggestion.get("transform") == "numeric_like":
            plot_df = plot_df.copy()
            plot_df[x_axis] = pd.to_numeric(plot_df[x_axis], errors="coerce")
            plot_df = plot_df.dropna(subset=[x_axis])
        return px.histogram(plot_df, x=x_axis, marginal="box", nbins=30, title=str(suggestion.get("title")))

    if chart_type == "Boxplot" and y_axis:
        return create_boxplot(df, x_axis if isinstance(x_axis, str) else None, str(y_axis))

    if chart_type == "Scatter" and x_axis and y_axis:
        correlation = suggestion.get("correlation")
        if correlation is None:
            correlation = df[[str(x_axis), str(y_axis)]].corr(numeric_only=True).iloc[0, 1]
        if pd.isna(correlation) or abs(float(correlation)) < 0.1:
            st.info("Scatter plot skipped: No meaningful relationship detected")
            st.warning("No meaningful relationship found between selected columns")
            return None
        if is_identifier_column(df, str(x_axis)) or is_identifier_column(df, str(y_axis)):
            st.info("Scatter plot skipped: Identifier columns do not provide meaningful analytical relationships")
            return None
        return create_scatter(df, str(x_axis), str(y_axis), None)

    if chart_type == "Heatmap":
        if len(get_meaningful_numerical_columns(df)) < 2:
            st.info("Correlation heatmap skipped: not enough non-identifier numerical columns.")
            return None
        return create_heatmap(df)

    if chart_type == "Bar" and x_axis:
        return create_bar(df, str(x_axis), str(y_axis) if y_axis not in [None, "Count"] else None, top_n=top_n)

    if chart_type == "Pie" and x_axis:
        profile = get_cardinality_profile(df, str(x_axis))
        if not profile["pie_allowed"]:
            st.warning(f"Column '{x_axis}' has too many unique values for a pie chart. Pie charts are limited to 10 categories.")
            return None
        return create_pie(df, str(x_axis), top_n=top_n)

    return None


@st.cache_data(show_spinner=False)
def generate_png_bytes_from_json(fig_json: str) -> bytes:
    fig = pio.from_json(fig_json)
    return fig.to_image(format="png")


def render_plot_download(fig: go.Figure, key_suffix: str) -> None:
    png_state_key = f"png_bytes_{key_suffix}"
    error_state_key = f"png_error_{key_suffix}"
    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")

    button_col, download_col = st.columns(2)

    if button_col.button("Generate PNG", key=f"generate_png_{key_suffix}"):
        try:
            st.session_state[png_state_key] = generate_png_bytes_from_json(fig.to_json())
            st.session_state[error_state_key] = None
        except Exception:
            st.session_state[png_state_key] = None
            st.session_state[error_state_key] = "PNG export failed. Try again or install compatible setup."

    if st.session_state.get(error_state_key):
        st.warning(st.session_state[error_state_key])

    if st.session_state.get(png_state_key):
        download_col.download_button(
            label="Download PNG",
            data=st.session_state[png_state_key],
            file_name=f"plot_{key_suffix}.png",
            mime="image/png",
            key=f"plot_download_{key_suffix}",
        )

    st.download_button(
        label="Download Interactive HTML",
        data=html_bytes,
        file_name=f"plot_{key_suffix}.html",
        mime="text/html",
        key=f"plot_html_{key_suffix}",
    )


def build_figure(
    df: pd.DataFrame,
    chart_type: str,
    x_axis: Optional[str],
    y_axis: Optional[str],
    color_col: Optional[str],
) -> Optional[go.Figure]:
    if df.empty:
        return None
    try:
        if chart_type == "Histogram" and x_axis:
            return create_histogram(df, x_axis)
        if chart_type == "Boxplot" and y_axis:
            return create_boxplot(df, x_axis, y_axis)
        if chart_type == "Scatter" and x_axis and y_axis:
            return create_scatter(df, x_axis, y_axis, color_col)
        if chart_type == "Bar" and x_axis:
            top_n = 10 if x_axis in detect_column_types(df)["Categorical"] else None
            return create_bar(df, x_axis, y_axis, top_n=top_n)
        if chart_type == "Pie" and x_axis:
            profile = get_cardinality_profile(df, x_axis)
            if not profile["pie_allowed"]:
                st.warning(f"Column '{x_axis}' has too many unique values for a pie chart. Pie charts are limited to 10 categories.")
                return None
            return create_pie(df, x_axis, top_n=10)
        if chart_type == "Line" and x_axis and y_axis:
            return create_line(df, x_axis, y_axis, color_col)
        if chart_type == "Heatmap" and len(df.select_dtypes(include=["number"]).columns) >= 2:
            if len(get_meaningful_numerical_columns(df)) < 2:
                st.info("Correlation heatmap skipped: not enough non-identifier numerical columns.")
                return None
            return create_heatmap(df)
    except Exception as exc:
        st.error(f"Could not create the chart: {exc}")
    return None


def visualization_module(df: pd.DataFrame) -> None:
    st.subheader("Visualization Engine")
    detected = detect_column_types(df)
    chart_types = ["Line", "Bar", "Scatter", "Histogram", "Boxplot", "Heatmap", "Pie"]

    auto_tab, manual_tab = st.tabs(["Auto Mode", "Manual Mode"])

    with auto_tab:
        st.markdown("Smart suggestions based on detected dataset structure.")
        suggestions = suggest_visualizations(df)
        if not suggestions:
            st.warning("No visualization suggestions are available for this dataset yet.")
        limited_suggestions = suggestions[:5]
        if len(suggestions) > 5:
            st.info("Auto Mode is limited to the first 5 insight-driven plots for better performance.")
        for index, suggestion in enumerate(limited_suggestions):
            label = suggestion["chart"] if suggestion["chart"] != "Skip" else "Warning"
            with st.expander(
                f"{label} suggestion {index + 1}: {suggestion['note']}",
                expanded=index == 0,
            ):
                suggestion_payload = {
                    "Chart": suggestion["chart"],
                    "X-axis": suggestion["x"],
                    "Y-axis": suggestion["y"],
                    "Note": suggestion["note"],
                }
                if suggestion.get("warning"):
                    suggestion_payload["Warning"] = suggestion["warning"]
                st.json(suggestion_payload)

                fig = plot_visualization(df, suggestion)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state["last_chart_type"] = suggestion["chart"]
                    render_learning_content(str(suggestion["chart"]))
                    render_plot_download(fig, f"auto_{index}")
                elif suggestion["chart"] == "Skip":
                    st.info("Skipping this column in Auto Mode because the categorical cardinality is too high.")

    with manual_tab:
        st.markdown("Take control and design your own chart.")
        selection_cols = st.columns(4)
        x_options = ["None"] + df.columns.tolist()
        y_options = ["None"] + df.columns.tolist()
        color_options = ["None"] + detected["Categorical"]
        x_axis_raw = selection_cols[1].selectbox("X-axis", x_options)
        y_axis = selection_cols[2].selectbox("Y-axis", y_options)
        color_col = selection_cols[3].selectbox("Color Group", color_options)

        available_chart_types = chart_types.copy()
        if x_axis_raw != "None":
            x_profile = get_cardinality_profile(df, x_axis_raw) if x_axis_raw in detected["Categorical"] else None
            if x_profile and not x_profile["pie_allowed"] and "Pie" in available_chart_types:
                available_chart_types.remove("Pie")
                st.info("Pie chart is disabled for this column because it has more than 10 categories.")

        chart_type = selection_cols[0].selectbox("Plot Type", available_chart_types)

        x_axis = None if x_axis_raw == "None" else x_axis_raw
        y_axis = None if y_axis == "None" else y_axis
        color_col = None if color_col == "None" else color_col

        if chart_type == "Bar" and x_axis in detected["Categorical"]:
            profile = get_cardinality_profile(df, x_axis)
            if profile["is_high_cardinality"]:
                st.warning("Too many unique values. Showing top 10 categories only.")

        fig = build_figure(df, chart_type, x_axis, y_axis, color_col)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            st.session_state["last_chart_type"] = chart_type
            render_learning_content(chart_type)
            render_plot_download(fig, "manual")
        else:
            st.info("Choose chart inputs that match the selected chart type.")


def learning_module(selected_chart_type: Optional[str]) -> None:
    st.subheader("Concept Learning Zone")
    all_chart_types = list(CHART_LEARNING_CONTENT.keys())
    default_index = all_chart_types.index(selected_chart_type) if selected_chart_type in all_chart_types else 0
    chosen_chart = st.selectbox("Select a chart to learn about", all_chart_types, index=default_index)
    render_learning_content(chosen_chart)

    with st.expander("Teaching tip", expanded=True):
        st.write(
            "Try viewing a chart in the Visualization section first. Then come back here to connect the visual pattern with the concept behind it."
        )


def generate_dataset_questions(df: pd.DataFrame, difficulty: str) -> List[Dict[str, object]]:
    detected = detect_column_types(df)
    numerical = detected["Numerical"]
    categorical = detected["Categorical"]
    datetime_cols = detected["Datetime"]

    question_bank: List[Dict[str, object]] = []

    if categorical:
        question_bank.append(
            {
                "question": f"Which chart is best to compare values across categories like `{categorical[0]}`?",
                "options": ["Scatter Plot", "Histogram", "Bar Chart", "Line Chart"],
                "answer": "Bar Chart",
                "explanation": "Bar charts compare values or frequencies across categories clearly.",
            }
        )
        question_bank.append(
            {
                "question": f"What chart is best to show the proportion of records by `{categorical[0]}`?",
                "options": ["Pie Chart", "Scatter Plot", "Heatmap", "Histogram"],
                "answer": "Pie Chart",
                "explanation": "Pie charts show how each category contributes to the whole when categories are few.",
            }
        )

    if numerical:
        question_bank.append(
            {
                "question": f"Which chart is best to study the distribution of `{numerical[0]}`?",
                "options": ["Histogram", "Line Chart", "Pie Chart", "Scatter Plot"],
                "answer": "Histogram",
                "explanation": "Histograms reveal how numeric values are distributed across ranges.",
            }
        )
        question_bank.append(
            {
                "question": f"What chart helps detect outliers in `{numerical[0]}`?",
                "options": ["Pie Chart", "Boxplot", "Bar Chart", "Heatmap"],
                "answer": "Boxplot",
                "explanation": "Boxplots summarize spread and make unusually high or low values easier to spot.",
            }
        )

    if len(numerical) >= 2:
        question_bank.append(
            {
                "question": f"Which chart is best to inspect the relationship between `{numerical[0]}` and `{numerical[1]}`?",
                "options": ["Bar Chart", "Histogram", "Scatter Plot", "Pie Chart"],
                "answer": "Scatter Plot",
                "explanation": "Scatter plots help us see correlation, clusters, and trends between two numeric variables.",
            }
        )

    if datetime_cols and numerical:
        question_bank.append(
            {
                "question": f"Which chart is usually best to show how `{numerical[0]}` changes over `{datetime_cols[0]}`?",
                "options": ["Line Chart", "Pie Chart", "Histogram", "Boxplot"],
                "answer": "Line Chart",
                "explanation": "Line charts are ideal for showing trends over time or other ordered sequences.",
            }
        )

    question_bank.extend(
        [
            {
                "question": "Which visualization is best for showing a correlation matrix?",
                "options": ["Heatmap", "Pie Chart", "Line Chart", "Histogram"],
                "answer": "Heatmap",
                "explanation": "Heatmaps use color intensity to summarize matrix values such as correlations.",
            },
            {
                "question": "Which chart is least appropriate for comparing exact values across many categories?",
                "options": ["Bar Chart", "Pie Chart", "Table", "Grouped Bar Chart"],
                "answer": "Pie Chart",
                "explanation": "Pie charts are weaker when you need precise comparison across many categories.",
            },
        ]
    )

    sample_size = 4 if difficulty == "Easy" else 6 if difficulty == "Medium" else 8
    random.shuffle(question_bank)
    return question_bank[: min(sample_size, len(question_bank))]


def quiz_module(df: pd.DataFrame) -> None:
    st.subheader("Quiz Mode")
    difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")

    if st.button("Generate New Quiz"):
        st.session_state["quiz_questions"] = generate_dataset_questions(df, difficulty)
        st.session_state["quiz_answers"] = {}
        st.session_state["quiz_feedback"] = {}
        st.session_state["quiz_submitted"] = set()

    if not st.session_state["quiz_questions"]:
        st.session_state["quiz_questions"] = generate_dataset_questions(df, difficulty)

    questions = st.session_state["quiz_questions"]

    for idx, q in enumerate(questions):
        st.markdown(f"**Q{idx + 1}. {q['question']}**")
        answer = st.radio("Choose one answer", q["options"], key=f"quiz_q_{idx}", index=None)

        if st.button(f"Submit Q{idx + 1}", key=f"submit_q_{idx}"):
            if answer is None:
                st.warning("Please choose an answer first.")
            else:
                is_correct = answer == q["answer"]
                st.session_state["quiz_answers"][idx] = answer
                st.session_state["quiz_feedback"][idx] = is_correct
                st.session_state["quiz_submitted"].add(idx)

        if idx in st.session_state["quiz_submitted"]:
            is_correct = st.session_state["quiz_feedback"][idx]
            chosen = st.session_state["quiz_answers"][idx]
            if is_correct:
                st.success(f"Correct. You chose `{chosen}`.")
            else:
                st.error(f"Incorrect. You chose `{chosen}`. Correct answer: `{q['answer']}`.")
            st.caption(q["explanation"])

        st.divider()

    answered_count = len(st.session_state["quiz_submitted"])
    if answered_count:
        final_correct = sum(1 for value in st.session_state["quiz_feedback"].values() if value)
        st.metric("Score", f"{final_correct}/{answered_count}")
        if answered_count == len(questions):
            st.info("Quiz complete. Generate a new quiz to practice again.")


def describe_missing_values(missing_total: int, mode: str) -> str:
    if mode == "Technical Insights":
        return f"⚠️ The dataset contains {missing_total} missing values that may affect analysis quality."
    return f"⚠️ Some information is missing in {missing_total} cells, so a few results may be incomplete or less reliable."


def describe_duplicates(duplicates: int, mode: str) -> str:
    if mode == "Technical Insights":
        return f"⚠️ There are {duplicates} duplicate rows that may need removal."
    return f"⚠️ {duplicates} rows look repeated, which may count the same event more than once."


def describe_distribution(col: str, skew_value: float, mode: str) -> str:
    if mode == "Technical Insights":
        direction = "right-skewed" if skew_value > 1 else "left-skewed"
        return f"📊 `{col}` is {direction} with skewness {skew_value:.2f}."
    if skew_value > 1:
        return f"📊 Most {col} values are small, but a few much larger values stand out and may drive totals, revenue, or averages."
    return f"📊 Most {col} values are on the higher side, while a smaller group of low values stands out and may need separate attention."


def describe_correlation(col_a: str, col_b: str, corr_value: float, mode: str) -> str:
    if mode == "Technical Insights":
        return f"📈 `{col_a}` and `{col_b}` have a correlation of {corr_value:.2f}."
    if corr_value >= 0.7:
        return f"📈 As {col_a} increases, {col_b} also tends to increase strongly, so improving one may raise the other."
    if corr_value <= -0.7:
        return f"📈 When {col_a} goes up, {col_b} usually goes down, which suggests a strong trade-off between them."
    if corr_value > 0:
        return f"💡 {col_a} and {col_b} move upward together, which may help predict one from the other."
    return f"💡 {col_a} rises when {col_b} falls, which may reveal a useful balancing effect."


def describe_dominant_category(col: str, top_category: str, top_share: float, mode: str) -> str:
    if mode == "Technical Insights":
        return f"💡 `{col}` is dominated by `{top_category}`, representing {top_share:.1%} of the non-null values."
    return (
        f"💡 Most records fall under {top_category} in {col}, so decisions and campaigns should focus on this group first."
    )


def generate_insights(df: pd.DataFrame, mode: str = "Simple Insights") -> Tuple[List[str], str]:
    insights: List[str] = []
    detected = detect_column_types(df)
    meaningful_numerical = get_meaningful_numerical_columns(df)
    missing_total = int(df.isna().sum().sum())

    if missing_total > 0:
        insights.append(describe_missing_values(missing_total, mode))

    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        insights.append(describe_duplicates(duplicates, mode))

    for col in meaningful_numerical:
        series = df[col].dropna()
        if len(series) < 3:
            continue
        skew_value = float(series.skew())
        if abs(skew_value) > 1:
            insights.append(describe_distribution(col, skew_value, mode))

    correlation_pairs = get_correlation_pairs(df, threshold=0.7)
    if correlation_pairs:
        top_pair = correlation_pairs[0]
        insights.append(describe_correlation(top_pair[0], top_pair[1], top_pair[2], mode))

    for col in detected["Categorical"]:
        top_counts = df[col].astype(str).value_counts(dropna=True)
        if not top_counts.empty:
            top_category = top_counts.index[0]
            top_share = top_counts.iloc[0] / top_counts.sum()
            if top_share >= 0.5:
                insights.append(describe_dominant_category(col, top_category, top_share, mode))

    if not insights:
        if mode == "Technical Insights":
            insights.append("💡 No major statistical warnings were detected in the current dataset view.")
        else:
            insights.append("💡 The data looks fairly balanced overall, with no major warning signs in the current view.")

    if mode == "Technical Insights":
        summary = "This dataset includes "
        summary += f"{len(meaningful_numerical)} meaningful numerical column(s) and {len(detected['Categorical'])} categorical column(s). "
        summary += "The insights below summarize quality checks, distribution patterns, and strong relationships."
    else:
        summary = "This dataset gives a quick picture of what stands out, where the data may need cleanup, and which patterns could matter for decisions."

    return insights[:6], summary


def insights_module(df: pd.DataFrame) -> None:
    st.subheader("Insight Generator")
    mode = st.radio(
        "Insight Mode",
        options=["Simple Insights", "Technical Insights"],
        horizontal=True,
    )
    insights, summary = generate_insights(df, mode=mode)
    st.markdown("### Auto Summary")
    st.write(summary)
    st.markdown("### Insight Cards")
    for insight in insights:
        st.info(insight)


def build_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    summary_frames: List[pd.DataFrame] = []

    numeric_cols = df.select_dtypes(include=["number"])
    categorical_cols = df.select_dtypes(include=["object", "category", "string"])
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"])

    if not numeric_cols.empty:
        numeric_summary = numeric_cols.describe().transpose().reset_index().rename(columns={"index": "Column"})
        numeric_summary.insert(1, "Type", "Numerical")
        summary_frames.append(numeric_summary)

    if not categorical_cols.empty:
        categorical_summary = []
        for col in categorical_cols.columns:
            mode_values = categorical_cols[col].mode(dropna=True)
            top_value = mode_values.iloc[0] if not mode_values.empty else None
            freq_value = int((categorical_cols[col] == top_value).sum()) if top_value is not None else 0
            categorical_summary.append(
                {
                    "Column": col,
                    "Type": "Categorical",
                    "count": int(categorical_cols[col].count()),
                    "unique": int(categorical_cols[col].nunique(dropna=True)),
                    "top": top_value,
                    "freq": freq_value,
                }
            )
        summary_frames.append(pd.DataFrame(categorical_summary))

    if not datetime_cols.empty:
        datetime_summary = []
        for col in datetime_cols.columns:
            datetime_summary.append(
                {
                    "Column": col,
                    "Type": "Datetime",
                    "count": int(datetime_cols[col].count()),
                    "min": datetime_cols[col].min(),
                    "max": datetime_cols[col].max(),
                }
            )
        summary_frames.append(pd.DataFrame(datetime_summary))

    if not summary_frames:
        return pd.DataFrame(columns=["Column", "Type"])

    summary_df = pd.concat(summary_frames, ignore_index=True, sort=False)
    summary_df = summary_df.where(pd.notnull(summary_df), None)

    for col in summary_df.columns:
        if summary_df[col].dtype == "object":
            summary_df[col] = summary_df[col].astype(str)

    return summary_df


def eda_section(df: pd.DataFrame) -> None:
    st.subheader("Exploratory Data Analysis")
    summary_tab, distribution_tab, correlation_tab = st.tabs(
        ["Summary Statistics", "Distribution Analysis", "Correlation Matrix"]
    )

    with summary_tab:
        st.dataframe(build_summary_statistics(df), use_container_width=True)

    with distribution_tab:
        numerical_cols = get_meaningful_numerical_columns(df)
        if numerical_cols:
            selected_col = st.selectbox("Choose a numerical column", numerical_cols, key="eda_dist_col")
            fig = create_histogram(df, selected_col)
            st.plotly_chart(fig, use_container_width=True)
            render_learning_content("Histogram")
        else:
            st.warning("No numerical columns are available for distribution analysis.")

    with correlation_tab:
        numerical_cols = get_meaningful_numerical_columns(df)
        if len(numerical_cols) >= 2:
            fig = create_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
            render_learning_content("Heatmap")
        else:
            st.warning("At least two numerical columns are needed for a correlation matrix.")


def render_upload_section() -> None:
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "json"])

    if uploaded_file is None:
        st.info("Upload a dataset to start exploring the Virtual Data Visualization Lab.")
        return

    uploaded_signature = f"{uploaded_file.name}:{uploaded_file.size}"
    df, error = data_loader(uploaded_file)
    if error:
        st.error(error)
        return

    df = detect_and_convert_datetime(df)
    if st.session_state.get("uploaded_signature") != uploaded_signature:
        st.session_state["cleaned_df"] = df.copy()
        st.session_state["quiz_questions"] = []
        st.session_state["quiz_answers"] = {}
        st.session_state["quiz_feedback"] = {}
        st.session_state["quiz_submitted"] = set()

    st.session_state["raw_df"] = df
    st.session_state["uploaded_file_name"] = uploaded_file.name
    st.session_state["uploaded_signature"] = uploaded_signature

    st.success(f"Loaded `{uploaded_file.name}` successfully.")
    render_dataset_overview(df)


def render_top_banner() -> None:
    st.markdown(
        """
        <style>
            .hero {
                padding: 1.2rem 1.4rem;
                border-radius: 18px;
                background: linear-gradient(135deg, #0f766e 0%, #164e63 50%, #1d4ed8 100%);
                color: white;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(2, 6, 23, 0.18);
            }
            .hero h1 {
                margin: 0;
                font-size: 2.1rem;
            }
            .hero p {
                margin: 0.45rem 0 0 0;
                font-size: 1rem;
                opacity: 0.95;
            }
            .section-note {
                padding: 0.8rem 1rem;
                border-radius: 12px;
                background: rgba(15, 118, 110, 0.08);
                border-left: 4px solid #0f766e;
                margin-bottom: 1rem;
            }
        </style>
        <div class="hero">
            <h1>Data Visualization Virtual Lab</h1>
            <p>Upload data, clean it, visualize it, learn why charts work, and test your understanding in one interactive lab.</p>
            <p><strong>Built by Team ADS | Computer Engineering</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👨‍💻 Built by")
    st.sidebar.markdown(
        """
        - Rushikesh Shembade (D12C/53)  
        - Aditya Upasani (D12C/62)  
        - Raziq Sarwar Mukadam (D12C/49)
        """
    )


def render_app_footer() -> None:
    st.markdown(
        """
        <hr style="margin-top:50px;">
        <center>
        © 2026 Data Visualization Virtual Lab
        </center>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    initialize_session_state()
    render_top_banner()

    st.sidebar.title("Lab Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Upload Data", "Overview", "Cleaning", "Visualization", "Learning Mode", "Quiz Mode", "Insights"],
    )
    render_sidebar_footer()

    if page == "Upload Data":
        render_upload_section()

    raw_df = st.session_state.get("raw_df")
    if raw_df is None and page != "Upload Data":
        st.warning("Please upload a dataset first from the Upload Data section.")
        return

    active_df = st.session_state.get("cleaned_df") if st.session_state.get("cleaned_df") is not None else raw_df
    filtered_df = build_sidebar_filters(active_df) if active_df is not None else None

    if filtered_df is not None:
        st.markdown(
            f"<div class='section-note'>Active view: <strong>{filtered_df.shape[0]:,}</strong> rows and <strong>{filtered_df.shape[1]:,}</strong> columns after cleaning and filters.</div>",
            unsafe_allow_html=True,
        )

    if page == "Overview":
        render_dataset_overview(filtered_df)
        eda_section(filtered_df)
        csv_bytes, csv_name, csv_mime = make_serializable_download(active_df, "csv")
        xlsx_bytes, xlsx_name, xlsx_mime = make_serializable_download(active_df, "xlsx")
        col1, col2 = st.columns(2)
        col1.download_button("Download Cleaned CSV", data=csv_bytes, file_name=csv_name, mime=csv_mime)
        col2.download_button("Download Cleaned Excel", data=xlsx_bytes, file_name=xlsx_name, mime=xlsx_mime)
    elif page == "Cleaning":
        cleaned = cleaning_module(raw_df)
        st.markdown("### Filtered Preview After Cleaning")
        st.dataframe(cleaned.head(10), use_container_width=True)
    elif page == "Visualization":
        visualization_module(filtered_df)
    elif page == "Learning Mode":
        learning_module(st.session_state.get("last_chart_type"))
    elif page == "Quiz Mode":
        quiz_module(filtered_df)
    elif page == "Insights":
        insights_module(filtered_df)

    render_app_footer()


if __name__ == "__main__":
    main()
