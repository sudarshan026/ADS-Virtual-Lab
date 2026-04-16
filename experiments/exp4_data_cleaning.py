"""Experiment 4 — Data Cleaning & Imputation.

The original experiment is a Flask API + HTML frontend, NOT Streamlit.
This wrapper provides the same Data Cleaning & Imputation workflow
as a native Streamlit interface using standard pandas / sklearn utilities.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
import warnings, os

warnings.filterwarnings("ignore")


def run():
    # ── Sidebar navigation ──────────────────────────────────────────────────
    st.sidebar.markdown("### 🧹 Lab Navigation")
    section = st.sidebar.radio(
        "Go to",
        ["Aim", "Theory", "Procedure", "Simulation", "Quiz", "References"],
        key="exp4_section",
    )

    # ── AIM ──────────────────────────────────────────────────────────────────
    if section == "Aim":
        st.header("🎯 Aim")
        st.markdown("""
        To understand and apply various **data cleaning** and **imputation** techniques
        on real-world datasets, and to measure how different strategies affect
        downstream model accuracy.

        ### Learning Outcomes
        - Identify and handle missing values, duplicates, and inconsistent entries
        - Compare imputation methods: **Mode**, **KNN**, and **MICE**
        - Understand how preprocessing quality impacts ML performance
        """)

    # ── THEORY ───────────────────────────────────────────────────────────────
    elif section == "Theory":
        st.header("📚 Theory")
        st.subheader("Why Data Cleaning Matters")
        st.markdown("""
        Raw data often contains **missing values**, **outliers**, **duplicates**, and
        **inconsistent formats**. Cleaning ensures that models train on reliable input.
        """)

        st.subheader("Imputation Strategies")
        cols = st.columns(3)
        with cols[0]:
            st.success("**Mode / Mean / Median**\n\nReplace missing values with the most frequent (categorical) or central (numerical) value. Fast but ignores feature relationships.")
        with cols[1]:
            st.info("**KNN Imputation**\n\nFinds the *k* nearest complete neighbours and averages their values. Captures local data structure but slower on large datasets.")
        with cols[2]:
            st.warning("**MICE (Iterative)**\n\nMultivariate imputation by chained equations. Models each feature as a function of others across multiple rounds. Most accurate but slowest.")

        st.subheader("Preprocessing Pipeline")
        st.markdown("""
        1. **Drop duplicates**
        2. **Handle missing values** (imputation)
        3. **Encode categoricals** (Label / One-Hot)
        4. **Scale numerics** (StandardScaler)
        5. **Split** into train / test sets
        """)

    # ── PROCEDURE ────────────────────────────────────────────────────────────
    elif section == "Procedure":
        st.header("⚙️ Procedure")
        st.markdown("""
        1. Upload a CSV dataset (or use the built-in Adult Census dataset)
        2. Explore dataset statistics and missing-value heatmap
        3. Choose an imputation method
        4. Run preprocessing and train a classifier
        5. Compare accuracy across imputation strategies
        """)

    # ── SIMULATION ───────────────────────────────────────────────────────────
    elif section == "Simulation":
        st.header("🔬 Interactive Simulation")

        # --- Data loading ---
        use_default = st.checkbox("Use built-in Adult Census dataset", value=True, key="exp4_default")
        df = None
        if use_default:
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "ADS_virtual_lab-main exp 4", "adult.csv",
            )
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                st.error(f"Default dataset not found at `{csv_path}`. Please upload a CSV.")
        else:
            up = st.file_uploader("Upload CSV", type=["csv"], key="exp4_upload")
            if up:
                df = pd.read_csv(up)

        if df is None:
            st.info("Load a dataset to begin.")
            return

        st.success(f"Loaded **{df.shape[0]}** rows × **{df.shape[1]}** columns")

        # --- Overview ---
        with st.expander("Dataset Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Cells", int(df.isnull().sum().sum()))
        c4.metric("Duplicates", int(df.duplicated().sum()))

        # --- Missing‑value heatmap ---
        st.subheader("Missing Value Heatmap")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            fig_m = px.bar(
                x=missing.index, y=missing.values,
                labels={"x": "Column", "y": "Missing Count"},
                title="Missing Values per Column",
            )
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("No missing values detected in this dataset.")

        # --- Cleaning ---
        st.subheader("Step 1 — Clean")
        remove_dups = st.checkbox("Remove duplicate rows", value=True, key="exp4_dup")
        if remove_dups:
            before = len(df)
            df = df.drop_duplicates()
            st.caption(f"Removed {before - len(df)} duplicate rows.")

        # Replace '?' with NaN (common in Adult dataset)
        df.replace("?", np.nan, inplace=True)
        df.replace(" ?", np.nan, inplace=True)

        # --- Imputation ---
        st.subheader("Step 2 — Impute Missing Values")
        method = st.selectbox("Imputation method", ["Mode / Median", "KNN", "MICE (Iterative)"], key="exp4_imp")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        df_imputed = df.copy()

        if method == "Mode / Median":
            for c in numeric_cols:
                df_imputed[c].fillna(df_imputed[c].median(), inplace=True)
            for c in cat_cols:
                mode_val = df_imputed[c].mode()
                if not mode_val.empty:
                    df_imputed[c].fillna(mode_val.iloc[0], inplace=True)
        elif method == "KNN":
            # Encode cats temporarily
            le_map = {}
            df_enc = df_imputed.copy()
            for c in cat_cols:
                le = LabelEncoder()
                mask = df_enc[c].notna()
                df_enc.loc[mask, c] = le.fit_transform(df_enc.loc[mask, c].astype(str))
                df_enc[c] = pd.to_numeric(df_enc[c], errors="coerce")
                le_map[c] = le
            imp = KNNImputer(n_neighbors=5)
            arr = imp.fit_transform(df_enc)
            df_imputed = pd.DataFrame(arr, columns=df.columns)
            for c in cat_cols:
                df_imputed[c] = df_imputed[c].round().astype(int)
                valid = df_imputed[c].clip(0, len(le_map[c].classes_) - 1).astype(int)
                df_imputed[c] = le_map[c].inverse_transform(valid)
        else:  # MICE
            le_map = {}
            df_enc = df_imputed.copy()
            for c in cat_cols:
                le = LabelEncoder()
                mask = df_enc[c].notna()
                df_enc.loc[mask, c] = le.fit_transform(df_enc.loc[mask, c].astype(str))
                df_enc[c] = pd.to_numeric(df_enc[c], errors="coerce")
                le_map[c] = le
            imp = IterativeImputer(max_iter=10, random_state=42)
            arr = imp.fit_transform(df_enc)
            df_imputed = pd.DataFrame(arr, columns=df.columns)
            for c in cat_cols:
                df_imputed[c] = df_imputed[c].round().astype(int)
                valid = df_imputed[c].clip(0, len(le_map[c].classes_) - 1).astype(int)
                df_imputed[c] = le_map[c].inverse_transform(valid)

        st.success(f"Imputation complete — **{int(df_imputed.isnull().sum().sum())}** missing cells remain.")

        # --- Preprocessing & Training ---
        st.subheader("Step 3 — Train & Evaluate")
        target = st.selectbox("Target column", df_imputed.columns.tolist(), index=len(df_imputed.columns) - 1, key="exp4_target")

        if st.button("Train Models", key="exp4_train"):
            with st.spinner("Preprocessing and training…"):
                X = df_imputed.drop(columns=[target])
                y = df_imputed[target]

                # Encode categoricals
                for c in X.select_dtypes(include=["object", "category"]).columns:
                    X[c] = LabelEncoder().fit_transform(X[c].astype(str))
                if y.dtype == "object":
                    y = LabelEncoder().fit_transform(y.astype(str))

                X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                results = {}
                for name, clf in [("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
                                  ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42))]:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    results[name] = acc

                r1, r2 = st.columns(2)
                for col, (mname, acc) in zip([r1, r2], results.items()):
                    col.metric(mname, f"{acc:.4f}")

                fig = px.bar(x=list(results.keys()), y=list(results.values()),
                             labels={"x": "Model", "y": "Accuracy"},
                             title=f"Model Accuracy (Imputation: {method})")
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

    # ── QUIZ ─────────────────────────────────────────────────────────────────
    elif section == "Quiz":
        st.header("🧠 Quiz")
        questions = [
            ("What does KNN imputation use to fill missing values?",
             ["Random numbers", "Column mean", "Nearest neighbours", "Zeros"], "Nearest neighbours"),
            ("MICE stands for?",
             ["Multiple Iterative Cleaning Equations", "Multivariate Imputation by Chained Equations",
              "Mean Imputation with Conditional Encoding", "None of the above"],
             "Multivariate Imputation by Chained Equations"),
            ("Removing duplicates helps prevent?",
             ["Overfitting", "Underfitting", "Feature scaling", "Dimensionality reduction"], "Overfitting"),
        ]
        score = 0
        for i, (q, opts, correct) in enumerate(questions):
            ans = st.radio(f"**Q{i+1}.** {q}", opts, key=f"exp4_q{i}")
            if ans == correct:
                score += 1
        if st.button("Submit Quiz", key="exp4_quiz_submit"):
            st.success(f"Score: {score}/{len(questions)}")

    # ── REFERENCES ──────────────────────────────────────────────────────────
    elif section == "References":
        st.header("📚 References")
        st.markdown("""
        - Buuren, S. van & Groothuis-Oudshoorn, K. (2011). *MICE: Multivariate Imputation by Chained Equations in R.*
        - Troyanskaya, O. et al. (2001). *Missing value estimation methods for DNA microarrays.* Bioinformatics.
        - scikit-learn documentation: https://scikit-learn.org/stable/modules/impute.html
        """)
