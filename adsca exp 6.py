
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Virtual Labs | Outlier Detection", layout="wide")


def apply_dark_plot_theme(fig, y_title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0f172a",
        font=dict(color="#e5e7eb"),
        title=dict(font=dict(color="#f8fafc"), x=0.02),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#475569",
            borderwidth=1,
            font=dict(color="#f8fafc"),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zerolinecolor="rgba(148, 163, 184, 0.28)",
        linecolor="#94a3b8",
        tickfont=dict(color="#e5e7eb"),
        title_font=dict(color="#e5e7eb"),
    )
    fig.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zerolinecolor="rgba(148, 163, 184, 0.28)",
        linecolor="#94a3b8",
        tickfont=dict(color="#e5e7eb"),
        title_font=dict(color="#e5e7eb"),
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.7, color="#e5e7eb")), opacity=0.9)
    return fig

# --- Forced Light Theme CSS & Virtual Labs Branding ---
st.markdown("""
    
    """, unsafe_allow_html=True)

# --- Top Header Section ---
st.markdown("""
    <div class="vlab-header">
        <div class="vlab-logo">Virtual Labs<br><span style="font-size:12px; color:#7f8c8d !important; font-weight:normal;">An MoE Govt of India Initiative</span></div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <span style="color: #f1c40f;">⭐⭐⭐⭐⭐</span>
            <button style="background-color:#3498db; color:white; border:none; padding:8px 15px; border-radius:5px; cursor:pointer;">Rate Me</button>
            <button style="background-color:#3498db; color:white; border:none; padding:8px 15px; border-radius:5px; cursor:pointer;">Report a Bug</button>
        </div>
    </div>
    <div class="breadcrumb">Computer Science and Engineering > Data Science > Outlier Detection Lab</div>
    """, unsafe_allow_html=True)

# --- QUESTION BANK ---
QUIZ_BANK = {
    "Pretest": {
        "Easy": [
            ("What is an outlier?", ["A typical data point", "An extreme observation", "The average value"], 1),
            ("Which measure is most sensitive to a single outlier?", ["Median", "Mode", "Mean"], 2),
            ("True or False: Outliers are always errors.", ["True", "False"], 1)
        ],
        "Medium": [
            ("The standard Z-score threshold for outliers is:", ["1", "3", "10"], 1),
            ("How is IQR calculated?", ["Q3 - Q1", "Mean - Median", "Max - Min"], 0),
            ("k-NN detects outliers based on:", ["Path length", "Distance to neighbors", "Tree splits"], 1)
        ],
        "Hard": [
            ("Why is Z-score 'non-robust'?", ["It is slow", "Outliers inflate Mean and SD", "It only works for integers"], 1),
            ("Isolation Forest works better than k-NN for:", ["Small datasets", "High-dimensional data", "Normally distributed data"], 1),
            ("In k-NN, if a point's average distance to $k$ neighbors is very high, it is:", ["A cluster center", "An outlier", "A median point"], 1)
        ]
    },
    "Posttest": {
        "Easy": [
            ("In a box plot, points beyond whiskers are:", ["Means", "Outliers", "Quartiles"], 1),
            ("Removing outliers usually makes variance:", ["Smaller", "Larger", "Infinite"], 0),
            ("Which plot is best for spotting outliers visually?", ["Pie Chart", "Box Plot", "Stacked Bar"], 1)
        ],
        "Medium": [
            ("The 'Lower Fence' in IQR is:", ["Q1 - 1.5*IQR", "Q3 - 1.5*IQR", "Mean - 2*SD"], 0),
            ("Z-score assumes the data follows a:", ["Poisson Dist", "Normal Dist", "Uniform Dist"], 1),
            ("If $k=1$ in k-NN, detection is most sensitive to:", ["Clusters", "Local Noise", "Global Trends"], 1)
        ],
        "Hard": [
            ("Isolation Forest isolates anomalies because they:", ["Are frequent", "Require fewer splits to isolate", "Are near the mean"], 1),
            ("±3 Z-scores in a Normal distribution cover roughly:", ["95%", "68%", "99.7%"], 2),
            ("k-NN outlier detection is computationally expensive because:", ["It builds trees", "It calculates all pair-wise distances", "It uses many GPUs"], 1)
        ]
    }
}

# --- Sidebar Navigation ---
st.sidebar.markdown("### Menu")
menu = st.sidebar.radio("Navigation", 
    ["Aim", "Theory", "Pretest", "Demo (Upload Data)", "Posttest", "Feedback"],
    label_visibility="collapsed")

# --- 1. AIM SECTION ---
if menu == "Aim":
    st.markdown('<h1 class="lab-title">Aim</h1>', unsafe_allow_html=True)
    st.markdown("""
    ### Objective
    To understand and implement various statistical and machine learning techniques for detecting anomalies in numerical datasets.
    
    ### Learning Outcomes
    * **Define** univariate and multivariate outliers.
    * **Calculate** thresholds using Z-Score and Interquartile Range (IQR).
    * **Implement** Distance-based (k-NN) and Tree-based (Isolation Forest) detection.
    * **Analyze** the strengths and weaknesses of each method.
    """)

# --- 2. THEORY SECTION ---
elif menu == "Theory":
    st.markdown('<h1 class="lab-title">Theory & Mathematical Models</h1>', unsafe_allow_html=True)

    tabs = st.tabs([
        "Outlier Concept",
        "Z-Score",
        "IQR",
        "Isolation Forest",
        "k-NN"
    ])

    # =========================================================
    # TAB 1: OUTLIER CONCEPT
    # =========================================================
    with tabs[0]:
        st.markdown("## What is an Outlier?")
        st.write("""
        An outlier is a data point that lies significantly far from other observations.

        These values may arise due to:
        - Measurement or data entry errors  
        - Natural variation  
        - Rare or abnormal events  

        Outliers can distort analysis and must be identified carefully.
        """)

        st.markdown("""
        ### Types of Outliers
        - Global: Far from entire dataset  
        - Contextual: Abnormal in a given context  
        - Collective: Group anomaly  
        """)

    # =========================================================
    # TAB 2: Z-SCORE
    # =========================================================
    with tabs[1]:
        st.markdown("## Z-Score Method")

        # THEORY FIRST
        st.write("""
        The Z-score method identifies outliers by measuring how many standard deviations 
        a data point is away from the mean.

        It assumes that the data follows a normal (Gaussian) distribution.
        Points far from the mean are considered anomalies.
        """)

        # IMAGE (RESIZED)
        st.image(
            "https://images.ctfassets.net/kj4bmrik9d6o/2JiE5Q4Joss2xc0XQdlkqc/ead8b9563912f0ae08fd52bdeea66cd1/Outlier_Blog_CHARTS_StandardDeviation_4_r1.png",
            width=500,
            caption="Normal distribution with outliers beyond ±3 standard deviations"
        )

        # FORMULA
        st.markdown("### Formula")
        st.latex(r"Z = \frac{x - \mu}{\sigma}")

        st.markdown("""
        Outlier Condition:
        - |Z| > 3 → Outlier  
        """)

        # PROS CONS
        col1, col2 = st.columns(2)

        with col1:
            st.success("""
            Pros:
            - Simple and fast  
            - Easy to interpret  
            - Works well for normal distributions  
            """)

        with col2:
            st.error("""
            Cons:
            - Sensitive to extreme values  
            - Assumes normal distribution  
            """)

        st.markdown("### References")
        st.markdown("""
        - Z-K-R: A Novel Framework in Intrusion Detection (2024) - Read Paper
        - Hybrid Statistical–ML Outlier Detection (2024) - https://ieeexplore.ieee.org/document/10403245
        - Robust Z-Score Methods for High-Dimensional Data (2023) - https://www.sciencedirect.com/science/article/pii/S095741742300876X
        - Adaptive Z-Score Thresholding for Streaming Data (2024) - https://dl.acm.org/doi/10.1145/3627673
        - Z-Score Based Financial Fraud Detection (2023) - https://link.springer.com/article/10.1007/s00500-023-08567-1
        """)

    # =========================================================
    # TAB 3: IQR
    # =========================================================
    with tabs[2]:
        st.markdown("## IQR Method")

        # THEORY
        st.write("""
        The Interquartile Range (IQR) method detects outliers by focusing on the 
        middle 50% of the data.

        It is robust and does not rely on distribution assumptions.
        Values far outside this range are considered outliers.
        """)

        # IMAGE
        st.image(
            "https://www.thedataschool.co.uk/content/images/2023/09/IQR-Illustration.png",
            width=500,
            caption="Boxplot showing quartiles and outliers"
        )

        # FORMULA
        st.markdown("### Formula")
        st.latex(r"IQR = Q_3 - Q_1")
        st.latex(r"\text{Lower Bound} = Q_1 - 1.5 \times IQR")
        st.latex(r"\text{Upper Bound} = Q_3 + 1.5 \times IQR")

        # PROS CONS
        col1, col2 = st.columns(2)

        with col1:
            st.success("""
            Pros:
            - Robust to outliers  
            - Works for skewed data  
            - No distribution assumptions  
            """)

        with col2:
            st.error("""
            Cons:
            - Less sensitive  
            - Not suitable for high-dimensional data  
            """)

        st.markdown("### References")
        st.markdown("""
        - IQR-Based Outlier Detection in Big Data (2023) - https://ieeexplore.ieee.org/document/10199872
        - Robust IQR Filtering for Noisy Datasets (2024) - https://www.sciencedirect.com/science/article/pii/S002002552400312X
        - Improved IQR for Skewed Distributions (2023) - https://link.springer.com/article/10.1007/s41060-023-00418-5
        - IQR + ML Hybrid Anomaly Detection (2024) - https://dl.acm.org/doi/10.1145/3605098
        - Efficient Outlier Detection using IQR in IoT (2025) - https://www.mdpi.com/1424-8220/25/2/567
        """)

    # =========================================================
    # TAB 4: ISOLATION FOREST
    # =========================================================
    with tabs[3]:
        st.markdown("## Isolation Forest")

        # THEORY
        st.write("""
        Isolation Forest is an unsupervised learning algorithm that isolates anomalies 
        instead of modeling normal data.

        It works by randomly splitting the data. Since outliers are rare and different, 
        they are isolated faster than normal points.
        """)

        # IMAGE
        st.image(
            "https://miro.medium.com/v2/resize:fit:1358/0*t9rBhursm8w2mnzj.png",
            width=500,
            caption="Isolation Forest splits leading to shorter paths for anomalies"
        )

        # FORMULA / CONCEPT
        st.markdown("### Concept")
        st.latex(r"h(x) = \text{path length}")

        # PROS CONS
        col1, col2 = st.columns(2)

        with col1:
            st.success("""
            Pros:
            - Efficient for large datasets  
            - Handles high dimensions  
            - No scaling required  
            """)

        with col2:
            st.error("""
            Cons:
            - Hard to interpret  
            - Results vary due to randomness  
            """)

        st.markdown("### References")
        st.markdown("""
        - Fuzzy Anomaly Scores for Isolation Forest (2024) - Read Paper
        - Isolation Forest for Clustering & Anomaly Detection (2024) - Read Paper
        - Isolation Forest in SDN Security (2024) - Read Paper
        - Signature Isolation Forest (2024) - https://arxiv.org/abs/2403.04405
        - siForest: Set-Structured Isolation Forest (2024) - https://arxiv.org/abs/2412.06015
        """)

    # =========================================================
    # TAB 5: KNN
    # =========================================================
    with tabs[4]:
        st.markdown("## k-Nearest Neighbors (k-NN)")

        # THEORY
        st.write("""
        k-NN detects outliers based on distance from neighboring points.

        Normal points are close to their neighbors, while outliers lie far away 
        from any cluster.
        """)

        # IMAGE
        st.image(
            "https://datascientest.com/wp-content/uploads/2020/11/Illu-2-KNN-2048x983.jpg",
            width=500,
            caption="Outliers are distant from dense clusters"
        )

        # FORMULA
        st.markdown("### Formula")
        st.latex(r"Score(x) = \frac{1}{k} \sum dist(x, n_i)")

        # PROS CONS
        col1, col2 = st.columns(2)

        with col1:
            st.success("""
            Pros:
            - Simple and intuitive  
            - No distribution assumptions  
            """)

        with col2:
            st.error("""
            Cons:
            - Computationally expensive  
            - Sensitive to scaling  
            """)

        st.markdown("### References")
        st.markdown("""
        - SPINEX: Explainable kNN-based Anomaly Detection (2024) - https://arxiv.org/abs/2407.04760
        - kNN-Based Outlier Detection in High Dimensions (2023) - https://ieeexplore.ieee.org/document/10144567
        - Hybrid kNN and Density-Based Anomaly Detection (2024) - https://www.sciencedirect.com/science/article/pii/S095070512400221X
        - Efficient kNN Outlier Detection for Big Data (2023) - https://dl.acm.org/doi/10.1145/3580305
        - Deep kNN for Anomaly Detection (2024) - https://link.springer.com/article/10.1007/s10994-024-06521-7
        """)


# --- 3. QUIZ ENGINE ---
def run_quiz_engine(test_type):
    st.markdown(f'<h1 class="lab-title">{test_type}</h1>', unsafe_allow_html=True)
    diff = st.select_slider(f"Level", options=["Easy", "Medium", "Hard"], key=f"slider_{test_type}")
    
    questions = QUIZ_BANK[test_type][diff]
    
    with st.form(f"form_{test_type}_{diff}"):
        user_ans = []
        for i, (q, opts, _) in enumerate(questions):
            st.write(f"**Q{i+1}:** {q}")
            ans = st.radio("Select:", opts, key=f"q_{test_type}_{i}", index=None)
            user_ans.append(ans)
        
        if st.form_submit_button("Submit"):
            if None in user_ans:
                st.error("Answer all questions!")
            else:
                score = sum(1 for i, a in enumerate(user_ans) if a == questions[i][1][questions[i][2]])
                pct = (score/len(questions))*100
                st.divider()
                st.metric("Score", f"{score}/{len(questions)}", f"{pct:.0f}%")
                if pct >= 60: st.success("Pass!")
                else: st.error("Review Theory and retry.")

if menu == "Pretest": run_quiz_engine("Pretest")
elif menu == "Posttest": run_quiz_engine("Posttest")

# --- 4. DEMO SECTION ---
elif menu == "Demo (Upload Data)":
    st.markdown('<h1 class="lab-title">Experimental Demo</h1>', unsafe_allow_html=True)

    up = st.file_uploader("Upload CSV", type="csv")

    if up:
        import pandas as pd
        import numpy as np
        from scipy import stats
        from sklearn.neighbors import NearestNeighbors
        from sklearn.ensemble import IsolationForest
        import plotly.express as px

        df = pd.read_csv(up)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if num_cols:
            target = st.selectbox("Select Feature to Analyze", num_cols)

            # Clean data
            data = df[target].fillna(df[target].median())
            data_arr = data.values.reshape(-1, 1)

            # =====================================================
            # METHOD 1: Z-SCORE
            # =====================================================
            z = np.abs(stats.zscore(data))
            df['Z_Outlier'] = z > 3

            # =====================================================
            # METHOD 2: IQR
            # =====================================================
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            df['IQR_Outlier'] = (data < lower) | (data > upper)

            # =====================================================
            # METHOD 3: ISOLATION FOREST
            # =====================================================
            iso = IsolationForest(contamination=0.05, random_state=42)
            iso_pred = iso.fit_predict(data_arr)

            df['IF_Outlier'] = iso_pred == -1  # -1 = outlier

            # =====================================================
            # METHOD 4: KNN
            # =====================================================
            k_val = st.slider("Select k for k-NN", 2, 20, 5)

            knn = NearestNeighbors(n_neighbors=k_val)
            knn.fit(data_arr)

            distances, _ = knn.kneighbors(data_arr)
            avg_dist = distances.mean(axis=1)

            df['KNN_Score'] = avg_dist
            threshold = np.percentile(avg_dist, 95)
            df['KNN_Outlier'] = avg_dist > threshold

            # =====================================================
            # SUMMARY TABLE
            # =====================================================
            st.subheader("Outlier Detection Results")

            st.dataframe(
                df[[target, 'Z_Outlier', 'IQR_Outlier', 'IF_Outlier', 'KNN_Outlier']].head(20)
            )

            # =====================================================
            # COUNTS
            # =====================================================
            st.subheader("Outlier Count by Method")

            counts = {
                "Z-Score": int(df['Z_Outlier'].sum()),
                "IQR": int(df['IQR_Outlier'].sum()),
                "Isolation Forest": int(df['IF_Outlier'].sum()),
                "k-NN": int(df['KNN_Outlier'].sum())
            }

            st.table(pd.DataFrame(counts.items(), columns=["Method", "Outliers Detected"]))

            # =====================================================
            # VISUALIZATIONS (ALL METHODS)
            # =====================================================
            st.subheader("Visualization for Each Method")

            # Z-SCORE PLOT
            fig1 = px.scatter(df, y=target, color='Z_Outlier',
                              color_discrete_map={True: '#f97316', False: '#38bdf8'},
                              title="Z-Score Outliers")
            apply_dark_plot_theme(fig1, y_title=target)
            st.plotly_chart(fig1, use_container_width=True)

            # IQR PLOT
            fig2 = px.scatter(df, y=target, color='IQR_Outlier',
                              color_discrete_map={True: '#f97316', False: '#38bdf8'},
                              title="IQR Outliers")
            apply_dark_plot_theme(fig2, y_title=target)
            st.plotly_chart(fig2, use_container_width=True)

            # ISOLATION FOREST PLOT
            fig3 = px.scatter(df, y=target, color='IF_Outlier',
                              color_discrete_map={True: '#f97316', False: '#38bdf8'},
                              title="Isolation Forest Outliers")
            apply_dark_plot_theme(fig3, y_title=target)
            st.plotly_chart(fig3, use_container_width=True)

            # KNN PLOT
            fig4 = px.scatter(df, y=target, color='KNN_Outlier',
                              color_discrete_map={True: '#f97316', False: '#38bdf8'},
                              title="k-NN Outliers")
            apply_dark_plot_theme(fig4, y_title=target)
            st.plotly_chart(fig4, use_container_width=True)

            # =====================================================
            # COMBINED VIEW
            # =====================================================
            st.subheader("Combined Outlier Detection")

            df['Total_Outliers'] = (
                df['Z_Outlier'].astype(int) +
                df['IQR_Outlier'].astype(int) +
                df['IF_Outlier'].astype(int) +
                df['KNN_Outlier'].astype(int)
            )

            fig_combined = px.scatter(
                df,
                y=target,
                color='Total_Outliers',
                title="Points flagged by multiple methods"
            )

            apply_dark_plot_theme(fig_combined, y_title=target)

            st.plotly_chart(fig_combined, use_container_width=True)

        else:
            st.error("No numeric columns found in dataset.")

# --- 6. FEEDBACK ---
elif menu == "Feedback":
    st.markdown('<h1 class="lab-title">Feedback</h1>', unsafe_allow_html=True)
    with st.form("fb"):
        st.select_slider("Rate Lab", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])
        st.text_area("Suggestions")
        st.form_submit_button("Submit")
