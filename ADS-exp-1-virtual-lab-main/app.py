import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

st.set_page_config(page_title="Virtual Lab", layout="wide")

# -------- CSS --------
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #f8fafc;
    font-family: 'Segoe UI', sans-serif;
}

/* TEXT SIZE */
p, li { font-size: 18px !important; color: #1e293b; }

h1 { font-size: 38px !important; color: #0f172a; }
h2 { font-size: 28px !important; color: #1e293b; }
h3 { font-size: 22px !important; color: #334155; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #0f172a;
}

[data-testid="stSidebar"] * {
    color: white !important;
    font-size: 18px !important;
}

/* SIDEBAR SELECTED ITEM */
.css-1d391kg {
    background-color: #2563eb !important;
    border-radius: 8px;
}

/* CARD DESIGN */
.card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* BUTTON */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
}

</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("<h1 style='text-align: center;'>📊 Virtual Lab: Statistical Analysis</h1>", unsafe_allow_html=True)
st.markdown("---")

# -------- SIDEBAR --------
st.sidebar.title("🧪 Virtual Lab")

menu = st.sidebar.radio(
    "Navigation",
    ["Aim", "Theory", "Procedure", "Simulation", "Observations", "Quiz", "References"]
)

# -------- AIM --------
if menu == "Aim":
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("🎯 Aim")
    st.write("To explore descriptive and inferential statistics on datasets using real-world data.")

    st.markdown("### 📖 Description")
    st.write("""
    The objective of this experiment is to develop a comprehensive understanding of statistical analysis through an interactive virtual lab environment. This system enables users to upload real-world datasets in CSV format and perform various statistical operations to extract meaningful insights.

The experiment focuses on applying descriptive statistics such as mean, median, mode, variance, standard deviation, and interquartile range to summarize and understand the characteristics of data. It also incorporates data visualization techniques like histograms, boxplots, and scatter plots to visually interpret patterns, distributions, and relationships within the dataset.

In addition, the experiment introduces inferential statistics, particularly correlation analysis and hypothesis testing using p-values. This helps in determining whether relationships between variables are statistically significant, thereby supporting data-driven decision-making.

Overall, this virtual lab provides a practical approach to learning statistics by combining theoretical concepts with hands-on implementation, making it easier to understand how statistical techniques are applied in real-world scenarios such as business analysis, research, and data science.
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- THEORY --------
elif menu == "Theory":
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("📚 Theory")

    st.subheader("Descriptive Statistics")
    st.markdown("""
    - Mean → Average  
    - Median → Middle value  
    - Mode → Most frequent    
    """)

    st.image("https://images.openai.com/static-rsc-4/kk23PUVvuow2r8jQ9_riytUkzTsradO9QjbCQPHVBS5jiwPzXmtBRHXfUiuckjmrn0mtbSv-69f-SNW6uqK_-M9NJITfZqnwAafMl9RlI2s9w92O2Ps50HIPRwc0TkxmHRHW-_c77wHskI6rmkZ2VTDV3YrUq_evFayVJx5SrIM-OohItL2ckCrr21WZuE3F?purpose=fullsize",
             caption="Statistical Workflow")


    st.markdown("---")

    st.subheader("Visualization")
    st.markdown("""
    - Histogram → Distribution   
    """)
    st.image("https://imgs.search.brave.com/9vEP50-9wHbcRQ7jmtJITOn7KCAwTMspKwZFnmR1uLo/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9xc3V0/cmEuY29tL3dwLWNv/bnRlbnQvdXBsb2Fk/cy8yMDIwLzEwL0hp/c3RvZ3JhbS5wbmc",
             caption="Statistical Workflow")
    



    st.markdown("""
    - Boxplot → Outliers   
    """)
    st.image("https://imgs.search.brave.com/64cSqkLS1SupoAULxH0owCQm6-dmTA-2KIfXXdTGV8Q/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93d3cu/bWF0aC1zYWxhbWFu/ZGVycy5jb20vaW1h/Z2UtZmlsZXMvYm94/LXBsb3QtZXhhbXBs/ZS0zYS5naWY.gif",
             caption="Statistical Workflow")
    


    
    st.markdown("""
    - Scatter Plot → Relationship  
    """)
    st.image("https://imgs.search.brave.com/-9Zgw3kdQcYP3a_8A9YvoFbmMC9i9Kvjen5Roua6Qvs/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zZWFi/b3JuLnB5ZGF0YS5v/cmcvX2ltYWdlcy9z/Y2F0dGVycGxvdF8x/NV8wLnBuZw",
             caption="Statistical Workflow")
    

    st.markdown("---")

    st.subheader("Inferential Statistics")
    st.markdown("""
    - Correlation → Strength of relation  
    - p-value → Significance  
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- PROCEDURE --------
elif menu == "Procedure":
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("⚙️ Procedure")

    st.write("""
    1. Upload CSV dataset  
    2. View dataset  
    3. Select column  
    4. Perform statistics  
    5. Visualize data  
    6. Analyze correlation  
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- SIMULATION --------
elif menu == "Simulation":
    st.header("🔬 Simulation")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.dataframe(df)

        numeric_cols = df.select_dtypes(include='number').columns

        if len(numeric_cols) > 0:
            col = st.selectbox("Select column", numeric_cols)

            mean = df[col].mean()
            std = df[col].std()

            st.write(f"Mean: {mean:.2f}")
            st.write(f"Std Dev: {std:.2f}")

            fig, ax = plt.subplots(figsize=(5,3))
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(5,2))
            sns.boxplot(x=df[col], ax=ax2)
            st.pyplot(fig2)

            if len(numeric_cols) >= 2:
                x = st.selectbox("X-axis", numeric_cols)
                y = st.selectbox("Y-axis", numeric_cols)

                fig3, ax3 = plt.subplots(figsize=(5,3))
                sns.scatterplot(x=df[x], y=df[y], ax=ax3)
                st.pyplot(fig3)

                corr, p = pearsonr(df[x], df[y])

                st.write(f"Correlation: {corr:.2f}")
                st.write(f"p-value: {p:.4f}")

                st.session_state["mean"] = mean
                st.session_state["std"] = std
                st.session_state["col_name"] = col
                st.session_state["corr"] = corr
                st.session_state["p_value"] = p

# -------- OBSERVATIONS --------
elif menu == "Observations":
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("📌 Observations")

    if "mean" in st.session_state:
        st.write(f"Mean: {st.session_state['mean']:.2f}")
        st.write(f"Std Dev: {st.session_state['std']:.2f}")

        if st.session_state["p_value"] is not None:
            if st.session_state["p_value"] < 0.05:
                st.success("Significant relationship")
            else:
                st.warning("No significant relationship")
    else:
        st.warning("Run simulation first")

    st.markdown('</div>', unsafe_allow_html=True)

# -------- QUIZ --------
elif menu == "Quiz":
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("🧠 Statistical Quiz")

    level = st.selectbox("Select Difficulty Level", ["Basic", "Moderate", "Advanced"])

    if level == "Basic":
        questions = [
            ("Mean represents?", ["Middle value", "Average", "Mode"], "Average"),
            ("Median is?", ["Average", "Middle value", "Highest"], "Middle value"),
            ("Mode is?", ["Frequent", "Average", "Middle"], "Frequent"),
            ("CSV file is?", ["Image", "Data file", "Audio"], "Data file")
        ]

    elif level == "Moderate":
        questions = [
            ("Standard deviation measures?", ["Spread", "Center", "Count"], "Spread"),
            ("Histogram shows?", ["Relation", "Distribution", "Outliers"], "Distribution"),
            ("Boxplot detects?", ["Trend", "Outliers", "Mean"], "Outliers"),
            ("Scatter plot shows?", ["Relation", "Distribution", "Mean"], "Relation"),
            ("Variance represents?", ["Spread", "Center", "Count"], "Spread")
        ]

    else:
        questions = [
            ("Correlation range?", ["0-1", "-1 to 1", "0-100"], "-1 to 1"),
            ("p-value < 0.05 means?", ["Significant", "Random", "None"], "Significant"),
            ("Null hypothesis means?", ["No relation", "Relation", "Random"], "No relation"),
            ("IQR stands for?", ["Inter Quartile Range", "Interval", "Internal"], "Inter Quartile Range"),
            ("Correlation measures?", ["Causation", "Relationship", "Distribution"], "Relationship")
        ]

    score = 0

    for i, (q, options, correct) in enumerate(questions):
        ans = st.radio(f"Q{i+1}. {q}", options, key=f"{level}_{i}")
        if ans == correct:
            score += 1

    if st.button("Submit Quiz"):
        st.success(f"Score: {score}/{len(questions)}")

        if score == len(questions):
            st.success("Excellent 🔥")
        elif score >= len(questions)//2:
            st.info("Good 👍")
        else:
            st.warning("Revise concepts!")

    st.markdown('</div>', unsafe_allow_html=True)

# -------- REFERENCES --------
elif menu == "References":
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("📚 References (Recent Research Papers)")

    st.markdown("""
- https://arxiv.org/pdf/2108.02497.pdf  
- https://arxiv.org/pdf/2002.07637.pdf  
- https://arxiv.org/pdf/2203.15556.pdf  
- https://www.mdpi.com/2227-7390/11/5/1234  

---

### 📘 Core Learning
- https://www.statlearning.com/  
- https://www.khanacademy.org/math/statistics-probability  
- https://pandas.pydata.org/docs/  
""")

    st.markdown('</div>', unsafe_allow_html=True)

# -------- FOOTER --------
st.markdown("""
<hr>
<p style='text-align:center; font-size:14px; color:gray;'>
© Members: 1. Vedika Dhamale | 2. Shravani Bhosale | 3. Akash Jadhav
</p>
""", unsafe_allow_html=True)