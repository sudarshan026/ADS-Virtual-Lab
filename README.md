<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Experiments-9-7c3aed?style=for-the-badge" alt="Experiments"/>
  <img src="https://img.shields.io/badge/Status-Active-00d4ff?style=for-the-badge" alt="Status"/>
</p>

<h1 align="center">🔬 Applied Data Science — Virtual Lab Platform</h1>

<p align="center">
  <strong>A next-generation, unified virtual laboratory for mastering Applied Data Science through 9 hands-on interactive experiments — from descriptive statistics to AutoML.</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-experiments">Experiments</a> •
  <a href="#%EF%B8%8F-architecture">Architecture</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Quick Start](#-quick-start)
- [Experiments](#-experiments)
- [Architecture](#%EF%B8%8F-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Configuration](#%EF%B8%8F-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🌟 Overview

The **Applied Data Science Virtual Lab** is a comprehensive, browser-based learning platform that consolidates 9 standalone data science experiments into a single, cohesive application. Built with Streamlit and designed with a futuristic dark-neon theme, it provides students and practitioners with an interactive environment to explore the complete data science pipeline.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏠 **Unified Landing Page** | Glassmorphism card grid with animated gradients and one-click access to all experiments |
| 🧭 **Persistent Navigation** | Sidebar navigation stays accessible from any experiment page |
| 🎨 **Consistent Theming** | Dark neon aesthetic with Inter font, glow effects, and smooth page transitions |
| 🔌 **Zero-Copy Integration** | Original experiment code is loaded dynamically at runtime — no duplication |
| 📊 **9 Complete Labs** | Statistics, cleaning, visualization, ML, SMOTE, outliers, time series, fusion, AutoML |
| 📱 **Responsive Layout** | Wide-mode Streamlit layout optimized for desktop and laptop screens |

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.9 or higher
- **pip** package manager
- **Git** (optional, for cloning)

### Installation

```bash
# 1. Clone or download the repository
git clone <repository-url>
cd VLADS

# 2. Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open automatically in your default browser at **http://localhost:8501**.

> **Tip:** To change the port, use `streamlit run app.py --server.port 8080`

---

## 🔬 Experiments

### Experiment 1 — 📊 Descriptive & Inferential Statistics

Explore fundamental statistical concepts including measures of central tendency (mean, median, mode), dispersion (variance, standard deviation), correlation analysis, and hypothesis testing on built-in and uploaded datasets.

**Key Topics:** Descriptive statistics, probability distributions, t-tests, chi-square tests, ANOVA

---

### Experiment 2 — 🧹 Data Cleaning & Imputation

A hands-on lab for understanding how data quality impacts ML model performance. Compare three imputation strategies — **Mode/Median**, **KNN**, and **MICE (Iterative)** — and train classifiers to measure the downstream effect.

**Key Topics:** Missing value handling, duplicate removal, imputation comparison, preprocessing pipelines

---

### Experiment 3 — 📈 Data Visualization

Build interactive visualizations using Plotly: histograms, scatter plots, box plots, heatmaps, pair plots, and more. Upload your own CSV or use built-in datasets to practice exploratory data analysis.

**Key Topics:** EDA, Plotly charts, distribution analysis, feature relationships

---

### Experiment 4 — 🎯 Model Evaluation Metrics

A full ML classification lab supporting Logistic Regression, KNN, SVM, Random Forest, and XGBoost. Includes ROC curves, confusion matrices, precision-recall curves, learning curves, validation curves, radar charts, and a carbon footprint tracker.

**Key Topics:** Classification metrics, cross-validation, hyperparameter tuning, model comparison

---

### Experiment 5 — ⚖️ SMOTE Technique

Tackle class imbalance using the Synthetic Minority Over-sampling Technique. Load real-world imbalanced datasets (Attrition, Bank, Credit Card, Diabetes), apply SMOTE, and compare model performance before and after balancing.

**Key Topics:** Class imbalance, SMOTE algorithm, oversampling vs undersampling, GAN comparison

---

### Experiment 6 — 🔍 Outlier Detection

Detect anomalies in datasets using multiple statistical and ML-based methods: **Z-Score**, **IQR (Interquartile Range)**, **Isolation Forest**, and **k-Nearest Neighbors**. Visualize outlier boundaries and compare detection rates.

**Key Topics:** Anomaly detection, statistical thresholds, ensemble methods, visualization

---

### Experiment 7 — 📉 Time Series Forecasting

Decompose time series into trend, seasonal, and residual components. Build and compare **Moving Average** and **ARIMA(p,d,q)** models. Evaluate with MAE, MSE, RMSE, and MAPE. Includes built-in Apple Stock and Chocolate Sales datasets.

**Key Topics:** Time series decomposition, stationarity, ARIMA, forecasting evaluation

---

### Experiment 8 — 🧬 Data Science Lifecycle (Multimodal Fusion)

An attention-based multimodal fusion lab that combines 5 data modalities (Image, Text, Audio, Sensor, Video/Social) to classify samples. Includes realistic noise injection, feature importance analysis, and ablation studies.

**Key Topics:** Multimodal learning, feature fusion, attention mechanisms, ablation analysis

---

### Experiment 9 — 🤖 AutoML Techniques

Run and compare three AutoML engines — **FLAML**, **AutoGluon**, and **H2O AutoML** — on uploaded datasets. Auto-detects classification vs regression, handles preprocessing, and generates comparative visualizations.

**Key Topics:** Automated ML, model selection, hyperparameter optimization, ensemble methods

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      app.py                               │
│               (Entry Point & Router)                      │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │  Landing     │  │   Sidebar    │  │  Dynamic      │   │
│  │  Page        │  │   Navigation │  │  Experiment   │   │
│  │  (3×3 Grid)  │  │   (9 items)  │  │  Loader       │   │
│  └─────────────┘  └──────────────┘  └───────┬───────┘   │
└──────────────────────────────────────────────┼───────────┘
                                               │
                    ┌──────────────────────────┴──────────┐
                    │         theme.py                     │
                    │   (CSS / Fonts / Utilities)          │
                    └──────────────────────────────────────┘
                                               │
          ┌────────────────────────────────────┼────────┐
          │            experiments/                      │
          │                                             │
          │  exp1_statistics.py ──→ ADS-exp-1-*/app.py  │
          │  exp2_model_eval.py ──→ ADS-VL-*/app.py     │
          │  exp3_visualization.py → app exp 3.py       │
          │  exp4_data_cleaning.py  (native Streamlit)  │
          │  exp5_smote.py ───────→ ADS_VirtualLab*/    │
          │  exp6_outlier.py ─────→ adsca exp 6.py      │
          │  exp7_timeseries.py ──→ ADS_Virtual*/exp7   │
          │  exp8_lifecycle.py ───→ VL-DS-*/app.py      │
          │  exp9_automl.py ─────→ Exp_9_*/app.py       │
          └─────────────────────────────────────────────┘
```

### Integration Strategy

| Mechanism | Purpose |
|-----------|---------|
| **`exec()` + `compile()`** | Dynamically loads original experiment code at runtime without duplicating files |
| **`st.set_page_config` monkeypatch** | After the main config call, replaced with a no-op lambda so experiments don't raise `StreamlitAPIException` |
| **`sys.path` injection** | Each wrapper adds the experiment's directory to Python's path so local imports (e.g., `utils/`, `models/`) resolve correctly |
| **`os.chdir()` for data files** | Experiments loading CSV files relative to their directory (e.g., exp7) temporarily switch the working directory |

---

## 📁 Project Structure

```
VLADS/
│
├── app.py                              # Main entry point — landing page & router
├── theme.py                            # Unified dark neon theme system
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── experiments/                        # Experiment wrapper modules
│   ├── __init__.py
│   ├── exp1_statistics.py              # → Descriptive & Inferential Statistics
│   ├── exp2_model_evaluation.py        # → Model Evaluation Metrics
│   ├── exp3_visualization.py           # → Data Visualization
│   ├── exp4_data_cleaning.py           # → Data Cleaning & Imputation (native)
│   ├── exp5_smote.py                   # → SMOTE Technique
│   ├── exp6_outlier.py                 # → Outlier Detection
│   ├── exp7_timeseries.py              # → Time Series Forecasting
│   ├── exp8_lifecycle.py               # → Data Science Lifecycle
│   └── exp9_automl.py                  # → AutoML Techniques
│
├── ADS-exp-1-virtual-lab-main/         # Original Experiment 1 source
├── ADS-VL-main exp 2/                  # Original Experiment 2 source
├── app exp 3.py                        # Original Experiment 3 source
├── ADS_virtual_lab-main exp 4/         # Original Experiment 4 source (Flask)
│   ├── api/                            # Flask API backend
│   ├── adult.csv                       # Adult Census dataset
│   └── virtual-lab-ui/                 # HTML frontend
├── ADS_VirtualLab_SMOTE-main exp 5/    # Original Experiment 5 source
│   ├── utils/                          # Data loaders, SMOTE handler, models
│   └── models/                         # Pre-trained model weights
├── adsca exp 6.py                      # Original Experiment 6 source
├── ADS_Virtual_Lab-main exp 7/         # Original Experiment 7 source
│   └── *.csv                           # Time series datasets
├── VL-DS-main exp8/                    # Original Experiment 8 source
└── Exp_9_ADSVirtualLab-main/           # Original Experiment 9 source
```

---

## 🛠 Tech Stack

### Core Framework

| Technology | Version | Purpose |
|------------|---------|---------|
| [Streamlit](https://streamlit.io/) | ≥ 1.30 | Web application framework |
| [Python](https://python.org/) | ≥ 3.9 | Runtime |

### Data Science Libraries

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `scikit-learn` | ML models, preprocessing, metrics |
| `scipy` | Statistical tests and distributions |
| `statsmodels` | Time series analysis (ARIMA, decomposition) |

### Visualization

| Library | Purpose |
|---------|---------|
| `plotly` | Interactive charts and dashboards |
| `matplotlib` | Static plots and figures |
| `seaborn` | Statistical visualizations |

### Specialized (per-experiment)

| Library | Experiment | Purpose |
|---------|------------|---------|
| `xgboost` | 4 | Gradient boosting classifier |
| `umap-learn` | 4 | Dimensionality reduction |
| `imbalanced-learn` | 5 | SMOTE implementation |
| `flaml` | 9 | Fast & Lightweight AutoML |
| `autogluon` | 9 | AutoGluon AutoML engine |
| `h2o` | 9 | H2O AutoML engine |

> **Note:** Experiments 9's AutoML libraries (FLAML, AutoGluon, H2O) are optional. The experiment gracefully handles missing libraries with informative error messages.

---

## ⚙️ Configuration

### Streamlit Settings

Create a `.streamlit/config.toml` file for custom configuration:

```toml
[server]
port = 8501
headless = true
maxUploadSize = 200    # MB — increase for large datasets

[theme]
primaryColor = "#00d4ff"
backgroundColor = "#0a0a1a"
secondaryBackgroundColor = "#0f0f2d"
textColor = "#e2e8f0"

[browser]
gatherUsageStats = false
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | `8501` | Port to serve the application |
| `STREAMLIT_SERVER_HEADLESS` | `false` | Set `true` for containerized deployment |

---

## 🐛 Troubleshooting

<details>
<summary><strong>❌ ModuleNotFoundError: No module named 'xxx'</strong></summary>

Install the missing dependency:
```bash
pip install <module-name>
```
For experiment 5 (SMOTE), ensure you're in the `VLADS/` root directory when launching — the wrapper adjusts `sys.path` relative to it.
</details>

<details>
<summary><strong>❌ StreamlitAPIException: set_page_config() can only be called once</strong></summary>

This should not occur in the unified app. If it does, ensure you're running `app.py` from the `VLADS/` root and not an individual experiment file:
```bash
# ✅ Correct
streamlit run app.py

# ❌ Wrong — will bypass the monkeypatch
streamlit run "ADS-exp-1-virtual-lab-main/app.py"
```
</details>

<details>
<summary><strong>❌ Experiment 5 — "No pre-trained models found"</strong></summary>

Experiment 5 requires pre-trained model weights in the `models/` directory. Run the training script first:
```bash
cd "ADS_VirtualLab_SMOTE-main exp 5"
python train_all_models.py
cd ..
```
</details>

<details>
<summary><strong>❌ Experiment 9 — AutoGluon / H2O errors</strong></summary>

These libraries have complex system dependencies:
- **AutoGluon:** Requires specific torch versions. See [AutoGluon install guide](https://auto.gluon.ai/stable/install.html)
- **H2O:** Requires Java Runtime (JRE 11+). See [H2O docs](https://docs.h2o.ai/)
- **FLAML** works standalone and is the most portable option
</details>

<details>
<summary><strong>⚠️ Deprecation warnings about use_container_width</strong></summary>

These are cosmetic warnings from original experiment code using the older Streamlit API. They do not affect functionality and will be resolved in future updates.
</details>

---

## 🤝 Contributing

Contributions are welcome! To add a new experiment:

1. **Create** your Streamlit experiment as a standalone `app.py`
2. **Place** it in a new directory under `VLADS/`
3. **Create** a wrapper in `experiments/exp10_name.py`:
   ```python
   import sys, os

   def run():
       base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
       exp_dir = os.path.join(base, "your-experiment-folder")
       if exp_dir not in sys.path:
           sys.path.insert(0, exp_dir)
       filepath = os.path.join(exp_dir, "app.py")
       with open(filepath, "r", encoding="utf-8") as f:
           code = f.read()
       exec(compile(code, filepath, "exec"),
            {"__name__": "__experiment__", "__file__": filepath})
   ```
4. **Register** it in `app.py`'s `EXPERIMENTS` dict and update `theme.py`'s metadata arrays
5. **Submit** a pull request

### Code Standards

- Follow [PEP 8](https://pep8.org/) for Python code
- Use type hints where practical
- Original experiment files must **not** be modified
- Wrapper files should be minimal (< 20 lines)

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **Streamlit** — for making data apps delightful to build
- **scikit-learn** — the backbone of ML experiments
- **Plotly** — powering interactive visualizations
- **Department of CSE (Data Science)** — for the experiment curricula

---

<p align="center">
  <sub>Built with ❤️ for the Applied Data Science community</sub>
</p>
