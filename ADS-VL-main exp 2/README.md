# 🧪 ML Classification Virtual Lab (2026 Edition)

A full-featured Streamlit virtual laboratory for comparing 5 classification models with carbon footprint tracking.

## Features
- **5 Models**: Logistic Regression, Random Forest, SVM, Decision Tree, k-NN
- **Carbon Tracking**: Live CO₂ and energy tracking via CodeCarbon (estimated fallback if unavailable)
- **6 Visualizations**: Heatmap, Radar, Bubble, ROC Curves, Boxplot, Parallel Coordinates
- **Dataset Support**: Upload any CSV, or use built-in sklearn datasets (Breast Cancer, Iris, Wine)
- **Recommended Dataset**: Indian Engineering Student Placement (upload your CSVs)

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Pages
| Page | Description |
|------|-------------|
| 🏠 Home | Overview and workflow guide |
| 📂 Dataset & EDA | Upload/select data, visualize distributions and correlations |
| ⚙️ Model Config | Tune hyperparameters per model, set CV folds/test split |
| 🚀 Training Lab | One-click training with live progress + carbon tracking |
| 📊 Results Dashboard | Heatmap, Radar, Bubble, ROC, Boxplot, Parallel Coordinates |
| 🌱 Carbon Footprint | Full emissions report with eco-efficiency score and tips |

## Dataset Recommendations

### Primary (your experiment dataset)
Upload `indianengineeringstudentplacement.csv` + `placementtargets.csv`, merge on StudentID, then set `placementstatus` as target.

### Quick demo datasets (built-in)
- **Breast Cancer** (sklearn) — 569 samples, 30 features, binary
- **Wine** (sklearn) — 178 samples, 13 features, 3-class
- **Iris** (sklearn) — 150 samples, 4 features, 3-class

## Carbon Footprint Tracking
Install `codecarbon` for real measurements:
```bash
pip install codecarbon
```
Without it, the lab uses time-based emission estimates (proportional to actual CodeCarbon measurements).
