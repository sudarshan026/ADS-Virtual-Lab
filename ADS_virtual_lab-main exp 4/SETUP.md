# ADS Virtual Lab - Complete Setup & Installation Guide

## Overview

This is a complete Advanced Data Science Virtual Lab with a Streamlit frontend and Python Flask backend. It implements all 10 steps of the data science pipeline:

1. **Load Dataset** - UCI Adult dataset (32K+ rows)
2. **Data Understanding** - Feature analysis and statistics
3. **Data Cleaning** - Handle missing values, duplicates, outliers
4. **Data Imputation** - Mode, KNN, MICE methods
5. **Data Preprocessing** - Encoding, scaling, train-test split
6. **Supervised Learning** - Logistic Regression, Random Forest, XGBoost
7. **Unsupervised Learning** - K-Means clustering with PCA
8. **Data Fusion** - 4 fusion techniques (Early, Late, Weighted, Hybrid)
9. **Evaluation** - Comprehensive metrics & comparison
10. **Visualization** - Interactive charts and dashboards

---

## Prerequisites

- **Python 3.8+** installed
- **Git** (optional, for version control)
- ~2GB disk space

---

## Installation

### Step 1: Backend Setup

```bash
# Navigate to API directory
cd api

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep flask
```

### Step 2: Frontend Setup (Streamlit)

```bash
# Install Streamlit UI dependencies in the API virtual environment
api/.venv/Scripts/python.exe -m pip install -r virtual-lab-ui/requirements.txt
```

---

## Running the Application

### Option A: Manual Start (2 Terminals)

**Terminal 1 - Backend API:**
```bash
cd api
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

**Terminal 2 - Frontend:**
```bash
api/.venv/Scripts/python.exe -m streamlit run virtual-lab-ui/app.py --server.port 8501
```

You should see:
```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
```

### Option B: One-Command Start (Windows)

Use `start.bat`:
```batch
start.bat
```

Then:
```bash
start.bat
```

### Option C: One-Command Start (Mac/Linux)

Create `start.sh`:
```bash
#!/bin/bash
cd api && python app.py &
python -m streamlit run virtual-lab-ui/app.py --server.port 8501 &
```

Then:
```bash
chmod +x start.sh
./start.sh
```

---

## Usage

### 1. Open the Application

Open your browser and go to: **http://localhost:8501**

You'll see the ADS Virtual Lab Streamlit dashboard with module navigation in the sidebar.

### 2. Complete Pipeline Walkthrough

#### **Module 1: Data Loader**
- Select **Data Loader** from the Streamlit sidebar
- View dataset statistics: 32,562 rows, 15 features
- See feature list and class distribution
- Click "Reload Dataset" to refresh data

#### **Module 2: Data Cleaning**
- Select **Data Cleaning** from the Streamlit sidebar
- View cleaning steps pipeline
- Handles: missing values, duplicates, category fixing, outlier detection

#### **Module 3: Preprocessing**
- Select **Preprocessing** from the Streamlit sidebar
- Shows: One-Hot Encoding, StandardScaler, 80-20 split
- Displays processed feature count and training samples

#### **Module 4: Model Training** ⭐
- Select **Model Training** from the Streamlit sidebar
- Click "Train All Models" button
- Trains 3 models: Logistic Regression, Random Forest, XGBoost
- Displays accuracy comparison chart
- Shows F1-score, ROC-AUC, training time for each model
- Shows confusion matrix for best model

#### **Module 5: Clustering**
- Select **Clustering** from the Streamlit sidebar
- Runs K-Means clustering (k=2,3,4,5)
- Shows Silhouette scores and Davies-Bouldin index
- Displays PCA visualization of clusters
- Shows elbow method curve

#### **Module 6: Data Fusion** ⭐
- Select **Data Fusion** from the Streamlit sidebar
- Click "Run Fusion Comparison" button
- Compares 5 approaches:
  - No Fusion (baseline)
  - Early Fusion (merge data)
  - Late Fusion (combine predictions)
  - Weighted Fusion (optimal weights)
  - Hybrid Fusion (multi-level)
- Shows accuracy improvement chart
- Detailed comparison table with metrics

---

## API Endpoints

### Data Operations

```
GET  /api/data/load              → Load & analyze dataset
GET  /api/data/sample?n=5        → Get sample rows
```

### Data Cleaning

```
POST /api/cleaning/run           → Run cleaning pipeline
GET  /api/cleaning/missing-values → Get missing value summary
POST /api/cleaning/impute        → Impute with MODE, KNN, or MICE
POST /api/cleaning/impute/compare → Compare imputation methods
```

### Preprocessing

```
POST /api/preprocessing/run      → Run encoding, scaling, split
```

### Model Training

```
POST /api/models/train           → Train single model
POST /api/models/compare         → Train all 3 models and compare
```

### Clustering

```
POST /api/clustering/analyze     → K-Means analysis with k=2-5
POST /api/clustering/pca         → Get PCA visualization
GET  /api/clustering/elbow       → Get elbow curve data
```

### Data Fusion

```
POST /api/fusion/compare         → Compare all 5 fusion techniques
POST /api/fusion/early           → Run early fusion
POST /api/fusion/late            → Run late fusion
POST /api/fusion/weighted        → Run weighted fusion
POST /api/fusion/hybrid          → Run hybrid fusion
```

### Pipeline

```
POST /api/pipeline/run-all       → Run complete pipeline (all steps)
GET  /api/health                 → Check API status
```

---

## Key Features

### Frontend
✅ Neobrutalism UI with bold borders and cartoony design
✅ Responsive design (mobile, tablet, desktop)
✅ Smooth animations and transitions
✅ Loading indicators and error handling
✅ Module-based architecture
✅ Real-time data visualization

### Backend
✅ Complete ML pipeline
✅ 3 ML models: LR, RF, XGBoost
✅ 3 imputation methods: Mode, KNN, MICE
✅ 5 data fusion techniques
✅ K-Means clustering + PCA
✅ Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC
✅ In-memory caching for performance

### Data
✅ UCI Adult dataset (32,562 samples)
✅ 15 features (mix categorical & numerical)
✅ Classification task: predict income >50K / ≤50K

---

## Expected Results

### Model Performance
- **Logistic Regression**: ~84-85% accuracy
- **Random Forest**: ~86-87% accuracy
- **XGBoost**: ~87-88% accuracy

### Fusion Results
- **No Fusion**: ~87% accuracy (baseline)
- **Early Fusion**: ~86-87% accuracy (slightly lower due to feature interaction)
- **Late Fusion**: ~86-87% accuracy (ensemble effect)
- **Weighted Fusion**: ~87-89% accuracy (best, optimized weights)
- **Hybrid Fusion**: ~87-88% accuracy (good balance)

### Clustering
- **Optimal K**: 3 clusters (by Silhouette score)
- **PCA Variance**: ~72% explained by first 2 components

---

## Troubleshooting

### Issue: Cannot connect to API
**Solution**: Make sure backend is running on port 5000
```bash
cd api && python app.py
```

### Issue: Port already in use
**Solution**: Kill the process and restart
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:5000 | xargs kill -9
```

### Issue: Module not found error in Python
**Solution**: Reinstall dependencies
```bash
cd api
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: streamlit command not found
**Solution**: Install UI dependencies in the same Python environment used to run backend/UI
```bash
api/.venv/Scripts/python.exe -m pip install -r api/requirements.txt -r virtual-lab-ui/requirements.txt
```

### Issue: Slow response from backend
**Solution**: First API call caches the dataset. Subsequent calls are faster.
- Initial data load: ~2-3 seconds
- Model training: ~10-15 seconds (first time)
- Fusion analysis: ~20-30 seconds

---

## File Structure

```
d:/DL/ADS_virtual_lab/
├── adult.csv                           # Dataset (32K rows)
├── api/                                # Backend Flask app
│   ├── app.py                         # Main Flask entry point
│   ├── requirements.txt                # Python dependencies
│   ├── .env                           # Environment config
│   ├── .gitignore                     # Git ignore rules
│   └── utils/                         # ML utilities
│       ├── cache.py                   # In-memory caching
│       ├── data_loader.py             # Dataset loading
│       ├── cleaner.py                 # Data cleaning
│       ├── imputer.py                 # Imputation methods
│       ├── preprocessor.py            # Feature engineering
│       ├── ml_models.py               # Model training
│       ├── fusion.py                  # Data fusion
│       └── clustering.py              # K-Means + PCA
└── virtual-lab-ui/                    # Frontend Streamlit app
   ├── app.py                         # Main Streamlit UI
   ├── requirements.txt               # UI dependencies
   ├── .gitignore
   └── README.md                      # Frontend README
```

---

## Performance Notes

- **First API call**: ~2-3s (loads and caches dataset)
- **Model training**: ~10-15s (all 3 models sequentially)
- **Data fusion**: ~20-30s (trains 5 different approaches)
- **Clustering**: ~5-8s (K-Means + PCA)
- **Subsequent API calls**: <1s (cached results)

All times are approximate and depend on your machine.

---

## Customization

### Change Dataset
Replace `adult.csv` with your own CSV file. Update `api/utils/data_loader.py`:
```python
_loader = DataLoader("../your_dataset.csv")
```

### Change Model Parameters
Edit `api/utils/ml_models.py`:
```python
model = RandomForestClassifier(n_estimators=200, random_state=42)  # Increase trees
```

### Change API Port
Edit `api/app.py`:
```python
if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Change port
```

Update the API base URL from the Streamlit sidebar field after launching `virtual-lab-ui/app.py`.

---

## Next Steps

1. **Add More Visualizations**
   - ROC curves
   - Feature importance plots
   - Correlation heatmaps

2. **Enhance Frontend**
   - Download results as CSV/JSON
   - Dark mode toggle
   - Real-time progress bars

3. **Expand Backend**
   - Support more ML models (SVM, Neural Networks)
   - Add cross-validation
   - Implement feature selection

4. **Deployment**
   - Docker containerization
   - Deploy to AWS/Azure/GCP
   - Add user authentication

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review API responses for error messages
3. Check console logs in browser (F12)
4. Check terminal output for backend errors

---

Built with 🎈 Streamlit & 🐍 Flask
ADS Virtual Lab - Advanced Data Science Pipeline
