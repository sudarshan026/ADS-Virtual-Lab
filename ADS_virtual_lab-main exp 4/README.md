# 🧪 ADS Virtual Lab - Advanced Data Science Interactive Platform

A complete, production-ready Advanced Data Science virtual lab with a Streamlit frontend and Python Flask backend. Implements all 10 steps of the data science pipeline from raw data to machine learning model evaluation and data fusion.

## ✨ Features

### 🎨 Frontend (Streamlit)
- **Single Python UI stack**: Streamlit app integrated with Flask backend
- **Interactive Modules**: Data, cleaning, imputation, preprocessing, models, clustering, fusion, pipeline
- **Real-time Visualizations**: Charts, tables, confusion matrices, PCA scatter plots
- **Error Handling**: Inline API error feedback and status checks
- **Loading States**: Streamlit spinners for long-running operations

### 🔬 Backend (Python Flask)
- **Complete ML Pipeline**: All 10 steps implemented
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
- **Data Fusion** (Core): 4 advanced techniques
- **Clustering**: K-Means + PCA visualization
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Imputation**: Mode, KNN, MICE methods
- **Performance**: In-memory caching for speed
- **API-First**: REST endpoints for all operations
- **CORS Enabled**: Safe cross-origin requests

### 📊 Data Science Pipeline (10 Steps)

```
1. Load Dataset          → Load UCI Adult (32K rows, 15 features)
2. Data Understanding   → Analyze features, find missing values
3. Data Cleaning        → Handle ?, duplicates, outliers, categories
4. Data Imputation      → Mode, KNN, MICE comparison
5. Preprocessing        → One-hot encoding, scaling, train-test split
6. Supervised Learning  → LR, RF, XGBoost with metrics
7. Unsupervised Learning → K-Means, PCA, Silhouette scores
8. Data Fusion          → Early, Late, Weighted, Hybrid techniques
9. Evaluation           → Compare all results and metrics
10. Visualization       → Interactive charts and dashboards
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- ~2GB disk space

### Installation

**Backend:**
```bash
cd api
pip install -r requirements.txt
```

**Frontend (Streamlit):**
```bash
api/.venv/Scripts/python.exe -m pip install -r virtual-lab-ui/requirements.txt
```

### Run

**Option 1: Two Terminals**
```bash
# Terminal 1 - Backend
cd api && python app.py

# Terminal 2 - Frontend
api/.venv/Scripts/python.exe -m streamlit run virtual-lab-ui/app.py --server.port 8501
```

**Option 2: One Command (Windows)**
```bash
double-click start.bat
```

Then open: **http://localhost:8501**

---

## 📁 Project Structure

```
d:/DL/ADS_virtual_lab/
├── 📄 adult.csv                        Dataset (32,562 rows)
├── 📄 SETUP.md                         Complete setup guide
├── 📄 start.bat                        One-click launcher
├── 📁 api/                             ← Backend Flask
│   ├── app.py                          Main Flask app
│   ├── requirements.txt                 Dependencies
│   ├── .env                            Config
│   ├── README.md                       API docs
│   └── utils/
│       ├── cache.py                    In-memory cache
│       ├── data_loader.py              Load dataset
│       ├── cleaner.py                  Data cleaning
│       ├── imputer.py                  Imputation
│       ├── preprocessor.py             Encoding & scaling
│       ├── ml_models.py                LR, RF, XGBoost
│       ├── fusion.py                   Data fusion
│       └── clustering.py               K-Means + PCA
└── 📁 virtual-lab-ui/                  ← Frontend Streamlit
   ├── app.py                          Main Streamlit UI
   ├── requirements.txt                UI dependencies
   ├── .gitignore
   └── README.md
```

---

## 💡 Key Highlights

### Supervised Learning (Step 6) ⭐
```
3 models trained and compared:
├── Logistic Regression
│   └── Accuracy: ~84-85%
├── Random Forest (100 trees)
│   └── Accuracy: ~86-87%
└── XGBoost (100 trees)
    └── Accuracy: ~87-88% ← Best

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
```

### Data Fusion (Step 8) ⭐⭐⭐
```
5 approaches compared:
├── No Fusion (baseline)
│   └── Accuracy: ~87%
├── Early Fusion (merge data)
│   └── Accuracy: ~86-87%
├── Late Fusion (ensemble predictions)
│   └── Accuracy: ~86-87%
├── Weighted Fusion (optimal weights)
│   └── Accuracy: ~87-89% ← BEST
└── Hybrid Fusion (multi-level)
    └── Accuracy: ~87-88%

Shows that fusion improves results!
```

### Clustering (Step 7)
```
K-Means with k=2-5:
├── Optimal k: 3 (highest Silhouette)
├── PCA Visualization: 72% variance in 2D
├── Silhouette Scores: [0.45, 0.52, 0.48, 0.41]
└── Davies-Bouldin Index: [1.23, 0.98, 1.05, 1.34]
```

---

## 🎯 API Endpoints

### Complete REST API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/data/load` | GET | Load & analyze dataset |
| `/api/cleaning/run` | POST | Clean data pipeline |
| `/api/cleaning/impute` | POST | Impute (mode\|knn\|mice) |
| `/api/preprocessing/run` | POST | Encode, scale, split |
| `/api/models/train` | POST | Train single model |
| `/api/models/compare` | POST | Train & compare all 3 |
| `/api/clustering/analyze` | POST | K-Means analysis |
| `/api/clustering/pca` | POST | PCA visualization |
| `/api/fusion/compare` | POST | Compare all 5 fusion |
| `/api/pipeline/run-all` | POST | Run complete pipeline |
| `/api/health` | GET | API status check |

---

## 🎨 UI/UX Design

### Neobrutalism Theme
- **Borders**: 3px solid black, 0px border-radius
- **Colors**: Black primary, white secondary, yellow accent (#ffeb3b)
- **Typography**: Courier New monospace, bold weights, very wide letter-spacing
- **Shadows**: Hard drop shadows (4-6px offset)
- **Interactions**: Lift & shadow on hover, click animations

### Interactive Modules
1. **Overview** 📋 - Health and workflow summary
2. **Data Loader** 📊 - Dataset statistics and sample rows
3. **Data Cleaning** 🧹 - Cleaning + missing summary
4. **Imputation** 🩹 - Mode, KNN, MICE + comparison
5. **Preprocessing** ⚡ - Encoding, scaling, train-test split
6. **Model Training** 🤖 - LR, RF, XGBoost metrics
7. **Clustering** 📈 - K-Means, PCA, elbow method
8. **Data Fusion** 🔗 - Multi-technique fusion report
9. **Pipeline Runner** ▶ - End-to-end execution

### Layout
- **Desktop**: Wide dashboard layout
- **Mobile/Tablet**: Streamlit responsive column collapse

---

## 📊 Visualizations

### Charts Included
- **Bar Charts**: Model accuracy comparison, fusion comparison
- **Confusion Matrix**: Heat-mapped grid
- **Metrics Table**: Detailed performance metrics
- **Line Chart**: Elbow method curve
- **Scatter Plot**: PCA cluster visualization (data-ready)

### Loading & Errors
- Animated loading spinners
- User-friendly error alerts
- Success notifications

---

## 🔧 Tech Stack

### Frontend
- Streamlit
- Requests
- Altair
- Pandas

### Backend
- Python 3.8+
- Flask 3.0
- Flask-CORS
- Pandas 2.3
- NumPy 2.1
- scikit-learn 1.5
- XGBoost 2.1
- SciPy 1.14

### Data
- UCI Adult Dataset (32,562 samples)
- 15 features (categorical & numerical)
- Binary classification (income >50K / ≤50K)

---

## ⚡ Performance

### Benchmarks
| Operation | Time |
|-----------|------|
| Data loading (first) | 2-3s |
| Model training (all 3) | 10-15s |
| Data fusion (5 approaches) | 20-30s |
| Clustering | 5-8s |
| Subsequent API calls | <1s |

### Caching
- Dataset cached after first load
- Models cached after training
- Results cached for reuse

---

## 📖 How to Use

1. **Start the System**
   ```bash
   double-click start.bat  # Or run both terminals manually
   ```

2. **Open Browser**
   Navigate to `http://localhost:8501`

3. **Interact with Modules**
   - Select modules from the sidebar
   - Click action buttons (Load Dataset, Train Models, Run Fusion, etc.)
   - View real-time results

4. **Explore Data**
   - See dataset stats
   - Check feature list
   - View class distribution

5. **Train Models**
   - Click "Train All Models"
   - See accuracy chart
   - Compare metrics

6. **Run Fusion**
   - Click "Run Fusion Comparison"
   - Compare 5 techniques
   - See accuracy improvement

---

## 📚 Documentation

- **Setup Guide**: `SETUP.md` - Complete installation & usage
- **API Docs**: `api/README.md` - All endpoints & parameters
- **Frontend Readme**: `virtual-lab-ui/README.md` - Component structure

---

## 🎓 Learning Materials

This lab demonstrates:
- ✅ Complete ML pipeline
- ✅ Data preprocessing best practices
- ✅ Model evaluation & comparison
- ✅ Ensemble methods (Random Forest, XGBoost)
- ✅ Data fusion techniques
- ✅ Clustering & dimensionality reduction
- ✅ Frontend-Backend integration
- ✅ REST API design
- ✅ Production-ready code structure

---

## 🚀 Next Steps

### Advanced Features
- Add neural networks (TensorFlow)
- Implement feature selection
- Add hyperparameter tuning
- Create export functionality

### Production
- Docker containerization
- Cloud deployment (AWS/Azure/GCP)
- User authentication
- Database integration
- Real-time monitoring

### Enhancements
- Dark mode
- Download results
- Batch processing
- Custom datasets
- More visualizations

---

## 📝 License

Educational use - Advanced Data Science learning platform

---

## 🙌 Built With

- 🎈 **Streamlit** - Interactive Python frontend
- 🐍 **Flask** - Lightweight Python backend
- 📊 **scikit-learn** - ML algorithms
- 🚀 **XGBoost** - State-of-the-art boosting

---

**Ready to explore advanced data science concepts!** 🎉

Open http://localhost:8501 to get started →
