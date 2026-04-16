# ADS Virtual Lab API Backend

Python Flask backend for the Advanced Data Science Virtual Lab. Implements complete ML pipeline with data fusion techniques.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

API will be available at: `http://localhost:5000`

## Features

### 1. Data Management
- Load UCI Adult dataset (32K+ rows)
- Analyze feature types and distributions
- Identify missing values

### 2. Data Cleaning
- Handle missing values ("?" markers)
- Remove duplicate records
- Fix inconsistent categories
- Detect outliers using IQR method

### 3. Data Imputation
- **Mode**: Fill categorical/numerical with mode/mean
- **KNN**: K-Nearest neighbors imputation (k=5)
- **MICE**: Multivariate imputation by chained equations
- Compare all methods with metrics

### 4. Preprocessing
- One-hot encoding for categorical features
- StandardScaler for numerical features
- Train-test split (80-20)

### 5. Supervised Learning
- **Logistic Regression**: Baseline linear classifier
- **Random Forest**: Ensemble (100 trees)
- **XGBoost**: Gradient boosting (100 trees)

Metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Confusion Matrix
- Feature Importance

### 6. Unsupervised Learning
- K-Means clustering (k=2-5)
- PCA visualization (2D)
- Silhouette Score, Davies-Bouldin Index
- Elbow method analysis

### 7. Data Fusion
- **Early Fusion**: Merge → train single model
- **Late Fusion**: Train separate → average predictions
- **Weighted Fusion**: Learn optimal weights
- **Hybrid Fusion**: Early (A+B) + Late (C)
- **Stacking**: Meta-model approach (optional)

### 8. Evaluation
Compare techniques with:
- Accuracy
- F1-Score
- ROC-AUC
- Training time

## API Endpoints

### Data
- `GET /api/data/load` - Load and analyze dataset
- `GET /api/data/sample?n=5` - Get sample rows

### Cleaning & Imputation
- `POST /api/cleaning/run` - Run cleaning pipeline
- `POST /api/cleaning/impute` - Impute missing values (mode|knn|mice)
- `POST /api/cleaning/impute/compare` - Compare all imputation methods
- `GET /api/cleaning/missing-values` - Get missing value summary

### Preprocessing
- `POST /api/preprocessing/run` - Encode, scale, split data

### Models
- `POST /api/models/train` - Train single model
- `POST /api/models/compare` - Train and compare all models

### Clustering
- `POST /api/clustering/analyze` - K-Means analysis
- `POST /api/clustering/pca` - Get PCA visualization
- `GET /api/clustering/elbow` - Elbow curve data

### Fusion
- `POST /api/fusion/compare` - Compare all techniques
- `POST /api/fusion/early` - Early fusion
- `POST /api/fusion/late` - Late fusion
- `POST /api/fusion/weighted` - Weighted fusion
- `POST /api/fusion/hybrid` - Hybrid fusion

### Pipeline
- `POST /api/pipeline/run-all` - Execute complete pipeline
- `GET /api/health` - Health check

## Architecture

```
api/
├── app.py                    # Flask app & routes
├── requirements.txt          # Python dependencies
├── .env                     # Environment config
└── utils/
    ├── cache.py             # In-memory caching
    ├── data_loader.py       # Load & analyze
    ├── cleaner.py           # Data cleaning
    ├── imputer.py           # Imputation (Mode/KNN/MICE)
    ├── preprocessor.py      # Encoding & scaling
    ├── ml_models.py         # Model training
    ├── fusion.py            # Data fusion
    └── clustering.py        # K-Means + PCA
```

## Performance

- **First API call**: ~2-3s (dataset caching)
- **Model training**: ~10-15s (all 3 models)
- **Data fusion**: ~20-30s (5 approaches)
- **Clustering**: ~5-8s
- **Subsequent calls**: <1s (cached)

## Configuration

Edit `.env` for:
```
FLASK_ENV=development
FLASK_DEBUG=True
API_PORT=5000
```

## Dependencies

Core ML libraries:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML models & metrics
- `xgboost` - Gradient boosting
- `scipy` - Scientific computing
- `flask` - Web framework
- `flask-cors` - Cross-origin requests

## Testing

Check health:
```bash
curl http://localhost:5000/api/health
```

Load data:
```bash
curl http://localhost:5000/api/data/load
```

Train models:
```bash
curl -X POST http://localhost:5000/api/models/compare
```

## Error Handling

All endpoints return JSON with errors:
```json
{
  "error": "Error message here",
  "type": "Exception type"
}
```

## Troubleshooting

**Port in use**: Kill process on 5000
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Missing dependencies**: Reinstall
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Memory issues**: Data is cached automatically
- First load: ~100MB RAM
- After: ~500MB RAM (cached models)

## Integration

Frontend connects via `http://localhost:5000/api`

CORS enabled for:
- `http://localhost:8501` (Streamlit UI)
- `http://localhost:3000` (alternative)

Update in `app.py` if needed:
```python
CORS(app, resources={r"/api/*": {"origins": ["your-url"]}})
```

---

For complete setup instructions, see: `../SETUP.md`
