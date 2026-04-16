# ADS Virtual Lab UI (Streamlit)

This folder now contains the Streamlit frontend for the ADS Virtual Lab.

The Streamlit app connects to the Flask backend in `api/` and exposes the full workflow:

1. Data loading
2. Data cleaning
3. Imputation (mode, knn, mice)
4. Preprocessing
5. Supervised model training
6. Clustering and PCA
7. Fusion techniques
8. Full pipeline execution

## Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Streamlit UI dependencies

## Run

From workspace root:

```bash
# 1) Start backend
api/.venv/Scripts/python.exe api/app.py

# 2) Start Streamlit UI
api/.venv/Scripts/python.exe -m streamlit run virtual-lab-ui/app.py --server.port 8501
```

Open:

- Backend API: `http://localhost:5000`
- Streamlit UI: `http://localhost:8501`

## Install dependencies

If needed, install both backend and UI dependencies in the same virtual environment:

```bash
api/.venv/Scripts/python.exe -m pip install -r api/requirements.txt -r virtual-lab-ui/requirements.txt
```

## Notes

- This folder is Streamlit-only and contains only Python UI files.
- The app allows changing the API base URL from the sidebar if your backend host/port changes.
