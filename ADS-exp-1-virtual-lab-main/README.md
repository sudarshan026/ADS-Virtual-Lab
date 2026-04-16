# ADS Exp 1 – Virtual Lab (Statistical Analysis)

A Streamlit-based virtual lab for exploring **descriptive** and **inferential** statistics using real-world CSV datasets.

## Features
- Upload a CSV dataset
- View the dataset in-app
- Descriptive stats (mean, standard deviation) for selected numeric columns
- Visualizations: histogram (with KDE) and box plot
- Scatter plot between two numeric columns
- Pearson correlation and p-value

## Requirements
- Python 3.9+ recommended

## Run locally (Windows)
```powershell
cd "d:\vs code\ads vl"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install streamlit pandas matplotlib seaborn scipy
streamlit run app.py
```

## Usage
1. Start the app with `streamlit run app.py`
2. Open the URL shown in the terminal
3. Go to **Simulation** and upload a `.csv` file

---

Repository: https://github.com/2023vedikadhamale-cell/ADS-exp-1-virtual-lab
