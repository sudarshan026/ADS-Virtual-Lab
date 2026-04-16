# ADS - Virtual Lab

Time Series Forecasting virtual laboratory using Streamlit.

## Features

- Step-wise learning flow (Aim, Theory, Dataset, Visualisation, Decomposition, Modelling, Results, Quiz, References)
- Built-in datasets and CSV upload support
- Forecasting with Moving Average and ARIMA
- Evaluation using MAE, MSE, RMSE, and MAPE
- Self-assessment quiz and curated learning references

## Run Locally (Windows CMD)

```cmd
cd /d "C:\Users\Anishka\Documents\Hanishka\Virtual lab\ADS_Virtual_Lab"
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install plotly
streamlit run exp7.py
```

## References

### Books and Papers

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. Time Series Analysis: Forecasting and Control. Wiley: https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021
2. Hyndman, R. J., and Athanasopoulos, G. Forecasting: Principles and Practice (open textbook): https://otexts.com/fpp3/
3. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2018). The M4 Competition: Results, findings, conclusion and way forward. International Journal of Forecasting: https://www.sciencedirect.com/science/article/pii/S0169207018300785
4. Hyndman, R. J., and Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R. Journal of Statistical Software: https://www.jstatsoft.org/article/view/v027i03

### Practical Documentation

- Statsmodels ARIMA API: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
- Statsmodels seasonal decomposition API: https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
- Pandas Time Series User Guide: https://pandas.pydata.org/docs/user_guide/timeseries.html

### YouTube Videos

- StatQuest: ARIMA clearly explained: https://www.youtube.com/watch?v=-aCF0_wfVwY
- freeCodeCamp: Time Series Forecasting full tutorial: https://www.youtube.com/watch?v=0E_31WqVzCY
- Krish Naik: ARIMA in Python practical: https://www.youtube.com/watch?v=8FCDpFhd1zk
