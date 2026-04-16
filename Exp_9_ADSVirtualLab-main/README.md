# 🤖 AutoML Virtual Lab - Complete Documentation

## Overview

AutoML Virtual Lab is a powerful web application built with Streamlit that enables users to automatically apply machine learning models to their datasets using multiple AutoML engines. The application intelligently detects the problem type (classification or regression) and runs state-of-the-art AutoML frameworks.

## 🎯 Features

### Data Management
- **CSV Upload**: Upload any CSV dataset
- **Sample Dataset**: Load a pre-built sample dataset for testing
- **Data Preview**: View dataset statistics and first few rows
- **Auto Preprocessing**: Automatic handling of missing values and categorical encoding

### AutoML Models
- **FLAML**: Fast and lightweight AutoML (fastest training)
- **AutoGluon**: High-accuracy ensemble methods
- **H2O AutoML**: Advanced ensemble-based approach

### Intelligent Features
- **Auto Task Detection**: Automatically detects classification vs regression
- **Smart Metrics**: Shows appropriate metrics based on task type:
  - Classification: Accuracy, F1-Score, Confusion Matrix
  - Regression: RMSE
- **Configurable Training**: Adjustable time budget and test size
- **Model Comparison**: Compare multiple models side-by-side

### Visualizations
- Model performance comparison charts
- Execution time comparison
- Confusion matrices (for classification)
- Prediction samples

### Output & Export
- Download predictions as CSV
- View sample predictions
- Performance metrics for each model
- Execution time tracking

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager
- At least 8GB RAM (AutoML packages are resource-intensive)

### Step 1: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements (takes 10-15 minutes)
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📊 How to Use

### 1. Load Data
- Click "Upload CSV Dataset" to select a CSV file, OR
- Click "Load Sample Dataset" to use a pre-built example

### 2. Configure
- Select the **Target Column** (the column you want to predict)
- The app automatically detects if it's classification or regression
- Adjust **Time Budget** (how long each model trains, 30-300 seconds)
- Adjust **Test Size** (portion of data for testing, 10-50%)

### 3. Select Models
- Choose which AutoML models to run:
  - ⚡ FLAML (recommended for speed)
  - 🏆 AutoGluon (recommended for accuracy)
  - 💧 H2O AutoML (recommended for ensemble)

### 4. Train Models
- Click "🎯 Start AutoML Training"
- Wait for models to complete (progress shown)
- View results as they complete

### 5. Analyze Results
- Compare model metrics (accuracy, RMSE, F1-score)
- View confusion matrices (classification only)
- Analyze execution times
- Review prediction samples

### 6. Export
- Download predictions as CSV file
- Contains test features, actual values, and predictions from each model

### 7. Reset
- Click "🔄 Reset" to clear results and start over

## 📁 Project Structure

```
ADSvirtuallab/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python package dependencies
└── README.md              # This file
```

## 🔧 Configuration Options

### In the Sidebar:

| Option | Range | Default | Description |
|--------|-------|---------|-------------|
| Target Column | Any column | - | The variable to predict |
| Time Budget | 30-300 sec | 120 | Max training time per model |
| Test Size | 0.1-0.5 | 0.2 | Fraction of data for testing |

## 📈 Understanding Results

### For Classification:
- **Accuracy**: Percentage of correct predictions (0-1)
- **F1-Score**: Balance between precision and recall (0-1)
- **Confusion Matrix**: Shows true positives, false positives, etc.

### For Regression:
- **RMSE**: Root Mean Squared Error (lower is better)
- Shows average prediction error

### Execution Time:
- Total time to train the model
- Useful for comparing training efficiency
- Includes preprocessing time

## ⚠️ Troubleshooting

### Issue: Module not found errors
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Issue: AutoGluon or H2O not working
**Solution**: Install with no build isolation:
```bash
pip install autogluon --no-build-isolation
pip install h2o --no-build-isolation
```

### Issue: Out of memory
**Solution**: AutoML models are memory-intensive. Try:
- Using a smaller dataset
- Reducing time budget
- Using only FLAML (most memory efficient)
- Close other applications

### Issue: Very slow performance
**Solution**: This is normal for AutoML:
- First run may be slower (model compilation)
- Larger datasets take longer
- H2O may take longer than FLAML
- Increase time budget if models seem incomplete

### Issue: Predictions differ between runs
**Solution**: This is normal - AutoML uses randomization. Set seed in code if reproducibility needed.

## 🎨 UI Features

- **Sidebar**: Data upload and model configuration
- **Main Panel**: Results and visualizations
- **Status Indicators**: ✅ Success, ❌ Error, 🔄 Loading
- **Metrics Cards**: Quick view of key performance metrics
- **Interactive Charts**: Plotly-powered visualizations
- **Responsive Layout**: Works on desktop and tablet

## 💡 Tips & Best Practices

1. **Data Quality**: Clean your data before uploading for better results
2. **Sample Size**: At least 100 samples recommended
3. **Training Time**: Longer time budget (120-300s) often yields better models
4. **Model Selection**: Use multiple models to find the best performer
5. **Test Size**: Use 20% for small datasets, up to 30% for large ones
6. **Missing Values**: App handles automatically, but fewer is better

## 🔬 Technical Details

### Data Preprocessing
- Missing numeric values: Filled with mean
- Missing categorical values: Filled with mode
- Categorical encoding: LabelEncoder
- Train-test split: Stratified for classification

### Model Parameters
- FLAML: Time-based optimization, random forest + gradient boosting
- AutoGluon: Ensemble stacking, supports all tree models
- H2O: Deep learning + tree models, ensemble voting

### Supported Data Types
- Numeric (int, float)
- Categorical (string, object)
- Binary (bool)
- Missing values (NaN)

## 📦 Dependencies

Key packages used:
- **Streamlit**: Web framework
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: Machine learning utilities
- **FLAML/AutoGluon/H2O**: AutoML engines
- **Plotly**: Interactive visualizations

## ✨ Advanced Features

### Automatic Task Detection
The app identifies the problem type by analyzing:
- Data type of target variable
- Number of unique values
- Ratio of unique values to samples

### Intelligent Preprocessing
- Handles all missing values automatically
- Encodes categorical variables
- Scales features appropriately for each model

### Performance Metrics
- Classification: Weighted F1-score for imbalanced data
- Stratified splitting for balanced train-test sets
- Execution time measurement

## 🎯 Use Cases

1. **Business Analytics**: Predict customer churn, sales forecasting
2. **Healthcare**: Disease prediction, patient outcome analysis
3. **Finance**: Credit risk assessment, fraud detection
4. **Operations**: Maintenance prediction, resource forecasting
5. **Education**: Student performance prediction
6. **Research**: Scientific data analysis, hypothesis testing

## 🔐 Security & Privacy

- All processing happens locally on your machine
- No data is sent to external servers
- Models are trained in temporary directories
- H2O cluster is properly shut down after use

## 📝 License & Credits

Built with ❤️ using:
- [Streamlit](https://streamlit.io/)
- [FLAML](https://microsoft.github.io/FLAML/)
- [AutoGluon](https://auto.gluon.ai/)
- [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

## 🤝 Contributing

To extend the application:
1. Add new AutoML models in `app.py`
2. Create new visualization functions
3. Add custom preprocessing steps
4. Implement additional metrics

## 📞 Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Verify all dependencies are installed
3. Ensure your dataset is in CSV format
4. Try with the sample dataset first
5. Check system resource availability

## 🎓 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [FLAML GitHub](https://github.com/microsoft/FLAML)
- [AutoGluon Documentation](https://auto.gluon.ai/)
- [H2O AutoML Guide](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

**AutoML Virtual Lab Project** | 2024 | Built for Automated Machine Learning

Made with 🤖 and ❤️
