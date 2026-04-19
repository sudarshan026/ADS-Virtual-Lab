r"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                     AUTOML VIRTUAL LAB - Web Application                       ║
║                    Powered by Streamlit & AutoML Engines                       ║
╚════════════════════════════════════════════════════════════════════════════════╝

INSTALLATION INSTRUCTIONS:
==========================
1. Create a Python virtual environment: python -m venv venv
2. Activate the virtual environment:
   - Windows: venv\Scripts\activate
   - macOS/Linux: source venv/bin/activate
3. Install dependencies: pip install -r requirements.txt

RUNNING THE APPLICATION:
========================
streamlit run app.py

The app will open in your default browser at http://localhost:8501

FEATURES:
=========
- Upload CSV datasets
- Auto-detect classification vs regression
- Run 3 AutoML models: FLAML, AutoGluon, H2O AutoML
- Compare model performance
- Visualize feature importance and confusion matrices
- Download predictions as CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, mean_squared_error
)
import plotly.graph_objects as go
import plotly.express as px
import warnings
import time
from io import StringIO
import base64

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AutoML Virtual Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

if 'df' not in st.session_state:
    st.session_state.df = None
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_task_type(y):
    """
    Auto-detect if the problem is classification or regression.
    
    Args:
        y: Target variable (pandas Series)
    
    Returns:
        str: 'classification' or 'regression'
    """
    # Check if target is numeric
    if y.dtype in ['object', 'category', 'bool']:
        return 'classification'
    
    # Check number of unique values
    n_unique = y.nunique()
    n_samples = len(y)
    
    # If few unique values relative to samples, likely classification
    if n_unique < (n_samples ** 0.5):
        return 'classification'
    
    return 'regression'


def preprocess_data(X, y, target_col):
    """
    Preprocess data: handle missing values and encode categorical variables.
    
    Args:
        X: Feature matrix
        y: Target variable
        target_col: Name of target column
    
    Returns:
        tuple: Preprocessed X, y, and encoders dict
    """
    encoders = {}
    
    # Handle missing values - fill with mean for numeric, mode for categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Fill numeric missing values
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].mean())
    
    # Fill categorical missing values
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    # Handle target variable
    if y.dtype in ['object', 'category']:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        encoders['target'] = le
    elif y.dtype == 'bool':
        y = y.astype(int)
    
    return X, y, encoders


def get_download_link(df, filename):
    """Generate a download link for CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOML MODEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_safe_flaml_estimators():
    """Check which FLAML estimators are actually available in the environment."""
    safe_list = ['rf', 'extra_tree'] # These are usually safe as they use sklearn
    
    # Try to detect others
    try:
        import lightgbm
        safe_list.append('lgbm')
    except ImportError:
        pass
        
    try:
        import xgboost
        safe_list.append('xgb_limitdepth')
    except ImportError:
        pass
        
    try:
        import catboost
        safe_list.append('catboost')
    except ImportError:
        pass
        
    return safe_list


def run_flaml_model(X_train, X_test, y_train, y_test, task_type, time_budget=120):
    """
    Run FLAML (Fast and Lightweight AutoML) model.
    
    Args:
        X_train, X_test, y_train, y_test: Train-test split data
        task_type: 'classification' or 'regression'
        time_budget: Time budget for AutoML in seconds
    
    Returns:
        dict: Results including predictions, metrics, and execution time
    """
    try:
        from flaml import AutoML
        
        start_time = time.time()
        
        # Determine internal task type for FLAML
        if task_type == 'classification':
            n_unique = len(np.unique(y_train))
            flaml_task = 'binary' if n_unique <= 2 else 'multi'
            metric = 'accuracy'
        else:
            flaml_task = 'regression'
            metric = 'rmse'
            
        # Get available estimators to avoid 'NoneType' callable errors (especially on Python 3.13)
        estimator_list = get_safe_flaml_estimators()
            
        # Initialize AutoML
        automl = AutoML()
        
        settings = {
            "time_budget": time_budget,
            "metric": metric,
            "task": flaml_task,
            "log_file_name": "flaml.log",
            "seed": 42,
            "estimator_list": estimator_list,
            "eval_method": "auto", # FLAML chooses, but we add safety layer below
        }
        
        try:
            # First attempt with determined estimators
            automl.fit(X_train, y_train, **settings)
        except (TypeError, AttributeError, Exception) as e:
            # Fallback for 'get_n_splits' (AttributeError) or missing estimators (TypeError)
            if any(msg in str(e) for msg in ["get_n_splits", "NoneType", "TypeError"]):
                # REDUCED COMPLEXITY FALLBACK: Use simple Native Holdout with Random Splitting
                # This is the most resilient mode against stratification and splitter crashes
                automl = AutoML() # Deep reset
                automl.fit(
                    X_train=X_train,
                    y_train=y_train,
                    time_budget=time_budget,
                    task=flaml_task,
                    metric=metric,
                    estimator_list=['rf', 'extra_tree'],
                    eval_method='holdout',
                    split_type='random', # Explicitly force simple random split to avoid CV/Stratification bugs
                    split_ratio=0.1,
                    seed=42,
                    log_file_name="flaml.log"
                )
            else:
                raise e
        
        # Validate if a model was actually found
        if automl.best_estimator is None:
            raise ValueError("FLAML could not find a suitable model within the time budget.")
            
        # Predict
        y_pred = automl.predict(X_test)
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics = {
                "Accuracy": round(accuracy, 4),
                "F1-Score": round(f1, 4),
            }
        else:
            # Force Accuracy calculation even for regression per user request
            accuracy = accuracy_score(y_true=np.round(y_test), y_pred=np.round(y_pred))
            metrics = {"Accuracy": round(accuracy, 4)}
        
        return {
            "model": automl,
            "predictions": y_pred,
            "metrics": metrics,
            "execution_time": round(execution_time, 2),
            "model_name": "FLAML",
            "y_true": y_test,
            "success": True,
            "error": None
        }
    
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}"
        if 'NoneType' in str(e):
            error_msg += f"\nNote: Error likely due to missing estimators in Python 3.13.\n{traceback.format_exc()}"
            
        return {
            "model": None,
            "predictions": None,
            "metrics": None,
            "execution_time": None,
            "model_name": "FLAML",
            "y_true": None,
            "success": False,
            "error": error_msg.strip()
        }


def run_autogluon_model(X_train, X_test, y_train, y_test, task_type, time_budget=120):
    """
    Run AutoGluon model with aggressive isolation and retry logic.
    """
    import tempfile
    import shutil
    import os
    import traceback
    import gc
    import time
    
    # AutoGluon specific task mapping
    if task_type == 'classification':
        n_unique = len(np.unique(y_train))
        ag_problem_type = 'binary' if n_unique <= 2 else 'multiclass'
    else:
        ag_problem_type = 'regression'

    import uuid
    
    # Track all temp dirs created for final cleanup
    created_paths = []
    
    def _train():
        # Generate a truly unique, collision-proof path for every attempt
        unique_id = str(uuid.uuid4())[:8]
        # Use a local directory to prevent "path is on mount 'D:', start on mount 'C:'" errors during stacking
        base_temp_dir = os.path.join(os.getcwd(), "temp_ag_models")
        os.makedirs(base_temp_dir, exist_ok=True)
        path = os.path.join(base_temp_dir, f"ag_{unique_id}")
        created_paths.append(path)
        
        from autogluon.tabular import TabularPredictor
        predictor = TabularPredictor(
            label='__target__',
            problem_type=ag_problem_type,
            path=path,
            verbosity=0
        )
        predictor.fit(
            train_data,
            time_limit=time_budget,
            presets='best_quality',
            dynamic_stacking=False # Bypass Ray worker state conflicts on Windows
        )
        return predictor

    # Force garbage collection to clear any lingering AutoGluon state
    gc.collect()
    time.sleep(1)
    
    train_data = X_train.copy()
    train_data['__target__'] = y_train
    
    try:
        start_time = time.time()
        try:
            # First attempt with a fresh UUID path
            predictor = _train()
        except AssertionError as ae:
            if "already fit" in str(ae):
                # NUCLEAR RESET: Multiple GC rounds and longer sleep for Windows lock release
                for _ in range(3): gc.collect()
                time.sleep(3)
                # Retry with a DIFFERENT fresh UUID path
                predictor = _train()
            else:
                raise ae
        
        y_pred = predictor.predict(X_test).values
        execution_time = time.time() - start_time
        
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics = {"Accuracy": round(accuracy, 4), "F1-Score": round(f1, 4)}
        else:
            # Force Accuracy calculation even for regression per user request
            accuracy = accuracy_score(y_true=np.round(y_test), y_pred=np.round(y_pred))
            metrics = {"Accuracy": round(accuracy, 4)}
        
        return {
            "model": predictor,
            "predictions": y_pred,
            "metrics": metrics,
            "execution_time": round(execution_time, 2),
            "model_name": "AutoGluon",
            "y_true": y_test,
            "success": True,
            "error": None
        }
    
    except Exception as e:
        full_traceback = traceback.format_exc()
        return {
            "model": None, "predictions": None, "metrics": None,
            "execution_time": None, "model_name": "AutoGluon", "y_true": None,
            "success": False, "error": f"{str(e)}\n\nDebug Info:\n{full_traceback}"
        }
    finally:
        # Cleanup all paths created during both attempts
        for path in created_paths:
            try:
                shutil.rmtree(path, ignore_errors=True)
            except:
                pass


def run_h2o_model(X_train, X_test, y_train, y_test, task_type, time_budget=120):
    """
    Run H2O AutoML model with ensemble methods.
    """
    try:
        import h2o
        from h2o.automl import H2OAutoML
        
        # Initialize H2O - Ensure cluster is reachable and properly typed
        h2o.init(ignore_config=True, strict_version_check=False)
        
        start_time = time.time()
        
        # Convert to H2O frame
        train_h2o = h2o.H2OFrame(X_train)
        target_frame = h2o.H2OFrame(pd.DataFrame({'target': y_train}))
        
        # In H2O, for classification, the target MUST be a factor
        if task_type == 'classification':
            target_frame['target'] = target_frame['target'].asfactor()
            
        train_h2o = train_h2o.cbind(target_frame)
        test_h2o = h2o.H2OFrame(X_test)
        
        # Run AutoML - Fix verbosity and parameter types
        aml = H2OAutoML(
            max_runtime_secs=int(time_budget),
            seed=42,
            verbosity=None # Set to None instead of 0 to avoid 'int' has no attribute 'lower'
        )
        
        aml.train(y='target', training_frame=train_h2o)
        
        # Predict
        predictions = aml.predict(test_h2o)
        
        # Handle predictions based on task type
        pred_df = predictions.as_data_frame()
        if task_type == "classification":
            y_pred = pred_df['predict'].values
        else:
            y_pred = pred_df['predict'].values
            
        execution_time = time.time() - start_time
        
        # Calculate metrics
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics = {
                "Accuracy": round(accuracy, 4),
                "F1-Score": round(f1, 4),
            }
        else:
            # Force Accuracy calculation even for regression per user request
            accuracy = accuracy_score(y_true=np.round(y_test), y_pred=np.round(y_pred))
            metrics = {"Accuracy": round(accuracy, 4)}
        
        # Shutdown H2O locally for this run
        try:
            h2o.cluster().shutdown()
        except:
            pass
            
        return {
            "model": aml,
            "predictions": y_pred,
            "metrics": metrics,
            "execution_time": round(execution_time, 2),
            "model_name": "H2O AutoML",
            "y_true": y_test,
            "success": True,
            "error": None
        }
    
    except Exception as e:
        import traceback
        full_error = f"{str(e)}\n{traceback.format_exc() if 'lower' in str(e) else ''}"
        try:
            import h2o
            h2o.cluster().shutdown()
        except:
            pass
        
        return {
            "model": None,
            "predictions": None,
            "metrics": None,
            "execution_time": None,
            "model_name": "H2O AutoML",
            "y_true": None,
            "success": False,
            "error": full_error.strip()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(results, metric_name):
    """Create bar chart comparing model performance."""
    models = []
    scores = []
    
    for model_name, result in results.items():
        if result['success'] and result['metrics']:
            models.append(model_name)
            if metric_name in result['metrics']:
                scores.append(result['metrics'][metric_name])
    
    if models:
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=scores,
                marker=dict(
                    color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)],
                    line=dict(color='black', width=1.5)
                ),
                text=scores,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f"Model Comparison: {metric_name}",
            xaxis_title="Model",
            yaxis_title=metric_name,
            height=400,
            showlegend=False
        )
        
        return fig
    return None


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Create confusion matrix heatmap for classification."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred {i}" for i in range(len(cm))],
        y=[f"True {i}" for i in range(len(cm))],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 12},
    ))
    
    fig.update_layout(
        title=f"Confusion Matrix - {model_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    
    return fig


def plot_execution_times(results):
    """Create bar chart for execution times."""
    models = []
    times = []
    
    for model_name, result in results.items():
        if result['success']:
            models.append(model_name)
            times.append(result['execution_time'])
    
    if models:
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=times,
                marker=dict(color='#d62728'),
                text=times,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Model Training Time (seconds)",
            xaxis_title="Model",
            yaxis_title="Time (seconds)",
            height=400,
            showlegend=False
        )
        
        return fig
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown('<h1 class="main-header">AutoML Virtual Lab</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #666;'>Automated Machine Learning Platform - Compare Multiple AutoML Engines</p>",
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR SECTION
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## Data Upload & Configuration")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=['csv'],
    help="Select a CSV file for analysis"
)

if uploaded_file is not None:
    # Read the dataset
    st.session_state.df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset loaded successfully!")
else:
    if st.sidebar.button("Load Sample Dataset"):
        # Create sample dataset
        np.random.seed(42)
        st.session_state.df = pd.DataFrame({
            'Age': np.random.randint(20, 70, 200),
            'Income': np.random.randint(20000, 150000, 200),
            'CreditScore': np.random.randint(300, 850, 200),
            'LoanAmount': np.random.randint(5000, 500000, 200),
            'Default': np.random.choice([0, 1], 200, p=[0.8, 0.2])
        })
        st.sidebar.success("Sample dataset loaded!")

# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.df is not None:
    
    # Target column selection
    st.sidebar.markdown("### Target Variable")
    st.session_state.target_col = st.sidebar.selectbox(
        "Select target column:",
        st.session_state.df.columns,
        help="The column to predict"
    )
    
    # Auto-detect task type
    if st.session_state.target_col:
        st.session_state.task_type = detect_task_type(st.session_state.df[st.session_state.target_col])
    
    # Task type display
    task_emoji = "📊" if st.session_state.task_type == "classification" else "📈"
    st.sidebar.info(f"{task_emoji} **Detected Task:** {st.session_state.task_type.capitalize()}")
    
    # Model selection
    st.sidebar.markdown("### Select AutoML Models")
    models_to_run = {
        "FLAML": st.sidebar.checkbox("FLAML (Fast & Lightweight)", value=True),
        "AutoGluon": st.sidebar.checkbox("AutoGluon (High Accuracy)", value=True),
        "H2O AutoML": st.sidebar.checkbox("H2O AutoML (Ensemble)", value=False)
    }
    
    # Time budget
    st.sidebar.markdown("### AutoML Configuration")
    time_budget = st.sidebar.slider(
        "Time budget (seconds):",
        min_value=30,
        max_value=300,
        value=120,
        step=30
    )
    
    # Test size
    test_size = st.sidebar.slider(
        "Test size ratio:",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05
    )
    
    # Main content area
    st.markdown('<h2 class="section-header">📊 Dataset Preview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Samples", st.session_state.df.shape[0])
    with col2:
        st.metric("Features", st.session_state.df.shape[1])
    with col3:
        st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
    with col4:
        st.metric("Data Types", st.session_state.df.dtypes.nunique())
    
    # Display first few rows
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    
    # Dataset statistics
    with st.expander("Dataset Statistics"):
        st.write(st.session_state.df.describe())
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RUN AUTOML BUTTON
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown('<h2 class="section-header">Run AutoML Models</h2>', unsafe_allow_html=True)
    
    col_run, col_reset = st.columns([3, 1])
    
    run_button = col_run.button(
        "Start AutoML Training",
        type="primary",
        use_container_width=True,
        key="run_button"
    )
    
    if col_reset.button("🔄 Reset", use_container_width=True):
        st.session_state.results = {}
        st.session_state.predictions = None
        st.rerun()
    
    # ═══════════════════════════════════════════════════════════════════════════
    if run_button:
        # Check if at least one model is selected
        if not any(models_to_run.values()):
            st.error("Please select at least one AutoML model!")
        else:
            # Prepare data
            X = st.session_state.df.drop(columns=[st.session_state.target_col])
            y = st.session_state.df[st.session_state.target_col]
            
            # Preprocess
            X, y, encoders = preprocess_data(X.copy(), y.copy(), st.session_state.target_col)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                stratify=y if st.session_state.task_type == "classification" else None
            )
            
            # Run selected models
            with st.spinner("Training models... This may take a few minutes..."):
                for model_name, should_run in models_to_run.items():
                    if should_run:
                        model_placeholder = st.empty()
                        model_placeholder.info(f"Running {model_name}...")
                        
                        if model_name == "FLAML":
                            result = run_flaml_model(
                                X_train, X_test, y_train, y_test,
                                st.session_state.task_type,
                                time_budget
                            )
                        elif model_name == "AutoGluon":
                            result = run_autogluon_model(
                                X_train, X_test, y_train, y_test,
                                st.session_state.task_type,
                                time_budget
                            )
                        elif model_name == "H2O AutoML":
                            result = run_h2o_model(
                                X_train, X_test, y_train, y_test,
                                st.session_state.task_type,
                                time_budget
                            )
                        
                        st.session_state.results[model_name] = result
                        
                        if result['success']:
                            model_placeholder.success(f"{model_name} completed!")
                        else:
                            model_placeholder.error(f"{model_name} failed: {result['error']}")
            
            # Store test data for predictions
            st.session_state.predictions = {
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': {name: res['predictions'] for name, res in st.session_state.results.items() if res['success']}
            }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS SECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    if st.session_state.results:
        st.markdown('<h2 class="section-header">Results & Analysis</h2>', unsafe_allow_html=True)
        
        # Filter successful results
        successful_results = {
            name: res for name, res in st.session_state.results.items()
            if res['success']
        }
        
        if successful_results:
            # Determine Best Model (Always using Accuracy per user request)
            best_model_name = max(successful_results, key=lambda x: successful_results[x]['metrics'].get('Accuracy', 0))
            best_metric_val = successful_results[best_model_name]['metrics'].get('Accuracy', 0)
            metric_unit = "Accuracy"
            
            # Best Model Announcement
            st.success(f"**Winner:** {best_model_name} is the best performing model with a {metric_unit} of **{best_metric_val:.4f}**!")
            st.balloons()
            
            # Metrics display
            st.markdown("### Model Metrics Leaderboard")
            
            metrics_cols = st.columns(len(successful_results))
            for idx, (model_name, result) in enumerate(successful_results.items()):
                with metrics_cols[idx]:
                    is_best = (model_name == best_model_name)
                    st.markdown(f"#### {'⭐ ' if is_best else ''}{model_name}")
                    st.markdown(f"**Time:** {result['execution_time']}s")
                    
                    for metric_name, metric_value in result['metrics'].items():
                        delta_str = ""
                        if metric_name == "Accuracy" and len(successful_results) > 1:
                            avg = np.mean([r['metrics'].get('Accuracy', 0) for r in successful_results.values()])
                            delta = metric_value - avg
                            delta_str = f"{delta:+.2%}"
                        st.metric(metric_name, f"{metric_value:.4f}", delta_str if delta_str else None)
            
            # Model comparison charts
            st.markdown("### Model Comparisons")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                metric_to_compare = "Accuracy"
                
                comparison_fig = plot_model_comparison(successful_results, metric_to_compare)
                if comparison_fig:
                    st.plotly_chart(comparison_fig, use_container_width=True)
            
            with chart_col2:
                time_fig = plot_execution_times(successful_results)
                if time_fig:
                    st.plotly_chart(time_fig, use_container_width=True)
            
            # Confusion matrices for classification
            if st.session_state.task_type == "classification":
                st.markdown("### Confusion Matrices (Classification)")
                
                cm_cols = st.columns(len(successful_results))
                for idx, (model_name, result) in enumerate(successful_results.items()):
                    with cm_cols[idx]:
                        cm_fig = plot_confusion_matrix(
                            result['y_true'],
                            result['predictions'],
                            model_name
                        )
                        st.plotly_chart(cm_fig, use_container_width=True)
            
            # Predictions sample
            st.markdown("### Prediction Samples (First 10)")
            
            predictions_df = st.session_state.predictions['X_test'].head(10).copy()
            predictions_df['Actual'] = st.session_state.predictions['y_test'][:10].values
            
            for model_name, predictions in st.session_state.predictions['y_pred'].items():
                predictions_df[f'{model_name}_Prediction'] = predictions[:10]
            
            st.dataframe(predictions_df, use_container_width=True)
            
            # Download predictions
            st.markdown("### Download Predictions")
            
            full_predictions_df = st.session_state.predictions['X_test'].copy()
            full_predictions_df['Actual'] = st.session_state.predictions['y_test'].values
            
            for model_name, predictions in st.session_state.predictions['y_pred'].items():
                full_predictions_df[f'{model_name}_Prediction'] = predictions
            
            csv = full_predictions_df.to_csv(index=False)
            st.download_button(
                label="Download Full Predictions (CSV)",
                data=csv,
                file_name="automl_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Failed models display
        failed_results = {
            name: res for name, res in st.session_state.results.items()
            if not res['success']
        }
        
        if failed_results:
            st.markdown("### Failed Attempts")
            for model_name, result in failed_results.items():
                with st.expander(f"{model_name}"):
                    st.error(f"**Error:** {result['error']}")
                    st.info("**Tip:** Ensure all required dependencies are installed. Check 'Installation Instructions' at the top of this file.")

else:
    st.info(
        "**Get Started:** Upload a dataset or load a sample dataset from the sidebar to begin!"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
    <div class="footer">
        <p>🤖 <b>AutoML Virtual Lab Project</p>
    </div>
""", unsafe_allow_html=True)
