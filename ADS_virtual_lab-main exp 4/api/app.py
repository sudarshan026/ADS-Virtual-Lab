from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import sys
import traceback

# Load environment variables
load_dotenv()

# Import utilities
from utils.data_loader import _loader
from utils.cleaner import cleaner
from utils.imputer import imputer
from utils.preprocessor import preprocessor
from utils.ml_models import trainer
from utils.fusion import fusion
from utils.clustering import analyzer
from utils.cache import cache

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:8501", "http://localhost:3000"]}})

# Error handler
@app.errorhandler(Exception)
def handle_error(error):
    """Handle all errors"""
    print(f"Error: {str(error)}")
    traceback.print_exc()
    return jsonify({
        "error": str(error),
        "type": type(error).__name__
    }), 500

# Health check
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "cache_size": cache.size()
    })

# ==================== DATA LOADING ====================
@app.route("/api/data/load", methods=["GET"])
def load_data():
    """Load and analyze dataset"""
    try:
        _loader.load()
        stats = _loader.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/data/sample", methods=["GET"])
def get_sample():
    """Get sample rows from dataset"""
    n = request.args.get("n", 5, type=int)
    sample = _loader.get_sample(n)
    return jsonify({"sample": sample})

# ==================== DATA CLEANING ====================
@app.route("/api/cleaning/run", methods=["POST"])
def run_cleaning():
    """Run data cleaning pipeline"""
    try:
        result = cleaner.clean()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/cleaning/missing-values", methods=["GET"])
def get_missing_values():
    """Get missing value summary"""
    summary = cleaner.get_missing_value_summary()
    return jsonify({"missing_values": summary})

# ==================== DATA IMPUTATION ====================
@app.route("/api/cleaning/impute", methods=["POST"])
def impute_data():
    """Impute missing values"""
    try:
        data = request.get_json()
        method = data.get("method", "mode")

        if method == "mode":
            result = imputer.impute_mode()
        elif method == "knn":
            result = imputer.impute_knn()
        elif method == "mice":
            result = imputer.impute_mice()
        else:
            return jsonify({"error": f"Unknown method: {method}"}), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/cleaning/impute/compare", methods=["POST"])
def compare_imputation():
    """Compare all imputation methods"""
    try:
        result = imputer.compare_methods()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ==================== PREPROCESSING ====================
@app.route("/api/preprocessing/run", methods=["POST"])
def preprocess():
    """Run preprocessing pipeline"""
    try:
        data = request.get_json() or {}
        imputation_method = data.get("imputation_method", "mode")
        test_size = data.get("test_size", 0.2)

        result = preprocessor.preprocess(imputation_method, test_size)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ==================== MODEL TRAINING ====================
@app.route("/api/models/train", methods=["POST"])
def train_model():
    """Train a single model"""
    try:
        data = request.get_json()
        model_name = data.get("model", "random_forest")

        result = trainer.train_model(model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/models/compare", methods=["POST"])
def compare_models():
    """Compare all models"""
    try:
        results = trainer.compare_models()
        return jsonify({"models": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ==================== CLUSTERING ====================
@app.route("/api/clustering/analyze", methods=["POST"])
def analyze_clustering():
    """Analyze K-Means clustering"""
    try:
        result = analyzer.analyze_clusters()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/clustering/pca", methods=["POST"])
def get_pca():
    """Get PCA visualization"""
    try:
        data = request.get_json() or {}
        k = data.get("k", 3)
        result = analyzer.get_pca_visualization(k=k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/clustering/elbow", methods=["GET"])
def get_elbow():
    """Get elbow curve data"""
    try:
        result = analyzer.get_elbow_curve()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ==================== DATA FUSION ====================
@app.route("/api/fusion/compare", methods=["POST"])
def compare_fusion():
    """Compare all fusion techniques"""
    try:
        results = fusion.compare_all_techniques()
        return jsonify({"fusion_results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/fusion/early", methods=["POST"])
def early_fusion():
    """Early fusion technique"""
    try:
        result = fusion.early_fusion()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/fusion/late", methods=["POST"])
def late_fusion():
    """Late fusion technique"""
    try:
        result = fusion.late_fusion()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/fusion/weighted", methods=["POST"])
def weighted_fusion():
    """Weighted fusion technique"""
    try:
        result = fusion.weighted_fusion()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/fusion/hybrid", methods=["POST"])
def hybrid_fusion():
    """Hybrid fusion technique"""
    try:
        result = fusion.hybrid_fusion()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ==================== PIPELINE ====================
@app.route("/api/pipeline/run-all", methods=["POST"])
def run_all_pipeline():
    """Run complete pipeline in sequence"""
    try:
        steps = []

        # 1. Load data
        _loader.load()
        steps.append({"step": "Data Loading", "status": "completed"})

        # 2. Clean data
        cleaner.clean()
        steps.append({"step": "Data Cleaning", "status": "completed"})

        # 3. Imputation
        imputer.impute_mode()
        steps.append({"step": "Data Imputation", "status": "completed"})

        # 4. Preprocessing
        preprocessor.preprocess()
        steps.append({"step": "Data Preprocessing", "status": "completed"})

        # 5. Model training
        trainer.compare_models()
        steps.append({"step": "Model Training", "status": "completed"})

        # 6. Clustering
        analyzer.analyze_clusters()
        steps.append({"step": "Clustering Analysis", "status": "completed"})

        # 7. Fusion
        fusion.compare_all_techniques()
        steps.append({"step": "Data Fusion", "status": "completed"})

        return jsonify({
            "status": "success",
            "message": "Complete pipeline executed successfully",
            "steps": steps
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
