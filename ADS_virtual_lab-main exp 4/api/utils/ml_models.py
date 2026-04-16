import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from .cache import cache
from .preprocessor import preprocessor


class MLModelTrainer:
    """Train and evaluate ML models"""

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

    def load_data(self):
        """Load preprocessed data"""
        if cache.exists("preprocessed_data"):
            data = cache.get("preprocessed_data")
        else:
            preprocessor.preprocess()
            data = cache.get("preprocessed_data")

        self.X_train = data["X_train"]
        self.X_test = data["X_test"]
        self.y_train = data["y_train"]
        self.y_test = data["y_test"]

    def train_model(self, model_name):
        """Train a single model"""
        if self.X_train is None:
            self.load_data()

        start_time = time.time()

        if model_name == "logistic_regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_name == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_name == "xgboost":
            model = XGBClassifier(n_estimators=100, random_state=42, verbose=0)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Train
        model.fit(self.X_train, self.y_train)
        training_time = (time.time() - start_time) * 1000

        # Predict
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        cm = confusion_matrix(self.y_test, y_pred)

        # Feature importance (if available)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_[:10]  # Top 10
            feature_importance = {f"Feature {i}": round(float(imp), 4) for i, imp in enumerate(importances)}

        # Cache model
        self.models[model_name] = model
        cache.set(f"model_{model_name}", model)

        result = {
            "model": model_name,
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4)
            },
            "confusion_matrix": cm.tolist(),
            "training_time_ms": round(training_time, 2),
            "feature_importance": feature_importance,
            "test_samples": len(self.y_test)
        }

        self.results[model_name] = result
        cache.set(f"result_{model_name}", result)

        return result

    def compare_models(self):
        """Train all models and compare"""
        results = []
        for model_name in ["logistic_regression", "random_forest", "xgboost"]:
            result = self.train_model(model_name)
            results.append({
                "model": result["model"],
                "accuracy": result["metrics"]["accuracy"],
                "precision": result["metrics"]["precision"],
                "recall": result["metrics"]["recall"],
                "f1_score": result["metrics"]["f1_score"],
                "roc_auc": result["metrics"]["roc_auc"],
                "training_time_ms": result["training_time_ms"]
            })

        # Find best
        best_model = max(results, key=lambda x: x["accuracy"])
        results.append({
            "best_model": best_model["model"],
            "best_accuracy": best_model["accuracy"]
        })

        cache.set("model_comparison", results)
        return results

    def predict(self, model_name, X):
        """Make predictions with trained model"""
        if model_name not in self.models:
            if cache.exists(f"model_{model_name}"):
                self.models[model_name] = cache.get(f"model_{model_name}")
            else:
                self.train_model(model_name)

        model = self.models[model_name]
        return model.predict(X)


trainer = MLModelTrainer()
