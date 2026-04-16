import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from .cache import cache
from .preprocessor import preprocessor


class DataFusion:
    """Data Fusion techniques for combining multiple sources"""

    def __init__(self):
        self.sources = {}
        self.study_guide = {
            "No Fusion (Baseline)": {
                "concept": "Single-source benchmark",
                "best_for": "Checking whether fusion really helps",
                "student_task": "Use this score as your baseline before testing fusion strategies."
            },
            "Early Fusion": {
                "concept": "Feature-level fusion",
                "best_for": "Strongly correlated sources",
                "student_task": "Inspect which feature groups dominate model performance after concatenation."
            },
            "Late Fusion": {
                "concept": "Decision-level fusion",
                "best_for": "When each source has independent predictive power",
                "student_task": "Compare averaged probabilities to each individual source output."
            },
            "Weighted Fusion": {
                "concept": "Reliability-aware ensemble",
                "best_for": "Uneven source quality",
                "student_task": "Analyze how source weights shift when one source is noisier than others."
            },
            "CNN Fusion": {
                "concept": "Deep feature interactions (tabular approximation)",
                "best_for": "Capturing nonlinear interactions across modalities",
                "student_task": "Compare this nonlinear approach against tree-based techniques."
            },
            "Attention Fusion": {
                "concept": "Sample-wise source emphasis",
                "best_for": "Cases where useful modality changes per row",
                "student_task": "Observe average attention weights and relate them to source reliability."
            },
            "Gated Fusion": {
                "concept": "Meta-learner controlled fusion",
                "best_for": "Combining confidence-aware source outputs",
                "student_task": "Inspect how gate inputs (probability + confidence) affect final predictions."
            },
            "Hybrid Fusion": {
                "concept": "Multi-stage ensemble",
                "best_for": "Balancing robustness and accuracy",
                "student_task": "Evaluate whether blending improves stability over any single strategy."
            },
        }

    def create_data_sources(self):
        """Create 3 synthetic data sources from the Adult dataset"""
        if cache.exists("preprocessed_data"):
            data = cache.get("preprocessed_data")
        else:
            preprocessor.preprocess()
            data = cache.get("preprocessed_data")

        X_train = data["X_train"].copy()
        X_test = data["X_test"].copy()
        y_train = data["y_train"].copy()
        y_test = data["y_test"].copy()

        # Encode labels to integers if they are strings
        if isinstance(y_train.iloc[0], str):
            unique_vals = sorted(y_train.unique())
            y_train = (y_train == unique_vals[1]).astype(int)
            y_test = (y_test == unique_vals[1]).astype(int)

        # Source A: Original features
        self.sources['A'] = {
            'X_train': X_train.iloc[:, :5],
            'X_test': X_test.iloc[:, :5],
            'y_train': y_train,
            'y_test': y_test,
            'name': 'Core Features'
        }

        # Source B: Synthetic wage features (engineered features)
        B_train = X_train.iloc[:, 5:10].copy() if len(X_train.columns) > 5 else X_train.iloc[:, :5].copy()
        B_test = X_test.iloc[:, 5:10].copy() if len(X_test.columns) > 5 else X_test.iloc[:, :5].copy()
        self.sources['B'] = {
            'X_train': B_train,
            'X_test': B_test,
            'y_train': y_train,
            'y_test': y_test,
            'name': 'Wage Features'
        }

        # Source C: Financial features
        C_cols = min(10, len(X_train.columns) - 10)
        if C_cols > 0:
            C_train = X_train.iloc[:, 10:10+C_cols].copy()
            C_test = X_test.iloc[:, 10:10+C_cols].copy()
        else:
            C_train = X_train.iloc[:, -5:].copy() if len(X_train.columns) > 5 else X_train.iloc[:, :5].copy()
            C_test = X_test.iloc[:, -5:].copy() if len(X_test.columns) > 5 else X_test.iloc[:, :5].copy()

        self.sources['C'] = {
            'X_train': C_train,
            'X_test': C_test,
            'y_train': y_train,
            'y_test': y_test,
            'name': 'Financial Features'
        }

        return self.sources

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate all metrics"""
        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
        }
        return metrics

    def _calculate_metrics_with_probabilities(self, y_true, y_pred, y_prob):
        """Calculate metrics including ROC-AUC when probabilities are available"""
        metrics = self._calculate_metrics(y_true, y_pred)
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except Exception:
            metrics["roc_auc"] = None
        return metrics

    def _build_source_model(self):
        """Reusable RF model for per-source training"""
        return RandomForestClassifier(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )

    def _attach_study_guide(self, result):
        """Attach educational notes for virtual-lab reporting"""
        guide = self.study_guide.get(result.get("technique"), {})
        return {**result, **guide}

    def _normalize_weights(self, weights):
        """Safely normalize weight vectors"""
        weight_array = np.asarray(weights, dtype=float)
        weight_array = np.maximum(weight_array, 1e-6)
        return weight_array / np.sum(weight_array)

    def _train_source_models(self):
        """Train one model per source and return predictions/probabilities"""
        if not self.sources:
            self.create_data_sources()

        models = {}
        train_probs = {}
        test_probs = {}
        oob_scores = {}

        for source_name in ['A', 'B', 'C']:
            model = self._build_source_model()
            model.fit(self.sources[source_name]['X_train'], self.sources[source_name]['y_train'])
            models[source_name] = model
            train_probs[source_name] = model.predict_proba(self.sources[source_name]['X_train'])[:, 1]
            test_probs[source_name] = model.predict_proba(self.sources[source_name]['X_test'])[:, 1]
            oob_scores[source_name] = float(getattr(model, 'oob_score_', 0.0))

        return {
            "models": models,
            "train_probs": train_probs,
            "test_probs": test_probs,
            "oob_scores": oob_scores
        }

    def _get_combined_features(self):
        """Get concatenated features from all sources"""
        X_train = pd.concat([
            self.sources['A']['X_train'],
            self.sources['B']['X_train'],
            self.sources['C']['X_train']
        ], axis=1)
        X_test = pd.concat([
            self.sources['A']['X_test'],
            self.sources['B']['X_test'],
            self.sources['C']['X_test']
        ], axis=1)
        return X_train, X_test

    def no_fusion_baseline(self, source_bundle=None):
        """Baseline: source A only (no fusion)"""
        if source_bundle is None:
            source_bundle = self._train_source_models()

        y_true = self.sources['A']['y_test']
        y_prob = source_bundle['test_probs']['A']
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = self._calculate_metrics_with_probabilities(y_true, y_pred, y_prob)

        result = {
            "technique": "No Fusion (Baseline)",
            "source_focus": "Source A only",
            **metrics
        }
        return self._attach_study_guide(result)

    def early_fusion(self):
        """Early Fusion: Merge all sources and train a single model"""
        if not self.sources:
            self.create_data_sources()

        X_train, X_test = self._get_combined_features()

        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, self.sources['A']['y_train'])
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = self._calculate_metrics_with_probabilities(self.sources['A']['y_test'], y_pred, y_prob)
        result = {
            "technique": "Early Fusion",
            **metrics
        }
        return self._attach_study_guide(result)

    def late_fusion(self, source_bundle=None):
        """Late Fusion: Average source-level probabilities"""
        if source_bundle is None:
            source_bundle = self._train_source_models()

        test_matrix = np.column_stack([
            source_bundle['test_probs']['A'],
            source_bundle['test_probs']['B'],
            source_bundle['test_probs']['C']
        ])
        late_prob = np.mean(test_matrix, axis=1)
        y_pred = (late_prob >= 0.5).astype(int)

        metrics = self._calculate_metrics_with_probabilities(self.sources['A']['y_test'], y_pred, late_prob)
        result = {
            "technique": "Late Fusion",
            "combiner": "Mean probability averaging",
            **metrics
        }
        return self._attach_study_guide(result)

    def weighted_fusion(self, source_bundle=None):
        """Weighted Fusion: Reliability-based weighted averaging"""
        if source_bundle is None:
            source_bundle = self._train_source_models()

        raw_weights = np.array([
            source_bundle['oob_scores']['A'],
            source_bundle['oob_scores']['B'],
            source_bundle['oob_scores']['C']
        ])
        weights = self._normalize_weights(raw_weights)

        weighted_prob = (
            weights[0] * source_bundle['test_probs']['A'] +
            weights[1] * source_bundle['test_probs']['B'] +
            weights[2] * source_bundle['test_probs']['C']
        )
        y_pred = (weighted_prob >= 0.5).astype(int)

        metrics = self._calculate_metrics_with_probabilities(self.sources['A']['y_test'], y_pred, weighted_prob)
        result = {
            "technique": "Weighted Fusion",
            "weights": {
                "Source A": round(float(weights[0]), 3),
                "Source B": round(float(weights[1]), 3),
                "Source C": round(float(weights[2]), 3)
            },
            **metrics
        }
        return self._attach_study_guide(result)

    def cnn_fusion(self):
        """CNN-inspired Fusion: MLP approximation for tabular data"""
        if not self.sources:
            self.create_data_sources()

        X_train, X_test = self._get_combined_features()
        model = MLPClassifier(
            hidden_layer_sizes=(96, 48),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=220,
            random_state=42,
            early_stopping=True
        )
        model.fit(X_train, self.sources['A']['y_train'])

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = self._calculate_metrics_with_probabilities(self.sources['A']['y_test'], y_pred, y_prob)
        result = {
            "technique": "CNN Fusion",
            "note": "Tabular approximation using a multi-layer perceptron.",
            **metrics
        }
        return self._attach_study_guide(result)

    def attention_fusion(self, source_bundle=None):
        """Attention Fusion: Per-row confidence weighted source probabilities"""
        if source_bundle is None:
            source_bundle = self._train_source_models()

        test_matrix = np.column_stack([
            source_bundle['test_probs']['A'],
            source_bundle['test_probs']['B'],
            source_bundle['test_probs']['C']
        ])

        confidence = np.abs(test_matrix - 0.5) * 4.0
        exp_scores = np.exp(confidence - confidence.max(axis=1, keepdims=True))
        attention_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        attention_prob = np.sum(attention_weights * test_matrix, axis=1)
        y_pred = (attention_prob >= 0.5).astype(int)

        metrics = self._calculate_metrics_with_probabilities(self.sources['A']['y_test'], y_pred, attention_prob)
        mean_attention = attention_weights.mean(axis=0)
        result = {
            "technique": "Attention Fusion",
            "average_attention": {
                "Source A": round(float(mean_attention[0]), 3),
                "Source B": round(float(mean_attention[1]), 3),
                "Source C": round(float(mean_attention[2]), 3)
            },
            **metrics
        }
        return self._attach_study_guide(result)

    def gated_fusion(self, source_bundle=None):
        """Gated Fusion: Train a meta-learner over source probabilities"""
        if source_bundle is None:
            source_bundle = self._train_source_models()

        train_probs = np.column_stack([
            source_bundle['train_probs']['A'],
            source_bundle['train_probs']['B'],
            source_bundle['train_probs']['C']
        ])
        test_probs = np.column_stack([
            source_bundle['test_probs']['A'],
            source_bundle['test_probs']['B'],
            source_bundle['test_probs']['C']
        ])

        train_conf = np.abs(train_probs - 0.5)
        test_conf = np.abs(test_probs - 0.5)
        meta_train = np.column_stack([train_probs, train_conf, train_probs * train_conf])
        meta_test = np.column_stack([test_probs, test_conf, test_probs * test_conf])

        gate = LogisticRegression(max_iter=400, C=0.7, random_state=42)
        gate.fit(meta_train, self.sources['A']['y_train'])
        gated_prob = gate.predict_proba(meta_test)[:, 1]
        y_pred = (gated_prob >= 0.5).astype(int)

        metrics = self._calculate_metrics_with_probabilities(self.sources['A']['y_test'], y_pred, gated_prob)
        result = {
            "technique": "Gated Fusion",
            "gate_type": "Logistic meta-learner",
            **metrics
        }
        return self._attach_study_guide(result)

    def hybrid_fusion(self, source_bundle=None):
        """Hybrid Fusion: Blend early, weighted, and gated predictions"""
        if source_bundle is None:
            source_bundle = self._train_source_models()

        X_train, X_test = self._get_combined_features()
        early_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        early_model.fit(X_train, self.sources['A']['y_train'])
        early_prob = early_model.predict_proba(X_test)[:, 1]

        weighted_result = self.weighted_fusion(source_bundle=source_bundle)
        gated_result = self.gated_fusion(source_bundle=source_bundle)

        weighted_prob = (
            weighted_result['weights']['Source A'] * source_bundle['test_probs']['A'] +
            weighted_result['weights']['Source B'] * source_bundle['test_probs']['B'] +
            weighted_result['weights']['Source C'] * source_bundle['test_probs']['C']
        )

        train_probs = np.column_stack([
            source_bundle['train_probs']['A'],
            source_bundle['train_probs']['B'],
            source_bundle['train_probs']['C']
        ])
        test_probs = np.column_stack([
            source_bundle['test_probs']['A'],
            source_bundle['test_probs']['B'],
            source_bundle['test_probs']['C']
        ])
        train_conf = np.abs(train_probs - 0.5)
        test_conf = np.abs(test_probs - 0.5)
        meta_train = np.column_stack([train_probs, train_conf, train_probs * train_conf])
        meta_test = np.column_stack([test_probs, test_conf, test_probs * test_conf])
        gate = LogisticRegression(max_iter=400, C=0.7, random_state=42)
        gate.fit(meta_train, self.sources['A']['y_train'])
        gated_prob = gate.predict_proba(meta_test)[:, 1]

        confidence_scores = self._normalize_weights([
            np.mean(np.abs(early_prob - 0.5)),
            np.mean(np.abs(weighted_prob - 0.5)),
            np.mean(np.abs(gated_prob - 0.5))
        ])

        hybrid_prob = (
            confidence_scores[0] * early_prob +
            confidence_scores[1] * weighted_prob +
            confidence_scores[2] * gated_prob
        )
        y_pred = (hybrid_prob >= 0.5).astype(int)

        metrics = self._calculate_metrics_with_probabilities(self.sources['A']['y_test'], y_pred, hybrid_prob)
        result = {
            "technique": "Hybrid Fusion",
            "blend_weights": {
                "Early": round(float(confidence_scores[0]), 3),
                "Weighted": round(float(confidence_scores[1]), 3),
                "Gated": round(float(confidence_scores[2]), 3)
            },
            "gated_reference_accuracy": gated_result.get('accuracy'),
            **metrics
        }
        return self._attach_study_guide(result)

    def compare_all_techniques(self):
        """Compare all implemented fusion techniques"""
        self.create_data_sources()
        source_bundle = self._train_source_models()

        results = [
            self.no_fusion_baseline(source_bundle=source_bundle),
            self.early_fusion(),
            self.late_fusion(source_bundle=source_bundle),
            self.weighted_fusion(source_bundle=source_bundle),
            self.cnn_fusion(),
            self.attention_fusion(source_bundle=source_bundle),
            self.gated_fusion(source_bundle=source_bundle),
            self.hybrid_fusion(source_bundle=source_bundle)
        ]

        results = sorted(results, key=lambda item: item.get('accuracy', 0), reverse=True)
        for rank, item in enumerate(results, start=1):
            item['rank'] = rank

        cache.set("fusion_comparison", results)
        return results


fusion = DataFusion()
