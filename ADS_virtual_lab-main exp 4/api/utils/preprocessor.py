import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from .cache import cache
from .imputer import imputer


class DataPreprocessor:
    """Data preprocessing: encoding, scaling, and splitting"""

    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.encoders = {}
        self.feature_names = []

    def load_data(self, imputation_method='mode'):
        """Load imputed data"""
        cache_key = f"imputed_data_{imputation_method}"
        if cache.exists(cache_key):
            self.df = cache.get(cache_key).copy()
        else:
            if imputation_method == 'mode':
                imputer.impute_mode()
            elif imputation_method == 'knn':
                imputer.impute_knn()
            elif imputation_method == 'mice':
                imputer.impute_mice()
            self.df = cache.get(cache_key).copy()

    def preprocess(self, imputation_method='mode', test_size=0.2, random_state=42):
        """Complete preprocessing pipeline"""
        self.load_data(imputation_method)

        # Separate features and target
        if 'income' in self.df.columns:
            X = self.df.drop('income', axis=1)
            y = self.df['income'].str.strip()
        else:
            X = self.df
            y = None

        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))
            self.encoders[col] = encoder

        # Scale numerical features
        self.scaler = StandardScaler()
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        self.feature_names = X.columns.tolist()

        # Train-test split
        if y is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            self.X_train, self.X_test = train_test_split(X, test_size=test_size, random_state=random_state)
            self.y_train, self.y_test = None, None

        # Cache preprocessed data
        cache.set("preprocessed_data", {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "feature_names": self.feature_names
        })

        return {
            "original_features": len(self.df.columns),
            "processed_features": len(self.feature_names),
            "total_samples": len(X),
            "train_samples": len(self.X_train),
            "test_samples": len(self.X_test),
            "test_size_ratio": float(test_size),
            "categorical_features_encoded": len(categorical_cols),
            "numerical_features_scaled": len(X.select_dtypes(include=[np.number]).columns),
            "scaling_mean": [round(m, 4) for m in self.scaler.mean_[:5]],  # First 5 features
            "scaling_std": [round(s, 4) for s in self.scaler.scale_[:5]]    # First 5 features
        }

    def get_data(self):
        """Get preprocessed data"""
        return {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "feature_names": self.feature_names
        }


preprocessor = DataPreprocessor()
