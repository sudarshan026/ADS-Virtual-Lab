import pandas as pd
import numpy as np
import time
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, f1_score
from .cache import cache
from .cleaner import cleaner


class DataImputer:
    """Imputation methods for missing values"""

    def __init__(self):
        self.df = None
        self.imputed_data = {
            'mode': None,
            'knn': None,
            'mice': None
        }

    def load_data(self):
        """Load cleaned data"""
        if cache.exists("cleaned_data"):
            self.df = cache.get("cleaned_data").copy()
        else:
            cleaner.clean()
            self.df = cache.get("cleaned_data").copy()

    def impute_mode(self):
        """Impute using mode for categorical, mean for numerical"""
        if self.df is None:
            self.load_data()

        start_time = time.time()
        df_imputed = self.df.copy()

        # Numerical columns: use mean
        numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)

        # Categorical columns: use mode
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_val = df_imputed[col].mode()[0] if len(df_imputed[col].mode()) > 0 else 'unknown'
            df_imputed[col].fillna(mode_val, inplace=True)

        exec_time = (time.time() - start_time) * 1000

        self.imputed_data['mode'] = df_imputed.copy()
        cache.set("imputed_data_mode", df_imputed.copy())

        return {
            "method": "mode",
            "execution_time_ms": round(exec_time, 2),
            "missing_values_remaining": int(df_imputed.isnull().sum().sum())
        }

    def impute_knn(self, k=5):
        """Impute using KNN imputer"""
        if self.df is None:
            self.load_data()

        start_time = time.time()
        df_imputed = self.df.copy()

        # Encode categorical to numerical for KNN
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns
        categorical_mapping = {}

        for col in categorical_cols:
            unique_vals = df_imputed[col].dropna().unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            categorical_mapping[col] = mapping
            df_imputed[col] = df_imputed[col].map(mapping)

        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=k)
        df_imputed = pd.DataFrame(imputer.fit_transform(df_imputed), columns=df_imputed.columns)

        # Decode back to categorical
        for col, mapping in categorical_mapping.items():
            inverse_mapping = {v: k for k, v in mapping.items()}
            df_imputed[col] = df_imputed[col].round().astype(int).map(inverse_mapping)

        exec_time = (time.time() - start_time) * 1000

        self.imputed_data['knn'] = df_imputed.copy()
        cache.set("imputed_data_knn", df_imputed.copy())

        return {
            "method": "knn",
            "k_neighbors": k,
            "execution_time_ms": round(exec_time, 2),
            "missing_values_remaining": int(df_imputed.isnull().sum().sum())
        }

    def impute_mice(self, max_iter=10):
        """Impute using MICE (Multivariate Imputation by Chained Equations)"""
        if self.df is None:
            self.load_data()

        start_time = time.time()
        df_imputed = self.df.copy()

        # Encode categorical to numerical
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns
        categorical_mapping = {}

        for col in categorical_cols:
            unique_vals = df_imputed[col].dropna().unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            categorical_mapping[col] = mapping
            df_imputed[col] = df_imputed[col].map(mapping)

        # Apply MICE imputation
        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        df_imputed = pd.DataFrame(imputer.fit_transform(df_imputed), columns=df_imputed.columns)

        # Decode back to categorical
        for col, mapping in categorical_mapping.items():
            inverse_mapping = {v: k for k, v in mapping.items()}
            df_imputed[col] = df_imputed[col].round().astype(int).map(inverse_mapping)

        exec_time = (time.time() - start_time) * 1000

        self.imputed_data['mice'] = df_imputed.copy()
        cache.set("imputed_data_mice", df_imputed.copy())

        return {
            "method": "mice",
            "max_iterations": max_iter,
            "execution_time_ms": round(exec_time, 2),
            "missing_values_remaining": int(df_imputed.isnull().sum().sum())
        }

    def compare_methods(self):
        """Compare all imputation methods"""
        mode_result = self.impute_mode()
        knn_result = self.impute_knn()
        mice_result = self.impute_mice()

        return {
            "methods_tested": 3,
            "results": [mode_result, knn_result, mice_result],
            "fastest": "mode",
            "recommended": "mice (most sophisticated)"
        }


imputer = DataImputer()
