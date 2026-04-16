import pandas as pd
import numpy as np
from pathlib import Path
from .cache import cache


class DataLoader:
    """Load and analyze UCI Adult dataset"""

    def __init__(self, csv_path="../adult.csv"):
        self.csv_path = Path(csv_path)
        self.df = None
        self.numerical_features = []
        self.categorical_features = []

    def load(self):
        """Load dataset from CSV"""
        # Check if already in cache
        if cache.exists("raw_data"):
            self.df = cache.get("raw_data")
            return self.df

        # Load CSV
        try:
            self.df = pd.read_csv(self.csv_path)
            # Strip whitespace from column names
            self.df.columns = self.df.columns.str.strip()

            # Identify feature types
            self._identify_features()

            # Cache dataset
            cache.set("raw_data", self.df)
            return self.df
        except FileNotFoundError:
            raise Exception(f"Dataset not found at {self.csv_path}")

    def _identify_features(self):
        """Identify numerical and categorical features"""
        self.numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()

    def get_statistics(self):
        """Get comprehensive dataset statistics"""
        if self.df is None:
            self.load()

        # Count missing values
        missing_values = {}
        for col in self.df.columns:
            missing_count = self.df[col].isin(['?', np.nan, 'NaN']).sum()
            if missing_count > 0:
                missing_values[col] = int(missing_count)

        # Class distribution (if 'income' exists)
        class_dist = {}
        if 'income' in self.df.columns:
            class_dist = self.df['income'].value_counts().to_dict()
            class_dist = {str(k).strip(): int(v) for k, v in class_dist.items()}

        return {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "features": self.df.columns.tolist(),
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "missing_values": missing_values,
            "class_distribution": class_dist,
            "data_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }

    def get_sample(self, n=5):
        """Get sample rows"""
        if self.df is None:
            self.load()
        return self.df.head(n).to_dict('records')

    def get_dataframe(self):
        """Get entire dataframe"""
        if self.df is None:
            self.load()
        return self.df.copy()


# Singleton loader
_loader = DataLoader("../adult.csv")
