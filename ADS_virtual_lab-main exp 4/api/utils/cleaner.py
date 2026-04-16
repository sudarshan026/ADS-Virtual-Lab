import pandas as pd
import numpy as np
from .cache import cache
from .data_loader import _loader


class DataCleaner:
    """Data cleaning pipeline"""

    def __init__(self):
        self.df = None
        self.original_rows = 0
        self.cleaned_rows = 0

    def load_data(self):
        """Load raw data from cache or loader"""
        if cache.exists("raw_data"):
            self.df = cache.get("raw_data").copy()
        else:
            _loader.load()
            self.df = cache.get("raw_data").copy()

        self.original_rows = len(self.df)
        return self.df

    def clean(self):
        """Run complete cleaning pipeline"""
        if self.df is None:
            self.load_data()

        # Step 1: Replace '?' with NaN
        missing_before = (self.df == '?').sum().sum()
        self.df = self.df.replace('?', np.nan)

        # Step 2: Remove duplicates
        duplicates_removed = len(self.df) - len(self.df.drop_duplicates())
        self.df = self.df.drop_duplicates()

        # Step 3: Fix category inconsistencies
        for col in self.df.select_dtypes(include=['object']).columns:
            if col != 'income':  # Don't modify target
                self.df[col] = self.df[col].str.strip().str.lower()

        # Step 4: Detect outliers using IQR method
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_indices = set()

        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].index
            outliers_indices.update(outliers)

        outliers_detected = len(outliers_indices)

        self.cleaned_rows = len(self.df)

        # Cache cleaned data
        cache.set("cleaned_data", self.df.copy())

        return {
            "original_rows": int(self.original_rows),
            "cleaned_rows": int(self.cleaned_rows),
            "duplicates_removed": int(duplicates_removed),
            "missing_values_replaced": int(missing_before),
            "outliers_detected": int(outliers_detected),
            "rows_with_outliers": int(len(outliers_indices)),
            "summary": f"Cleaned dataset ready. Removed {duplicates_removed} duplicates, detected {outliers_detected} outliers."
        }

    def get_missing_value_summary(self):
        """Get missing values summary"""
        if self.df is None:
            self.load_data()

        missing = self.df.isnull().sum()
        missing = missing[missing > 0].to_dict()
        return {col: int(count) for col, count in missing.items()}


cleaner = DataCleaner()
