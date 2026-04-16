import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from .cache import cache
from .preprocessor import preprocessor


class ClusteringAnalyzer:
    """K-Means clustering and visualization"""

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None

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

    def analyze_clusters(self, k_range=range(2, 6)):
        """Analyze K-Means with different k values"""
        if self.X_train is None:
            self.load_data()

        # Combine train and test for clustering
        X = pd.concat([self.X_train, self.X_test], axis=0)

        results = {
            "silhouette_scores": [],
            "davies_bouldin_scores": [],
            "inertias": [],
            "optimal_k": None,
            "k_values": []
        }

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            sil_score = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            inertia = kmeans.inertia_

            results["silhouette_scores"].append(round(sil_score, 4))
            results["davies_bouldin_scores"].append(round(db_score, 4))
            results["inertias"].append(round(inertia, 2))
            results["k_values"].append(k)

            if results["optimal_k"] is None or sil_score > max(results["silhouette_scores"][:-1]):
                results["optimal_k"] = k

        cache.set("clustering_analysis", results)
        return results

    def get_pca_visualization(self, n_components=2, k=3):
        """Get 2D PCA visualization of clusters"""
        if self.X_train is None:
            self.load_data()

        # Combine train and test
        X = pd.concat([self.X_train, self.X_test], axis=0)

        # Apply K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        # Prepare visualization data
        points = []
        for i, (point, label) in enumerate(zip(X_pca, labels)):
            points.append({
                "x": round(float(point[0]), 4),
                "y": round(float(point[1] if n_components > 1 else 0), 4),
                "cluster": int(label)
            })

        explained_variance = [round(float(var), 4) for var in pca.explained_variance_ratio_]

        visualization = {
            "points": points[:1000],  # Limit to 1000 points for performance
            "total_points": len(points),
            "k": k,
            "explained_variance": explained_variance,
            "cumulative_variance": round(sum(explained_variance), 4)
        }

        cache.set("pca_visualization", visualization)
        return visualization

    def get_elbow_curve(self):
        """Get elbow method data"""
        results = cache.get("clustering_analysis")
        if not results:
            results = self.analyze_clusters()

        elbow_data = []
        for k, inertia in zip(results["k_values"], results["inertias"]):
            elbow_data.append([k, inertia])

        return {
            "elbow_method": elbow_data,
            "optimal_k": results["optimal_k"]
        }


analyzer = ClusteringAnalyzer()
