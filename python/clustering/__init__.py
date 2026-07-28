"""
High-performance C++ clustering engine with Python bindings.
"""
import numpy as np

try:
    from clustering._clustering import (
        KMeans as _KMeans,
        OnlineKMeans as _OnlineKMeans,
        MiniBatchKMeans as _MiniBatchKMeans,
        PCA as _PCA,
        TSNE as _TSNE,
        TSNEConfig,
        DriftDetector as _DriftDetector,
        DriftMetrics,
        VersionManager as _VersionManager,
        FeatureStore as _FeatureStore,
        OnlineConfig,
        KMeansConfig,
        l2_distance,
    )
    _HAS_C = True
except ImportError:
    _HAS_C = False


class KMeans:
    def __init__(self, k=8, max_iter=300, tol=1e-4, max_threads=0):
        if _HAS_C:
            self._impl = _KMeans(k)
        else:
            self._k, self._max_iter, self._tol = k, max_iter, tol
            self._labels = self._centroids = None
            self._inertia = self._n_iter = 0

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if _HAS_C:
            self._impl.fit(X)
        else:
            self._fit_numpy(X)
        return self

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self._impl.predict(X) if _HAS_C else self._predict_numpy(X)

    def partial_fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if _HAS_C:
            self._impl.partial_fit(X)
        return self

    @property
    def labels(self):
        return self._impl.labels if _HAS_C else self._labels

    @property
    def centroids(self):
        return self._impl.centroids if _HAS_C else self._centroids

    @property
    def inertia(self):
        return self._impl.inertia if _HAS_C else self._inertia

    @property
    def n_iter(self):
        return self._impl.n_iter if _HAS_C else self._n_iter

    def _fit_numpy(self, X):
        n, d = X.shape
        self._centroids = X[np.random.choice(n, self._k, replace=False)].copy()
        for it in range(self._max_iter):
            dists = np.linalg.norm(X[:, None] - self._centroids, axis=2)
            self._labels = np.argmin(dists, axis=1)
            new_c = np.array([X[self._labels == i].mean(0) if np.any(self._labels == i) else self._centroids[i] for i in range(self._k)])
            if np.allclose(self._centroids, new_c, atol=self._tol):
                break
            self._centroids = new_c
        dists = np.linalg.norm(X[:, None] - self._centroids, axis=2)
        self._inertia = float(np.sum(np.min(dists, axis=1) ** 2))
        self._n_iter = it + 1

    def _predict_numpy(self, X):
        return np.argmin(np.linalg.norm(X[:, None] - self._centroids, axis=2), axis=1)


class OnlineKMeans:
    def __init__(self, k=8, window_size=1000, forgetting_factor=0.99,
                 auto_retrain=True, drift_threshold=0.1):
        if _HAS_C:
            self._impl = _OnlineKMeans(k)
        else:
            self._k, self._centroids, self._points_seen = k, None, 0

    def partial_fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if _HAS_C:
            self._impl.partial_fit(X)
        elif self._centroids is None:
            self._centroids = X[np.random.choice(X.shape[0], self._k, replace=False)].copy()
            self._points_seen = X.shape[0]
        return self

    @property
    def points_seen(self):
        return self._impl.points_seen if _HAS_C else self._points_seen

    @property
    def centroids(self):
        return self._impl.centroids if _HAS_C else self._centroids

    @property
    def labels(self):
        return self._impl.labels if _HAS_C else None

    def set_window_size(self, s):
        if _HAS_C: self._impl.set_window_size(s)

    def set_forgetting_factor(self, f):
        if _HAS_C: self._impl.set_forgetting_factor(f)


class MiniBatchKMeans(KMeans):
    def __init__(self, k=8, batch_size=100):
        if _HAS_C:
            self._impl = _MiniBatchKMeans(k, batch_size)
            self._labels = self._centroids = None
            self._inertia = self._n_iter = 0
        else:
            super().__init__(k)


class PCA:
    def __init__(self, n_components):
        self._n = n_components
        self._impl = _PCA(n_components) if _HAS_C else None

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if _HAS_C:
            self._impl.fit(X)
        else:
            self._fit_numpy(X)
        return self

    def transform(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self._impl.transform(X) if _HAS_C else X @ self._components.T

    def fit_transform(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self._impl.fit_transform(X) if _HAS_C else self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self._impl.inverse_transform(X) if _HAS_C else X @ self._components + self._mean

    @property
    def components(self):
        return self._impl.components if _HAS_C else self._components

    @property
    def explained_variance_ratio(self):
        return self._impl.explained_variance_ratio if _HAS_C else self._var_ratio

    @property
    def total_explained_variance_ratio(self):
        return self._impl.total_explained_variance_ratio if _HAS_C else float(np.sum(self._var_ratio))

    def _fit_numpy(self, X):
        n, d = X.shape
        k = self._n

        self._mean = X.mean(axis=0)
        X_c = X - self._mean

        cov = (X_c.T @ X_c) / (n - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][:k]

        self._components = eigvecs[:, idx].T
        total_var = np.sum(eigvals)
        self._var_ratio = np.array([eigvals[i] / total_var for i in idx], dtype=np.float32)


class TSNE:
    def __init__(self, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
        if _HAS_C:
            cfg = TSNEConfig()
            cfg.n_components, cfg.perplexity = n_components, perplexity
            cfg.learning_rate, cfg.n_iter = learning_rate, n_iter
            self._impl = _TSNE(cfg)
        else:
            self._impl = None

    def fit(self, X):
        if _HAS_C:
            self._impl.fit(np.ascontiguousarray(X, dtype=np.float32))
        return self

    def fit_transform(self, X):
        if _HAS_C:
            return self._impl.fit_transform(np.ascontiguousarray(X, dtype=np.float32))
        return X[:, :2]

    @property
    def embedding(self):
        return self._impl.embedding if _HAS_C else None

    @property
    def kl_divergence(self):
        return self._impl.kl_divergence if _HAS_C else 0.0


class DriftDetector:
    def __init__(self, threshold=0.1, window_size=10):
        self._impl = _DriftDetector() if _HAS_C else None
        if self._impl:
            self._impl.set_threshold(threshold)
            self._impl.set_window_size(window_size)

    def check(self, X, labels, centroids):
        if _HAS_C:
            return self._impl.check(
                np.ascontiguousarray(X, dtype=np.float32),
                np.ascontiguousarray(labels, dtype=np.float32),
                np.ascontiguousarray(centroids, dtype=np.float32)
            )
        return DriftMetrics()

    @property
    def is_drifting(self):
        return self._impl.is_drifting if _HAS_C else False


class VersionManager:
    def __init__(self, path="./versions"):
        self._impl = _VersionManager() if _HAS_C else None
        if self._impl:
            self._impl.set_storage_path(path)

    def save(self, centroids, labels):
        if _HAS_C:
            return self._impl.save(
                np.ascontiguousarray(centroids, dtype=np.float32),
                np.ascontiguousarray(labels, dtype=np.float32)
            )
        return 0

    def load(self, version_id):
        if _HAS_C:
            return self._impl.load(version_id)
        return None, None


class FeatureStore:
    def __init__(self, path="./cache"):
        self._impl = _FeatureStore() if _HAS_C else None
        if self._impl:
            self._impl.set_cache_path(path)

    def preprocess(self, X, operation):
        if _HAS_C:
            return self._impl.preprocess(
                np.ascontiguousarray(X, dtype=np.float32), operation
            )
        return X
