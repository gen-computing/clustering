# API Reference

## Headers

```cpp
#include <clustering/clustering.h>  // All-in-one
// or individual:
#include <clustering/kmeans.h>
#include <clustering/online.h>
#include <clustering/mini_batch.h>
#include <clustering/pca.h>
#include <clustering/tsne.h>
#include <clustering/drift.h>
#include <clustering/versioning.h>
#include <clustering/feature_store.h>
#include <clustering/renderer.h>
```

---

## KMeans

Batch KMeans clustering with KMeans++ initialization.

### Constructor

```cpp
KMeans(size_t k = 8);
KMeans(const KMeansConfig& config);
```

### KMeansConfig

```cpp
struct KMeansConfig {
    size_t k = 8;                    // Number of clusters
    size_t max_iter = 300;           // Max iterations
    float tol = 1e-4f;              // Convergence tolerance
    size_t max_threads = 0;          // 0 = auto-detect
    bool enable_versioning = false;
    bool enable_drift_detection = false;
    std::string feature_store_path;
};
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(X)` | void | Train on matrix X (n_samples × n_features) |
| `predict(X)` | Vector | Predict cluster labels for new data |
| `partial_fit(X)` | void | Incremental update |
| `labels()` | const Vector& | Cluster assignments after fit |
| `centroids()` | const Matrix& | Cluster centers |
| `n_iter()` | size_t | Iterations executed |
| `inertia()` | float | Sum of squared distances |

### Example

```cpp
Matrix X(1000, 32);  // 1000 points, 32 dimensions
// ... fill X ...

KMeans km(5);  // 5 clusters
km.fit(X);

Vector labels = km.labels();      // [0, 2, 1, 4, 0, ...]
Matrix centroids = km.centroids(); // 5 × 32 matrix
float inertia = km.inertia();     // Total within-cluster sum
```

---

## OnlineKMeans

Online KMeans with sliding window and forgetting factor.

### OnlineConfig

```cpp
struct OnlineConfig {
    size_t k = 8;
    size_t window_size = 1000;      // Sliding window size (0 = no window)
    float forgetting_factor = 0.99; // Weight decay (0-1)
    bool auto_retrain = true;       // Retrain on drift
    float drift_threshold = 0.1f;
    size_t retrain_interval = 100;  // Check every N points
};
```

### Methods

Inherits all KMeans methods, plus:

| Method | Description |
|--------|-------------|
| `set_window_size(n)` | Set sliding window size |
| `set_forgetting_factor(f)` | Set forgetting factor (0-1) |
| `set_auto_retrain(b)` | Enable/disable auto-retrain |
| `points_seen()` | Total points processed |

### Example

```cpp
OnlineConfig config;
config.k = 5;
config.window_size = 500;
config.forgetting_factor = 0.95;
config.auto_retrain = true;

OnlineKMeans km(config);

// Process streaming data
for (int i = 0; i < 10000; ++i) {
    Matrix batch = get_next_batch(100);
    km.partial_fit(batch);
}
```

---

## MiniBatchKMeans

Mini-batch KMeans for large datasets.

### Constructor

```cpp
MiniBatchKMeans(size_t k = 8, size_t batch_size = 100);
```

### Example

```cpp
MiniBatchKMeans mb(5, 200);  // 5 clusters, batch size 200
mb.fit(X);
mb.partial_fit(new_batch);
```

---

## PCA

Principal Component Analysis for dimensionality reduction.

### Constructor

```cpp
PCA(size_t n_components);
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(X)` | void | Fit PCA model |
| `transform(X)` | Matrix | Project to lower dimensions |
| `fit_transform(X)` | Matrix | Fit + transform |
| `inverse_transform(X)` | Matrix | Reconstruct from reduced |
| `components()` | const Matrix& | Principal axes (n_components × n_features) |
| `explained_variance()` | const Vector& | Variance per component |
| `explained_variance_ratio()` | const Vector& | % variance per component |
| `total_explained_variance_ratio()` | float | Total % variance retained |

### Example

```cpp
Matrix X(1000, 50);  // 50 dimensions

PCA pca(2);  // Reduce to 2D
Matrix Y = pca.fit_transform(X);  // 1000 × 2

float variance = pca.total_explained_variance_ratio();  // e.g., 0.85
Matrix X_reconstructed = pca.inverse_transform(Y);       // 1000 × 50
```

---

## t-SNE

t-Distributed Stochastic Neighbor Embedding.

### TSNEConfig

```cpp
struct TSNEConfig {
    size_t n_components = 2;
    size_t perplexity = 30;
    float learning_rate = 200.0f;
    size_t n_iter = 1000;
    size_t early_exaggeration = 12;
    float min_gradient_norm = 1e-7f;
};
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(X)` | void | Compute embedding |
| `fit_transform(X)` | Matrix | Fit + return embedding |
| `embedding()` | const Matrix& | 2D embedding |
| `kl_divergence()` | float | KL divergence (lower = better) |

### Example

```cpp
TSNEConfig config;
config.n_components = 2;
config.perplexity = 30;

TSNE tsne(config);
Matrix embedding = tsne.fit_transform(X);  // n_samples × 2
```

---

## DriftDetector

Concept drift detection with auto-retrain.

### Methods

| Method | Description |
|--------|-------------|
| `set_threshold(t)` | Drift sensitivity (default: 0.1) |
| `set_window_size(n)` | History window size |
| `check(X, labels, centroids)` | Compute metrics, detect drift |
| `is_drifting()` | Current drift status |

### DriftMetrics

```cpp
struct DriftMetrics {
    float silhouette_score;        // [-1, 1], higher = better
    float davies_bouldin_index;    // Lower = better
    float calinski_harabasz_score; // Higher = better
    float cluster_stability;       // [0, 1], higher = more balanced
    bool drift_detected;           // Threshold exceeded
};
```

### Example

```cpp
DriftDetector detector;
detector.set_threshold(0.05f);
detector.set_window_size(10);

// In production loop
DriftMetrics m = detector.check(X, labels, centroids);
if (m.drift_detected) {
    km.fit(X);  // Retrain
}
```

---

## VersionManager

Centroid versioning and rollback.

### Methods

| Method | Description |
|--------|-------------|
| `set_storage_path(path)` | Set storage directory |
| `save(centroids, labels)` | Save version, returns version ID |
| `load(id, centroids, labels)` | Load version by ID |
| `list_versions()` | List all version IDs |
| `latest_version()` | Get latest version ID |
| `has_versions()` | Check if any versions exist |

### Example

```cpp
VersionManager vm;
vm.set_storage_path("./versions");

// Save
size_t v1 = vm.save(km.centroids(), km.labels());

// Load
Matrix centroids;
Vector labels;
vm.load(v1, centroids, labels);

// List
auto versions = vm.list_versions();
```

---

## FeatureStore

Preprocessing with caching.

### Methods

| Method | Description |
|--------|-------------|
| `set_cache_path(path)` | Set cache directory |
| `preprocess(X, operation)` | Preprocess with cache |
| `cache_hit(key)` | Check if cached |

### Operations

| Operation | Description |
|-----------|-------------|
| `"normalize"` | Zero mean, unit variance |
| `"standardize"` | Scale to [0, 1] |
| `"minmax"` | Min-max scaling |

### Example

```cpp
FeatureStore store;
store.set_cache_path("./cache");

Matrix normalized = store.preprocess(X, "normalize");
Matrix standardized = store.preprocess(X, "standardize");
```

---

## Renderer

OpenGL 3.3 visualization.

### RendererConfig

```cpp
struct RendererConfig {
    int width = 1280;
    int height = 720;
    std::title = "Clustering";
    float point_size = 3.0f;
    bool show_centroids = true;
    bool show_axes = true;
    int background[3] = {30, 30, 40};
};
```

### Methods

| Method | Description |
|--------|-------------|
| `init()` | Initialize OpenGL context |
| `set_data(X, labels, centroids)` | Set visualization data |
| `set_metrics(inertia, iterations)` | Set metrics overlay |
| `run()` | Enter render loop |
| `screenshot(filename)` | Save PNG screenshot |

### Controls

| Input | Action |
|-------|--------|
| Mouse drag | Rotate |
| Scroll | Zoom |
| R | Reset view |
| ESC | Exit |

---

## Distance Functions

```cpp
#include <clustering/distance.h>

float d = l2_distance_avx2(a, b, dim);     // AVX2 optimized
Matrix dist_matrix = compute_distance_matrix(X, Y, threads);  // Batch
```
