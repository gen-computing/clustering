# Examples

## Basic Clustering

```cpp
#include <iostream>
#include "clustering/clustering.h"

using namespace clustering;

int main() {
    // Generate sample data
    Matrix X(500, 3);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < 500; ++i) {
        X[i][0] = dist(gen);
        X[i][1] = dist(gen);
        X[i][2] = dist(gen);
    }

    // Cluster into 5 groups
    KMeans km(5);
    km.fit(X);

    std::cout << "Clusters: " << 5 << std::endl;
    std::cout << "Inertia: " << km.inertia() << std::endl;
    std::cout << "Iterations: " << km.n_iter() << std::endl;

    // Get results
    Vector labels = km.labels();
    Matrix centroids = km.centroids();

    for (size_t i = 0; i < 5; ++i) {
        size_t count = 0;
        for (size_t j = 0; j < labels.size(); ++j) {
            if (labels[j] == i) count++;
        }
        std::cout << "Cluster " << i << ": " << count << " points" << std::endl;
    }

    return 0;
}
```

---

## Streaming Online Learning

```cpp
#include <iostream>
#include "clustering/clustering.h"

using namespace clustering;

int main() {
    OnlineConfig config;
    config.k = 5;
    config.window_size = 1000;
    config.forgetting_factor = 0.99;
    config.auto_retrain = true;

    OnlineKMeans km(config);

    // Simulate streaming data
    for (int batch = 0; batch < 100; ++batch) {
        Matrix X(100, 32);
        std::mt19937 gen(batch);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // Data distribution shifts over time
        float shift = batch * 0.01f;

        for (size_t i = 0; i < 100; ++i) {
            for (size_t d = 0; d < 32; ++d) {
                X[i][d] = dist(gen) + shift;
            }
        }

        km.partial_fit(X);

        if (batch % 10 == 0) {
            std::cout << "Batch " << batch
                      << " | Points seen: " << km.points_seen()
                      << " | Inertia: " << km.inertia() << std::endl;
        }
    }

    return 0;
}
```

---

## Dimensionality Reduction + Clustering

```cpp
#include <iostream>
#include "clustering/clustering.h"

using namespace clustering;

int main() {
    // High-dimensional data
    Matrix X(1000, 100);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < 1000; ++i) {
        for (size_t d = 0; d < 100; ++d) {
            X[i][d] = dist(gen);
        }
    }

    // Reduce to 10 dimensions
    PCA pca(10);
    Matrix Y = pca.fit_transform(X);

    std::cout << "100d -> 10d" << std::endl;
    std::cout << "Variance retained: "
              << pca.total_explained_variance_ratio() * 100 << "%" << std::endl;

    // Cluster in reduced space
    KMeans km(5);
    km.fit(Y);

    std::cout << "Inertia: " << km.inertia() << std::endl;

    // Reconstruct original space
    Matrix X_reconstructed = pca.inverse_transform(Y);

    float error = 0.0f;
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t d = 0; d < 100; ++d) {
            float diff = X[i][d] - X_reconstructed[i][d];
            error += diff * diff;
        }
    }
    error = std::sqrt(error / (1000 * 100));

    std::cout << "Reconstruction RMSE: " << error << std::endl;

    return 0;
}
```

---

## Drift Detection + Auto-Retrain

```cpp
#include <iostream>
#include "clustering/clustering.h"

using namespace clustering;

int main() {
    // Initial training data
    Matrix X(500, 10);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < 500; ++i) {
        for (size_t d = 0; d < 10; ++d) {
            X[i][d] = dist(gen);
        }
    }

    KMeans km(3);
    km.fit(X);

    // Setup drift detection
    DriftDetector detector;
    detector.set_threshold(0.1f);
    detector.set_window_size(10);

    // Simulate production with drift
    for (int t = 0; t < 100; ++t) {
        // Generate data with drift
        Matrix batch(100, 10);
        float drift = (t > 50) ? 5.0f : 0.0f;  # Drift at t=50

        for (size_t i = 0; i < 100; ++i) {
            for (size_t d = 0; d < 10; ++d) {
                batch[i][d] = dist(gen) + drift;
            }
        }

        // Check drift
        Vector labels = km.predict(batch);
        DriftMetrics m = detector.check(batch, labels, km.centroids());

        if (m.drift_detected) {
            std::cout << "Drift detected at t=" << t << "! Retraining..." << std::endl;
            km.fit(batch);  // Retrain on new data
        }

        if (t % 10 == 0) {
            std::cout << "t=" << t
                      << " | Silhouette: " << m.silhouette_score
                      << " | Drift: " << m.drift_detected << std::endl;
        }
    }

    return 0;
}
```

---

## Visualization

```cpp
#include "clustering/clustering.h"

using namespace clustering;

int main() {
    // Generate clustered data
    Matrix X(500, 3);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < 500; ++i) {
        X[i][0] = dist(gen) + (i / 100) * 3.0f;
        X[i][1] = dist(gen) + (i % 5) * 2.0f;
        X[i][2] = dist(gen);
    }

    // Cluster
    KMeans km(5);
    km.fit(X);

    // Visualize
    RendererConfig config;
    config.width = 1280;
    config.height = 720;
    config.title = "KMeans Clustering";
    config.point_size = 4.0f;
    config.show_centroids = true;

    Renderer renderer(config);
    if (renderer.init()) {
        renderer.set_data(X, km.labels(), km.centroids());
        renderer.set_metrics(km.inertia(), km.n_iter());
        renderer.run();  // Blocks until window closes
    }

    return 0;
}
```

---

## Feature Store Caching

```cpp
#include <iostream>
#include "clustering/clustering.h"

using namespace clustering;

int main() {
    FeatureStore store;
    store.set_cache_path("./cache");

    Matrix X(1000, 50);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 10.0f);

    for (size_t i = 0; i < 1000; ++i) {
        for (size_t d = 0; d < 50; ++d) {
            X[i][d] = dist(gen);
        }
    }

    // First call: computes and caches
    auto t1 = std::chrono::high_resolution_clock::now();
    Matrix norm1 = store.preprocess(X, "normalize");
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms1 = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Second call: loads from cache
    t1 = std::chrono::high_resolution_clock::now();
    Matrix norm2 = store.preprocess(X, "normalize");
    t2 = std::chrono::high_resolution_clock::now();
    double ms2 = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "First call: " << ms1 << "ms" << std::endl;
    std::cout << "Cached call: " << ms2 << "ms" << std::endl;
    std::cout << "Speedup: " << ms1 / ms2 << "x" << std::endl;

    return 0;
}
```

---

## Versioning + Rollback

```cpp
#include <iostream>
#include "clustering/clustering.h"

using namespace clustering;

int main() {
    VersionManager vm;
    vm.set_storage_path("./versions");

    // Train v1
    Matrix X1(500, 10);
    // ... fill X1 ...
    KMeans km(3);
    km.fit(X1);
    size_t v1 = vm.save(km.centroids(), km.labels());
    std::cout << "Saved version " << v1 << std::endl;

    // Train v2 (different data)
    Matrix X2(500, 10);
    // ... fill X2 ...
    km.fit(X2);
    size_t v2 = vm.save(km.centroids(), km.labels());
    std::cout << "Saved version " << v2 << std::endl;

    // Rollback to v1
    Matrix centroids;
    Vector labels;
    vm.load(v1, centroids, labels);
    std::cout << "Rolled back to version " << v1 << std::endl;

    return 0;
}
```
