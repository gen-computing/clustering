#include <iostream>
#include <random>
#include "clustering/clustering.h"
#include "clustering/online.h"

using namespace clustering;

int main() {
    std::cout << "=== Streaming Clustering Example ===" << std::endl;

    const size_t n_features = 32;
    const size_t n_clusters = 3;
    const size_t batch_size = 100;
    const size_t n_batches = 10;

    // Create online KMeans engine
    OnlineKMeans engine(n_clusters);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::cout << "Processing " << n_batches << " batches of " << batch_size << " points..."
              << std::endl;

    for (size_t batch = 0; batch < n_batches; ++batch) {
        // Generate batch
        Matrix X(batch_size, n_features);
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t d = 0; d < n_features; ++d) {
                X[i][d] = dist(gen) + static_cast<float>(batch % n_clusters) * 3.0f;
            }
        }

        // Partial fit
        engine.partial_fit(X);

        std::cout << "Batch " << batch + 1 << "/" << n_batches
                  << " completed. Iterations: " << engine.n_iter() << std::endl;
    }

    // Print final centroids
    std::cout << "\nFinal centroids:" << std::endl;
    for (size_t c = 0; c < n_clusters; ++c) {
        std::cout << "  Cluster " << c << ": [";
        for (size_t d = 0; d < std::min(n_features, size_t(5)); ++d) {
            if (d > 0) std::cout << ", ";
            std::cout << engine.centroids()[c][d];
        }
        if (n_features > 5) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }

    // Test prediction on new data
    Matrix test_points(5, n_features);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t d = 0; d < n_features; ++d) {
            test_points[i][d] = dist(gen);
        }
    }

    Vector predictions = engine.predict(test_points);
    std::cout << "\nPredictions for test points:" << std::endl;
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "  Point " << i << " -> Cluster " << predictions[i] << std::endl;
    }

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
