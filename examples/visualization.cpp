#include <iostream>
#include <random>
#include "clustering/clustering.h"

using namespace clustering;

Matrix generate_clusters(size_t n_points, size_t n_clusters, size_t n_features) {
    Matrix X(n_points, n_features);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    size_t points_per_cluster = n_points / n_clusters;

    for (size_t c = 0; c < n_clusters; ++c) {
        std::vector<float> center(n_features);
        for (size_t d = 0; d < n_features; ++d) {
            center[d] = dist(gen) * 5.0f;
        }

        for (size_t i = 0; i < points_per_cluster; ++i) {
            size_t idx = c * points_per_cluster + i;
            for (size_t d = 0; d < n_features; ++d) {
                X[idx][d] = center[d] + dist(gen) * 0.5f;
            }
        }
    }

    return X;
}

int main() {
    std::cout << "=== Visualization Example ===" << std::endl;

    // Generate 3D data for visualization
    const size_t n_points = 500;
    const size_t n_clusters = 4;
    const size_t n_features = 3;

    std::cout << "Generating " << n_points << " points with "
              << n_clusters << " clusters in 3D..." << std::endl;

    Matrix X = generate_clusters(n_points, n_clusters, n_features);

    // Run KMeans
    KMeans engine(n_clusters);
    engine.fit(X);

    std::cout << "Clustering completed. Inertia: " << engine.inertia() << std::endl;

    // Visualize
    RendererConfig config;
    config.width = 1280;
    config.height = 720;
    config.title = "KMeans Clustering - 3D Visualization";
    config.point_size = 4.0f;
    config.show_centroids = true;
    config.show_axes = true;

    Renderer renderer(config);

    if (!renderer.init()) {
        std::cerr << "Failed to initialize renderer" << std::endl;
        return 1;
    }

    renderer.set_data(X, engine.labels(), engine.centroids());
    renderer.set_metrics(engine.inertia(), engine.n_iter());
    renderer.run();

    std::cout << "=== Done ===" << std::endl;
    return 0;
}
