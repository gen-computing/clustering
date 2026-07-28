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

int main(int argc, char* argv[]) {
    bool use_viz = (argc > 1 && std::string(argv[1]) == "--viz");

    // Generate 3D data for visualization
    const size_t n_points = 500;
    const size_t n_clusters = 5;
    const size_t n_features = use_viz ? 3 : 32;

    std::cout << "=== Basic Clustering Example ===" << std::endl;
    std::cout << "Generating " << n_points << " points with "
              << n_clusters << " clusters in " << n_features << " dimensions..."
              << std::endl;

    Matrix X = generate_clusters(n_points, n_clusters, n_features);

    KMeans engine(n_clusters);
    engine.fit(X);

    std::cout << "Clustering completed in " << engine.n_iter() << " iterations" << std::endl;
    std::cout << "Inertia: " << engine.inertia() << std::endl;

    // Count points per cluster
    std::vector<size_t> counts(n_clusters, 0);
    for (size_t i = 0; i < engine.labels().size(); ++i) {
        counts[static_cast<size_t>(engine.labels()[i])]++;
    }

    for (size_t c = 0; c < n_clusters; ++c) {
        std::cout << "Cluster " << c << ": " << counts[c] << " points" << std::endl;
    }

    // Visualize if requested
    if (use_viz) {
        std::cout << "\nOpening visualization window..." << std::endl;
        std::cout << "Controls: Mouse=Rotate, Scroll=Zoom, R=Reset, ESC=Exit" << std::endl;

        RendererConfig config;
        config.width = 1280;
        config.height = 720;
        config.title = "KMeans Clustering";
        config.point_size = 8.0f;
        config.show_centroids = true;

        Renderer renderer(config);
        if (renderer.init()) {
            renderer.set_data(X, engine.labels(), engine.centroids());
            renderer.set_metrics(engine.inertia(), engine.n_iter());
            renderer.run();
        } else {
            std::cerr << "Failed to initialize renderer" << std::endl;
            return 1;
        }
    } else {
        std::cout << "\nRun with --viz flag for 3D visualization" << std::endl;
    }

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
