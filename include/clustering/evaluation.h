#pragma once

#include "matrix.h"
#include <vector>
#include <string>

namespace clustering {

struct EvalResult {
    int k;
    float inertia;
    float silhouette_score;
    float davies_bouldin;
    float calinski_harabasz;
    size_t noise = 0;   // DBSCAN: number of noise points (0 for KMeans-family)
};

class ClusterEvaluator {
public:
    ClusterEvaluator();
    ~ClusterEvaluator() = default;

    // Elbow method: run KMeans for k=min_k..max_k, return inertia per k.
    // The "elbow" point is where adding more clusters gives diminishing returns.
    std::vector<EvalResult> elbow(const Matrix& X, size_t min_k = 1, size_t max_k = 15,
                                   size_t max_iter = 100, size_t max_threads = 0);

    // Full evaluation: run KMeans for k=min_k..max_k, return all metrics.
    // Includes silhouette, Davies-Bouldin, and Calinski-Harabasz scores.
    std::vector<EvalResult> evaluate(const Matrix& X, size_t min_k = 2, size_t max_k = 15,
                                      size_t max_iter = 100, size_t max_threads = 0);

    // Find the optimal k based on the best silhouette score.
    size_t best_k_silhouette(const std::vector<EvalResult>& results) const;

    // Find the optimal k based on the best Davies-Bouldin index.
    size_t best_k_db(const std::vector<EvalResult>& results) const;

    // Find the optimal k based on the "elbow" (maximum curvature).
    size_t best_k_elbow(const std::vector<EvalResult>& results) const;
};

} // namespace clustering
