// ============================================================================
// ClusterEvaluator -- Find optimal k using quality metrics.
//
// Exploratory clustering asks: "What's the right number of clusters?"
// These methods answer that by running KMeans at different k values and
// evaluating the resulting clustering quality.
//
// METHODS:
//   1. ELBOW METHOD: Plot inertia vs k. The "elbow" (point of diminishing
//      returns) suggests the optimal k. Inertia always decreases with more
//      clusters, but the rate of decrease drops sharply at the right k.
//
//   2. SILHOUETTE SCORE: Measures how well-separated clusters are.
//      Range [-1,1]. Peak silhouette suggests optimal k.
//
//   3. DAVIES-BOULDIN INDEX: Average similarity between each cluster and
//      its most similar neighbor. Lower = better. Minimum suggests optimal k.
//
//   4. CALINSKI-HARABASZ SCORE: Ratio of between-cluster to within-cluster
//      variance. Higher = better. Peak suggests optimal k.
// ============================================================================

#include "clustering/evaluation.h"
#include "clustering/kmeans.h"
#include "clustering/drift.h"
#include "clustering/logging.h"
#include <algorithm>
#include <cmath>

namespace clustering {

ClusterEvaluator::ClusterEvaluator() = default;

std::vector<EvalResult> ClusterEvaluator::elbow(const Matrix& X, size_t min_k, size_t max_k,
                                                  size_t max_iter, size_t max_threads) {
    std::vector<EvalResult> results;
    if (X.rows() < 2) return results;

    size_t max_possible = std::min(max_k, X.rows() - 1);
    min_k = std::max(min_k, size_t(1));

    for (size_t k = min_k; k <= max_possible; ++k) {
        EvalResult r;
        r.k = (int)k;

        KMeansConfig cfg;
        cfg.k = k;
        cfg.max_iter = max_iter;
        cfg.max_threads = max_threads;
        KMeans km(cfg);
        km.fit(X);

        r.inertia = km.inertia();
        r.silhouette_score = 0;
        r.davies_bouldin = 0;
        r.calinski_harabasz = 0;
        results.push_back(r);
    }
    return results;
}

std::vector<EvalResult> ClusterEvaluator::evaluate(const Matrix& X, size_t min_k, size_t max_k,
                                                     size_t max_iter, size_t max_threads) {
    std::vector<EvalResult> results;
    if (X.rows() < 3) return results;

    size_t max_possible = std::min(max_k, X.rows() - 1);
    min_k = std::max(min_k, size_t(2));
    LOG_INFO("ClusterEvaluator::evaluate: testing k=%zu..%zu on n=%zu x d=%zu",
             min_k, max_possible, X.rows(), X.cols());

    for (size_t k = min_k; k <= max_possible; ++k) {
        EvalResult r;
        r.k = (int)k;

        KMeansConfig cfg;
        cfg.k = k;
        cfg.max_iter = max_iter;
        cfg.max_threads = max_threads;
        KMeans km(cfg);
        km.fit(X);

        r.inertia = km.inertia();

        // Compute quality metrics using fresh DriftDetector per k
        // (shared detector accumulates history across k values, contaminating results)
        DriftDetector detector;
        DriftMetrics dm = detector.check(X, km.labels(), km.centroids());
        r.silhouette_score = dm.silhouette_score;
        r.davies_bouldin = dm.davies_bouldin_index;
        r.calinski_harabasz = dm.calinski_harabasz_score;

        results.push_back(r);
    }
    return results;
}

size_t ClusterEvaluator::best_k_silhouette(const std::vector<EvalResult>& results) const {
    if (results.empty()) return 2;
    size_t best = 0;
    float best_score = -2.0f;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].silhouette_score > best_score) {
            best_score = results[i].silhouette_score;
            best = i;
        }
    }
    return results[best].k;
}

size_t ClusterEvaluator::best_k_db(const std::vector<EvalResult>& results) const {
    if (results.empty()) return 2;
    size_t best = 0;
    float best_score = std::numeric_limits<float>::max();
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].davies_bouldin < best_score) {
            best_score = results[i].davies_bouldin;
            best = i;
        }
    }
    return results[best].k;
}

size_t ClusterEvaluator::best_k_elbow(const std::vector<EvalResult>& results) const {
    if (results.size() < 3) return results.empty() ? 2 : (size_t)results[0].k;

    // Find k with maximum curvature (second derivative) of inertia curve
    // curvature = inertia[k-1] + inertia[k+1] - 2*inertia[k]
    size_t best = 0;
    float max_curvature = -std::numeric_limits<float>::max();

    for (size_t i = 1; i < results.size() - 1; ++i) {
        float curvature = results[i - 1].inertia + results[i + 1].inertia
                        - 2.0f * results[i].inertia;
        if (curvature > max_curvature) {
            max_curvature = curvature;
            best = i;
        }
    }
    return results[best].k;
}

} // namespace clustering
