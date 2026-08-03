// ============================================================================
// DBSCAN implementation — Density-Based Spatial Clustering of Applications
// with Noise. Uses nanoflann KD-tree for O(n log n) radius queries.
// ============================================================================

#include "clustering/dbscan.h"
#include "clustering/validate.h"
#include "clustering/logging.h"
#include <nanoflann.hpp>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace clustering {

// Use a fixed dimension of 3 for the KD-tree (pad 2D data with zeros)
// This avoids template issues with variable dimensions.
static const int KD_DIM = 3;

using KDTreeIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, DBSCAN::KDTreeAdaptor>,
    DBSCAN::KDTreeAdaptor, KD_DIM>;

// ============================================================================
// CONSTRUCTORS / DESTRUCTOR
// ============================================================================

DBSCAN::DBSCAN()
    : n_clusters_(0), n_noise_(0), fitted_(false) {}

DBSCAN::DBSCAN(const DBSCANConfig& config)
    : config_(config), n_clusters_(0), n_noise_(0), fitted_(false) {}

DBSCAN::~DBSCAN() = default;

// ============================================================================
// fit() — MAIN DBSCAN ALGORITHM
// ============================================================================
void DBSCAN::fit(const Matrix& X) {
    LOG_INFO("DBSCAN::fit: n=%zu, d=%zu, eps=%.3f, min_pts=%zu",
             X.rows(), X.cols(), config_.epsilon, config_.min_pts);

    validate_matrix(X);
    validate_dbscan(config_.epsilon, config_.min_pts);

    // Store training data
    X_ = X;

    // Build KD-tree once for all queries
    auto adaptor = std::make_unique<KDTreeAdaptor>(X_);
    auto* index = new KDTreeIndex(3, *adaptor);
    index->buildIndex();

    // Store index and adaptor for reuse across all region_query calls
    if (kdtree_index_) {
        delete static_cast<KDTreeIndex*>(kdtree_index_);
    }
    kdtree_adaptor_ = std::move(adaptor);
    kdtree_index_ = index;

    size_t n = X.rows();
    labels_.resize(n);
    labels_.fill(-1.0f);
    visited_.assign(n, false);
    n_clusters_ = 0;
    n_noise_ = 0;

    // ---- MAIN LOOP ----
    for (size_t i = 0; i < n; ++i) {
        if (visited_[i]) continue;
        visited_[i] = true;

        std::vector<size_t> neighbors;
        region_query(i, neighbors);

        if (neighbors.size() < config_.min_pts) {
            labels_[i] = -2.0f;  // Noise
            n_noise_++;
        } else {
            n_clusters_++;
            expand_cluster(i, n_clusters_);
        }
    }

    // Relabel: noise (-2) -> 0, clusters 1..k stay as-is
    for (size_t i = 0; i < n; ++i) {
        if (labels_[i] == -2.0f) labels_[i] = 0.0f;
    }

    fitted_ = true;
    LOG_INFO("DBSCAN::fit done: clusters=%zu, noise=%zu", n_clusters_, n_noise_);
}

// ============================================================================
// region_query() — FIND NEIGHBORS USING KD-TREE
// ============================================================================
void DBSCAN::region_query(size_t point_idx, std::vector<size_t>& neighbors) const {
    neighbors.clear();
    if (!kdtree_index_) return;

    auto* index = static_cast<KDTreeIndex*>(kdtree_index_);

    // Pad query point to KD_DIM (3) dimensions
    float query_point[KD_DIM] = {0, 0, 0};
    for (size_t d = 0; d < X_.cols() && d < KD_DIM; ++d) {
        query_point[d] = X_[point_idx][d];
    }

    float search_radius = config_.epsilon * config_.epsilon;
    std::vector<std::pair<size_t, float>> result_pairs;

    index->radiusSearch(query_point, search_radius, result_pairs,
                        nanoflann::SearchParams(10, 0, false));

    neighbors.reserve(result_pairs.size());
    for (const auto& p : result_pairs) {
        neighbors.push_back(p.first);
    }
}

// ============================================================================
// expand_cluster() — BFS EXPANSION
// ============================================================================
void DBSCAN::expand_cluster(size_t point_idx, size_t cluster_id) {
    labels_[point_idx] = (float)cluster_id;
    if (labels_[point_idx] == -2.0f) n_noise_--;

    std::vector<size_t> seeds;
    region_query(point_idx, seeds);

    // Use pre-allocated member instead of per-call vector allocation
    in_seeds_.assign(X_.rows(), 0);
    for (size_t s : seeds) in_seeds_[s] = 1;

    for (size_t si = 0; si < seeds.size(); ++si) {
        size_t current = seeds[si];

        if (!visited_[current]) {
            visited_[current] = true;
            std::vector<size_t> current_neighbors;
            region_query(current, current_neighbors);

            if (current_neighbors.size() >= config_.min_pts) {
                for (size_t nbr : current_neighbors) {
                    if (!in_seeds_[nbr]) {
                        in_seeds_[nbr] = 1;
                        seeds.push_back(nbr);
                    }
                }
            }
        }

        if (labels_[current] < 0) {
            if (labels_[current] == -2.0f) n_noise_--;
            labels_[current] = (float)cluster_id;
        }
    }
}

// ============================================================================
// predict() — ASSIGN NEW POINTS USING KD-TREE
// ============================================================================
Vector DBSCAN::predict(const Matrix& X) const {
    validate_fitted(fitted_);

    size_t n = X.rows();
    Vector result(n);

    // Use cached KD-tree from fit() (no rebuild)
    auto* index = static_cast<KDTreeIndex*>(kdtree_index_);

    for (size_t i = 0; i < n; ++i) {
        float query_point[KD_DIM] = {0, 0, 0};
        for (size_t d = 0; d < X.cols() && d < KD_DIM; ++d) {
            query_point[d] = X[i][d];
        }

        size_t nearest_idx;
        float nearest_dist_sq;
        index->knnSearch(query_point, 1, &nearest_idx, &nearest_dist_sq);

        if (std::sqrt(nearest_dist_sq) <= config_.epsilon) {
            result[i] = labels_[nearest_idx];
        } else {
            result[i] = 0.0f;
        }
    }

    return result;
}

// ============================================================================
// estimate_epsilon()
// ============================================================================
float DBSCAN::estimate_epsilon(const Matrix& X, size_t min_pts, size_t sample_size) {
    size_t n = X.rows();
    if (n < 2) return 0.5f;

    KDTreeAdaptor adaptor(X);
    KDTreeIndex index(3, adaptor);
    index.buildIndex();

    size_t sn = std::min(sample_size, n);
    size_t step = std::max(n / sn, (size_t)1);
    float sum_knn = 0.0f;
    size_t count = 0;

    for (size_t si = 0; si < sn; ++si) {
        size_t pi = si * step;
        if (pi >= n) break;

        float query_point[3];
        for (size_t d = 0; d < X.cols(); ++d) {
            query_point[d] = X[pi][d];
        }

        size_t k = std::min(min_pts, n - 1);
        std::vector<size_t> indices(k);
        std::vector<float> dists_sq(k);
        index.knnSearch(query_point, k, indices.data(), dists_sq.data());

        sum_knn += std::sqrt(dists_sq[k - 1]);
        count++;
    }

    return count > 0 ? sum_knn / count : 0.5f;
}

} // namespace clustering
