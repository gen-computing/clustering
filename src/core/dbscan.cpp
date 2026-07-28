// ============================================================================
// DBSCAN -- Density-Based Spatial Clustering of Applications with Noise.
//
// Unlike KMeans (which needs k pre-specified), DBSCAN discovers clusters
// based on point DENSITY. It can find arbitrarily-shaped clusters and
// automatically detects noise points.
//
// KEY CONCEPTS:
//   epsilon (eps): Maximum distance between two points to be "neighbors".
//   minPts: Minimum number of points in a neighborhood to form a cluster.
//   Core point: Has >= minPts neighbors within epsilon.
//   Border point: Has < minPts neighbors but is within epsilon of a core point.
//   Noise point: Neither core nor border (isolated).
//
// ALGORITHM:
//   1. For each unvisited point:
//      a. Find all neighbors within epsilon distance.
//      b. If neighbors >= minPts: start a new cluster, expand it.
//         (Add all density-reachable core points and their border points)
//      c. Otherwise: mark as noise (may later become border of another cluster).
//   2. Stop when all points visited.
//
// COMPLEXITY: O(n²) without spatial index, O(n log n) with KD-tree.
//             Our implementation is O(n²) per point for simplicity.
//
// ADVANTAGES over KMeans:
//   - No need to specify number of clusters
//   - Can find non-spherical clusters
//   - Robust to outliers (marks them as noise)
//   - Deterministic (no random init)
//
// DISADVANTAGES:
//   - eps and minPts need tuning
//   - Struggles with varying density clusters
//   - O(n²) distance computation
// ============================================================================

#include "clustering/dbscan.h"
#include "clustering/distance.h"
#include "clustering/logging.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace clustering {

DBSCAN::DBSCAN() : n_clusters_(0), n_noise_(0), fitted_(false) {}

DBSCAN::DBSCAN(const DBSCANConfig& config)
    : config_(config), n_clusters_(0), n_noise_(0), fitted_(false) {}

DBSCAN::~DBSCAN() = default;

void DBSCAN::fit(const Matrix& X) {
    LOG_INFO("DBSCAN::fit: n=%zu, d=%zu, eps=%.3f, min_pts=%zu", X.rows(), X.cols(), config_.epsilon, config_.min_pts);
    if (X.rows() == 0) throw std::runtime_error("Empty input matrix");
    if (X.cols() == 0) throw std::runtime_error("No features");

    // Store training data for predict()
    X_ = X;

    size_t n = X.rows();
    labels_.resize(n);
    labels_.fill(-1.0f);  // All points start as UNCLASSIFIED (-1)
    visited_.assign(n, false);
    n_clusters_ = 0;
    n_noise_ = 0;

    for (size_t i = 0; i < n; ++i) {
        if (visited_[i]) continue;
        visited_[i] = true;

        // Find all points within epsilon of point i
        std::vector<size_t> neighbors;
        region_query(X, i, neighbors);

        if (neighbors.size() < config_.min_pts) {
            labels_[i] = -2.0f;  // Noise
            n_noise_++;
            LOG_DEBUG("DBSCAN: point %zu -> noise (%zu neighbors < %zu min_pts)", i, neighbors.size(), config_.min_pts);
        } else {
            n_clusters_++;
            LOG_DEBUG("DBSCAN: point %zu -> CORE (cluster %zu, %zu neighbors)", i, n_clusters_, neighbors.size());
            expand_cluster(X, i, n_clusters_);
        }
    }

    // Relabel: convert -2 (noise) to actual cluster ID 0, and shift real clusters 1..k
    // to 1..k (so noise = 0, clusters = 1,2,3...)
    for (size_t i = 0; i < n; ++i) {
        if (labels_[i] == -2.0f) labels_[i] = 0.0f;  // Noise = cluster 0
    }

    fitted_ = true;
}

void DBSCAN::region_query(const Matrix& X, size_t point_idx, std::vector<size_t>& neighbors) const {
    neighbors.clear();
    size_t n = X.rows();
    size_t d = X.cols();
    const float* point = X[point_idx];

    for (size_t j = 0; j < n; ++j) {
        float dist = l2_distance_avx2(point, X[j], d);
        if (dist <= config_.epsilon) {
            neighbors.push_back(j);
        }
    }
}

void DBSCAN::expand_cluster(const Matrix& X, size_t point_idx, size_t cluster_id) {
    labels_[point_idx] = (float)cluster_id;
    if (labels_[point_idx] == -2.0f) n_noise_--;

    std::vector<size_t> seeds;
    region_query(X, point_idx, seeds);

    // O(1) membership tracking: avoids O(n) linear scan per neighbor.
    // Uses uint8_t NOT bool — vector<bool> is a packed bitset with proxy references
    // that causes heap corruption when mixed with regular vector operations.
    std::vector<uint8_t> in_seeds(X.rows(), 0);
    for (size_t s : seeds) in_seeds[s] = 1;

    for (size_t si = 0; si < seeds.size(); ++si) {
        size_t current = seeds[si];

        if (!visited_[current]) {
            visited_[current] = true;
            std::vector<size_t> current_neighbors;
            region_query(X, current, current_neighbors);

            if (current_neighbors.size() >= config_.min_pts) {
                for (size_t nbr : current_neighbors) {
                    if (!in_seeds[nbr]) {
                        in_seeds[nbr] = 1;
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

Vector DBSCAN::predict(const Matrix& X) const {
    if (!fitted_) throw std::runtime_error("DBSCAN not fitted");

    size_t n = X.rows();
    Vector result(n);

    // For new points: find nearest training point and return its cluster label.
    // If nearest training point is noise (label 0), return noise.
    for (size_t i = 0; i < n; ++i) {
        float min_dist = std::numeric_limits<float>::max();
        size_t nearest_idx = 0;

        for (size_t j = 0; j < X_.rows(); ++j) {
            float dist = l2_distance_avx2(X[i], X_[j], X.cols());
            if (dist < min_dist) {
                min_dist = dist;
                nearest_idx = j;
            }
        }

        // If nearest training point is within epsilon, assign its label
        if (min_dist <= config_.epsilon) {
            result[i] = labels_[nearest_idx];
        } else {
            result[i] = 0.0f;  // 0 = noise
        }
    }

    return result;
}

float DBSCAN::estimate_epsilon(const Matrix& X, size_t min_pts, size_t sample_size) {
    size_t n = X.rows();
    if (n < 2) return 0.5f;

    size_t sn = std::min(sample_size, n);
    size_t step = n / sn;
    float sum_knn = 0.0f;
    size_t count = 0;

    for (size_t si = 0; si < sn; ++si) {
        size_t pi = si * step;
        const float* point = X[pi];
        std::vector<float> dists;
        dists.reserve(n);
        for (size_t j = 0; j < n; ++j) {
            if (j == pi) continue;
            float d = l2_distance_avx2(point, X[j], X.cols());
            dists.push_back(d);
        }
        std::nth_element(dists.begin(), dists.begin() + std::min(min_pts, dists.size()), dists.end());
        sum_knn += dists[std::min(min_pts, dists.size()) - 1];
        count++;
    }

    return count > 0 ? sum_knn / count : 0.5f;
}

} // namespace clustering
