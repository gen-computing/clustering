// ============================================================================
// MiniBatchKMeans — Fast KMeans using random subsets with AVX2 distances.
// ============================================================================

#include "clustering/mini_batch.h"
#include "clustering/validate.h"
#include "clustering/distance.h"
#include <algorithm>

namespace clustering {

MiniBatchKMeans::MiniBatchKMeans(size_t k, size_t batch_size)
    : KMeans(k), batch_size_(batch_size), n_updates_(0),
      rng_(std::random_device{}()) {}

MiniBatchKMeans::~MiniBatchKMeans() = default;

void MiniBatchKMeans::partial_fit(const Matrix& X) {
    validate_matrix(X);

    if (!fitted_) {
        fit(X);
        return;
    }

    size_t n_centroids = centroids().rows();
    size_t d = X.cols();
    size_t actual_batch = std::min(batch_size_, X.rows());

    // ---- RANDOM SAMPLING (reuse pre-allocated vector) ----
    indices_.resize(actual_batch);
    std::uniform_int_distribution<size_t> dist(0, X.rows() - 1);
    for (size_t i = 0; i < actual_batch; ++i) {
        indices_[i] = dist(rng_);
    }

    // ---- COMPUTE DISTANCES (AVX2) ----
    // Reuse flat buffer instead of allocating Matrix per call.
    // Layout: dist_buffer_[i * n_centroids + c] = distance from point i to centroid c.
    dist_buffer_.resize(actual_batch * n_centroids);

    for (size_t i = 0; i < actual_batch; ++i) {
        const float* point = X[indices_[i]];
        for (size_t c = 0; c < n_centroids; ++c) {
            dist_buffer_[i * n_centroids + c] = l2_distance_avx2(point, centroids()[c], d);
        }
    }

    // ---- UPDATE CENTROIDS ----
    for (size_t i = 0; i < actual_batch; ++i) {
        // Find nearest centroid (scan flat buffer)
        size_t nearest = 0;
        float min_dist = dist_buffer_[i * n_centroids];
        for (size_t c = 1; c < n_centroids; ++c) {
            float d = dist_buffer_[i * n_centroids + c];
            if (d < min_dist) { min_dist = d; nearest = c; }
        }

        // Learning rate: per-point decay (Sculley 2010)
        float lr = 1.0f / static_cast<float>(n_updates_ + 1);

        // Centroid update: centroid += lr * (point - centroid)
        float* cent = centroids_mut()[nearest];
        const float* point = X[indices_[i]];
        for (size_t d = 0; d < X.cols(); ++d) {
            cent[d] += lr * (point[d] - cent[d]);
        }

        ++n_updates_;
    }
}

} // namespace clustering
