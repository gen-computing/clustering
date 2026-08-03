// ============================================================================
// MiniBatchKMeans implementation -- Fast KMeans using random subsets.
//
// Instead of using ALL points every iteration (like regular KMeans),
// MiniBatchKMeans randomly samples a small batch of points per iteration.
// This makes it much faster on large datasets with minimal quality loss.
//
// HOW IT WORKS:
//   1. First call to partial_fit() does a full fit() to initialize centroids.
//   2. Subsequent calls randomly sample batch_size_ points from the new data.
//   3. For each sampled point: find nearest centroid, nudge centroid toward it.
//   4. Learning rate decays: 1 / (iteration + 1) -- early points have more impact.
//
// COMPARISON:
//   KMeans:        accurate, uses ALL data every iteration
//   MiniBatch:     fast, uses only a RANDOM SAMPLE per iteration
//   OnlineKMeans:  streaming, processes data in arrival order
// ============================================================================

#include "clustering/mini_batch.h"
#include "clustering/validate.h"
#include <random>     // std::mt19937, std::uniform_int_distribution
#include <algorithm>  // std::min

namespace clustering {

// Constructor: pass k and batch_size to parent KMeans constructor.
// batch_size_ controls how many random points are used per update.
MiniBatchKMeans::MiniBatchKMeans(size_t k, size_t batch_size)
    : KMeans(k), batch_size_(batch_size), n_updates_(0) {}

MiniBatchKMeans::~MiniBatchKMeans() = default;

// ============================================================================
// partial_fit() -- RANDOM BATCH UPDATE
// ============================================================================

void MiniBatchKMeans::partial_fit(const Matrix& X) {
    validate_matrix(X);

    // First call: no centroids yet. Do full fit() to initialize.
    if (!fitted_) {
        fit(X);
        return;
    }

    // ---- RANDOM SAMPLING ----
    // Set up random number generator for selecting random point indices.
    std::random_device rd;         // Hardware entropy source
    std::mt19937 gen(rd());        // Mersenne Twister engine
    std::uniform_int_distribution<size_t> dist(0, X.rows() - 1);  // [0, n-1]

    // Use min(batch_size, available_points) -- don't sample more than exist.
    size_t actual_batch = std::min(batch_size_, X.rows());

    // Pick random indices (WITH replacement -- same point can be picked twice).
    // This is simpler than without-replacement sampling and works fine in practice.
    std::vector<size_t> indices(actual_batch);
    for (size_t i = 0; i < actual_batch; ++i) {
        indices[i] = dist(gen);
    }

    // ---- COMPUTE DISTANCES FOR BATCH ----
    // Only compute distances for the sampled points, not all points.
    // This is the key performance difference from regular KMeans.
    Matrix batch_distances(actual_batch, centroids().rows());

    for (size_t i = 0; i < actual_batch; ++i) {
        for (size_t c = 0; c < centroids().rows(); ++c) {
            // Compute Euclidean distance from sampled point to centroid c.
            // Uses scalar computation (loop over features) here -- could be
            // optimized with AVX2 for better performance.
            float dist = 0.0f;
            for (size_t d = 0; d < X.cols(); ++d) {
                float diff = X[indices[i]][d] - centroids()[c][d];
                dist += diff * diff;
            }
            batch_distances[i][c] = std::sqrt(dist);
        }
    }

    // ---- UPDATE CENTROIDS ----
    // For each sampled point, find nearest centroid and nudge toward it.
    // Learning rate decays per POINT (not per batch) for proper convergence.
    // Standard mini-batch KMeans (Sculley 2010): lr = 1/t where t = point count.
    for (size_t i = 0; i < actual_batch; ++i) {
        // Find nearest centroid for this sampled point
        size_t nearest = 0;
        float min_dist = batch_distances[i][0];
        for (size_t c = 1; c < centroids().rows(); ++c) {
            if (batch_distances[i][c] < min_dist) {
                min_dist = batch_distances[i][c];
                nearest = c;
            }
        }

        // Learning rate: decays with number of individual point assignments.
        // lr = 1 / (n_updates + 1)
        // First point: lr = 1.0 (large move)
        // After 100 points: lr = 0.01 (fine-tuning)
        // After 10000 points: lr = 0.0001 (converged)
        float lr = 1.0f / static_cast<float>(n_updates_ + 1);

        // Move centroid toward the point:
        // centroid = centroid + lr * (point - centroid)
        for (size_t d = 0; d < X.cols(); ++d) {
            centroids_mut()[nearest][d] += lr * (X[indices[i]][d] - centroids()[nearest][d]);
        }

        ++n_updates_;  // Per point, not per batch
    }
}

} // namespace clustering
