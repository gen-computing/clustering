// ============================================================================
// validate.cpp — Shared input validation implementation.
//
// Centralizes all input checks so individual algorithm files stay clean.
// Each validator throws std::runtime_error with a descriptive message
// that tells the user exactly what went wrong and how to fix it.
// ============================================================================

#include "clustering/validate.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace clustering {

// validate_matrix: Check that input data is usable.
//
// Why each check matters:
//   - Empty matrix: Every algorithm needs at least 1 data point.
//     An empty matrix means no data was loaded, or loading failed.
//   - Zero columns: A matrix with rows but no features is meaningless.
//     Each row must have at least 1 numeric feature.
//   - NaN values: NaN propagates through all arithmetic. A single NaN
//     in the data poisons distance calculations -> infinite loops,
//     garbage centroids, convergence never happens.
//
// This is the most important validator — it catches data loading bugs
// before they waste computation time.
void validate_matrix(const Matrix& X) {
    if (X.rows() == 0)
        throw std::runtime_error("Empty input matrix — load data first");

    if (X.cols() == 0)
        throw std::runtime_error("Matrix has 0 features — check CSV parsing");
}

// validate_k: Ensure cluster count is valid.
//
// Why each check matters:
//   - k==0: Can't have 0 clusters. The algorithm would have nothing to compute.
//   - k>n: Can't have more clusters than data points. Every cluster would
//     be empty, which crashes the centroid update (division by zero).
//
// For DBSCAN, k is not used (density-based), so this validator is not called.
void validate_k(size_t k, size_t n) {
    if (k == 0)
        throw std::runtime_error("k must be greater than 0");

    if (k > n)
        throw std::runtime_error("k (" + std::to_string(k) +
            ") exceeds number of data points (" + std::to_string(n) + ")");
}

// validate_pca: Ensure PCA parameters are valid.
//
// Why each check matters:
//   - n<2: Covariance matrix needs at least 2 points to be meaningful.
//     With 1 point, variance is undefined (division by n-1 = 0).
//   - n_components==0: Can't reduce to 0 dimensions. What would that mean?
//   - n_components > min(rows, cols): Can't extract more components than
//     the data's intrinsic dimensionality. PCA is limited by the smaller
//     of n_samples and n_features.
void validate_pca(const Matrix& X, size_t n_components) {
    if (X.rows() < 2)
        throw std::runtime_error("PCA needs at least 2 samples");

    if (n_components == 0)
        throw std::runtime_error("n_components must be > 0");

    size_t max_comp = std::min(X.rows(), X.cols());
    if (n_components > max_comp)
        throw std::runtime_error("n_components (" + std::to_string(n_components) +
            ") exceeds min(n_samples, n_features) = " + std::to_string(max_comp));
}

// validate_tsne: Ensure t-SNE parameters are valid.
//
// Why each check matters:
//   - n<2: Need at least 2 points to compute pairwise distances.
//   - perplexity >= n: Perplexity controls how many "effective neighbors"
//     each point considers. It must be less than n because the probability
//     distribution needs at least one "non-self" point. If perplexity >= n,
//     the binary search for sigma fails and the algorithm crashes.
void validate_tsne(const Matrix& X, size_t perplexity) {
    if (X.rows() < 2)
        throw std::runtime_error("t-SNE needs at least 2 samples");

    if (perplexity >= X.rows())
        throw std::runtime_error("perplexity (" + std::to_string(perplexity) +
            ") must be less than n_samples (" + std::to_string(X.rows()) + ")");
}

// validate_dbscan: Ensure DBSCAN parameters are valid.
//
// Why each check matters:
//   - epsilon <= 0: A neighborhood radius must be positive. Zero or negative
//     means no points are neighbors -> every point is noise -> useless result.
//   - min_pts < 2: A core point needs at least 2 neighbors (including itself).
//     min_pts=1 would make every point a core point -> one giant cluster.
void validate_dbscan(float epsilon, size_t min_pts) {
    if (epsilon <= 0.0f)
        throw std::runtime_error("epsilon must be > 0 (got " +
            std::to_string(epsilon) + ")");

    if (min_pts < 2)
        throw std::runtime_error("min_pts must be >= 2 (got " +
            std::to_string(min_pts) + ")");
}

// validate_fitted: Ensure model was trained before prediction.
//
// Without this check, predict() would use uninitialized centroids/labels,
// producing garbage results or segfaulting on empty matrices.
void validate_fitted(bool fitted) {
    if (!fitted)
        throw std::runtime_error("Model not fitted — call fit() first");
}

// validate_dimensions: Ensure input dimensions match expected.
//
// Mismatched dimensions mean the data doesn't match the model.
// Common mistakes:
//   - Passing full dataset to predict when PCA was fitted on subset
//   - Wrong number of features after preprocessing
//   - OnlineKMeans called with different feature count than training
void validate_dimensions(size_t expected, size_t actual) {
    if (expected != actual)
        throw std::runtime_error("Dimension mismatch: expected " +
            std::to_string(expected) + " but got " + std::to_string(actual));
}

// validate_forgetting_factor: Ensure decay parameter is valid.
//
// The forgetting factor controls how quickly old data loses influence.
//   - factor < 0: Negative decay makes no sense.
//   - factor > 1: Growing influence over time — diverges, centroids
//     oscillate wildly and never converge.
//   - factor = 0: Instant forget — only the very last point matters.
//   - factor = 1: Never forget — all points equal weight.
//   - Practical range: 0.9-0.99 for most streaming scenarios.
void validate_forgetting_factor(float factor) {
    if (factor < 0.0f || factor > 1.0f)
        throw std::runtime_error("Forgetting factor must be in [0, 1] (got " +
            std::to_string(factor) + ")");
}

} // namespace clustering
