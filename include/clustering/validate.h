#pragma once

#include "matrix.h"
#include <cstddef>

namespace clustering {

// ============================================================================
// validate.h — Shared input validation for all clustering algorithms.
//
// Every public method that accepts data should call these validators
// before doing any computation. This prevents:
//   - NaN poisoning (infinite loops, garbage results)
//   - Empty matrix crashes
//   - Dimension mismatches (segfaults)
//   - Invalid parameters (k=0, k>n, eps<=0)
//
// All validators throw std::runtime_error with descriptive messages.
// ============================================================================

// Validate input data matrix.
// Checks: empty matrix, zero columns, NaN values.
// Used by: all algorithms that accept a Matrix input.
void validate_matrix(const Matrix& X);

// Validate cluster count k against dataset size.
// Checks: k==0, k>n (where n = number of data points).
// Used by: KMeans, MiniBatchKMeans, OnlineKMeans.
void validate_k(size_t k, size_t n);

// Validate PCA parameters.
// Checks: n<2, n_components==0, n_components > min(n_samples, n_features).
// Used by: PCA::fit().
void validate_pca(const Matrix& X, size_t n_components);

// Validate t-SNE parameters.
// Checks: n<2, perplexity >= n (must be less than sample count).
// Used by: TSNE::fit().
void validate_tsne(const Matrix& X, size_t perplexity);

// Validate DBSCAN parameters.
// Checks: epsilon <= 0, min_pts < 2.
// Used by: DBSCAN constructor/fit().
void validate_dbscan(float epsilon, size_t min_pts);

// Check that model has been fitted before prediction.
// Checks: fitted == false.
// Used by: predict(), transform(), inverse_transform() on all algorithms.
void validate_fitted(bool fitted);

// Validate dimension match between expected and actual.
// Checks: expected != actual.
// Used by: OnlineKMeans::partial_fit_point(), PCA transform/inverse_transform.
void validate_dimensions(size_t expected, size_t actual);

// Validate forgetting factor for OnlineKMeans.
// Checks: factor not in [0.0, 1.0].
// Used by: OnlineKMeans::set_forgetting_factor().
void validate_forgetting_factor(float factor);

} // namespace clustering
