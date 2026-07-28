#pragma once

#include "matrix.h"
#include <vector>

namespace clustering {

// ============================================================================
// PCA -- Principal Component Analysis (dimensionality reduction).
//
// WHAT IT DOES:
//   Takes high-dimensional data (e.g., 100 features) and projects it into
//   fewer dimensions (e.g., 2 or 3) while preserving as much VARIANCE
//   (information/spread) as possible.
//
// ANALOGY:
//   Imagine a cloud of points in 3D space that actually lies on a flat 2D sheet.
//   PCA finds that sheet and gives you the 2D coordinates of each point on it.
//
// REAL WORLD USES:
//   - Visualizing 100-dimensional data in 2D or 3D plots
//   - Speeding up ML algorithms by reducing features (less computation)
//   - Removing noise (discarding dimensions with tiny variance)
//   - Data compression (reconstruct approximate original from fewer dimensions)
//
// HOW IT WORKS (internally using Eigen3 + BLAS):
//   1. Center the data (subtract mean from each feature)
//   2. Compute the covariance matrix (captures how features vary together)
//   3. Find eigenvectors/eigenvalues of the covariance matrix
//      (these are the "principal components" -- the directions of maximum variance)
//   4. Keep only the top k eigenvectors (those with largest eigenvalues)
//   5. Project data onto those eigenvectors to get reduced dimensions
//
// KEY TERMS:
//   Explained variance: How much of the original data's "spread" each component
//                       captures. 0.35 means 35% of all variance.
//   Total explained variance ratio: Sum of explained variances of kept components.
//                                   0.85 means we kept 85% of the information.
//   Inverse transform: Reconstruct approximate original data from reduced form.
//                       Not perfect (some info lost), but close.
// ============================================================================
class PCA {
public:
    // Default constructor: creates unconfigured PCA (set n_components later if needed).
    PCA() = default;

    // Constructor: specify how many dimensions to reduce TO.
    // Example: PCA pca(2);  // reduce data to 2 dimensions
    // n_components must be <= min(n_samples, n_features) and >= 1.
    explicit PCA(size_t n_components);

    // Convenience: also accepts int (so PCA(3) works without casting).
    explicit PCA(int n_components) : PCA(static_cast<size_t>(n_components)) {}

    ~PCA() = default;

    // ----- STANDARD API (uses Matrix class) -----

    // fit: LEARN the transformation from data.
    // Computes mean, covariance, eigenvectors. Does NOT transform the data.
    // Call once on training data. Then call transform() on any data.
    // Throws: runtime_error if X is empty, 0 features, <2 samples,
    //         n_components=0, or n_components > min(rows, cols).
    void fit(const Matrix& X);

    // transform: APPLY the learned transformation to new data.
    // Projects data from original dimension down to n_components dimensions.
    // Must call fit() first.
    // Returns: Matrix with same rows as X, n_components columns.
    // Throws: runtime_error if not fitted, or feature count mismatch.
    Matrix transform(const Matrix& X) const;

    // fit_transform: LEARN AND TRANSFORM in one call.
    // Convenience method -- equivalent to fit(X) then transform(X).
    // Returns: The reduced-dimension representation of X.
    Matrix fit_transform(const Matrix& X);

    // inverse_transform: RECONSTRUCT approximate original data from reduced form.
    // Goes from n_components dimensions BACK to original n_features_ dimensions.
    // The reconstruction is approximate (some info was lost during reduction).
    // Formula: X_reconstructed = Y @ components + mean
    //           (project back into original space, then add back mean)
    // Throws: runtime_error if not fitted, or component count mismatch.
    Matrix inverse_transform(const Matrix& X) const;

    // ----- RAW POINTER API (used by Python bindings for speed) -----
    // These operate directly on float* arrays, avoiding Matrix copies.
    // Used when integrating with numpy through pybind11.

    // fit_raw: Learn PCA from raw float array.
    // data   -- pointer to raw float data (row-major layout)
    // n_rows -- number of data points
    // n_cols -- number of features per point
    void fit_raw(const float* data, size_t n_rows, size_t n_cols);

    // transform_raw: Apply PCA to raw data, write result to pre-allocated buffer.
    // data   -- input data (n_rows x n_features_)
    // n_rows -- number of points to transform
    // out    -- output buffer (must be n_rows * n_components floats)
    void transform_raw(const float* data, size_t n_rows, float* out) const;

    // fit_transform_raw: Learn and transform in one call (raw pointers).
    // data   -- input (n_rows x n_cols)
    // n_cols -- original feature count
    // out    -- output buffer (n_rows * n_components)
    void fit_transform_raw(const float* data, size_t n_rows, size_t n_cols, float* out);

    // inverse_transform_raw: Reverse transform using raw pointers.
    // data   -- reduced data (n_rows x n_components)
    // out    -- output buffer (n_rows * n_features_)
    void inverse_transform_raw(const float* data, size_t n_rows, float* out) const;

    // ----- ACCESSORS (read results after fitting) -----
    size_t n_components() const { return n_components_; }

    // components: Principal axes (directions of maximum variance).
    //              Matrix of shape (n_components x n_features_).
    //              Each row is one principal component vector.
    const Matrix& components() const { return components_; }

    // explained_variance: How much variance EACH component captures (absolute value).
    //                      The eigenvalue for each component. Larger = more important.
    const Vector& explained_variance() const { return explained_variance_; }

    // explained_variance_ratio: Variance captured by each component as FRACTION (0-1).
    //                            explained_variance_ratio[0] is for the first component.
    const Vector& explained_variance_ratio() const { return explained_variance_ratio_; }

    // total_explained_variance_ratio: SUM of all explained_variance_ratios.
    //                                  Tells you what fraction of total information
    //                                  was preserved. 0.95 = kept 95% of data spread.
    float total_explained_variance_ratio() const;

private:
    size_t n_components_;              // Target dimension (how many components to keep)
    size_t n_features_;                // Original dimension (set during fit)
    Matrix components_;                // Principal component vectors (n_comp x n_feat)
    Vector explained_variance_;        // Eigenvalues (one per component)
    Vector explained_variance_ratio_;  // Eigenvalues as fraction of total
    Vector mean_;                      // Mean of each original feature (for centering)
    bool fitted_;                      // Has fit() been called successfully?
};

} // namespace clustering
