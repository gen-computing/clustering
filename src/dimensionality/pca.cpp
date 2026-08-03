// ============================================================================
// PCA implementation -- Principal Component Analysis using Eigen3 + BLAS.
//
// PCA reduces the number of features (dimensions) while preserving as much
// variance (information) as possible. It finds the "principal components" --
// directions in the data that capture the most variation.
//
// HOW IT WORKS (mathematically):
//   1. Center the data (subtract the mean of each feature)
//   2. Compute the covariance matrix: cov = X_centered^T * X_centered / (n-1)
//      (The covariance matrix captures how features vary together.)
//   3. Perform eigenvalue decomposition on cov.
//      (Eigenvectors = principal directions, Eigenvalues = variance captured.)
//   4. Sort by eigenvalue descending, keep top k.
//   5. Transform: Y = X_centered * components^T
//   6. Inverse transform: X_approx = Y * components + mean
//
// DEPENDENCIES:
//   Eigen3: C++ template library for linear algebra, used for:
//     - Matrix multiplication (BLAS-accelerated via OpenBLAS)
//     - Eigenvalue decomposition (SelfAdjointEigenSolver)
//     - Column-wise operations (rowwise(), colwise())
//
//   OpenBLAS: Provides optimized BLAS routines that Eigen can use.
//     BLAS (Basic Linear Algebra Subprograms) is the standard low-level
//     API for vector/matrix operations. OpenBLAS uses hand-tuned assembly
//     for each CPU architecture.
//
// MEMORY LAYOUT:
//   Row-major: data[i * cols + j] is element at row i, column j.
//   This matches numpy's default layout, making zero-copy interop possible.
// ============================================================================

#include "clustering/pca.h"
#include "clustering/validate.h"
#include <Eigen/Dense>           // Core Eigen matrix types
#include <Eigen/Eigenvalues>     // SelfAdjointEigenSolver
#include <Eigen/SVD>             // BDCSVD for randomized SVD path
#include <cmath>                 // std::max
#include <cstring>               // std::memcpy
#include <algorithm>             // std::min, std::max
#include <numeric>               // std::accumulate
#include <random>                // std::mt19937 (unused in current impl, for future)
#include <stdexcept>             // std::runtime_error for input validation

namespace clustering {

// Define a type alias for row-major float matrices.
// Eigen::Matrix<float, Dynamic, Dynamic, Eigen::RowMajor> means:
//   float:     element type
//   Dynamic:   rows and cols determined at runtime
//   RowMajor:  stored row-by-row (numpy compatible)
//
// This is important: Eigen defaults to COLUMN-major (Fortran order),
// but numpy uses ROW-major (C order). Without RowMajor, data would
// appear transposed when shared with Python.
using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ============================================================================
// CONSTRUCTOR
// ============================================================================

// Store the number of components we want to keep.
// n_features_ is set during fit(); components_ and explained_variance_ are
// empty until training.
PCA::PCA(size_t n_components)
    : n_components_(n_components), n_features_(0), fitted_(false) {}

// ============================================================================
// fit() -- LEARN THE PCA TRANSFORMATION (Matrix API)
// ============================================================================

void PCA::fit(const Matrix& X) {
    validate_matrix(X);
    validate_pca(X, n_components_);

    // Delegate to the raw pointer version (the actual implementation).
    fit_raw(X.data(), X.rows(), X.cols());
}

// ============================================================================
// fit_raw() -- LEARN PCA FROM RAW FLOAT ARRAY (the real implementation)
//
// This is the core PCA algorithm. Uses Eigen for matrix operations.
// ============================================================================

void PCA::fit_raw(const float* data, size_t n_rows, size_t n_cols) {
    if (n_rows < 2) throw std::runtime_error("PCA needs at least 2 samples");
    if (n_cols == 0) throw std::runtime_error("Matrix has 0 features");
    if (n_components_ == 0) throw std::runtime_error("n_components must be > 0");
    if (n_components_ > std::min(n_rows, n_cols))
        throw std::runtime_error("n_components exceeds min(rows, cols)");

    // Rename for clarity in formulas.
    size_t n = n_rows;   // Number of samples
    size_t d = n_cols;   // Number of features (original dimensionality)
    n_features_ = d;      // Store for later use by transform/inverse_transform
    size_t k = n_components_;  // Number of components to keep

    // ---- STEP 1: Map raw data to Eigen matrix ----
    // Eigen::Map creates an Eigen matrix VIEW over existing memory.
    // No copy happens -- the Eigen matrix reads/writes directly from `data`.
    // This is zero-overhead: same performance as raw C arrays.
    // Syntax: Eigen::Map<MatrixType>(pointer, rows, cols)
    Eigen::Map<const RowMajorMatrixXf> X(data, n, d);

    // ---- STEP 2: Center the data ----
    Eigen::VectorXf mean = X.colwise().mean();
    Eigen::MatrixXf Xc = X.rowwise() - mean.transpose();

    // ---- STEP 3: Eigendecomposition (auto-select method) ----
    //
    // For d <= 500: use covariance matrix + SelfAdjointEigenSolver (O(d³)).
    //   Faster for small d because covariance matrix fits in cache.
    //
    // For d > 500: use BDCSVD on centered data directly (O(n*d*k)).
    //   Avoids forming the d×d covariance matrix entirely.
    //   Much faster when d >> k (e.g., d=10000, k=2 → ~100x speedup).
    //
    Eigen::MatrixXf eigvecs;
    Eigen::VectorXf eigvals_full;
    float total_var = 0.0f;

    if (d <= 500) {
        // ---- Path A: Covariance + Eigendecomposition (small d) ----
        Eigen::MatrixXf cov = (Xc.adjoint() * Xc) / (float)(n - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(cov);
        eigvecs = solver.eigenvectors();
        eigvals_full = solver.eigenvalues();
        total_var = cov.trace();
    } else {
        // ---- Path B: BDCSVD (large d, avoids d×d covariance) ----
        // BDCSVD computes thin SVD: Xc = U * S * V^T
        // The right singular vectors V are the principal components.
        // Singular values s relate to eigenvalues: eigenvalue = s² / (n-1)
        Eigen::BDCSVD<Eigen::MatrixXf> svd(Xc, Eigen::ComputeThinV);

        // Right singular vectors (columns of V) = principal directions
        eigvecs = svd.matrixV();

        // Eigenvalues from singular values: λ = σ² / (n-1)
        Eigen::VectorXf S = svd.singularValues();
        eigvals_full = S.array().square() / (float)(n - 1);

        // Total variance = sum of all eigenvalues
        total_var = eigvals_full.sum();
    }

    // ---- STEP 4: Extract top-k components ----
    components_.resize(k, d);
    explained_variance_.resize(k);
    explained_variance_ratio_.resize(k);

    // Get eigenvalue indices: SelfAdjointEigenSolver returns ascending order,
    // BDCSVD returns descending order. Handle both.
    bool ascending = (d <= 500);  // SelfAdjointEigenSolver = ascending

    for (size_t i = 0; i < k; ++i) {
        size_t idx = ascending ? (d - 1 - i) : i;  // Pick i-th largest

        float ev = std::max(0.0f, eigvals_full(idx));
        explained_variance_[i] = ev;
        explained_variance_ratio_[i] = total_var > 0 ? ev / total_var : 0.0f;

        // Copy eigenvector (column idx) as row i of components_
        for (size_t j = 0; j < d; ++j)
            components_[i][j] = eigvecs(j, idx);
    }

    // ---- STEP 6: Store mean for later centering ----
    // We need the mean for transforming new data: X_centered = X - mean
    // And for inverse_transform: X_reconstructed = Y @ components + mean
    mean_ = Vector(d);
    for (size_t j = 0; j < d; ++j) mean_[j] = mean(j);

    fitted_ = true;
}

// ============================================================================
// transform() -- REDUCE DIMENSIONALITY OF DATA (Matrix API)
// ============================================================================

Matrix PCA::transform(const Matrix& X) const {
    validate_fitted(fitted_);

    // Feature count must match what we trained on.
    if (X.cols() != n_features_)
        throw std::runtime_error("Feature count mismatch");

    // ---- Map input to Eigen ----
    Eigen::Map<const RowMajorMatrixXf> Xin(X.data(), X.rows(), X.cols());

    // ---- Center using stored mean ----
    // Convert our Vector mean to an Eigen vector.
    Eigen::VectorXf mean_eig(n_features_);
    for (size_t j = 0; j < n_features_; ++j) mean_eig(j) = mean_[j];

    // Subtract mean from each row: X_centered = X - mean
    Eigen::MatrixXf Xc = Xin.rowwise() - mean_eig.transpose();

    // ---- Build components matrix (k × d) in Eigen format ----
    // components_ is stored row-major: components_[i][j] = i-th component, j-th feature.
    size_t k = n_components_;
    Eigen::MatrixXf comps(k, n_features_);
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < n_features_; ++j)
            comps(i, j) = components_[i][j];

    // ---- Project onto components ----
    // Y = X_centered * components^T
    //   = (n × d) * (d × k) = (n × k)
    //
    // comps.adjoint() = transpose = (d × k) matrix.
    // This BLAS-accelerated matrix multiply computes the principal component scores.
    Eigen::MatrixXf Y = Xc * comps.adjoint();

    // ---- Copy to output Matrix ----
    // std::memcpy is the fastest way to copy raw bytes between same-format arrays.
    // Y.size() returns total elements (rows * cols).
    // sizeof(float) = 4 bytes per element.
    Matrix result(Y.rows(), Y.cols());
    std::memcpy(result.data(), Y.data(), Y.size() * sizeof(float));
    return result;
}

// ============================================================================
// fit_transform() -- FIT AND TRANSFORM IN ONE CALL
// ============================================================================

Matrix PCA::fit_transform(const Matrix& X) {
    fit(X);           // Learn the transformation
    return transform(X);  // Apply it to the same data
}

// ============================================================================
// transform_raw() -- TRANSFORM USING RAW POINTERS (for Python bindings)
// ============================================================================

void PCA::transform_raw(const float* data, size_t n_rows, float* out) const {
    validate_fitted(fitted_);

    // Same logic as transform() but avoids Matrix copies.
    // Input data is at `data` pointer, output goes to `out` pointer.
    Eigen::Map<const RowMajorMatrixXf> Xin(data, n_rows, n_features_);

    // Center
    Eigen::VectorXf mean_eig(n_features_);
    for (size_t j = 0; j < n_features_; ++j) mean_eig(j) = mean_[j];
    Eigen::MatrixXf Xc = Xin.rowwise() - mean_eig.transpose();

    // Build components
    size_t k = n_components_;
    Eigen::MatrixXf comps(k, n_features_);
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < n_features_; ++j)
            comps(i, j) = components_[i][j];

    // Project
    RowMajorMatrixXf Y = Xc * comps.adjoint();

    // Copy to output buffer (n_rows * k floats)
    std::memcpy(out, Y.data(), n_rows * k * sizeof(float));
}

// ============================================================================
// fit_transform_raw() -- FIT AND TRANSFORM IN ONE CALL (raw pointers)
// ============================================================================

void PCA::fit_transform_raw(const float* data, size_t n_rows, size_t n_cols, float* out) {
    fit_raw(data, n_rows, n_cols);         // Learn
    transform_raw(data, n_rows, out);       // Apply
}

// ============================================================================
// inverse_transform_raw() -- RECONSTRUCT APPROXIMATE ORIGINAL DATA
//
// Goes from reduced dimension (k) back to original dimension (d).
// X_approx = Y * components + mean
//
// The reconstruction is APPROXIMATE because we threw away (d - k)
// components during the forward transform. The error depends on how
// much variance those discarded components captured.
// ============================================================================

void PCA::inverse_transform_raw(const float* data, size_t n_rows, float* out) const {
    validate_fitted(fitted_);

    // Map input (n_rows × k) to Eigen.
    Eigen::Map<const RowMajorMatrixXf> Yin(data, n_rows, n_components_);

    size_t k = n_components_;
    size_t d = n_features_;

    // Build components matrix (k × d).
    Eigen::MatrixXf comps(k, d);
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < d; ++j)
            comps(i, j) = components_[i][j];

    // Reconstruct: X_approx = Y * components
    // (n × k) * (k × d) = (n × d)
    RowMajorMatrixXf X = Yin * comps;

    // Add back the mean that was subtracted during forward transform.
    // This is essential: without this, all points would be centered at origin.
    Eigen::VectorXf mean_eig(d);
    for (size_t j = 0; j < d; ++j) mean_eig(j) = mean_[j];

    // rowwise() += adds the mean to every row.
    X.rowwise() += mean_eig.transpose();

    // Copy to output buffer (n_rows * d floats).
    std::memcpy(out, X.data(), n_rows * d * sizeof(float));
}

// ============================================================================
// inverse_transform() -- RECONSTRUCT (Matrix API)
// ============================================================================

Matrix PCA::inverse_transform(const Matrix& X) const {
    validate_fitted(fitted_);

    // Input must have the right number of columns (n_components).
    if (X.cols() != n_components_)
        throw std::runtime_error("Component count mismatch");

    Matrix result(X.rows(), n_features_);
    inverse_transform_raw(X.data(), X.rows(), result.data());
    return result;
}

// ============================================================================
// total_explained_variance_ratio() -- SUMMARIZE HOW MUCH INFO WAS RETAINED
//
// Returns: sum of explained_variance_ratio_ for all k components.
// 0.95 = "we kept 95% of the original data's information".
// This is the key number for deciding if k is big enough.
// ============================================================================

float PCA::total_explained_variance_ratio() const {
    validate_fitted(fitted_);
    float total = 0.0f;
    for (size_t i = 0; i < n_components_; ++i) total += explained_variance_ratio_[i];
    return total;
}

} // namespace clustering
