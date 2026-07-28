// ============================================================================
// FeatureStore implementation -- Data preprocessing with disk caching.
//
// Transforms raw data using common preprocessing operations and caches
// the results to disk. On subsequent calls with the same data dimensions
// and operation, loads from cache instead of recomputing.
//
// SUPPORTED OPERATIONS:
//   "normalize":   Scale each feature so its L2 norm = 1.
//                  value = value / sqrt(sum of squares of column)
//                  Result: each feature column has unit length.
//
//   "standardize": Make each feature have mean=0 and std=1 (z-score).
//                  value = (value - mean) / standard_deviation
//                  Result: each feature column has mean=0, variance=1.
//                  This is the most common preprocessing for ML.
//
//   "minmax":      Rescale each feature to [0, 1].
//                  value = (value - min) / (max - min)
//                  Result: all values are between 0 and 1.
//
// CACHE KEY: Based on dimensions + operation name.
//   Example: "10000_50_normalize.bin" for 10k×50 data normalized.
//   LIMITATION: Two different 10k×50 datasets would collide on the same key.
//   For production, use a content hash (SHA-256 or MD5 of data bytes).
// ============================================================================

#include "clustering/feature_store.h"
#include <fstream>       // std::ifstream, std::ofstream
#include <filesystem>    // std::filesystem::create_directories, exists
#include <algorithm>     // std::min, std::max
#include <cmath>         // std::sqrt

namespace clustering {

FeatureStore::FeatureStore() = default;
FeatureStore::~FeatureStore() = default;

// ============================================================================
// set_cache_path() -- CONFIGURE CACHE DIRECTORY
//
// Creates the directory if it doesn't exist (like VersionManager).
// ============================================================================

void FeatureStore::set_cache_path(const std::string& path) {
    cache_path_ = path;
    std::filesystem::create_directories(path);
}

// ============================================================================
// preprocess() -- MAIN ENTRY POINT: TRANSFORM WITH CACHING
//
// 1. Compute a cache key from matrix dimensions and operation name.
// 2. Check if cached result exists on disk. If yes, load and return it.
// 3. If not cached, compute the transformation.
// 4. Save the result to cache for next time.
// 5. Return the transformed matrix.
//
// Unknown operation names return the input unchanged (no transformation).
// ============================================================================

Matrix FeatureStore::preprocess(const Matrix& X, const std::string& operation) {
    // ---- Step 1: Check cache ----
    std::string key = compute_key(X, operation);
    Matrix cached;
    if (load_from_cache(key, cached)) {
        return cached;  // Cache hit! Return immediately -- no computation needed.
    }

    // ---- Step 2: Apply the transformation ----
    Matrix result;

    // Dispatch based on operation name string.
    // Using if-else chain (simple, fast enough for 3 operations).
    // For more operations, a map<string, function> would be cleaner.
    if (operation == "normalize") {
        result = normalize(X);
    } else if (operation == "standardize") {
        result = standardize(X);
    } else if (operation == "minmax") {
        result = minmax_scale(X);
    } else {
        // Unknown operation: return unchanged data.
        // This is graceful degradation -- the caller might have a typo.
        result = X;
    }

    // ---- Step 3: Save to cache for future calls ----
    save_to_cache(key, result);

    return result;
}

// ============================================================================
// cache_hit() -- CHECK IF A KEY EXISTS IN CACHE (NO LOAD)
//
// Just checks file existence -- fast (O(1) filesystem call).
// Use before preprocess() if you want to know whether it'll be cached.
// ============================================================================

bool FeatureStore::cache_hit(const std::string& key) const {
    std::string filename = cache_path_ + "/" + key + ".bin";
    return std::filesystem::exists(filename);
}

// ============================================================================
// normalize() -- L2 NORMALIZATION PER FEATURE COLUMN
//
// For each feature (column):
//   norm_j = sqrt(sum of squares of all values in column j)
//   result[i][j] = X[i][j] / norm_j
//
// After normalization: each column has L2 norm = 1.
// This makes all features comparable in magnitude -- important for algorithms
// that use distance (like KMeans), where a feature with large values would
// dominate the distance calculation.
// ============================================================================

Matrix FeatureStore::normalize(const Matrix& X) {
    size_t n = X.rows();
    size_t d = X.cols();

    Matrix result(n, d);

    // Process each feature independently.
    for (size_t j = 0; j < d; ++j) {
        // ---- Compute L2 norm of column j ----
        // L2 norm = sqrt(x1² + x2² + ... + xn²)
        float norm = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            norm += X[i][j] * X[i][j];
        }
        norm = std::sqrt(norm);

        // ---- Divide each value by the norm ----
        // If norm > 0 (non-zero column): scale to unit norm.
        // If norm == 0 (all zeros): result stays 0 (skip division).
        if (norm > 0) {
            for (size_t i = 0; i < n; ++i) {
                result[i][j] = X[i][j] / norm;
            }
        }
        // If norm == 0, the result column remains all zeros (default from resize).
    }

    return result;
}

// ============================================================================
// standardize() -- Z-SCORE NORMALIZATION (zero mean, unit variance)
//
// For each feature:
//   mean_j = (1/n) * sum_i X[i][j]
//   std_j  = sqrt((1/n) * sum_i (X[i][j] - mean_j)²)
//   result[i][j] = (X[i][j] - mean_j) / std_j
//
// After standardization:
//   - Each column has mean ≈ 0
//   - Each column has standard deviation = 1
//   - About 68% of values fall in [-1, 1] (assuming normal distribution)
//
// This is the most commonly used preprocessing in ML because many algorithms
// (including neural networks, SVM, and distance-based methods) work better
// when features are on the same scale.
// ============================================================================

Matrix FeatureStore::standardize(const Matrix& X) {
    size_t n = X.rows();
    size_t d = X.cols();

    Matrix result(n, d);

    // Process each feature independently.
    for (size_t j = 0; j < d; ++j) {
        // ---- Step 1: Compute mean of column j ----
        // mean = (1/n) * sum of all values in column j
        float mean = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            mean += X[i][j];
        }
        mean /= n;

        // ---- Step 2: Compute standard deviation of column j ----
        // std = sqrt((1/n) * sum of (value - mean)²)
        float std_dev = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float diff = X[i][j] - mean;
            std_dev += diff * diff;
        }
        std_dev = std::sqrt(std_dev / n);

        // ---- Step 3: Standardize (z-score) ----
        // value = (original - mean) / std_dev
        // If std_dev is 0 (all values identical), result stays 0.
        if (std_dev > 0) {
            for (size_t i = 0; i < n; ++i) {
                result[i][j] = (X[i][j] - mean) / std_dev;
            }
        }
    }

    return result;
}

// ============================================================================
// minmax_scale() -- RESCALE TO [0, 1]
//
// For each feature:
//   min_j, max_j = minimum and maximum value in column j
//   result[i][j] = (X[i][j] - min_j) / (max_j - min_j)
//
// After scaling: all values are between 0 and 1 (inclusive).
//
// PRO: Preserves exactly the original distribution shape.
// CON: Sensitive to outliers (one extreme value stretches the scale).
// ============================================================================

Matrix FeatureStore::minmax_scale(const Matrix& X) {
    size_t n = X.rows();
    size_t d = X.cols();

    Matrix result(n, d);

    // Process each feature independently.
    for (size_t j = 0; j < d; ++j) {
        // ---- Find min and max of column j ----
        // Initialize with first value, then scan the rest.
        float min_val = X[0][j];
        float max_val = X[0][j];
        for (size_t i = 1; i < n; ++i) {
            min_val = std::min(min_val, X[i][j]);
            max_val = std::max(max_val, X[i][j]);
        }

        // ---- Scale to [0, 1] ----
        // range = max - min. If range is 0, all values are identical -> result = 0.
        float range = max_val - min_val;
        if (range > 0) {
            for (size_t i = 0; i < n; ++i) {
                // Linear transformation: (x - min) / range maps [min,max] to [0,1].
                result[i][j] = (X[i][j] - min_val) / range;
            }
        }
        // If range == 0, result column remains zeros.
    }

    return result;
}

// ============================================================================
// compute_key() -- GENERATE CACHE KEY FROM DIMENSIONS AND OPERATION
//
// Format: "{rows}_{cols}_{operation}"
// Example: "10000_50_normalize"
//
// CAVEAT: This key only considers dimensions, not content. Two different
// matrices of the same size will collide. For production use, compute a
// hash of the actual data bytes (e.g., using std::hash or SHA-256).
// ============================================================================

std::string FeatureStore::compute_key(const Matrix& X, const std::string& operation) {
    // Simple key: rows + columns + operation name.
    // Concatenates using underscores as separators.
    return std::to_string(X.rows()) + "_" + std::to_string(X.cols()) + "_" + operation;
}

// ============================================================================
// load_from_cache() -- READ CACHED MATRIX FROM DISK
//
// Cache file format (same binary layout as data, with metadata header):
//   [rows: size_t] [cols: size_t]                -- dimensions (16 bytes)
//   [data: rows*cols floats]                      -- the actual values
//
// Returns: true if cached file found and loaded successfully.
//          false if file doesn't exist or read failed.
// ============================================================================

bool FeatureStore::load_from_cache(const std::string& key, Matrix& result) const {
    std::string filename = cache_path_ + "/" + key + ".bin";

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;  // Cache miss: file doesn't exist
    }

    // ---- Read dimensions ----
    size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // ---- Allocate and read data ----
    result.resize(rows, cols);
    file.read(reinterpret_cast<char*>(result.data()), rows * cols * sizeof(float));

    file.close();
    return true;
}

// ============================================================================
// save_to_cache() -- WRITE MATRIX TO CACHE FILE
//
// Same binary format as load_from_cache (symmetric).
// File is overwritten if it already exists for this key.
// ============================================================================

void FeatureStore::save_to_cache(const std::string& key, const Matrix& data) {
    std::string filename = cache_path_ + "/" + key + ".bin";

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return;  // Can't write (disk full, permission denied) -- silently skip
    }

    // ---- Write dimensions ----
    size_t rows = data.rows();
    size_t cols = data.cols();

    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // ---- Write data ----
    file.write(reinterpret_cast<const char*>(data.data()), rows * cols * sizeof(float));

    file.close();
}

} // namespace clustering
