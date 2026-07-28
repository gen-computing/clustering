#pragma once

#include "matrix.h"
#include <string>

namespace clustering {

// ============================================================================
// FeatureStore -- Preprocesses data and caches results to disk.
//
// WHAT IT DOES:
//   Raw data often needs to be transformed before clustering:
//   - Normalize (scale each feature to unit L2 norm)
//   - Standardize (zero mean, unit variance)
//   - MinMax scaling (rescale to [0, 1] range)
//
//   FeatureStore does the transformation AND saves the result to disk.
//   Next time you ask for the same operation on the same-sized data,
//   it loads the cached result INSTEAD of recomputing -- much faster.
//
// CACHE KEY:
//   The cache key is based on matrix dimensions + operation name.
//   Example: data with 10000 rows, 50 columns, operation "normalize"
//            -> key is "10000_50_normalize"
//   This is SIMPLE but NOT collision-proof: two different 10000x50 datasets
//   with the same operation would get the same key. For production use,
//   you'd want a content-based hash (e.g., SHA-256 of the data).
//
// WHY USE IT:
//   - Preprocessing large datasets can be slow (O(n*d))
//   - If you run multiple experiments on the same data, caching saves time
//   - Useful in ML pipelines where preprocessing is repeated
// ============================================================================
class FeatureStore {
public:
    FeatureStore();
    ~FeatureStore();

    // set_cache_path: Where to store cached .bin files.
    // Creates directory if it doesn't exist.
    void set_cache_path(const std::string& path);

    // preprocess: Apply an operation with caching.
    // If the result is already cached, loads from disk (fast).
    // Otherwise, computes the transformation and saves to cache.
    //
    // Parameters:
    //   X         -- input data matrix (n_samples x n_features)
    //   operation -- one of: "normalize", "standardize", "minmax"
    //                (unknown operation names return X unchanged)
    //
    // Returns: The transformed matrix (same dimensions as input).
    Matrix preprocess(const Matrix& X, const std::string& operation);

    // cache_hit: Check if a given key exists in the cache without loading it.
    // Useful for checking before deciding whether to call preprocess.
    bool cache_hit(const std::string& key) const;

private:
    // ----- PRIVATE TRANSFORMATION METHODS -----

    // normalize: Scale each column (feature) independently so its L2 norm = 1.
    //   For each column j: norm = sqrt(sum of squares), then divide all values by norm.
    //   Result: each feature column has unit length.
    Matrix normalize(const Matrix& X);

    // standardize: Make each feature have mean=0 and standard deviation=1.
    //   For each column: compute mean, compute std_dev,
    //   then transform: (value - mean) / std_dev.
    //   This is also called "z-score normalization" or "standard scaling".
    Matrix standardize(const Matrix& X);

    // minmax_scale: Rescale each feature to the range [0, 1].
    //   For each column: find min and max value.
    //   Then transform: (value - min) / (max - min).
    Matrix minmax_scale(const Matrix& X);

    // ----- PRIVATE CACHE METHODS -----

    // compute_key: Generate a cache key from matrix dimensions and operation name.
    // Example: Matrix 100x50 + "normalize" -> key "100_50_normalize"
    std::string compute_key(const Matrix& X, const std::string& operation);

    // load_from_cache: Try to load cached data from disk.
    // Returns true if found and loaded into `result`, false if not cached.
    bool load_from_cache(const std::string& key, Matrix& result) const;

    // save_to_cache: Write transformed data to cache file.
    void save_to_cache(const std::string& key, const Matrix& data);

    std::string cache_path_;  // Directory for cached .bin files
};

} // namespace clustering
