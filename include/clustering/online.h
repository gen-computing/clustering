#pragma once

#include "kmeans.h"    // OnlineKMeans inherits from KMeans (is-a relationship)
#include "drift.h"     // For DriftDetector (auto-detects when data patterns change)
#include <deque>       // std::deque -- double-ended queue for sliding window

namespace clustering {

// ============================================================================
// OnlineConfig -- Settings for streaming/online clustering.
//
// Regular KMeans needs ALL data at once. OnlineKMeans processes data point by
// point or in small batches as it arrives -- like a stream. This is essential
// for real-time applications where data is continuously generated.
// ============================================================================
struct OnlineConfig {
    size_t k = 8;                       // Number of clusters (same as regular KMeans)

    size_t window_size = 1000;          // SLIDING WINDOW: Only remember the most recent
                                        // N data points. Old points are forgotten.
                                        // This makes the model adapt to changing data.
                                        // Set to 0 for NO window (remembers everything).

    float forgetting_factor = 0.99f;   // FORGETTING FACTOR: How quickly old data
                                        // loses influence. Range: [0.0, 1.0].
                                        // 1.0 = never forget (all points equal weight)
                                        // 0.9 = recent points matter MUCH more
                                        // 0.0 = completely forget everything (useless)

    bool auto_retrain = true;           // AUTO-RETRAIN: If drift is detected, should
                                        // the model automatically do a full retrain
                                        // from the sliding window data?

    float drift_threshold = 0.1f;      // DRIFT SENSITIVITY: How much silhouette score
                                        // degradation before triggering retrain.
                                        // Lower = more sensitive to drift.
                                        // 0.1 = scores must drop by 0.1 to trigger.

    size_t retrain_interval = 1000;     // Check for drift every N points.
                                        // Checking every point would be too slow.

    // Iteration callback: called after each partial_fit batch.
    // Same signature as KMeansConfig::iter_callback.
    std::function<bool(size_t, const Matrix&, const Vector&)> iter_callback;
};

// ============================================================================
// OnlineKMeans -- Streaming/online clustering on data that arrives over time.
//
// DIFFERENCE FROM REGULAR KMEANS:
//   Regular KMeans: needs ALL data at once, computes everything from scratch.
//   OnlineKMeans:   processes data POINT BY POINT as it arrives. Updates
//                   centroids incrementally. Adapts to changing patterns.
//
// REAL WORLD USE CASES:
//   - Sensor data that keeps arriving (temperature, pressure, vibration)
//   - User behavior that changes over time (shopping patterns shift with seasons)
//   - Stock market data (market regimes change)
//
// KEY FEATURES:
//   1. SLIDING WINDOW: Only recent N points matter. Old data is forgotten.
//   2. FORGETTING FACTOR: Gradual decay of old influence.
//   3. AUTO-DETECT DRIFT: Notices when data patterns change significantly.
//   4. AUTO-RETRAIN: If drift detected, recomputes from scratch on window data.
//
// HOW IT INHERITS: OnlineKMeans IS-A KMeans. It reuses fit(), predict(),
// centroids(), labels(), inertia() from the parent class but REPLACES
// partial_fit() with its own streaming logic.
// ============================================================================
class OnlineKMeans : public KMeans {
public:
    // ----- CONSTRUCTORS -----

    // Simple: just specify k. Uses default OnlineConfig values.
    explicit OnlineKMeans(size_t k = 8);

    // Full: pass a complete OnlineConfig to customize all behavior.
    explicit OnlineKMeans(const OnlineConfig& config);

    // Destructor: `override` keyword tells compiler "I'm replacing the parent's
    // destructor". If parent's destructor signature changes, compiler will error
    // here instead of silently creating a new, unrelated destructor.
    ~OnlineKMeans() override;

    // ----- CORE METHODS (overriding KMeans) -----

    // partial_fit: Process a BATCH of new data points.
    // On first call (model not yet fitted): runs full fit() to initialize.
    // On subsequent calls: feeds each point through partial_fit_point().
    // This is `override` -- it REPLACES KMeans::partial_fit.
    // Parameter X: Matrix where each row = one new data point.
    // Throws: runtime_error if X is empty, has 0 features, or dimension mismatch.
    void partial_fit(const Matrix& X) override;

    // partial_fit_point: Process a SINGLE new data point.
    // This is the atomic operation -- the core of online learning.
    // Goes through these steps:
    //   1. If window_size > 0: add point to sliding window, remove oldest if full.
    //   2. Find which centroid is nearest to this point.
    //   3. Apply forgetting factor to that cluster's count.
    //   4. Move centroid slightly toward this point (learning rate = 1/count).
    //   5. If auto_retrain and interval reached: check for drift.
    //
    // Parameters:
    //   point -- pointer to float array (the feature values of one data point)
    //   dim   -- number of features (must match centroids().cols())
    // Throws: runtime_error if model not fitted, or dimension mismatch.
    void partial_fit_point(const float* point, size_t dim);

    // ----- CONFIGURATION SETTERS -----

    // Change sliding window size at runtime.
    void set_window_size(size_t size);

    // Change forgetting factor. Must be in range [0.0, 1.0].
    // Throws: runtime_error if factor is outside [0, 1].
    void set_forgetting_factor(float factor);

    // Enable or disable automatic retrain on drift detection.
    void set_auto_retrain(bool enable);

    // ----- STATE ACCESSORS -----

    // How many points have been processed total (since creation)?
    size_t points_seen() const { return points_seen_; }

    // How many points are currently assigned to each cluster?
    // Index 0 = cluster 0's count, etc.
    // These counts are affected by the forgetting factor (older counts decay).
    const std::vector<float>& cluster_counts() const { return cluster_counts_; }

private:
    // ----- INTERNAL METHODS -----

    // update_with_window: Handle the sliding window logic.
    // Adds the point to the window deque. If window is full, removes the oldest.
    // Then updates centroids same as the direct path.
    void update_with_window(const float* point, size_t dim);

    // check_drift_and_retrain: Sample the window data and check if drift occurred.
    // If DriftDetector reports drift, calls retrain().
    // Only runs if window has at least 10 points (need enough data to assess).
    void check_drift_and_retrain();

    // retrain: Do a full KMeans fit() on the current sliding window contents.
    // This resets all centroids based on the most recent data only.
    void retrain();

    // compute_window_inertia: Calculate inertia only within the sliding window.
    // Used for monitoring cluster quality on recent data.
    float compute_window_inertia() const;

    // ----- MEMBER VARIABLES -----

    OnlineConfig online_config_;
    size_t points_seen_;
    std::vector<float> cluster_counts_;

    // Sliding window: flat ring buffer (no per-point heap allocation).
    // Stores points contiguously: window_data_[write_pos * dim .. write_pos * dim + dim - 1]
    // write_pos_ cycles 0..window_size-1. window_count_ tracks how many valid entries.
    std::vector<float> window_data_;     // Flat buffer: window_size * dim floats
    size_t write_pos_ = 0;              // Write position in ring buffer
    size_t window_count_ = 0;           // Number of valid entries in window
    size_t window_dim_ = 0;             // Feature dimension (set on first use)

    DriftDetector drift_detector_;
    size_t since_last_retrain_;
};

} // namespace clustering
