// ============================================================================
// OnlineKMeans implementation -- Streaming clustering with window & forgetting.
//
// OnlineKMeans extends KMeans for STREAMING data. Instead of needing all data
// at once (batch mode), it processes data point-by-point or in small batches
// as it arrives. This is essential for real-time applications.
//
// Key differences from regular KMeans:
//   1. SLIDING WINDOW: Only remembers the most recent N points. Old data
//      is discarded (pop_front from the deque). This lets the model adapt
//      to changing patterns.
//   2. FORGETTING FACTOR: Each cluster's "count" is multiplied by the factor
//      before adding the new point. A factor of 0.99 means old points have
//      99% of their previous weight after each update.
//   3. AUTO-DRIFT DETECTION: Periodically checks for concept drift using
//      DriftDetector. If detected, retrains from scratch on window data.
//   4. INCREMENTAL UPDATE: Each point slightly moves its nearest centroid.
//      The learning rate is 1/count -- first points move centroid a lot,
//      later points move it very little.
// ============================================================================

#include "clustering/online.h"
#include "clustering/distance.h"
#include "clustering/validate.h"
#include <algorithm>   // std::min
#include <cmath>       // std::sqrt
#include <stdexcept>   // std::runtime_error

namespace clustering {

// ============================================================================
// CONSTRUCTORS
// ============================================================================

// Simple constructor: just specify k. Uses default OnlineConfig values.
// Calls KMeans(k) to initialize the parent class.
// cluster_counts_ starts as a vector of k zeros.
OnlineKMeans::OnlineKMeans(size_t k)
    : KMeans(k), points_seen_(0), cluster_counts_(k, 0), since_last_retrain_(0) {
    // Configure the drift detector with our default threshold.
    drift_detector_.set_threshold(online_config_.drift_threshold);
}

// Config-based constructor: copy all settings from OnlineConfig.
// Also passes config.k to the KMeans parent constructor.
OnlineKMeans::OnlineKMeans(const OnlineConfig& config)
    : KMeans(config.k), online_config_(config), points_seen_(0),
      cluster_counts_(config.k, 0), since_last_retrain_(0) {
    // Configure drift detector from the user's settings.
    drift_detector_.set_threshold(config.drift_threshold);
    drift_detector_.set_window_size(config.window_size);

    // Pass iteration callback to base class for real-time visualization
    if (config.iter_callback) {
        config_.iter_callback = config.iter_callback;
    }
}

OnlineKMeans::~OnlineKMeans() = default;

// ============================================================================
// partial_fit() -- PROCESS A BATCH OF NEW DATA
// ============================================================================

void OnlineKMeans::partial_fit(const Matrix& X) {
    validate_matrix(X);

    // First call: not yet trained. Do full fit() to initialize centroids.
    // After fitting, record which cluster each initial point went to.
    if (!fitted_) {
        fit(X);                     // Use parent class fit() to learn initial centroids
        points_seen_ = X.rows();     // Track total points seen
        for (size_t i = 0; i < X.rows(); ++i) {
            size_t c = static_cast<size_t>(labels()[i]);
            cluster_counts_[c]++;    // Increment per-cluster count
        }
        return;
    }

    // Already trained: feed each point one-by-one through the streaming logic.
    // partial_fit_point() handles the sliding window, forgetting factor,
    // centroid update, and drift detection for a single point.
    for (size_t i = 0; i < X.rows(); ++i) {
        partial_fit_point(X[i], X.cols());
    }
}

// ============================================================================
// partial_fit_point() -- PROCESS A SINGLE NEW DATA POINT
//
// This is the atomic operation of online learning. For each point:
// 1. If sliding window is enabled: add point to window, pop oldest if full.
// 2. Find nearest centroid using AVX2 distance.
// 3. Apply forgetting factor to that cluster's count.
// 4. Move centroid toward this point with learning rate = 1/count.
// 5. If auto_retrain enabled and interval reached: check for drift.
// ============================================================================

void OnlineKMeans::partial_fit_point(const float* point, size_t dim) {
    validate_fitted(fitted_);
    validate_dimensions(centroids().cols(), dim);

    // ---- PATH 1: Sliding window enabled ----
    // Add point to the window buffer. If window is full, remove oldest point.
    // This creates a "forgetting by truncation" effect -- only recent data matters.
    if (online_config_.window_size > 0) {
        update_with_window(point, dim);
    } else {
        // ---- PATH 2: No sliding window, direct update ----
        // Find which existing centroid is nearest to this new point.
        // This is an O(k * d) operation: k distance calculations.
        size_t nearest = 0;
        float min_dist = l2_distance_avx2(point, centroids()[0], dim);

        for (size_t c = 1; c < centroids().rows(); ++c) {
            float dist = l2_distance_avx2(point, centroids()[c], dim);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = c;  // Found a closer centroid
            }
        }

        // Apply FORGETTING FACTOR: decay the count before adding the new point.
        // Old count * factor means old points lose influence over time.
        // Example with factor=0.99: after 100 updates, the very first point
        // has weight 0.99^100 ≈ 0.37 (37% of original influence).
        cluster_counts_[nearest] *= online_config_.forgetting_factor;

        // Add this new point (counts as 1.0, not yet decayed).
        cluster_counts_[nearest] += 1.0f;

        // LEARNING RATE: 1 / count means:
        //   - First point: lr = 1/1 = 1.0 (centroids JUMP to point)
        //   - After 100 points: lr = 1/100 = 0.01 (centroid moves 1% toward point)
        // This makes centroids stabilize quickly and then adjust slowly.
        float lr = 1.0f / cluster_counts_[nearest];

        // Move the centroid toward the new point:
        // new_pos = old_pos + lr * (point - old_pos)
        // This is gradient descent in the original data space.
        for (size_t d = 0; d < dim; ++d) {
            centroids_mut()[nearest][d] += lr * (point[d] - centroids()[nearest][d]);
        }
    }

    // Increment total points counter.
    points_seen_++;

    // ---- DRIFT DETECTION ----
    // Only check periodically (every retrain_interval points), not every point.
    // Checking every point would be far too expensive (drift check is O(n²)).
    if (online_config_.auto_retrain) {
        since_last_retrain_++;  // Count points since last check
        if (since_last_retrain_ >= online_config_.retrain_interval) {
            check_drift_and_retrain();  // Sample window, compute metrics, maybe retrain
            since_last_retrain_ = 0;    // Reset counter
        }
    }
}

// ============================================================================
// update_with_window() -- SLIDING WINDOW MANAGEMENT
//
// Maintains a window (FIFO buffer) of the most recent N points.
// When a new point arrives:
//   1. Push it to the back of the deque.
//   2. If window size exceeded, remove from the front (oldest point).
//   3. Update centroids (same as the no-window path).
// ============================================================================

void OnlineKMeans::update_with_window(const float* point, size_t dim) {
    // Lazy-init ring buffer on first call
    if (window_data_.empty()) {
        window_dim_ = dim;
        window_data_.resize(online_config_.window_size * dim);
    }

    // Write point into ring buffer at current position (no heap allocation)
    float* dst = &window_data_[write_pos_ * window_dim_];
    for (size_t d = 0; d < dim; ++d) dst[d] = point[d];

    // Advance write pointer (circular)
    write_pos_ = (write_pos_ + 1) % online_config_.window_size;
    if (window_count_ < online_config_.window_size) window_count_++;

    // ---- Find nearest centroid ----
    size_t nearest = 0;
    float min_dist = l2_distance_avx2(point, centroids()[0], dim);

    for (size_t c = 1; c < centroids().rows(); ++c) {
        float dist = l2_distance_avx2(point, centroids()[c], dim);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = c;
        }
    }

    // ---- Apply forgetting factor + learning rate update ----
    cluster_counts_[nearest] *= online_config_.forgetting_factor;
    cluster_counts_[nearest] += 1.0f;
    float lr = 1.0f / cluster_counts_[nearest];

    for (size_t d = 0; d < dim; ++d) {
        centroids_mut()[nearest][d] += lr * (point[d] - centroids()[nearest][d]);
    }
}

// ============================================================================
// check_drift_and_retrain() -- PERIODIC DRIFT ASSESSMENT
//
// Called every retrain_interval points (if auto_retrain is enabled).
// Samples the sliding window, computes clustering quality metrics, and
// triggers a full retrain if drift is detected.
// ============================================================================

void OnlineKMeans::check_drift_and_retrain() {
    if (window_count_ < 10) return;

    size_t n = std::min(window_count_, static_cast<size_t>(100));
    size_t d = centroids().cols();
    Matrix X(n, d);

    // Evenly spaced sampling from ring buffer
    size_t step = window_count_ / n;
    for (size_t i = 0; i < n; ++i) {
        size_t src_pos = (write_pos_ + i * step) % online_config_.window_size;
        const float* pt = &window_data_[src_pos * window_dim_];
        for (size_t j = 0; j < d; ++j) {
            X[i][j] = pt[j];
        }
    }

    Vector labels = predict(X);
    DriftMetrics metrics = drift_detector_.check(X, labels, centroids());

    if (metrics.drift_detected) {
        retrain();
    }
}

// ============================================================================
// retrain() -- FULL RETRAIN ON SLIDING WINDOW DATA
//
// Converts the sliding window deque into a Matrix, then calls fit()
// to learn completely new centroids from the current window contents.
// This is the "reset button" when the data pattern has changed.
// ============================================================================

void OnlineKMeans::retrain() {
    if (window_count_ < config_.k) return;

    size_t n = window_count_;
    size_t d = centroids().cols();
    Matrix X(n, d);

    // Copy from ring buffer into Matrix
    // Ring buffer wraps: entries 0..write_pos_-1 are newest, write_pos_..end are older
    // But for retrain we just need the data in order, so iterate linearly
    for (size_t i = 0; i < n; ++i) {
        size_t src_pos = (write_pos_ + i) % online_config_.window_size;
        const float* pt = &window_data_[src_pos * window_dim_];
        for (size_t j = 0; j < d; ++j) {
            X[i][j] = pt[j];
        }
    }

    fit(X);
}

// ============================================================================
// compute_window_inertia() -- INERTIA WITHIN SLIDING WINDOW ONLY
//
// Measures clustering quality on recent data only (in the window).
// This is more informative than global inertia when the data distribution
// has shifted over time.
// ============================================================================

float OnlineKMeans::compute_window_inertia() const {
    if (window_count_ == 0) return 0.0f;

    float inertia = 0.0f;
    for (size_t i = 0; i < window_count_; ++i) {
        size_t src_pos = (write_pos_ + i) % online_config_.window_size;
        const float* pt = &window_data_[src_pos * window_dim_];
        size_t nearest = 0;
        float min_dist = l2_distance_avx2(pt, centroids()[0], window_dim_);

        for (size_t c = 1; c < centroids().rows(); ++c) {
            float dist = l2_distance_avx2(pt, centroids()[c], window_dim_);
            if (dist < min_dist) { min_dist = dist; nearest = c; }
        }
        inertia += min_dist;
    }

    return inertia;
}

// ============================================================================
// CONFIGURATION SETTERS
// ============================================================================

void OnlineKMeans::set_window_size(size_t size) {
    online_config_.window_size = size;
}

void OnlineKMeans::set_forgetting_factor(float factor) {
    validate_forgetting_factor(factor);
    online_config_.forgetting_factor = factor;
}

void OnlineKMeans::set_auto_retrain(bool enable) {
    online_config_.auto_retrain = enable;
}

} // namespace clustering
