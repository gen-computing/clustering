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
}

OnlineKMeans::~OnlineKMeans() = default;

// ============================================================================
// partial_fit() -- PROCESS A BATCH OF NEW DATA
// ============================================================================

void OnlineKMeans::partial_fit(const Matrix& X) {
    // Validate input: same checks as KMeans.
    if (X.rows() == 0) throw std::runtime_error("Empty input matrix");
    if (X.cols() == 0) throw std::runtime_error("No features");

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
    // Guard: can't update a model that hasn't been initialized.
    if (!fitted_) throw std::runtime_error("Model not fitted");

    // Guard: the incoming point must have the same number of features as the centroids.
    // Otherwise we'd be comparing points in different spaces (meaningless).
    if (dim != centroids().cols()) throw std::runtime_error("Dimension mismatch");

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
    // Copy the raw float array into a std::vector<float>.
    // std::vector<float>(pointer, pointer + size) copies the range.
    // We need to copy because the original pointer may be temporary.
    std::vector<float> pt(point, point + dim);

    // Add to the back of the sliding window.
    window_.push_back(pt);

    // If window is too large, remove the OLDEST point from the front.
    // std::deque::pop_front() is O(1) -- constant time.
    while (window_.size() > online_config_.window_size) {
        window_.pop_front();
    }

    // ---- Find nearest centroid (same as the no-window path) ----
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
    // Need enough data in the window to make a meaningful assessment.
    // With fewer than 10 points, the metrics are noise.
    if (window_.size() < 10) return;

    // Sample up to 100 points from the window for the drift check.
    // Using all points might be O(n²) and slow; 100 is enough for a
    // statistical check of cluster quality.
    size_t n = std::min(window_.size(), static_cast<size_t>(100));
    size_t d = centroids().cols();

    Matrix X(n, d);  // Sampled data matrix for the drift detector

    // Evenly spaced sampling: take every (window_size / n)-th point.
    // This gives a representative sample without clustering bias.
    size_t step = window_.size() / n;
    for (size_t i = 0; i < n; ++i) {
        size_t idx = i * step;
        for (size_t j = 0; j < d; ++j) {
            X[i][j] = window_[idx][j];
        }
    }

    // Predict labels using the CURRENT centroids.
    Vector labels = predict(X);

    // Compute drift metrics (silhouette, Davies-Bouldin, Calinski-Harabasz, stability).
    DriftMetrics metrics = drift_detector_.check(X, labels, centroids());

    // If drift is detected (silhouette score degraded beyond threshold),
    // do a full retrain from the sliding window data.
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
    // Need at least as many points as clusters. Can't train k=5 with 3 points.
    if (window_.size() < config_.k) return;

    // Convert the window (deque<vector<float>>) into a Matrix.
    size_t n = window_.size();
    size_t d = centroids().cols();
    Matrix X(n, d);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            X[i][j] = window_[i][j];
        }
    }

    // Full KMeans fit on the window data.
    // This will recompute centroids from scratch, potentially finding
    // a completely different cluster structure if the data has drifted.
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
    if (window_.empty()) return 0.0f;

    float inertia = 0.0f;
    // For each point in the window...
    for (const auto& pt : window_) {
        // Find nearest centroid
        size_t nearest = 0;
        float min_dist = l2_distance_avx2(pt.data(), centroids()[0], pt.size());

        for (size_t c = 1; c < centroids().rows(); ++c) {
            float dist = l2_distance_avx2(pt.data(), centroids()[c], pt.size());
            if (dist < min_dist) {
                min_dist = dist;
                nearest = c;
            }
        }

        // Sum the distance (not squared) to nearest centroid.
        // This is a simplified inertia -- usually we'd sum squared distances.
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
    // Validate: factor must be between 0.0 and 1.0 (inclusive).
    // 0.0 = forget everything immediately (useless).
    // 1.0 = never forget (all points equal weight).
    if (factor < 0.0f || factor > 1.0f)
        throw std::runtime_error("Forgetting factor must be in [0, 1]");
    online_config_.forgetting_factor = factor;
}

void OnlineKMeans::set_auto_retrain(bool enable) {
    online_config_.auto_retrain = enable;
}

} // namespace clustering
