// ============================================================================
// DriftDetector implementation -- Concept drift detection for clustering.
//
// Concept drift: when the statistical properties of data change over time.
// Example: user behavior shifts with seasons, sensor values change as equipment
// wears, market regimes change. A model trained on old data becomes stale.
//
// DriftDetector monitors clustering quality using 4 different metrics and
// detects degradation trends over time.
//
// THE 4 METRICS:
//   1. SILHOUETTE SCORE: How well-separated are clusters? Range [-1, 1].
//      Score(i) = (b(i) - a(i)) / max(a(i), b(i))
//      a(i) = avg distance to own cluster, b(i) = avg distance to nearest other cluster.
//
//   2. DAVIES-BOULDIN INDEX: Average similarity between each cluster and its
//      most similar neighbor. LOWER = better clustering. Range [0, ∞).
//      DB = (1/k) * sum_i max_{j≠i} {(scatter_i + scatter_j) / distance(ci, cj)}
//
//   3. CALINSKI-HARABASZ SCORE: Ratio of between-cluster to within-cluster
//      dispersion. HIGHER = better. Range [0, ∞).
//      CH = (between / (k-1)) / (within / (n-k))
//
//   4. CLUSTER STABILITY: How balanced are cluster sizes? Range [0, 1].
//      Measured as normalized entropy of cluster size distribution.
//      1.0 = perfectly equal sizes, 0.0 = one cluster has everything.
//
// DRIFT DETECTION LOGIC:
//   - Maintains a sliding window of past DriftMetrics.
//   - Splits history into older half and recent half.
//   - If recent average silhouette is significantly lower than older average
//     (by more than threshold_), flags drift_detected = true.
// ============================================================================

#include "clustering/drift.h"
#include <cmath>       // std::sqrt, std::log2, std::min, std::max
#include <algorithm>   // std::min, std::max
#include <numeric>     // (used for future accumulation refactor)

namespace clustering {

DriftDetector::DriftDetector()
    : threshold_(0.1f), window_size_(10), drifting_(false) {}

DriftDetector::~DriftDetector() = default;

void DriftDetector::set_threshold(float threshold) {
    threshold_ = threshold;
}

void DriftDetector::set_window_size(size_t window_size) {
    window_size_ = window_size;
}

// ============================================================================
// check() -- COMPUTE ALL METRICS AND DETECT DRIFT
//
// This is the main entry point. Call once per evaluation cycle (not per point).
// Returns a DriftMetrics struct with all scores and the drift_detected flag.
//
// Steps:
//   1. Compute all 4 quality metrics.
//   2. Add to history (maintain sliding window size).
//   3. Compare recent vs older silhouette scores.
//   4. If degradation exceeds threshold, flag drift.
// ============================================================================

DriftMetrics DriftDetector::check(const Matrix& X, const Vector& labels, const Matrix& centroids) {
    DriftMetrics metrics;

    // Compute the 4 quality metrics. Each is O(n²) or O(n*k*d) computation.
    metrics.silhouette_score = compute_silhouette(X, labels, centroids);
    metrics.davies_bouldin_index = compute_davies_bouldin(X, labels, centroids);
    metrics.calinski_harabasz_score = compute_calinski_harabasz(X, labels, centroids);
    metrics.cluster_stability = compute_cluster_stability(labels, centroids.rows());

    // ---- Maintain sliding window of history ----
    history_.push_back(metrics);
    if (history_.size() > window_size_) {
        history_.erase(history_.begin());  // Remove oldest entry
    }

    // ---- Compute drift by comparing silhouette trends ----
    // We need at least 3 historical data points to do a meaningful split
    // into "older" and "recent" halves.
    if (history_.size() >= 3) {
        float recent_avg = 0.0f;
        float older_avg = 0.0f;
        size_t mid = history_.size() / 2;  // Split point: first half vs second half

        // Average silhouette of recent half (higher indices = more recent).
        for (size_t i = mid; i < history_.size(); ++i) {
            recent_avg += history_[i].silhouette_score;
        }

        // Average silhouette of older half (lower indices = older).
        for (size_t i = 0; i < mid; ++i) {
            older_avg += history_[i].silhouette_score;
        }

        recent_avg /= (history_.size() - mid);
        older_avg /= mid;

        // Drift if older average MINUS recent average > threshold.
        // Meaning: the silhouette score has DROPPED by more than threshold.
        // A drop in silhouette = clusters are less well-separated = data changed.
        metrics.drift_detected = (older_avg - recent_avg) > threshold_;
    } else {
        metrics.drift_detected = false;  // Not enough history yet
    }

    drifting_ = metrics.drift_detected;
    return metrics;
}

// ============================================================================
// compute_silhouette() -- HOW WELL-SEPARATED ARE THE CLUSTERS?
//
// For each point, compute:
//   a(i): average distance to all OTHER points in the SAME cluster
//   b(i): minimum average distance to points in a DIFFERENT cluster
//   s(i) = (b(i) - a(i)) / max(a(i), b(i))
//
// s(i) = 1  means point is much closer to own cluster than others (perfect).
// s(i) = 0  means point is on the boundary between clusters.
// s(i) < 0  means point might be in wrong cluster.
//
// Final score = average of s(i) over all points.
//
// COMPLEXITY: O(n² * d) -- for each point, compare to ALL other points.
// This is the slowest metric and the bottleneck for large datasets.
// ============================================================================

float DriftDetector::compute_silhouette(const Matrix& X, const Vector& labels, const Matrix& centroids) {
    size_t n = X.rows();
    size_t k = centroids.rows();

    // Degenerate cases: no points or only one cluster.
    if (n == 0 || k <= 1) return 0.0f;

    float total_silhouette = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        size_t cluster_i = static_cast<size_t>(labels[i]);

        // ---- Compute a(i): average distance within same cluster ----
        float a_i = 0.0f;
        size_t count_same = 0;
        for (size_t j = 0; j < n; ++j) {
            // Only consider OTHER points in the SAME cluster.
            if (i != j && static_cast<size_t>(labels[j]) == cluster_i) {
                // Compute Euclidean distance between point i and point j.
                float dist = 0.0f;
                for (size_t d = 0; d < X.cols(); ++d) {
                    float diff = X[i][d] - X[j][d];
                    dist += diff * diff;
                }
                a_i += std::sqrt(dist);
                count_same++;
            }
        }
        // a(i) = average distance to same-cluster points.
        a_i = count_same > 0 ? a_i / count_same : 0.0f;

        // ---- Compute b(i): minimum average distance to other clusters ----
        // For each OTHER cluster, compute average distance to all points in it.
        // Then take the MINIMUM of those averages (nearest other cluster).
        float b_i = std::numeric_limits<float>::max();

        for (size_t c = 0; c < k; ++c) {
            if (c == cluster_i) continue;  // Skip own cluster

            float avg_dist = 0.0f;
            size_t count = 0;
            for (size_t j = 0; j < n; ++j) {
                if (static_cast<size_t>(labels[j]) == c) {
                    float dist = 0.0f;
                    for (size_t d = 0; d < X.cols(); ++d) {
                        float diff = X[i][d] - X[j][d];
                        dist += diff * diff;
                    }
                    avg_dist += std::sqrt(dist);
                    count++;
                }
            }
            if (count > 0) {
                avg_dist /= count;
                b_i = std::min(b_i, avg_dist);  // Nearest other cluster
            }
        }

        // ---- Compute silhouette coefficient for point i ----
        float s_i = 0.0f;
        if (std::max(a_i, b_i) > 0) {
            // b_i > a_i: point is closer to own cluster -> positive score.
            // b_i < a_i: point is closer to other cluster -> negative score.
            s_i = (b_i - a_i) / std::max(a_i, b_i);
        }
        total_silhouette += s_i;
    }

    return total_silhouette / n;  // Average over all points
}

// ============================================================================
// compute_davies_bouldin() -- AVERAGE WORST-CASE CLUSTER SIMILARITY
//
// For each cluster i:
//   R(i,j) = (scatter_i + scatter_j) / distance(centroid_i, centroid_j)
//   D_i = max_{j≠i} R(i,j)    (worst similarity to any other cluster)
//
// Davies-Bouldin index = (1/k) * sum_i D_i
//
// Lower = better. 0 means all clusters are infinitely far apart.
//
// Scatter = average distance from points in cluster to their centroid.
// This measures how "tight" each cluster is.
// ============================================================================

float DriftDetector::compute_davies_bouldin(const Matrix& X, const Vector& labels, const Matrix& centroids) {
    size_t k = centroids.rows();
    size_t n = X.rows();

    if (k <= 1) return 0.0f;  // Only one cluster: Davies-Bouldin is undefined

    // ---- Step 1: Count points per cluster ----
    std::vector<size_t> sizes(k, 0);
    for (size_t i = 0; i < n; ++i) {
        sizes[static_cast<size_t>(labels[i])]++;
    }

    // ---- Step 2: Compute scatter (within-cluster dispersion) ----
    // scatter[c] = average distance from points in cluster c to centroid c.
    std::vector<float> scatter(k, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        size_t c = static_cast<size_t>(labels[i]);
        float dist = 0.0f;
        for (size_t d = 0; d < X.cols(); ++d) {
            float diff = X[i][d] - centroids[c][d];
            dist += diff * diff;
        }
        scatter[c] += std::sqrt(dist);  // Sum distances
    }
    // Convert sum to average.
    for (size_t c = 0; c < k; ++c) {
        if (sizes[c] > 0) {
            scatter[c] /= sizes[c];
        }
    }

    // ---- Step 3: Compute Davies-Bouldin index ----
    float db_index = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        float max_ratio = 0.0f;  // Worst similarity for cluster i

        for (size_t j = 0; j < k; ++j) {
            if (i == j) continue;  // Don't compare cluster to itself

            // Distance between centroids i and j.
            float dist = 0.0f;
            for (size_t d = 0; d < centroids.cols(); ++d) {
                float diff = centroids[i][d] - centroids[j][d];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);

            // R(i,j) = (scatter_i + scatter_j) / distance_between_centroids.
            // Tight clusters that are far apart -> small R -> good.
            // Loose clusters that overlap -> large R -> bad.
            if (dist > 0) {
                float ratio = (scatter[i] + scatter[j]) / dist;
                max_ratio = std::max(max_ratio, ratio);
            }
        }
        db_index += max_ratio;
    }

    return db_index / k;  // Average over all clusters
}

// ============================================================================
// compute_calinski_harabasz() -- VARIANCE RATIO CRITERION
//
// CH = (between-cluster dispersion / (k-1)) / (within-cluster dispersion / (n-k))
//    = (SS_B / (k-1)) / (SS_W / (n-k))
//
// BETWEEN-CLUSTER (SS_B): How far are centroids from the global mean?
//   SS_B = sum_c (size_c * ||centroid_c - global_mean||²)
//
// WITHIN-CLUSTER (SS_W): How far are points from their own centroid?
//   SS_W = sum_i ||x_i - centroid_{label_i}||²
//
// Higher CH = better (clusters are compact AND well-separated).
// Range: [0, ∞).
// ============================================================================

float DriftDetector::compute_calinski_harabasz(const Matrix& X, const Vector& labels, const Matrix& centroids) {
    size_t n = X.rows();
    size_t k = centroids.rows();

    if (k <= 1 || n <= k) return 0.0f;  // Degenerate: need at least 2 clusters

    // ---- Step 1: Compute global centroid (mean of ALL points) ----
    // Iterate over all points and sum their coordinates.
    std::vector<float> global_centroid(X.cols(), 0.0f);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < X.cols(); ++d) {
            global_centroid[d] += X[i][d];
        }
    }
    // Divide by n to get the mean.
    for (size_t d = 0; d < X.cols(); ++d) {
        global_centroid[d] /= n;
    }

    // ---- Step 2: Count points per cluster ----
    std::vector<size_t> sizes(k, 0);
    for (size_t i = 0; i < n; ++i) {
        sizes[static_cast<size_t>(labels[i])]++;
    }

    // ---- Step 3: Between-cluster dispersion (SS_B) ----
    // For each cluster, compute: size_c * ||centroid_c - global_mean||²
    // Larger clusters and clusters far from the global mean contribute more.
    float between = 0.0f;
    for (size_t c = 0; c < k; ++c) {
        float dist = 0.0f;
        for (size_t d = 0; d < X.cols(); ++d) {
            float diff = centroids[c][d] - global_centroid[d];
            dist += diff * diff;
        }
        between += sizes[c] * dist;  // Weighted by cluster size
    }

    // ---- Step 4: Within-cluster dispersion (SS_W) ----
    // For each point, compute distance to its own centroid.
    float within = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        size_t c = static_cast<size_t>(labels[i]);
        float dist = 0.0f;
        for (size_t d = 0; d < X.cols(); ++d) {
            float diff = X[i][d] - centroids[c][d];
            dist += diff * diff;
        }
        within += dist;  // Note: no sqrt here, uses squared distances
    }

    if (within == 0) return 0.0f;  // All points exactly at centroids (perfect fit)

    // CH = (between / (k-1)) / (within / (n-k))
    // Normalized by degrees of freedom.
    return (between / (k - 1)) / (within / (n - k));
}

// ============================================================================
// compute_cluster_stability() -- HOW BALANCED ARE THE CLUSTER SIZES?
//
// Measured as normalized ENTROPY of the cluster size distribution.
//
// Entropy = -sum_c (p_c * log2(p_c))
// where p_c = size(c) / n (proportion of points in cluster c).
//
// Normalized entropy = entropy / max_entropy
// where max_entropy = log2(k) (achieved when all clusters have equal size).
//
// 1.0 = perfectly balanced (all clusters same size).
// 0.0 = maximally unbalanced (one cluster has all points).
//
// This is simpler and faster than silhouette/DB/CH -- O(n + k) instead of O(n²).
// ============================================================================

float DriftDetector::compute_cluster_stability(const Vector& labels, size_t k) {
    if (k <= 1) return 1.0f;  // One cluster = trivially stable

    size_t n = labels.size();

    // Count points per cluster.
    std::vector<size_t> sizes(k, 0);
    for (size_t i = 0; i < n; ++i) {
        sizes[static_cast<size_t>(labels[i])]++;
    }

    // Compute Shannon entropy of cluster sizes.
    // entropy = -sum(p * log2(p))
    float entropy = 0.0f;
    for (size_t c = 0; c < k; ++c) {
        if (sizes[c] > 0) {
            float p = static_cast<float>(sizes[c]) / n;  // Proportion in this cluster
            entropy -= p * std::log2(p);  // -p * log2(p) -- information content
        }
    }

    // Normalize by maximum possible entropy (all clusters equal size).
    // max_entropy = log2(k). Division gives range [0, 1].
    float max_entropy = std::log2(k);
    return max_entropy > 0 ? entropy / max_entropy : 1.0f;
}

} // namespace clustering
