#pragma once

#include "matrix.h"
#include <vector>

namespace clustering {

// ============================================================================
// DriftMetrics -- Results from a single drift detection check.
//
// These metrics assess how "good" the current clustering is. When a metric
// degrades significantly, it signals that the data pattern has changed
// (concept drift) and the model needs retraining.
// ============================================================================
struct DriftMetrics {
    // silhouette_score: How well-separated are the clusters?
    //   Range: [-1, 1]. Higher = better defined clusters.
    //   1.0 = perfect separation (points much closer to own cluster than others).
    //   0.0 = clusters overlap.
    //   <0 = points may be assigned to wrong clusters.
    float silhouette_score;

    // davies_bouldin_index: Average similarity between each cluster and its
    //   most similar neighbor. LOWER = better.
    //   Measures ratio of within-cluster scatter to between-cluster separation.
    float davies_bouldin_index;

    // calinski_harabasz_score: Ratio of between-cluster variance to within-cluster
    //   variance. HIGHER = better defined clusters (dense and well-separated).
    float calinski_harabasz_score;

    // cluster_stability: How balanced are the cluster sizes?
    //   Range: [0, 1]. 1.0 = perfectly equal sizes.
    //   Measured as normalized entropy of cluster size distribution.
    //   0.5 means moderate imbalance, 0.0 means one cluster has all points.
    float cluster_stability;

    // drift_detected: True if the silhouette score has degraded more than
    //   the configured threshold since recent history. This is the signal
    //   that triggers retraining in OnlineKMeans.
    bool drift_detected;
};

// ============================================================================
// DriftDetector -- Monitors clustering quality and detects concept drift.
//
// WHAT IS CONCEPT DRIFT?
//   When the statistical properties of your data change over time. Example:
//   - Customer behavior changing with seasons (summer vs winter buying).
//   - Sensor readings shifting as equipment wears down.
//   - Stock market regime changes (bull vs bear markets).
//
// HOW IT WORKS:
//   1. Keeps a history (sliding window) of recent DriftMetrics.
//   2. Each time check() is called, computes 4 quality metrics.
//   3. Compares recent average silhouette to older average.
//   4. If the degradation exceeds threshold, flags drift_detected = true.
//   5. The caller (typically OnlineKMeans) responds by retraining.
// ============================================================================
class DriftDetector {
public:
    DriftDetector();
    ~DriftDetector();

    // set_threshold: How much the silhouette score must drop to trigger drift.
    //   Default: 0.1. Lower = more sensitive (triggers more often).
    //   The check compares: (older_avg - recent_avg) > threshold.
    void set_threshold(float threshold);

    // set_window_size: How many historical checks to keep for trend analysis.
    //   Default: 10. Larger = smoother trends but slower to detect sudden drift.
    //   Must be at least 3 (need enough history to split into older/recent halves).
    void set_window_size(size_t window_size);

    // check: The main method. Compute all metrics and detect drift.
    // Parameters:
    //   X         -- the data points (n_samples x n_features)
    //   labels    -- which cluster each point belongs to (0 to k-1)
    //   centroids -- the cluster center coordinates (k x n_features)
    // Returns: DriftMetrics with all scores and the drift_detected flag.
    DriftMetrics check(const Matrix& X, const Vector& labels, const Matrix& centroids);

    // is_drifting: Returns the result of the MOST RECENT check() call.
    bool is_drifting() const { return drifting_; }

private:
    // Compute the 4 quality metrics. Each is O(n²) or O(n*k) complexity.

    // Silhouette score: For each point, compute how close it is to its own
    // cluster vs the nearest other cluster. Average over all points.
    // Score(i) = (b(i) - a(i)) / max(a(i), b(i))
    //   where a(i) = avg distance to own cluster
    //         b(i) = avg distance to nearest other cluster
    float compute_silhouette(const Matrix& X, const Vector& labels, const Matrix& centroids);

    // Davies-Bouldin index: Average of the worst similarity ratio for each cluster.
    // Di = max_j ((scatter_i + scatter_j) / distance(centroid_i, centroid_j))
    // DB = average(Di)
    float compute_davies_bouldin(const Matrix& X, const Vector& labels, const Matrix& centroids);

    // Calinski-Harabasz: Ratio of between-cluster to within-cluster dispersion.
    // CH = (between / (k-1)) / (within / (n-k))
    // Higher = better (clusters are compact and far apart).
    float compute_calinski_harabasz(const Matrix& X, const Vector& labels, const Matrix& centroids);

    // Cluster stability: Normalized entropy of cluster size distribution.
    // Perfectly balanced clusters = 1.0.
    // All points in one cluster = 0.0.
    float compute_cluster_stability(const Vector& labels, size_t k);

    float threshold_;                     // How much silhouette must drop to trigger
    size_t window_size_;                  // How many historical checks to keep
    bool drifting_;                       // Most recent drift status
    std::vector<DriftMetrics> history_;   // Rolling history of past checks
};

} // namespace clustering
