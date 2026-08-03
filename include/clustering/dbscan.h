#pragma once

#include "matrix.h"
#include <vector>
#include <cstddef>
#include <memory>

namespace clustering {

struct DBSCANConfig {
    float epsilon = 0.5f;       // Neighborhood radius. Points within this distance are "neighbors".
                                 // Larger = more points per neighborhood = fewer clusters.
    size_t min_pts = 5;         // Minimum neighbors to be a "core point".
                                 // Higher = stricter cluster definition = more points marked as noise.
    size_t max_threads = 0;     // 0 = auto-detect threads for distance computation.
};

class DBSCAN {
public:
    DBSCAN();
    explicit DBSCAN(const DBSCANConfig& config);
    ~DBSCAN();

    void fit(const Matrix& X);
    Vector predict(const Matrix& X) const;

    const Vector& labels() const { return labels_; }
    size_t n_clusters() const { return n_clusters_; }
    size_t n_noise() const { return n_noise_; }

    void set_epsilon(float eps) { config_.epsilon = eps; }
    void set_min_pts(size_t m) { config_.min_pts = m; }

    static float estimate_epsilon(const Matrix& X, size_t min_pts = 5, size_t sample_size = 500);

    // nanoflann adapter — public so KDTreeIndex type alias can reference it
    // KD-tree uses DIM=3 internally; 2D data gets z=0 padding.
    struct KDTreeAdaptor {
        const Matrix& data;
        size_t dim;
        KDTreeAdaptor(const Matrix& d) : data(d), dim(d.cols()) {}
        inline size_t kdtree_get_point_count() const { return data.rows(); }
        inline float kdtree_get_pt(const size_t idx, const size_t j) const {
            return (j < dim) ? data[idx][j] : 0.0f;
        }
        template<typename BBOX>
        bool kdtree_get_bbox(BBOX&) const { return false; }
    };

private:
    void region_query(size_t point_idx, std::vector<size_t>& neighbors) const;
    void expand_cluster(size_t point_idx, size_t cluster_id);

    DBSCANConfig config_;
    Vector labels_;
    Matrix X_;
    size_t n_clusters_;
    size_t n_noise_;
    bool fitted_;
    std::vector<bool> visited_;

    std::unique_ptr<KDTreeAdaptor> kdtree_adaptor_;
    void* kdtree_index_ = nullptr;
};

} // namespace clustering
