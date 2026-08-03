#pragma once

#include "kmeans.h"
#include <random>

namespace clustering {

class MiniBatchKMeans : public KMeans {
public:
    explicit MiniBatchKMeans(size_t k = 8, size_t batch_size = 100);
    ~MiniBatchKMeans() override;

    void partial_fit(const Matrix& X) override;

private:
    size_t batch_size_;
    size_t n_updates_;

    // Pre-allocated members (avoid per-call heap allocation)
    std::mt19937 rng_;                       // RNG seeded once
    std::vector<size_t> indices_;            // Reusable batch indices
    std::vector<float> dist_buffer_;         // Reusable distance storage
};

} // namespace clustering
