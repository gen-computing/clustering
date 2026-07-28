#pragma once

#include "kmeans.h"

namespace clustering {

// ============================================================================
// MiniBatchKMeans -- KMeans for very large datasets using random subsets.
//
// PROBLEM IT SOLVES:
//   Regular KMeans must compute distances from EVERY point to EVERY centroid
//   in each iteration. For millions of points, this is too slow.
//
// SOLUTION:
//   Instead of using all data each iteration, MiniBatchKMeans randomly picks
//   a small batch (e.g., 100 points) per iteration. It updates centroids
//   using only those points, then picks a new random batch next iteration.
//
// TRADE-OFF:
//   - Much FASTER than regular KMeans on large data
//   - Results are APPROXIMATE (not exactly the optimal clustering)
//   - Good enough for most practical purposes
//
// HOW IT INHERITS:
//   MiniBatchKMeans IS-A KMeans. Reuses fit() for initialization, then overrides
//   partial_fit() with batch-based logic.
//
// PARAMETER: batch_size -- how many random points to sample per iteration.
//   Larger = closer to regular KMeans but slower.
//   Smaller = faster but less accurate.
//   Typical values: 100-1000 depending on dataset size.
// ============================================================================
class MiniBatchKMeans : public KMeans {
public:
    // Constructor: Specify number of clusters (k) and batch size.
    // Example: MiniBatchKMeans mb(5, 200);  // 5 clusters, 200-point batches
    explicit MiniBatchKMeans(size_t k = 8, size_t batch_size = 100);

    ~MiniBatchKMeans() override;

    // partial_fit: On first call, does full fit() to initialize centroids.
    // On subsequent calls: randomly samples batch_size_ points from X,
    // computes their nearest centroids, and nudges centroids toward them.
    // Uses a decaying learning rate: 1 / (iteration + 1).
    //
    // Parameter X: New batch of data. Only batch_size_ random points from X
    //              are actually used for the update -- the rest are ignored.
    void partial_fit(const Matrix& X) override;

private:
    size_t batch_size_;  // How many random points to use per update step
    size_t n_updates_;   // Number of partial_fit calls (for learning rate decay)
};

} // namespace clustering
