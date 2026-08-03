#pragma once

#include "matrix.h"
#include <vector>

namespace clustering {

// ============================================================================
// TSNEConfig -- Settings for t-SNE (t-Distributed Stochastic Neighbor Embedding).
// ============================================================================
struct TSNEConfig {
    size_t n_components = 2;           // Target dimensions (usually 2 for visualization).

    size_t perplexity = 30;            // PERPLEXITY: Roughly "how many neighbors to consider".
                                       // Lower (5-20) = focus on very local structure.
                                       // Higher (50-100) = consider broader patterns.
                                       // Must be less than number of samples.
                                       // Typical range: 5-50. Default 30 works for most cases.

    float learning_rate = 200.0f;     // How fast the optimizer moves points.
                                       // Too high = unstable, points fly apart.
                                       // Too low = slow, may get stuck.
                                       // Automatic adjustment handles most cases.

    size_t n_iter = 1000;              // Number of optimization iterations.
                                       // More = better embedding but slower.
                                       // 250-1000 typical for small data, more for large.

    size_t early_exaggeration = 12;    // Number of initial iterations where
                                       // attractive forces are multiplied by 4.
                                       // Helps form well-separated clusters early on.
                                       // After this, forces return to normal.

    float min_gradient_norm = 1e-7f;  // Convergence threshold (stop if gradient
                                       // drops below this). Not always used.

    int random_seed = -1;              // Random seed for reproducibility.
                                       // -1 = use std::random_device (non-deterministic).
                                       // >= 0 = fixed seed (reproducible results).
};

// ============================================================================
// TSNE -- t-Distributed Stochastic Neighbor Embedding.
//
// WHAT IT DOES:
//   Converts high-dimensional data (many features) into 2D or 3D coordinates
//   for VISUALIZATION. Unlike PCA (linear), t-SNE preserves local neighborhoods
//   -- points that are close in high dimensions stay close in the 2D map.
//
// ANALOGY:
//   PCA is like taking a photo of a 3D object from the best angle.
//   t-SNE is like flattening a crumpled paper -- neighborhoods stay intact
//   but distances between far-apart groups may not be meaningful.
//
// HOW IT WORKS (simplified):
//   1. In HIGH dimension: compute probability that point i would pick point j
//      as its neighbor (using Gaussian kernel with adaptive bandwidth).
//   2. In LOW dimension (2D): start with random positions, compute same
//      probabilities but using Student's t-distribution (heavier tails).
//   3. Use gradient descent to move points so the low-dim probabilities
//      match the high-dim probabilities as closely as possible.
//   4. KL DIVERGENCE: Measures mismatch between the two probability distributions.
//      Lower KL divergence = better embedding.
//
// IMPORTANT NOTES:
//   - t-SNE is for VISUALIZATION, not for downstream ML tasks.
//   - Results are random (different runs give different layouts).
//   - Distances BETWEEN clusters are not meaningful (only within-cluster).
//   - O(n²) complexity: slow for >10,000 points without approximations.
//   - The transform() method is a placeholder -- real t-SNE has no transform.
// ============================================================================
class TSNE {
public:
    TSNE() = default;
    explicit TSNE(const TSNEConfig& config);
    ~TSNE() = default;

    // ----- CORE METHODS -----

    // fit: Compute the 2D/3D embedding from high-dimensional data.
    // Steps through: compute pairwise probabilities, initialize random embedding,
    // then run n_iter gradient descent steps.
    // Throws: runtime_error if X is empty, has 0 features, or fewer than 2 samples.
    void fit(const Matrix& X);

    // transform: For new points, assign them positions near their closest
    // neighbor in the training set. This is a SIMPLE PLACEHOLDER -- real t-SNE
    // does not support transform (you must rerun fit on combined data).
    Matrix transform(const Matrix& X) const;

    // fit_transform: Fit and return the embedding in one call.
    // Returns: Matrix of shape (n_samples x n_components) -- the 2D/3D coordinates.
    Matrix fit_transform(const Matrix& X);

    // ----- ACCESSORS (read results after fitting) -----

    // embedding: The final low-dimensional coordinates of each point.
    //            embedding()[i][0] = x-coordinate, embedding()[i][1] = y-coordinate.
    const Matrix& embedding() const { return embedding_; }

    // kl_divergence: How well the embedding preserves original distances.
    //                 Lower = better. 0 would mean perfect preservation (impossible).
    float kl_divergence() const { return kl_divergence_; }

private:
    // ----- INTERNAL ALGORITHM STEPS -----

    // compute_pairwise_probabilities: Step 1 of t-SNE.
    //   - Compute all pairwise Euclidean distances between points.
    //   - For each point i, find sigma such that the conditional probability
    //     distribution P(j|i) has a fixed perplexity.
    //   - Uses binary search to find the right sigma for each point.
    void compute_pairwise_probabilities(const Matrix& X);

    // symmetricize_probabilities: Step 2 of t-SNE.
    //   Make the probability matrix symmetric: P(i,j) = (P(j|i) + P(i|j)) / (2 * n).
    //   This makes the probabilities mutually consistent.
    void symmetricize_probabilities();

    // initialize_embedding: Step 3 of t-SNE.
    //   Create random starting positions for all points in 2D space.
    //   Also initializes gains (adaptive learning rates per point) and velocity (momentum).
    void initialize_embedding(const Matrix& X);

    // gradient_step: Step 4 of t-SNE (called repeatedly).
    //   Compute low-dimensional probabilities Q(i,j) using Student's t-distribution.
    //   Compute gradient of KL divergence w.r.t. each point's position.
    //   Apply gradient descent with momentum and adaptive gains.
    //   Center the embedding (remove mean drift).
    // Parameters: iter -- current iteration number (affects early exaggeration, learning rate).
    void gradient_step(size_t iter);

    // compute_kl_divergence: Calculate current KL divergence.
    //   KL(P||Q) = Sum(P_ij * log(P_ij / Q_ij)) for all i != j.
    //   Only computed when needed (not every iteration).
    float compute_kl_divergence() const;

    // ----- MEMBER VARIABLES -----

    TSNEConfig config_;      // All settings (perplexity, learning rate, etc.)
    size_t n_samples_;       // Number of data points
    size_t n_features_;      // Original feature count (set during fit)
    Matrix P_;               // High-dimensional pairwise probabilities (n x n)
    Matrix Q_;               // Low-dimensional pairwise probabilities (n x n)
    Matrix embedding_;       // The 2D/3D embedding coordinates (n x n_components)
    Matrix gains_;           // Per-point, per-dimension adaptive learning rate
    Matrix velocity_;        // Momentum accumulator for gradient descent
    Matrix X_;               // Original training data (stored for transform)
    float kl_divergence_;    // Current KL divergence (optimization target)
    bool fitted_;            // Has fit() been called?
};

} // namespace clustering
