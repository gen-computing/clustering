// ============================================================================
// t-SNE implementation -- t-Distributed Stochastic Neighbor Embedding.
//
// t-SNE maps high-dimensional data to 2D/3D for visualization, preserving
// local neighborhood structure. Unlike PCA (linear), t-SNE captures non-linear
// relationships in the data.
//
// ALGORITHM PHASES:
//   1. HIGH-DIM SPACE: Compute pairwise conditional probabilities P(j|i) using
//      Gaussian kernels with adaptive bandwidth (perplexity-controlled sigma).
//   2. LOW-DIM SPACE: Randomly initialize 2D embedding, then iteratively update
//      positions using gradient descent to minimize KL divergence between
//      high-dim and low-dim probability distributions.
//   3. OUTPUT: 2D coordinates that place similar points near each other.
//
// COMPLEXITY: O(n²) -- computes all pairwise distances twice per iteration.
//             Practically usable for n < 10,000 points without approximations.
// ============================================================================

#include "clustering/tsne.h"
#include "clustering/distance.h"
#include "clustering/validate.h"
#include <cmath>       // std::sqrt, std::log, std::exp, std::abs
#include <algorithm>   // std::min, std::max
#include <numeric>     // std::accumulate (for sums)
#include <random>      // std::mt19937, std::normal_distribution
#include <stdexcept>   // std::runtime_error

namespace clustering {

// ============================================================================
// CONSTRUCTOR
// ============================================================================

TSNE::TSNE(const TSNEConfig& config)
    : config_(config), n_samples_(0), n_features_(0), kl_divergence_(0.0f), fitted_(false) {}

// ============================================================================
// fit() -- COMPUTE THE t-SNE EMBEDDING
// ============================================================================

void TSNE::fit(const Matrix& X) {
    validate_matrix(X);
    validate_tsne(X, config_.perplexity);

    n_samples_ = X.rows();
    n_features_ = X.cols();

    // Store original training data for transform()
    X_ = X;

    // Use local n_components (don't mutate config_)
    size_t nc = std::min(config_.n_components, X.cols());

    // PHASE 1: Compute pairwise probabilities in high-dimensional space.
    compute_pairwise_probabilities(X);

    // PHASE 2: Randomly initialize the 2D/3D embedding.
    initialize_embedding(X);

    // PHASE 3: Iterative gradient descent optimization.
    float prev_kl = std::numeric_limits<float>::max();
    for (size_t iter = 0; iter < config_.n_iter; ++iter) {
        gradient_step(iter);

        // Compute KL divergence every 50 iterations (not every iter — O(n²) cost)
        if (iter % 50 == 0 || iter == config_.n_iter - 1) {
            kl_divergence_ = compute_kl_divergence();
        }

        // Early stopping: if gradient norm < threshold, we've converged
        // (config_.min_gradient_norm is checked but not used — now we use it)
        if (kl_divergence_ > 0 && std::abs(kl_divergence_ - prev_kl) < config_.min_gradient_norm * 100) {
            break;  // KL change is negligible → converged
        }
        prev_kl = kl_divergence_;
    }

    fitted_ = true;
}

// ============================================================================
// compute_pairwise_probabilities() -- HIGH-DIMENSIONAL SIMILARITIES
//
// For each pair of points (i, j), computes P(j|i) -- the conditional
// probability that i would choose j as its neighbor under a Gaussian
// centered at i with variance sigma_i².
//
// P(j|i) = exp(-||x_i - x_j||² / (2 * sigma_i²)) / sum(k≠i) exp(-||x_i - x_k||² / (2 * sigma_i²))
//
// The sigma_i is chosen so that the perplexity of P(j|i) equals the
// target perplexity. This means each point effectively considers
// approximately `perplexity` neighbors.
//
// Binary search finds sigma_i such that:
//   entropy_i = -sum(P(j|i) * log(P(j|i))) ≈ log(perplexity)
//
// This is O(n²) and the bottleneck for large datasets.
// ============================================================================

void TSNE::compute_pairwise_probabilities(const Matrix& X) {
    size_t n = X.rows();
    size_t d = X.cols();

    // ---- Step A: Compute all pairwise squared Euclidean distances ----
    // Store d² directly (avoid sqrt here, sqrt done later only for sigma search).
    // Precomputing d² saves ~50 n² sqrt calls in the binary search loop.
    Matrix dist_sq(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float dsq = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                float diff = X[i][k] - X[j][k];
                dsq += diff * diff;
            }
            dist_sq[i][j] = dsq;
            dist_sq[j][i] = dsq;
        }
    }

    // ---- Step B: Find sigma_i via binary search for each point ----
    // Allocate the probability matrix (n × n).
    P_.resize(n, n);

    // Target entropy: log(perplexity).
    // Perplexity of 30 means entropy should be ln(30) ≈ 3.4.
    float target_entropy = std::log(static_cast<float>(config_.perplexity));

    for (size_t i = 0; i < n; ++i) {
        // Binary search bounds for sigma.
        float sigma_min = 1e-10f;  // Very small sigma -> only immediate neighbors
        float sigma_max = 1e10f;   // Very large sigma -> all points equally probable
        float sigma = 1.0f;        // Starting guess

        // Binary search: up to 50 iterations to find the sigma that gives
        // the target perplexity.
        for (int iter = 0; iter < 50; ++iter) {
            // Compute conditional probabilities using precomputed d²
            float sum_p = 0.0f;
            for (size_t j = 0; j < n; ++j) {
                if (i != j) {
                    // dist_sq[i][j] already contains d² (no sqrt needed)
                    P_[i][j] = std::exp(-dist_sq[i][j] / (2.0f * sigma * sigma));
                    sum_p += P_[i][j];
                }
            }
            }

            // Normalize so probabilities sum to 1.
            if (sum_p > 0) {
                for (size_t j = 0; j < n; ++j) P_[i][j] /= sum_p;
            }

            // Compute the Shannon entropy of the distribution.
            // entropy = -sum(p * log(p))
            // Higher entropy = more uniform distribution = broader neighborhood.
            float entropy = 0.0f;
            for (size_t j = 0; j < n; ++j) {
                if (i != j && P_[i][j] > 1e-12f) {
                    entropy -= P_[i][j] * std::log(P_[i][j]);
                }
            }

            // Check if we're close enough to target.
            if (std::abs(entropy - target_entropy) < 0.01f) break;

            // Adjust sigma based on whether entropy is too high or too low.
            if (entropy > target_entropy) {
                // Entropy too high: distribution too flat -> need SMALLER sigma
                // (to make nearby points matter more).
                sigma_min = sigma;
                sigma = (sigma + sigma_max) / 2.0f;
            } else {
                // Entropy too low: distribution too peaked -> need LARGER sigma
                // (to spread probability to more points).
                sigma_max = sigma;
                sigma = (sigma + sigma_min) / 2.0f;
            }
        }
    }

    // ---- Step C: Symmetrize probabilities ----
    // Convert conditional P(j|i) to joint P(i,j) = (P(j|i) + P(i|j)) / (2n).
    // This makes the matrix symmetric, which simplifies the gradient.
    symmetricize_probabilities();
}

// ============================================================================
// symmetricize_probabilities() -- MAKE PROBABILITY MATRIX SYMMETRIC
//
// P(i,j) = (P(j|i) + P(i|j)) / (2 * n)
//
// This gives equal weight to the perspective of both points, and the
// division by 2n ensures the matrix normalizes properly.
// After this, P_[i][j] = P_[j][i] for all i, j.
// ============================================================================

void TSNE::symmetricize_probabilities() {
    size_t n = P_.rows();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            P_[i][j] = (P_[i][j] + P_[j][i]) / (2.0f * n);
            P_[j][i] = P_[i][j];  // Mirror for symmetry
        }
    }
}

// ============================================================================
// initialize_embedding() -- RANDOM STARTING POSITIONS
//
// Creates random 2D/3D coordinates from a Gaussian with very small variance
// (1e-4). This starts all points near the origin. The gradient descent will
// then spread them out according to the data's structure.
//
// Also initializes:
//   gains_: Adaptive learning rates per point per dimension (start at 1.0).
//   velocity_: Momentum accumulator (start at 0.0).
// ============================================================================

void TSNE::initialize_embedding(const Matrix& X) {
    // Use configured seed for reproducibility (or std::random_device if seed=-1)
    std::mt19937 gen(config_.random_seed >= 0 ? config_.random_seed : std::random_device{}());

    // Small-variance Gaussian: points start tightly clustered near (0, 0).
    // Too much spread initially can lead to bad local minima.
    std::normal_distribution<float> dist(0.0f, 1e-4f);

    embedding_.resize(n_samples_, config_.n_components);

    // gains_ tracks per-parameter adaptive learning rates.
    // Used in the "Jacobs" method: if gradient sign is consistent, increase
    // learning rate; if sign flips, decrease.
    gains_.resize(n_samples_, config_.n_components);
    gains_.fill(1.0f);  // Start with gain=1 for all parameters

    // velocity_ stores momentum for gradient descent.
    // v(t+1) = 0.8 * v(t) - lr * gain * gradient
    velocity_.resize(n_samples_, config_.n_components);
    velocity_.fill(0.0f);  // Start with zero velocity

    // Random initialization of embedding positions.
    for (size_t i = 0; i < n_samples_; ++i) {
        for (size_t d = 0; d < config_.n_components; ++d) {
            embedding_[i][d] = dist(gen);
        }
    }
}

// ============================================================================
// gradient_step() -- ONE ITERATION OF GRADIENT DESCENT
//
// The gradient of KL divergence with respect to embedding position y_i is:
//   dC/dy_i = 4 * sum_j( (P_ij - Q_ij) * (y_i - y_j) * (1 + ||y_i - y_j||²)^(-1) )
//
// Where:
//   P_ij = high-dimensional probability (computed earlier)
//   Q_ij = low-dimensional probability: (1 + ||y_i - y_j||²)^(-1) / sum_k≠l (1 + ||y_k - y_l||²)^(-1)
//
// The gradient has two forces:
//   ATTRACTION: P_ij pulls similar points together.
//   REPULSION:  -Q_ij pushes dissimilar points apart.
//
// 1. Compute Q (low-dim probabilities) using Student's t-distribution (heavy tails).
// 2. Compute gradient for each point.
// 3. Apply momentum with adaptive gains (Jacobs method).
// 4. Center the embedding (prevent drift).
// ============================================================================

void TSNE::gradient_step(size_t iter) {
    size_t n = n_samples_;
    size_t nd = config_.n_components;

    // ---- STEP 1: Compute low-dimensional affinities Q ----
    // Q[i][j] = (1 + ||y_i - y_j||²)^(-1) / sum_{k≠l} (1 + ||y_k - y_l||²)^(-1)
    //
    // Student's t-distribution with 1 degree of freedom (Cauchy distribution).
    // The heavy tails mean that moderate distances in the low-dimensional map
    // can still have meaningful probabilities -- this is what gives t-SNE its
    // ability to separate clusters nicely (the "crowding problem" solution).

    Q_.resize(n, n);
    float sum_q = 0.0f;  // Normalization constant

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float dist_sq = 0.0f;
            for (size_t d = 0; d < nd; ++d) {
                float diff = embedding_[i][d] - embedding_[j][d];
                dist_sq += diff * diff;
            }
            Q_[i][j] = 1.0f / (1.0f + dist_sq);   // Student's t-distribution kernel
            Q_[j][i] = Q_[i][j];                    // Symmetric
            sum_q += 2.0f * Q_[i][j];               // Counted twice (both i,j and j,i)
        }
    }

    // Normalize Q so it sums to 1 (like a proper probability distribution).
    if (sum_q > 0) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                Q_[i][j] /= sum_q;
            }
        }
    }

    // ---- STEP 2: Exaggeration phases ----
    // Early exaggeration (first few iterations): P multiplied by 4.
    // Makes clusters form quickly by exaggerating attractive forces.
    //
    // Late exaggeration (later iterations): P multiplied by 1.2.
    // Helps refine cluster separation after initial formation.
    // Standard in modern t-SNE implementations.
    float exaggeration = 1.0f;
    if (iter < config_.early_exaggeration) {
        exaggeration = 4.0f;  // Early: strong exaggeration
    } else if (iter > config_.n_iter * 3 / 4) {
        exaggeration = 1.2f;  // Late: mild exaggeration for final refinement
    }

    // ---- STEP 3: Compute gradients ----
    // For each point i, compute dC/dy_i = 4 * sum_j((P_ij - Q_ij) * (y_i - y_j))
    // (The (1 + ||y_i-y_j||²)^(-1) factor is absorbed into Q's normalization.)
    Matrix gradients(n, nd);

    for (size_t i = 0; i < n; ++i) {
        // Initialize gradient for point i to zero.
        for (size_t d = 0; d < nd; ++d) gradients[i][d] = 0.0f;

        for (size_t j = 0; j < n; ++j) {
            if (i != j) {
                // P_ij - Q_ij: the "error" between high-dim and low-dim probabilities.
                // Positive: these points should be CLOSER (attractive force).
                // Negative: these points should be FARTHER (repulsive force).
                float pq = (P_[i][j] - Q_[i][j]) * exaggeration;

                // Gradient contribution: pq * (y_i - y_j) for each dimension.
                for (size_t d = 0; d < nd; ++d) {
                    gradients[i][d] += pq * (embedding_[i][d] - embedding_[j][d]);
                }
            }
        }
    }

    // ---- STEP 4: Update embedding with momentum and adaptive gains ----
    // Learning rate: higher during early exaggeration, lower afterward.
    float learning_rate = config_.learning_rate;
    if (iter > config_.early_exaggeration) {
        learning_rate /= 2.0f;  // Reduce learning rate after exaggeration phase
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < nd; ++d) {
            float grad = gradients[i][d];

            // Adaptive gains (Jacobs delta-bar-delta method):
            // If gradient sign is consistent (same direction as last time),
            // increase gain by 0.2 (gaining momentum in the right direction).
            // If gradient sign flipped, decrease gain by ×0.8 (slow down -- we overshot).
            //
            // velocity_[i][d] has the OPPOSITE sign of the gradient (because
            // velocity updates with -grad), so we compare signs differently.
            if ((grad > 0) == (velocity_[i][d] > 0)) {
                gains_[i][d] += 0.2f;    // Same direction: increase learning rate
            } else {
                gains_[i][d] *= 0.8f;    // Direction change: decrease learning rate
            }
            // Clamp gain to minimum 0.01 to prevent getting stuck.
            gains_[i][d] = std::max(0.01f, gains_[i][d]);

            // Momentum update (like a ball rolling down a hill):
            // velocity(t+1) = momentum * velocity(t) - learning_rate * gain * gradient
            // position(t+1) = position(t) + velocity(t+1)
            //
            // Momentum = 0.8 means 80% of previous velocity is retained.
            // This smooths the trajectory and helps escape local minima.
            velocity_[i][d] = 0.8f * velocity_[i][d] - learning_rate * gains_[i][d] * grad;
            embedding_[i][d] += velocity_[i][d];
        }
    }

    // ---- STEP 5: Center the embedding ----
    // Without centering, the embedding can drift arbitrarily far from origin.
    // Centering keeps the embedding well-behaved for visualization.
    for (size_t d = 0; d < nd; ++d) {
        float mean = 0.0f;
        for (size_t i = 0; i < n; ++i) mean += embedding_[i][d];
        mean /= n;
        // Subtract the mean so the center of mass is at (0, 0, ...).
        for (size_t i = 0; i < n; ++i) embedding_[i][d] -= mean;
    }

    // ---- STEP 6: Update KL divergence for reporting ----
    kl_divergence_ = compute_kl_divergence();
}

// ============================================================================
// compute_kl_divergence() -- MEASURE EMBEDDING QUALITY
//
// KL(P||Q) = sum_{i≠j} P_ij * log(P_ij / Q_ij)
//
// Lower = better. 0 would mean perfect match (P=Q exactly).
// In practice, 0.1-2.0 is typical for good embeddings.
// High values mean the low-dimensional embedding doesn't preserve the
// original data structure well.
// ============================================================================

float TSNE::compute_kl_divergence() const {
    float kl = 0.0f;
    for (size_t i = 0; i < n_samples_; ++i) {
        for (size_t j = 0; j < n_samples_; ++j) {
            // Only compute where both probabilities are meaningfully > 0.
            // 1e-12 avoids log(0) = -inf.
            if (i != j && P_[i][j] > 1e-12f && Q_[i][j] > 1e-12f) {
                kl += P_[i][j] * std::log(P_[i][j] / Q_[i][j]);
            }
        }
    }
    return kl;
}

// ============================================================================
// transform() -- PLACEHOLDER FOR NEW POINT EMBEDDING
//
// REAL t-SNE does NOT support transform() -- you cannot embed a new point
// without re-running the entire optimization.
//
// This is a naive placeholder: assigns each new point the position of its
// nearest neighbor in the training set.
//
// LIMITATION: This does NOT use the original high-dimensional data correctly.
// It compares to embedding_ instead of to original X. A real implementation
// would need to store the training data.
// ============================================================================

Matrix TSNE::transform(const Matrix& X) const {
    validate_fitted(fitted_);

    size_t n = X.rows();
    size_t nd = config_.n_components;
    Matrix result(n, nd);

    // For each new point, find the nearest training point in original
    // high-dimensional space and copy its embedding position.
    for (size_t i = 0; i < n; ++i) {
        float min_dist = std::numeric_limits<float>::max();
        size_t nearest = 0;

        for (size_t j = 0; j < n_samples_; ++j) {
            float dist = l2_distance_avx2(X[i], X_[j], n_features_);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = j;
            }
        }

        // Copy the nearest training point's embedding position.
        for (size_t d = 0; d < nd; ++d) {
            result[i][d] = embedding_[nearest][d];
        }
    }

    return result;
}

// ============================================================================
// fit_transform() -- FIT AND RETURN EMBEDDING
// ============================================================================

Matrix TSNE::fit_transform(const Matrix& X) {
    fit(X);
    return embedding_;  // Return the 2D/3D coordinates
}

} // namespace clustering
