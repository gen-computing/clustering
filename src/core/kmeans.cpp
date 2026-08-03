// ============================================================================
// KMeans implementation -- The core batch clustering algorithm.
//
// This file contains the actual step-by-step logic that the header declares.
// The implementation is separate from the header to keep declarations clean
// and to enable faster compilation (changing .cpp doesn't recompile
// everything that includes the .h).
//
// ALGORITHM OVERVIEW (what fit() does):
//   1. VALIDATE input (not empty, k valid, etc.)
//   2. INITIALIZE k centroids using KMeans++ strategy
//   3. LOOP until convergence or max_iter:
//      a. Compute distances from all points to all centroids
//      b. Assign each point to its nearest centroid
//      c. Update centroids to be the mean of their assigned points
//      d. Check if centroids stopped moving (convergence)
//   4. Mark model as fitted
// ============================================================================

#include "clustering/kmeans.h"
#include "clustering/distance.h"     // For AVX2 distance functions
#include "clustering/thread_pool.h"  // For parallel distance computation
#include "clustering/validate.h"     // For shared input validation
#include "clustering/logging.h"
#include <algorithm>                 // For std::min, std::abs, std::fill
#include <random>                    // For std::mt19937 (Mersenne Twister random engine)
#include <cmath>                     // For std::sqrt, std::abs
#include <stdexcept>                 // For std::runtime_error (input validation)

namespace clustering {

// ============================================================================
// CONSTRUCTORS
// ============================================================================

// Simple constructor: only specify k.
// Member variables are initialized in the "initializer list" (after the colon).
// The syntax `n_iter_(0)` means "set n_iter_ to 0".
// This runs BEFORE the function body, which is empty here.
KMeans::KMeans(size_t k) : config_{}, n_iter_(0), inertia_(0.0f), fitted_(false) {
    config_.k = k;  // Override the default (8) with the user's value
}

// Config-based constructor: copy the entire config struct.
// config_(config) copies all fields. Other members initialized to defaults.
KMeans::KMeans(const KMeansConfig& config)
    : config_(config), n_iter_(0), inertia_(0.0f), fitted_(false) {}

// Virtual destructor: = default means "use the default compiler-generated one".
// Marked virtual because this class is a base class for OnlineKMeans and
// MiniBatchKMeans. Without virtual, deleting a derived object through a
// KMeans* pointer would NOT call the derived destructor -> memory leak.
KMeans::~KMeans() = default;

// ============================================================================
// fit() -- TRAIN THE MODEL
// ============================================================================

void KMeans::fit(const Matrix& X) {
    // ----- INPUT VALIDATION -----
    validate_matrix(X);
    validate_k(config_.k, X.rows());

    LOG_INFO("KMeans::fit starting: n=%zu x d=%zu, k=%zu, threads=%zu", X.rows(), X.cols(), config_.k, config_.max_threads);

    // ----- INITIALIZATION -----

    // Pick k starting centroids using the KMeans++ strategy.
    // KMeans++ picks centroids that are spread out, which helps the algorithm
    // converge faster and to a better solution than pure random.
    initialize_centroids(X);

    // Allocate space for the labels vector (one float per data point).
    // Each element will be set to the cluster ID (0.0f, 1.0f, ... k-1.0f).
    labels_.resize(X.rows());

    // Allocate the counts vector (one float per cluster).
    // This tracks how many points are assigned to each cluster.
    counts_.resize(config_.k);

    // Allocate the distances matrix (n rows x k columns).
    // distance_matrix[i][j] = Euclidean distance from point i to centroid j.
    // This is the "working memory" of the algorithm -- recomputed each iteration.
    distances_.resize(X.rows(), config_.k);

    // ----- MAIN ITERATION LOOP -----
    // This is the heart of KMeans. Repeat until converged or max_iter reached.

    // n_iter_ starts at 0 and increments each loop. The loop runs at most
    // config_.max_iter times (default 300) to prevent infinite loops.
    for (n_iter_ = 0; n_iter_ < config_.max_iter; ++n_iter_) {
        // STEP 1: Compute all point-to-centroid distances.
        // This is the most expensive operation (O(n * k * d)).
        // Uses AVX2 SIMD for speed and ThreadPool for parallelism.
        // Fills the distances_ matrix.
        compute_distances(X);

        // STEP 2: Assign each point to its nearest centroid.
        // For each row in distances_, find the column with minimum distance.
        // Sets labels_[i] = cluster_index and increments counts_[cluster_index].
        assign_clusters();

        // STEP 3: Move centroids to the mean of their assigned points.
        // For each cluster, sum up all points assigned to it, divide by count.
        // This moves the centroid to the "center of mass" of its points.
        update_centroids(X);

        // Real-time visualization callback (GUI toggle)
        if (config_.iter_callback) {
            if (config_.iter_callback(n_iter_, centroids_, labels_)) break;
        }

        // STEP 4: Check if we're done (convergence).
        // Compares current inertia to previous inertia. If the change is
        // smaller than config_.tol * inertia_, the centroids have stabilized
        // and we can stop early.
        if (check_convergence()) {
            break;  // Exit the loop -- model has converged
        }
    }

    // Mark the model as trained.
    // This flag is checked by predict() -- if false, predict() throws an error.
    LOG_INFO("KMeans::fit done: iter=%zu, inertia=%.4f", n_iter_, inertia_);
    fitted_ = true;
}

// ============================================================================
// predict() -- ASSIGN NEW DATA TO CLUSTERS
// ============================================================================

Vector KMeans::predict(const Matrix& X) const {
    validate_fitted(fitted_);

    // Create output: one label per input point.
    Vector labels(X.rows());

    // Create scratch space for distances (n rows x k columns).
    Matrix distances(X.rows(), config_.k);

    // Compute distance from every new point to every centroid.
    // Uses the same parallel/AVX2 function as fit() does.
    compute_distance_matrix(X, centroids_, distances, config_.max_threads);

    // For each point, find which centroid is closest.
    // This is a linear scan over the k distances.
    for (size_t i = 0; i < X.rows(); ++i) {
        size_t best = 0;                  // Index of the closest centroid so far
        float best_dist = distances[i][0]; // Distance to the current best centroid

        // Check all k centroids (starting from index 1 since we already checked 0).
        for (size_t j = 1; j < config_.k; ++j) {
            if (distances[i][j] < best_dist) {
                best_dist = distances[i][j];  // Found a closer centroid
                best = j;                      // Update best index
            }
        }

        // Store the cluster ID. Cast to float because Vector stores floats.
        // A float like 2.0f means "cluster 2".
        labels[i] = static_cast<float>(best);
    }

    return labels;
}

// ============================================================================
// partial_fit() -- INCREMENTAL UPDATE OF EXISTING MODEL
// ============================================================================

void KMeans::partial_fit(const Matrix& X) {
    // If we haven't been trained yet, do a full fit to initialize.
    // This gives us initial centroids to work with.
    if (!fitted_) {
        fit(X);
        return;
    }

    // INCREMENTAL UPDATE: Nudge centroids toward new data points.
    //
    // This is a simplified version of online learning. For each new point:
    // 1. Find the nearest centroid
    // 2. Move that centroid slightly toward the point
    //
    // The learning rate (lr) controls how much we move.
    // lr = 1 / sqrt(iteration + 1) -- large at first, then decays.
    //
    // This is NOT the same as fitting from scratch. The centroids will drift
    // toward the most recent data. For proper online learning, use OnlineKMeans
    // which has sliding windows and forgetting factors.

    // Learning rate: how aggressively to move centroids.
    // Decays with sqrt(n_iter) so early points have more influence.
    // Example: n_iter=0 -> lr=1.0, n_iter=99 -> lr=0.1
    float lr = 1.0f / std::sqrt(static_cast<float>(n_iter_ + 1));

    for (size_t i = 0; i < X.rows(); ++i) {
        // Find which centroid is closest to this point.
        // nearest_centroid uses AVX2 distance for speed.
        size_t nearest = nearest_centroid(X[i], centroids_);

        // Nudge the centroid toward this point:
        // new_centroid = old_centroid + lr * (point - old_centroid)
        //
        // This is "stochastic gradient descent" style:
        // If lr=1.0: centroid jumps to the point's exact location.
        // If lr=0.01: centroid moves 1% of the way toward the point.
        //
        // Each feature dimension is updated independently.
        for (size_t d = 0; d < X.cols(); ++d) {
            centroids_[nearest][d] += lr * (X[i][d] - centroids_[nearest][d]);
        }
    }

    // Count this as an iteration (even though we processed multiple points,
    // the learning rate decay is per-call to partial_fit).
    n_iter_++;
}

// ============================================================================
// initialize_centroids() -- KMEANS++ INITIALIZATION
//
// Picks k starting centroids so they're spread apart. Why?
// - Pure random initialization can put two centroids close together,
//   causing the algorithm to converge to a bad local minimum.
// - KMeans++ ensures centroids are diverse, leading to better clustering.
//
// The algorithm:
// 1. Pick first centroid randomly from the data.
// 2. For each subsequent centroid:
//    a. For each data point, compute distance to nearest already-chosen centroid.
//    b. Choose the next centroid with probability proportional to distance².
//       (Points FAR from existing centroids are MORE likely to be chosen.)
// 3. Repeat until k centroids are chosen.
// ============================================================================

void KMeans::initialize_centroids(const Matrix& X) {
    // Set up the random number generator.
    // std::random_device: uses hardware entropy (truly random).
    // std::mt19937: Mersenne Twister engine (fast, good quality randomness).
    std::random_device rd;
    std::mt19937 gen(rd());

    // Uniform distribution over [0, X.rows()-1] for picking random point indices.
    std::uniform_int_distribution<size_t> dist(0, X.rows() - 1);

    // Allocate space for k centroids, each with d features.
    centroids_.resize(config_.k, X.cols());

    // ----- PICK FIRST CENTROID -----
    // Randomly select any data point as the first centroid.
    size_t first = dist(gen);
    for (size_t d = 0; d < X.cols(); ++d) {
        centroids_[0][d] = X[first][d];  // Copy the chosen point
    }

    // ----- PICK REMAINING CENTROIDS -----
    // Track each point's squared distance to the nearest centroid.
    // Initialize with "infinity" (largest possible float).
    // As we add centroids, we update these distances.
    std::vector<float> min_distances(X.rows(), std::numeric_limits<float>::max());

    for (size_t c = 1; c < config_.k; ++c) {
        // ---- Update minimum distances ----
        // For each point, check if the NEW centroid (just chosen at index c-1)
        // is closer than the previous nearest.
        for (size_t i = 0; i < X.rows(); ++i) {
            float d = l2_distance_avx2(X[i], centroids_[c - 1], X.cols());
            min_distances[i] = std::min(min_distances[i], d);
        }

        // ---- Build probability distribution ----
        // Each point's probability = (distance to nearest centroid)² / total
        // Points far from any centroid get higher probability.
        float total = 0.0f;
        for (float d : min_distances) {
            total += d * d;  // Square makes distant points MUCH more likely
        }

        // ---- Sample according to the distribution ----
        // Generate a random number in [0, total).
        std::uniform_real_distribution<float> prob_dist(0.0f, total);
        float r = prob_dist(gen);

        // Walk through points, accumulating probabilities until we exceed r.
        // This is "roulette wheel selection": each point gets a slice of the
        // [0, total] line proportional to its distance².
        float cumulative = 0.0f;
        size_t chosen = dist(gen);  // Fallback: random point if rounding issues
        for (size_t i = 0; i < X.rows(); ++i) {
            cumulative += min_distances[i] * min_distances[i];
            if (cumulative >= r) {
                chosen = i;
                break;
            }
        }
        for (size_t d = 0; d < X.cols(); ++d) {
            centroids_[c][d] = X[chosen][d];
        }
    }
}

// ============================================================================
// compute_distances() -- FILL THE DISTANCE MATRIX
//
// For every point and every centroid, compute the Euclidean distance.
// This is O(n * k * d) -- the bottleneck of KMeans.
// Uses compute_distance_matrix() which internally uses ThreadPool for
// parallelism and AVX2 SIMD for speed.
// ============================================================================

void KMeans::compute_distances(const Matrix& X) {
    compute_distance_matrix(X, centroids_, distances_, config_.max_threads);
}

// ============================================================================
// assign_clusters() -- ASSIGN EACH POINT TO NEAREST CENTROID (E-step)
//
// For each point, scan the k distances and find the minimum.
// This is O(n * k) -- linear scan of the distance matrix.
//
// Also resets counts_ to track how many points per cluster.
// ============================================================================

void KMeans::assign_clusters() {
    size_t n = distances_.rows();  // Number of data points
    size_t k = distances_.cols();  // Number of clusters

    labels_.resize(n);
    counts_.fill(0.0f);  // Reset all cluster counts to 0

    // For each point...
    for (size_t i = 0; i < n; ++i) {
        size_t best = 0;
        float best_dist = distances_[i][0];  // Distance to centroid 0

        // Find the centroid with minimum distance
        for (size_t j = 1; j < k; ++j) {
            if (distances_[i][j] < best_dist) {
                best_dist = distances_[i][j];
                best = j;
            }
        }

        // Assign point i to cluster `best`
        labels_[i] = static_cast<float>(best);

        // Increment the count for that cluster
        counts_[best] += 1.0f;
    }
}

// ============================================================================
// update_centroids() -- MOVE CENTROIDS TO MEAN POSITION (M-step)
//
// For each cluster, compute the AVERAGE of all points assigned to it.
// This average becomes the new centroid position.
//
// Uses a two-pass approach:
//   1. Reset all centroids to zero, then accumulate (sum) all assigned points.
//   2. Divide each centroid by its point count to get the mean.
// ============================================================================

void KMeans::update_centroids(const Matrix& X) {
    size_t n = X.rows();       // Number of data points
    size_t d = X.cols();       // Number of features
    size_t k = config_.k;      // Number of clusters

    // ---- Step 1: Reset centroids to (0, 0, ..., 0) ----
    centroids_.fill(0.0f);

    // ---- Step 2: Accumulate all assigned points ----
    // For each point, add its feature values to its cluster's centroid.
    // After this loop, centroids_[c][d] = SUM of feature d for all points in cluster c.
    for (size_t i = 0; i < n; ++i) {
        size_t cluster = static_cast<size_t>(labels_[i]);
        for (size_t j = 0; j < d; ++j) {
            centroids_[cluster][j] += X[i][j];
        }
    }

    // ---- Step 3: Divide by count to get the mean ----
    // If a cluster has 0 points (unlikely but possible with poor initialization),
    // we leave its centroid unchanged (at zero, which is the reset value).
    for (size_t c = 0; c < k; ++c) {
        if (counts_[c] > 0) {
            for (size_t j = 0; j < d; ++j) {
                centroids_[c][j] /= counts_[c];
            }
        }
        // If counts_[c] == 0, the centroid stays at zero.
        // This is a degenerate case -- ideally all clusters have points.
    }
}

// ============================================================================
// check_convergence() -- HAS THE ALGORITHM STABILIZED?
//
// The algorithm converges when centroids stop moving significantly.
// We measure this indirectly: when inertia stops changing.
//
// Inertia = sum of squared distances from each point to its centroid.
// Lower inertia = tighter clusters = better fit.
//
// Strategy: track inertia across iterations. If the change is smaller
// than (tol * current_inertia), we've converged.
//
// Example: tol=1e-4, inertia=1000, change must be < 0.1 to converge.
// ============================================================================

bool KMeans::check_convergence() {
    // Need at least 2 iterations to compare (no "previous" inertia on iter 0).
    if (n_iter_ < 2) return false;

    // Save the old inertia before recomputing.
    float old_inertia = inertia_;

    // Compute current inertia:
    // Sum of (distance from point i to its assigned centroid)².
    // We use distances_[i][cluster] which was computed in compute_distances()
    // and the labels from assign_clusters().
    inertia_ = 0.0f;
    for (size_t i = 0; i < distances_.rows(); ++i) {
        size_t cluster = static_cast<size_t>(labels_[i]);
        inertia_ += distances_[i][cluster] * distances_[i][cluster];
    }

    // Check if the absolute change is below the tolerance threshold.
    // std::abs() gives the absolute value (ignoring sign).
    // tol * inertia_ makes it RELATIVE: 0.01% of current value.
    // This works better than absolute threshold for datasets of different scales.
    if (n_iter_ > 0 && std::abs(old_inertia - inertia_) < config_.tol * inertia_) {
        return true;  // Converged!
    }

    return false;  // Keep iterating
}

// ============================================================================
// CONFIGURATION HELPERS
// ============================================================================

void KMeans::enable_versioning(bool enable) {
    config_.enable_versioning = enable;
}

void KMeans::enable_drift_detection(bool enable) {
    config_.enable_drift_detection = enable;
}

void KMeans::set_feature_store(const std::string& path) {
    config_.feature_store_path = path;
}

} // namespace clustering
