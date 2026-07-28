#pragma once

#include "matrix.h"
#include <vector>     // std::vector for dynamic arrays
#include <cstddef>    // size_t for sizes
#include <functional> // std::function for callbacks
#include <string>     // std::string for file paths

namespace clustering {

// ============================================================================
// KMeansConfig -- Settings that control how the KMeans algorithm behaves.
//
// Pass this to the KMeans constructor to customize the algorithm.
// You can also just pass a number (k=5) for the simplest case.
// ============================================================================
struct KMeansConfig {
    size_t k = 8;                     // Number of clusters to find.
                                      // How many groups should the data be split into?
                                      // Example: k=5 splits data into 5 groups.

    size_t max_iter = 300;            // Maximum number of iterations (passes over data).
                                      // Algorithm stops early if converged before this limit.
                                      // Prevents infinite loops.

    float tol = 1e-4f;              // Convergence tolerance (1e-4 means 0.0001).
                                      // If centroids change by less than this amount
                                      // between iterations, the algorithm stops.
                                      // Smaller = more precise but slower.

    size_t max_threads = 0;           // Number of CPU threads to use for parallel work.
                                      // 0 = auto-detect (uses number of CPU cores, capped at 8).
                                      // 1 = single-threaded. 4 = use exactly 4 threads.

    bool enable_versioning = false;   // If true, saves centroids to disk after each fit.
                                      // Allows you to rollback to previous clustering results.

    bool enable_drift_detection = false; // If true, monitors for "concept drift"
                                         // (when the data pattern changes over time).

    std::string feature_store_path;   // Directory path for caching preprocessed data.
                                       // Empty = no caching.

    // Iteration callback: called after each clustering iteration.
    // Receives (iter, centroids, labels). Return true to stop early.
    // Used for real-time visualization in the GUI.
    std::function<bool(size_t, const Matrix&, const Vector&)> iter_callback;
};

// ============================================================================
// KMeans -- The main clustering algorithm.
//
// WHAT IT DOES:
//   Given a set of points (data), KMeans finds K groups (clusters) where
//   points in the same group are close together, and points in different
//   groups are far apart.
//
// HOW IT WORKS (simplified):
//   1. Pick K random starting points as "centroids" (cluster centers)
//   2. Assign every data point to its nearest centroid
//   3. Move each centroid to the average (mean) of all points assigned to it
//   4. Repeat steps 2-3 until centroids stop moving (convergence)
//
// REAL WORLD EXAMPLE:
//   You have customer data (age, income, spending). You want to group
//   them into 5 types. KMeans finds the 5 natural clusters.
//
// KEY METHODS:
//   fit(X)       -- train the model on data matrix X
//   predict(X)   -- for new data, tell which cluster each point belongs to
//   labels()     -- after fitting, get the cluster assignment for each point
//   centroids()  -- get the center point of each cluster
//   inertia()    -- total squared distance from each point to its centroid
//                   (lower = tighter clusters = better fit)
// ============================================================================
class KMeans {
public:
    // ----- CONSTRUCTORS -----

    // Simple constructor: just specify how many clusters (k).
    // Other settings use defaults (max 300 iterations, auto threads, etc.)
    // Example: KMeans km(5);  // find 5 clusters
    explicit KMeans(size_t k = 8);

    // Advanced constructor: pass a full KMeansConfig with custom settings.
    // Example: KMeansConfig cfg; cfg.k = 10; cfg.max_iter = 500; KMeans km(cfg);
    explicit KMeans(const KMeansConfig& config);

    // Virtual destructor: Important because OnlineKMeans inherits from KMeans.
    // "virtual" ensures the correct destructor chain is called when deleting
    // through a base class pointer. Without it, memory leaks can happen.
    virtual ~KMeans();

    // ----- CORE METHODS -----

    // fit: TRAIN the model on data.
    // Parameter X: Matrix where each row = one data point, each column = one feature.
    //              Must have at least k rows (more points than clusters).
    // After calling fit(), you can use labels(), centroids(), inertia(), predict().
    // Throws: runtime_error if X is empty, has 0 features, k=0, or k > number of rows.
    void fit(const Matrix& X);

    // predict: Assign existing or NEW data points to the nearest cluster.
    // Returns: Vector of length X.rows(), each element is the cluster ID (0 to k-1).
    //          Cluster ID is stored as float (e.g., 2.0 means cluster 2).
    // Throws: runtime_error if called before fit() (model not yet trained).
    // Example: Vector preds = km.predict(new_data);  // which cluster for each new point?
    Vector predict(const Matrix& X) const;

    // partial_fit: Update the model INCREMENTALLY with new data.
    // Unlike fit() which starts fresh, partial_fit ADJUSTS existing centroids.
    // Uses a decaying learning rate (newer points have less influence as time goes on).
    // If model not yet fitted, falls back to full fit(X).
    // VIRTUAL: OnlineKMeans and MiniBatchKMeans override this with specialized logic.
    virtual void partial_fit(const Matrix& X);

    // ----- RESULT ACCESSORS (read results after fitting) -----

    // labels: Which cluster (0 to k-1) each data point belongs to.
    // After fit(), labels().size() == number of input points.
    const Vector& labels() const { return labels_; }

    // centroids: The center point of each cluster.
    // centroids().rows() == k, centroids().cols() == number of features.
    // centroids()[2][0] = first feature of cluster 2's center.
    const Matrix& centroids() const { return centroids_; }

    // centroids_mut: Mutable (writable) access to centroids.
    // Used internally by OnlineKMeans to update centroids point-by-point.
    Matrix& centroids_mut() { return centroids_; }

    // n_iter: How many iterations the algorithm ran before converging.
    // Lower is usually better (means centroids stabilized quickly).
    size_t n_iter() const { return n_iter_; }

    // inertia: Sum of squared distances from each point to its assigned centroid.
    // This is the algorithm's "cost function" -- the number it tries to minimize.
    // Lower inertia = tighter, more compact clusters.
    // Comparing inertia between different runs with same k: lower is better.
    float inertia() const { return inertia_; }

    // ----- CONFIGURATION -----

    void enable_versioning(bool enable);
    void enable_drift_detection(bool enable);
    void set_feature_store(const std::string& path);

protected:
    // Protected methods: accessible by this class and subclasses (OnlineKMeans, MiniBatchKMeans),
    // but NOT by external code. These are the internal building blocks of the algorithm.

    // initialize_centroids: Pick k starting points using KMeans++ strategy.
    // KMeans++ picks centroids that are far apart, leading to better/faster convergence
    // than pure random selection. First centroid is random. Each subsequent centroid
    // is chosen with probability proportional to its distance from the nearest
    // already-chosen centroid.
    void initialize_centroids(const Matrix& X);

    // compute_distances: Calculate distance from every data point to every centroid.
    // Fills the distances_ matrix (n_samples rows x k columns).
    // Uses AVX2 SIMD for speed and ThreadPool for parallel execution.
    // distance_matrix[i][j] = Euclidean distance from point i to centroid j.
    void compute_distances(const Matrix& X);

    // assign_clusters: For each data point, find which centroid is nearest.
    // Writes results to labels_ (the cluster assignment per point)
    // and counts_ (how many points in each cluster).
    // This is the "E-step" (Expectation step) of the algorithm.
    void assign_clusters();

    // update_centroids: Move each centroid to the average position of all points
    // assigned to it. If a cluster has ZERO points, the centroid stays where it was.
    // This is the "M-step" (Maximization step) of the algorithm.
    void update_centroids(const Matrix& X);

    // check_convergence: Determine if the algorithm should stop.
    // Compares current inertia to previous inertia. If the change is smaller
    // than config_.tol * inertia_, returns true (converged).
    // Returns false for the first 2 iterations (not enough history to compare).
    bool check_convergence();

    // ----- MEMBER VARIABLES -----

    KMeansConfig config_;    // All settings packed together
    Matrix centroids_;       // k rows x d columns -- the cluster centers
    Vector labels_;          // n elements -- which cluster each point belongs to
    Vector counts_;          // k elements -- how many points in each cluster
    Matrix distances_;       // n x k matrix -- distance from each point to each centroid
    size_t n_iter_;          // Counter: how many iterations we've done
    float inertia_;          // Current total squared distance (the cost being minimized)
    bool fitted_;            // Has fit() been called successfully? Guards predict().
};

} // namespace clustering
