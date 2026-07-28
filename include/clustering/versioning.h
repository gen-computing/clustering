#pragma once

#include "matrix.h"
#include <vector>
#include <string>
#include <chrono>   // For timestamps (std::chrono::system_clock)

namespace clustering {

// ============================================================================
// VersionManager -- Save and load clustering results (centroids + labels).
//
// WHAT IT DOES:
//   Like "git for centroids". After each fit(), you can save the centroids
//   and cluster assignments to disk. Later you can list all saved versions,
//   load a specific version, or rollback to a previous one.
//
// WHY YOU NEED IT:
//   - Compare different runs (did v3's centroids improve over v2?)
//   - Rollback if a new model performs worse
//   - Audit trail (track model evolution over time)
//   - A/B testing (load different versions for comparison)
//
// STORAGE FORMAT:
//   Each version is saved as a binary file: {version_id}.bin
//   File structure:
//     [k: size_t] [d: size_t] [n: size_t]  -- metadata (cluster count, dims, labels count)
//     [centroid data: k*d floats]           -- centroid coordinates
//     [label data: n floats]                -- cluster assignments
//   Binary format is fast to read/write but not human-readable.
// ============================================================================
class VersionManager {
public:
    VersionManager();
    ~VersionManager();

    // set_storage_path: Where to save/load version files.
    // Creates the directory if it doesn't exist.
    // Scans existing .bin files to populate the version list.
    // Example: vm.set_storage_path("./models/clustering_versions");
    void set_storage_path(const std::string& path);

    // save: Write current centroids and labels to disk.
    // Returns: The new version ID (auto-incremented, starting at 1).
    //          Returns 0 if file could not be opened.
    // Parameters:
    //   centroids -- k rows x d columns matrix of cluster centers
    //   labels    -- n-element vector of cluster assignments
    size_t save(const Matrix& centroids, const Vector& labels);

    // load: Read a saved version from disk into existing matrices.
    // Returns: true if successful, false if version not found or file corrupt.
    // Parameters:
    //   version_id -- which version to load (returned by save())
    //   centroids  -- output: filled with loaded centroid data
    //   labels     -- output: filled with loaded label data
    bool load(size_t version_id, Matrix& centroids, Vector& labels) const;

    // list_versions: Get all saved version IDs in ascending order.
    // Example: auto versions = vm.list_versions(); // [1, 2, 3, 5]
    std::vector<size_t> list_versions() const;

    // latest_version: Get the highest version ID.
    // Returns 0 if no versions exist.
    size_t latest_version() const;

    // has_versions: Check if any versions have been saved.
    bool has_versions() const;

private:
    std::string storage_path_;        // Directory where .bin files are stored
    std::vector<size_t> versions_;    // Cached list of available version IDs
};

// ============================================================================
// Version -- Full version record with metadata (for potential future use).
//
// Stores not just centroids/labels but also timestamp and inertia.
// Useful for tracking model quality over time.
// ============================================================================
struct Version {
    size_t id;                                              // Version number
    Matrix centroids;                                       // Cluster centers
    Vector labels;                                          // Point assignments
    std::chrono::system_clock::time_point timestamp;        // When was this saved?
    float inertia;                                          // Clustering quality score
};

} // namespace clustering
