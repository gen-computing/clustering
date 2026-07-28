// ============================================================================
// Rollback utilities -- Convenience wrappers around VersionManager.
//
// Rollback: restoring a previously saved clustering model (centroids + labels)
// when a newer model performs worse or when you want to compare versions.
//
// These are thin wrappers. The actual save/load logic is in VersionManager.
// This split keeps the public API clean: VersionManager handles storage,
// rollback functions handle the "restore model" workflow.
//
// USE CASE EXAMPLE:
//   1. Train model, save as version 1.
//   2. Get new data, retrain, save as version 2.
//   3. Evaluate version 2 -- it's worse! (lower silhouette, higher inertia).
//   4. Rollback to version 1: load version 1's centroids and labels.
//   5. Continue with the better model.
// ============================================================================

#include "clustering/versioning.h"
#include <stdexcept>

namespace clustering {

// Rollback functionality is integrated into VersionManager.
// This file provides additional rollback utilities.

// ============================================================================
// rollback_to_version() -- RESTORE A PREVIOUS MODEL STATE
//
// Thin convenience wrapper: loads version_id's centroids and labels from
// the VersionManager's storage.
//
// Parameters:
//   vm          -- VersionManager with storage_path already set
//   version_id  -- which version to restore (returned by vm.save())
//   centroids   -- output: filled with loaded centroid data
//   labels      -- output: filled with loaded label data
//
// Returns: true if successful, false if version not found.
//
// Example usage in a KMeans workflow:
//   rollback_to_version(vm, version_id, engine.centroids_mut(), labels);
//   // Now engine's state is restored to that version.
// ============================================================================ 
bool rollback_to_version(
    VersionManager& vm,
    size_t version_id,
    Matrix& centroids,
    Vector& labels
) {
    return vm.load(version_id, centroids, labels);
}

// ============================================================================
// get_available_versions() -- LIST ALL SAVED VERSIONS
//
// Convenience wrapper: returns all version IDs available for rollback.
// Useful for building a version history UI or selecting which version to restore.
//
// Example: auto versions = get_available_versions(vm);
//          // versions = [1, 2, 3, 4]
// ============================================================================
std::vector<size_t> get_available_versions(const VersionManager& vm) {
    return vm.list_versions();
}

} // namespace clustering
