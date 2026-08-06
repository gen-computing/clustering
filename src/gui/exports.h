#pragma once

#include "clustering/matrix.h"
#include <string>
#include <vector>

namespace clustering {

class DataTable;

// Write cluster labels as "id,cluster" CSV rows. Returns true on success.
bool export_labels_csv(const DataTable& table, const Vector& labels, const std::string& path);

// Write centroids with a "f0,f1,..." header. Returns true on success.
bool export_centroids_csv(const Matrix& centroids, const std::string& path);

// Write the preprocessed table (column-name header, empty cell for missing values).
bool export_preprocessed_csv(const DataTable& table, const std::string& path);

// Write a human-readable text report. Returns true on success.
bool export_report(const DataTable& table, const std::vector<int>& selected_cols,
                   int algo, int k, float inertia, int n_iter,
                   float silhouette, float davies_bouldin,
                   bool clustering_done, const std::string& path);

} // namespace clustering
