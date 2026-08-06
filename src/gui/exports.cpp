#include "exports.h"
#include "data_table.h"
#include <cstdio>

namespace clustering {

bool export_labels_csv(const DataTable& table, const Vector& labels, const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    fprintf(f, "id,cluster\n");
    for (size_t i = 0; i < labels.size(); ++i)
        fprintf(f, "%zu,%.0f\n", i, labels[i]);
    fclose(f);
    return true;
}

bool export_centroids_csv(const Matrix& centroids, const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    for (size_t j = 0; j < centroids.cols(); ++j) {
        if (j > 0) fprintf(f, ",");
        fprintf(f, "f%zu", j);
    }
    fprintf(f, "\n");
    for (size_t i = 0; i < centroids.rows(); ++i) {
        for (size_t j = 0; j < centroids.cols(); ++j) {
            if (j > 0) fprintf(f, ",");
            fprintf(f, "%f", centroids[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return true;
}

bool export_preprocessed_csv(const DataTable& table, const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;

    for (size_t j = 0; j < table.cols(); ++j) {
        if (j > 0) fprintf(f, ",");
        fprintf(f, "%s", table.column_names()[j].c_str());
    }
    fprintf(f, "\n");

    for (size_t i = 0; i < table.rows(); ++i) {
        for (size_t j = 0; j < table.cols(); ++j) {
            if (j > 0) fprintf(f, ",");
            if (!table.is_missing(i, j))
                fprintf(f, "%f", table.data()[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return true;
}

bool export_report(const DataTable& table, const std::vector<int>& selected_cols,
                   int algo, int k, float inertia, int n_iter,
                   float silhouette, float davies_bouldin,
                   bool clustering_done, const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;

    fprintf(f, "Clustering Engine Report\n");
    fprintf(f, "========================\n\n");
    fprintf(f, "Dataset: %zu rows x %zu cols\n", table.rows(), table.cols());
    fprintf(f, "Selected columns: ");
    for (size_t j = 0; j < selected_cols.size() && j < table.cols(); ++j)
        if (selected_cols[j]) fprintf(f, "%s ", table.column_names()[j].c_str());
    fprintf(f, "\n\n");

    if (clustering_done) {
        fprintf(f, "Algorithm: %d  k: %d\n", algo, k);
        fprintf(f, "Inertia: %.4f  Iterations: %d\n", inertia, n_iter);
        fprintf(f, "Silhouette: %.4f  Davies-Bouldin: %.4f\n", silhouette, davies_bouldin);
    }
    fclose(f);
    return true;
}

} // namespace clustering
