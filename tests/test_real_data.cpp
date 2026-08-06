// ============================================================================
// test_real_data.cpp -- Golden-value tests against scikit-learn references.
//
// Compares engine output on real datasets against values produced by
// benchmarks/sklearn_benchmark.py --golden (benchmarks/data/golden.json).
// Datasets: iris (150x4), wine (178x13), synthetic_10k (10000x32).
//
// Reference values are deterministic (random_state=42). KMeans is run 5x
// (best inertia kept) to mirror sklearn's n_init=10 behavior.
// ============================================================================

#include <gtest/gtest.h>
#include "clustering/matrix.h"
#include "clustering/kmeans.h"
#include "clustering/dbscan.h"
#include "clustering/pca.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace clustering;

namespace {

// Locate a file relative to the repo root (works from build/ and repo root).
std::string find_file(const std::vector<std::string>& candidates) {
    for (const auto& c : candidates) {
        std::ifstream f(c);
        if (f.good()) return c;
    }
    return "";
}

Matrix load_csv(const std::string& path) {
    std::vector<std::vector<float>> rows;
    std::ifstream in(path);
    if (!in.good()) return Matrix();  // empty -> rows()==0
    std::string line;
    while (std::getline(in, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            char* end = nullptr;
            float v = std::strtof(cell.c_str(), &end);
            if (end == cell.c_str()) break;  // non-numeric cell: stop, keep row
            row.push_back(v);
        }
        if (!row.empty()) rows.push_back(std::move(row));
    }
    Matrix m(rows.size(), rows[0].size());
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j)
            m[i][j] = rows[i][j];
    return m;
}

// Minimal golden.json accessor: returns the string value of `key` inside the
// object block that starts after `anchor` (dataset name).
std::string golden_value(const std::string& json, const std::string& anchor,
                         const std::string& key) {
    auto pos = json.find('"' + anchor + '"');
    if (pos == std::string::npos) return "";
    auto open = json.find('{', pos);
    if (open == std::string::npos) return "";
    auto close = json.find('}', open);
    auto key_pos = json.find('"' + key + '"', open);
    if (key_pos == std::string::npos || key_pos > close) return "";
    auto colon = json.find(':', key_pos);
    if (colon == std::string::npos) return "";
    auto val_start = json.find_first_not_of(" \t\r\n", colon + 1);
    if (val_start == std::string::npos) return "";
    size_t val_end;
    if (json[val_start] == '[') {
        // Array value: scan to the matching ']' (no nested arrays in golden).
        val_end = json.find(']', val_start + 1);
        if (val_end == std::string::npos) return "";
        val_end++;
    } else {
        val_end = json.find_first_of(",}", val_start);
    }
    std::string v = json.substr(val_start, val_end - val_start);
    if (!v.empty() && v.front() == '"') {
        v = v.substr(1);
        if (!v.empty() && v.back() == '"') v.pop_back();
    }
    return v;
}

// Cluster size multiset from labels (label < 0 = noise, excluded).
std::vector<size_t> cluster_sizes(const Vector& labels, size_t n_clusters) {
    std::vector<size_t> sizes(n_clusters, 0);
    for (size_t i = 0; i < labels.size(); ++i)
        if (labels[i] >= 0) sizes[(size_t)labels[i]]++;
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

struct GoldenData {
    std::string json;
    Matrix iris, wine, syn;
    bool loaded = false;

    bool load() {
        std::string data_dir = find_file({"benchmarks/data/", "data/", "../benchmarks/data/"});
        std::string json_path = find_file({data_dir + "golden.json"});
        if (json_path.empty()) return false;
        std::ifstream f(json_path);
        std::stringstream ss;
        ss << f.rdbuf();
        json = ss.str();
        iris = load_csv(find_file({data_dir + "iris.csv"}));
        wine = load_csv(find_file({data_dir + "wine.csv"}));
        syn = load_csv(find_file({data_dir + "synthetic_10k_32d.csv"}));
        loaded = iris.rows() > 0 && wine.rows() > 0 && syn.rows() > 0;
        return loaded;
    }
};

double best_kmeans_inertia(const Matrix& X, size_t k) {
    double best = std::numeric_limits<double>::max();
    KMeansConfig cfg;
    cfg.k = k;
    cfg.max_threads = 4;
    for (int run = 0; run < 5; ++run) {
        KMeans km(cfg);
        km.fit(X);
        best = std::min(best, (double)km.inertia());
    }
    return best;
}

} // namespace

class RealDataTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!golden.load()) {
            GTEST_SKIP() << "golden.json / data CSVs not found; run "
                            "`python3 benchmarks/sklearn_benchmark.py --golden` from repo root";
        }
    }
    GoldenData golden;
};

TEST_F(RealDataTest, KMeansInertiaIris) {
    double ref = std::stod(golden_value(golden.json, "iris", "kmeans_inertia"));
    double got = best_kmeans_inertia(golden.iris, 3);
    EXPECT_NEAR(got, ref, ref * 0.02) << "iris k=3 inertia vs sklearn";
}

TEST_F(RealDataTest, KMeansInertiaWine) {
    double ref = std::stod(golden_value(golden.json, "wine", "kmeans_inertia"));
    double got = best_kmeans_inertia(golden.wine, 3);
    EXPECT_NEAR(got, ref, ref * 0.02) << "wine k=3 inertia vs sklearn";
}

TEST_F(RealDataTest, KMeansInertiaSynthetic10k) {
    double ref = std::stod(golden_value(golden.json, "synthetic_10k", "kmeans_inertia"));
    double got = best_kmeans_inertia(golden.syn, 10);
    EXPECT_NEAR(got, ref, ref * 0.02) << "synthetic_10k k=10 inertia vs sklearn";
}

TEST_F(RealDataTest, KMeansClusterSizesIris) {
    // Cluster partition shape (multiset of sizes) should match sklearn.
    double ref_inertia = std::stod(golden_value(golden.json, "iris", "kmeans_inertia"));
    KMeansConfig cfg;
    cfg.k = 3;
    cfg.max_threads = 4;
    KMeans best_km(cfg);
    double best = std::numeric_limits<double>::max();
    for (int run = 0; run < 5; ++run) {
        KMeans km(cfg);
        km.fit(golden.iris);
        if (km.inertia() < best) { best = km.inertia(); best_km = std::move(km); }
    }
    auto ref_sizes_str = golden_value(golden.json, "iris", "cluster_sizes");
    // Parse "[38, 50, 62]" into a multiset.
    std::vector<size_t> ref_sizes;
    size_t pos = 0;
    std::string s = ref_sizes_str;
    while ((pos = s.find_first_of("0123456789", pos)) != std::string::npos) {
        ref_sizes.push_back(std::stoul(s.substr(pos)));
        pos = s.find_first_of(",]", pos);
        if (pos == std::string::npos) break;
        pos++;
    }
    std::sort(ref_sizes.begin(), ref_sizes.end());
    auto got_sizes = cluster_sizes(best_km.labels(), 3);
    EXPECT_EQ(got_sizes, ref_sizes) << "iris cluster size multiset vs sklearn "
                                    << "(inertia " << best << " vs " << ref_inertia << ")";
}

TEST_F(RealDataTest, DBSCANIrisMatchesSklearn) {
    int ref_clusters = std::stoi(golden_value(golden.json, "iris", "dbscan_n_clusters"));
    int ref_noise = std::stoi(golden_value(golden.json, "iris", "dbscan_n_noise"));
    DBSCAN db;
    db.set_epsilon(0.5f);
    db.set_min_pts(5);
    db.fit(golden.iris);
    EXPECT_EQ((int)db.n_clusters(), ref_clusters)
        << "iris DBSCAN cluster count vs sklearn";
    // Border-point tie-breaking can differ slightly; allow small noise delta.
    EXPECT_LE(std::abs((int)db.n_noise() - ref_noise), 5)
        << "iris DBSCAN noise count vs sklearn (got " << db.n_noise()
        << ", ref " << ref_noise << ")";
}

TEST_F(RealDataTest, DBSCANWineBothNoise) {
    // Raw-scale wine is all noise for eps=0.5: no clusters in either engine.
    DBSCAN db;
    db.set_epsilon(0.5f);
    db.set_min_pts(5);
    db.fit(golden.wine);
    EXPECT_EQ((int)db.n_clusters(), 0);
    EXPECT_EQ(db.n_noise(), golden.wine.rows());
}

TEST_F(RealDataTest, PCAVarianceIris) {
    double ref = std::stod(golden_value(golden.json, "iris", "pca2_variance"));
    PCA pca(2);
    pca.fit(golden.iris);
    double got = 100.0 * (double)(pca.explained_variance_ratio()[0] +
                                  pca.explained_variance_ratio()[1]);
    EXPECT_NEAR(got, ref, 0.2) << "iris PCA 2-component variance vs sklearn";
}

TEST_F(RealDataTest, PCAVarianceSynthetic10k) {
    double ref = std::stod(golden_value(golden.json, "synthetic_10k", "pca2_variance"));
    PCA pca(2);
    pca.fit(golden.syn);
    double got = 100.0 * (double)(pca.explained_variance_ratio()[0] +
                                  pca.explained_variance_ratio()[1]);
    EXPECT_NEAR(got, ref, 0.2) << "synthetic_10k PCA 2-component variance vs sklearn";
}
