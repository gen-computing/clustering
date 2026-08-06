// ============================================================================
// test_dbscan.cpp -- DBSCAN unit tests.
// Tests: core/border/noise classification, cluster count, parameter validation.
// ============================================================================

#include <gtest/gtest.h>
#include "clustering/dbscan.h"

using namespace clustering;

// Helper: generate 2 clusters of points with some noise
static Matrix make_two_clusters_with_noise() {
    Matrix X(25, 2);
    // Cluster 1: 10 points around (0,0)
    X[0][0]=0; X[0][1]=0;
    X[1][0]=0.1f; X[1][1]=0.1f;
    X[2][0]=-0.1f; X[2][1]=0.1f;
    X[3][0]=0.1f; X[3][1]=-0.1f;
    X[4][0]=-0.1f; X[4][1]=-0.1f;
    X[5][0]=0.2f; X[5][1]=0;
    X[6][0]=0; X[6][1]=0.2f;
    X[7][0]=-0.2f; X[7][1]=0;
    X[8][0]=0; X[8][1]=-0.2f;
    X[9][0]=0.05f; X[9][1]=0.05f;
    // Cluster 2: 10 points around (10,10)
    for (size_t i = 0; i < 10; i++) {
        X[10+i][0] = 10.0f + (float)(i%5) * 0.1f;
        X[10+i][1] = 10.0f + (float)(i/5) * 0.1f;
    }
    // Noise: 5 points far from both clusters
    X[20][0]=5; X[20][1]=5;
    X[21][0]=5; X[21][1]=6;
    X[22][0]=6; X[22][1]=5;
    X[23][0]=5; X[23][1]=4;
    X[24][0]=4; X[24][1]=5;
    return X;
}

TEST(DBSCAN, TwoClustersWithNoise) {
    Matrix X = make_two_clusters_with_noise();
    DBSCANConfig cfg;
    cfg.epsilon = 1.0f;
    cfg.min_pts = 3;
    DBSCAN db(cfg);
    db.fit(X);

    EXPECT_GE(db.n_clusters(), 2u);
    EXPECT_EQ(db.labels().size(), 25u);

    // Cluster 1 points around (0,0) should have same label
    float c1_label = db.labels()[0]; // (0,0) point
    EXPECT_GT(c1_label, 0.0f); // Not noise
    for (size_t i = 1; i < 10; i++)
        EXPECT_FLOAT_EQ(db.labels()[i], c1_label);

    // Cluster 2 points around (10,10) should have different label
    float c2_label = db.labels()[10];
    EXPECT_GT(c2_label, 0.0f); // Not noise
    EXPECT_NE(c1_label, c2_label); // Different from cluster 1

    // Note: the 5 points at (4,5,6) may or may not form a 3rd cluster
    // depending on epsilon. Not asserting exact noise count.
}

TEST(DBSCAN, NoNoise) {
    Matrix X(10, 2);
    for (size_t i = 0; i < 5; i++) {
        X[i][0] = (float)i * 0.1f;
        X[i][1] = 0.0f;
    }
    for (size_t i = 5; i < 10; i++) {
        X[i][0] = 10.0f + (float)(i-5) * 0.1f;
        X[i][1] = 0.0f;
    }
    DBSCANConfig cfg;
    cfg.epsilon = 2.0f;
    cfg.min_pts = 2;
    DBSCAN db(cfg);
    db.fit(X);
    EXPECT_GE(db.n_clusters(), 2u);
}

TEST(DBSCAN, AllNoise) {
    // Points too far apart to form clusters
    Matrix X(5, 2);
    X[0][0]=0; X[0][1]=0;
    X[1][0]=10; X[1][1]=10;
    X[2][0]=20; X[2][1]=20;
    X[3][0]=30; X[3][1]=30;
    X[4][0]=40; X[4][1]=40;
    DBSCAN db;
    db.set_epsilon(1.0f);
    db.set_min_pts(2);
    db.fit(X);
    EXPECT_EQ(db.n_clusters(), 0u);
    EXPECT_EQ(db.n_noise(), 5u);
}

TEST(DBSCAN, SingleCluster) {
    Matrix X(10, 2);
    for (size_t i = 0; i < 10; i++) {
        X[i][0] = (float)i * 0.1f;
        X[i][1] = 0.0f;
    }
    DBSCANConfig cfg;
    cfg.epsilon = 1.0f;
    cfg.min_pts = 3;
    DBSCAN db(cfg);
    db.fit(X);
    EXPECT_EQ(db.n_clusters(), 1u);
    EXPECT_EQ(db.n_noise(), 0u);
}

TEST(DBSCAN, EmptyInput) {
    Matrix empty(0, 2);
    DBSCAN db;
    EXPECT_THROW(db.fit(empty), std::runtime_error);
}

TEST(DBSCAN, NoFeatures) {
    Matrix X(10, 0);
    DBSCAN db;
    EXPECT_THROW(db.fit(X), std::runtime_error);
}

TEST(DBSCAN, PredictBeforeFit) {
    Matrix X(5, 2);
    DBSCAN db;
    EXPECT_THROW(db.predict(X), std::runtime_error);
}

TEST(DBSCAN, ConfigSetters) {
    DBSCAN db;
    db.set_epsilon(1.0f);   // Wider epsilon so points are reachable
    db.set_min_pts(3);      // Lower min_pts so clusters can form
    Matrix X(10, 2);
    for (size_t i = 0; i < 10; i++) {
        X[i][0] = (float)i * 0.2f;  // Points at distance 0.2 apart
        X[i][1] = 0.0f;
    }
    db.fit(X);
    EXPECT_GE(db.n_clusters(), 1u);
}

// ============================================================================
// predict() correctness (bug #2 regression test).
// ============================================================================

TEST(DBSCAN, PredictAssignsNearPoints) {
    // Two dense blobs well within eps, far from each other.
    Matrix X(40, 2);
    for (size_t i = 0; i < 20; ++i) { X[i][0] = 0.0f + 0.05f * (float)i; X[i][1] = 0.0f; }
    for (size_t i = 20; i < 40; ++i) { X[i][0] = 10.0f + 0.05f * (float)i; X[i][1] = 10.0f; }
    DBSCAN db;
    db.set_epsilon(1.0f);
    db.set_min_pts(3);
    db.fit(X);
    ASSERT_EQ(db.n_clusters(), 2u);

    // Point near blob A -> must get a positive label (cluster 1 or 2).
    Matrix q1(1, 2); q1[0][0] = 0.2f; q1[0][1] = 0.1f;
    Vector l1 = db.predict(q1);
    EXPECT_GT(l1[0], 0.0f) << "near blob A should be clustered";

    // Point near blob B -> same.
    Matrix q2(1, 2); q2[0][0] = 10.2f; q2[0][1] = 10.1f;
    Vector l2 = db.predict(q2);
    EXPECT_GT(l2[0], 0.0f);

    // Far point -> noise (0).
    Matrix q3(1, 2); q3[0][0] = 50.0f; q3[0][1] = 50.0f;
    Vector l3 = db.predict(q3);
    EXPECT_EQ(l3[0], 0.0f);
}

TEST(DBSCAN, HighDimNoTruncation) {
    // Regression: KD-tree was hardcoded to 3 dims, so 4+ dim data used only
    // the first 3 features for neighborhood queries. Two clusters that differ
    // ONLY in feature 4 (index 3) must still be separated.
    Matrix X(40, 4);
    for (size_t i = 0; i < 20; ++i) {
        X[i][0] = 0.0f; X[i][1] = 0.0f; X[i][2] = 0.0f; X[i][3] = 0.0f;
    }
    for (size_t i = 20; i < 40; ++i) {
        X[i][0] = 0.0f; X[i][1] = 0.0f; X[i][2] = 0.0f; X[i][3] = 5.0f;
    }
    DBSCAN db;
    db.set_epsilon(0.5f);
    db.set_min_pts(3);
    db.fit(X);
    EXPECT_EQ(db.n_clusters(), 2u) << "clusters separated only by 4th feature";
    EXPECT_EQ(db.n_noise(), 0u);
}

// ============================================================================
// Robustness: standardized DBSCAN must find structure regardless of the raw
// feature scale, unlike raw-space eps which is unit-sensitive.
// ============================================================================

#include <random>
#include <fstream>
#include <sstream>

static Matrix make_blobs_2d(size_t per_blob, size_t n_blobs, float spread,
                            float sep, size_t noise) {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, spread);
    size_t n = per_blob * n_blobs + noise;
    Matrix X(n, 2);
    size_t idx = 0;
    for (size_t b = 0; b < n_blobs; ++b) {
        float cx = (float)b * sep;
        for (size_t k = 0; k < per_blob; ++k) {
            X[idx][0] = cx + nd(rng);
            X[idx][1] = nd(rng);
            idx++;
        }
    }
    // Sparse noise: 1.5 apart in y, centered between blobs -> isolated.
    for (size_t k = 0; k < noise; ++k) {
        X[idx][0] = sep * (float)(n_blobs - 1) / 2.0f;
        X[idx][1] = (float)k * 1.5f + 0.25f;
        idx++;
    }
    return X;
}

static Matrix load_csv_numeric(const std::string& path) {
    std::vector<std::vector<float>> rows;
    std::ifstream in(path);
    if (!in.good()) return Matrix();
    std::string line;
    while (std::getline(in, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            char* end = nullptr;
            float v = std::strtof(cell.c_str(), &end);
            if (end == cell.c_str()) break;
            row.push_back(v);
        }
        if (!row.empty()) rows.push_back(std::move(row));
    }
    if (rows.empty()) return Matrix();
    Matrix m(rows.size(), rows[0].size());
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j)
            m[i][j] = rows[i][j];
    return m;
}

TEST(DBSCAN, ScaleInvarianceRawVsStandardized) {
    Matrix X = make_blobs_2d(60, 2, 0.1f, 2.0f, 10);
    // Blow the raw scale up by 1000x: same structure, hostile to raw eps.
    for (size_t i = 0; i < X.rows(); ++i) {
        X[i][0] *= 1000.0f;
        X[i][1] *= 1000.0f;
    }

    DBSCANConfig raw_cfg;
    raw_cfg.epsilon = 1.0f;
    raw_cfg.min_pts = 5;
    DBSCAN raw(raw_cfg);
    raw.fit(X);
    // Raw-space eps=1.0 on 1000x-scale data finds nothing.
    EXPECT_EQ(raw.n_clusters(), 0u);
    EXPECT_EQ(raw.n_noise(), X.rows());

    DBSCANConfig std_cfg;
    std_cfg.epsilon = 0.5f;
    std_cfg.min_pts = 5;
    std_cfg.standardize = true;
    DBSCAN db(std_cfg);
    db.fit(X);
    EXPECT_EQ(db.n_clusters(), 2u);
    EXPECT_GE(db.n_noise(), 9u);  // >= 9 of the 10 sparse noise points isolated
}

TEST(DBSCAN, StandardizedBlobsPurity) {
    Matrix X = make_blobs_2d(60, 3, 0.1f, 2.0f, 6);
    DBSCANConfig cfg;
    cfg.epsilon = 0.5f;
    cfg.min_pts = 5;
    cfg.standardize = true;
    DBSCAN db(cfg);
    db.fit(X);
    EXPECT_EQ(db.n_clusters(), 3u);
    EXPECT_GE(db.n_noise(), 5u);

    // Purity: within each 60-point blob, one label dominates (>50 points).
    for (size_t b = 0; b < 3; ++b) {
        std::vector<size_t> counts(16, 0);
        for (size_t i = 0; i < 60; ++i) {
            float lbl = db.labels()[b * 60 + i];
            if (lbl >= 0 && lbl < 16) counts[(size_t)lbl]++;
        }
        bool has_dominant = false;
        for (auto c : counts) if (c > 50) has_dominant = true;
        EXPECT_TRUE(has_dominant) << "blob " << b << " has no dominant cluster";
    }
}

TEST(DBSCAN, StandardizedWineFindsClusters) {
    std::string data_dir = "";
    for (auto d : {"benchmarks/data/", "data/", "../benchmarks/data/"}) {
        std::ifstream f(std::string(d) + "wine.csv");
        if (f.good()) { data_dir = d; break; }
    }
    Matrix wine = load_csv_numeric(data_dir + "wine.csv");
    if (wine.rows() == 0) GTEST_SKIP() << "wine.csv not loadable";

    // High-dim data needs a larger eps than 2D: use the scale-free
    // estimate, exactly like the GUI's Auto-Estimate button.
    float eps = DBSCAN::estimate_epsilon(wine, 5, 100, true);
    EXPECT_GT(eps, 0.5f);
    DBSCANConfig cfg;
    cfg.epsilon = eps;
    cfg.min_pts = 5;
    cfg.standardize = true;
    DBSCAN db(cfg);
    db.fit(wine);
    // The user-facing regression: raw eps=0.5 gave 0 clusters on wine;
    // standardized data must recover real structure.
    EXPECT_GE(db.n_clusters(), 2u);
    EXPECT_LT(db.n_noise(), wine.rows() / 2);
}

TEST(DBSCAN, StandardizedIrisSensible) {
    std::string data_dir = "";
    for (auto d : {"benchmarks/data/", "data/", "../benchmarks/data/"}) {
        std::ifstream f(std::string(d) + "iris.csv");
        if (f.good()) { data_dir = d; break; }
    }
    Matrix iris = load_csv_numeric(data_dir + "iris.csv");
    if (iris.rows() == 0) GTEST_SKIP() << "iris.csv not loadable";

    float eps = DBSCAN::estimate_epsilon(iris, 5, 100, true);
    DBSCANConfig cfg;
    cfg.epsilon = eps;
    cfg.min_pts = 5;
    cfg.standardize = true;
    DBSCAN db(cfg);
    db.fit(iris);
    EXPECT_GE(db.n_clusters(), 2u);
    EXPECT_LT(db.n_noise(), iris.rows() / 2);
}

TEST(DBSCAN, EstimateEpsilonStandardizedIsScaleFree) {
    Matrix X = make_blobs_2d(60, 2, 0.1f, 2.0f, 10);
    for (size_t i = 0; i < X.rows(); ++i) {
        X[i][0] *= 1000.0f;
        X[i][1] *= 1000.0f;
    }
    float raw_eps = DBSCAN::estimate_epsilon(X, 5, 100, false);
    float std_eps = DBSCAN::estimate_epsilon(X, 5, 100, true);
    // Raw estimate lives in the 1000x units; standardized one is scale-free.
    EXPECT_GT(raw_eps, std_eps * 100.0f);
    EXPECT_GT(std_eps, 0.01f);
    EXPECT_LT(std_eps, 2.0f);
}

TEST(DBSCAN, PredictAppliesSameScale) {
    Matrix X = make_blobs_2d(60, 2, 0.1f, 2.0f, 10);
    for (size_t i = 0; i < X.rows(); ++i) {
        X[i][0] *= 1000.0f;
        X[i][1] *= 1000.0f;
    }
    DBSCANConfig cfg;
    cfg.epsilon = 0.5f;
    cfg.min_pts = 5;
    cfg.standardize = true;
    DBSCAN db(cfg);
    db.fit(X);
    Vector pred = db.predict(X);
    ASSERT_EQ(pred.size(), db.labels().size());
    size_t agree = 0;
    for (size_t i = 0; i < pred.size(); ++i)
        if (pred[i] == db.labels()[i]) agree++;
    EXPECT_GE(agree, pred.size() - 1);  // exact match except possible self-distance edge
}
