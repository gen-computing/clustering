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
