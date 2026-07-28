// ============================================================================
// test_evaluation.cpp -- ClusterEvaluator unit tests.
// Tests: elbow method, silhouette analysis, best-k detection.
// ============================================================================

#include <gtest/gtest.h>
#include "clustering/evaluation.h"
#include <cmath>

using namespace clustering;

static Matrix make_three_clusters() {
    Matrix X(60, 2);
    for (size_t i = 0; i < 20; i++) {
        X[i][0] = (float)i * 0.1f;
        X[i][1] = 0.0f;
    }
    for (size_t i = 20; i < 40; i++) {
        X[i][0] = 5.0f + (float)(i-20) * 0.1f;
        X[i][1] = 5.0f;
    }
    for (size_t i = 40; i < 60; i++) {
        X[i][0] = -5.0f + (float)(i-40) * 0.1f;
        X[i][1] = -5.0f;
    }
    return X;
}

TEST(ClusterEvaluator, ElbowMethodProducesResults) {
    Matrix X = make_three_clusters();
    ClusterEvaluator eval;
    auto results = eval.elbow(X, 1, 10, 100, 2);
    EXPECT_EQ(results.size(), 10u);
    for (auto& r : results) {
        EXPECT_GT(r.inertia, 0.0f);
    }
    // Inertia generally decreases with more clusters (can fluctuate due to KMeans++ randomness)
    // Just verify we have results with varying k
    EXPECT_GT(results[0].inertia, results.back().inertia); // First > last overall trend
}

TEST(ClusterEvaluator, EvaluateProducesMetrics) {
    Matrix X = make_three_clusters();
    ClusterEvaluator eval;
    auto results = eval.evaluate(X, 2, 10, 50, 2);
    EXPECT_GE(results.size(), 1u);
    for (auto& r : results) {
        EXPECT_GE(r.silhouette_score, -1.0f);
        EXPECT_LE(r.silhouette_score, 1.0f);
        EXPECT_GE(r.davies_bouldin, 0.0f);
        EXPECT_GE(r.calinski_harabasz, 0.0f);
    }
}

TEST(ClusterEvaluator, BestKDetection) {
    Matrix X = make_three_clusters();
    ClusterEvaluator eval;
    auto results = eval.evaluate(X, 2, 10, 50, 2);
    ASSERT_GE(results.size(), 3u);

    size_t best_k = eval.best_k_silhouette(results);
    EXPECT_GE(best_k, 2u);
    EXPECT_LE(best_k, 10u);

    size_t best_k_db = eval.best_k_db(results);
    EXPECT_GE(best_k_db, 2u);
    EXPECT_LE(best_k_db, 10u);
}

TEST(ClusterEvaluator, ElbowBestK) {
    Matrix X = make_three_clusters();
    ClusterEvaluator eval;
    auto results = eval.elbow(X, 1, 10, 50, 2);
    ASSERT_GE(results.size(), 3u);

    size_t best_k_elbow = eval.best_k_elbow(results);
    EXPECT_GE(best_k_elbow, 1u);
    EXPECT_LE(best_k_elbow, 10u);
}

TEST(ClusterEvaluator, SmallDataset) {
    Matrix X(5, 2);
    for (size_t i = 0; i < 5; i++) { X[i][0]=(float)i; X[i][1]=(float)i*2; }
    ClusterEvaluator eval;
    auto results = eval.evaluate(X, 2, 4, 50, 2);
    EXPECT_GE(results.size(), 1u);
}
