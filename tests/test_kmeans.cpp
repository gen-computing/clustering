#include <gtest/gtest.h>
#include "clustering/kmeans.h"
#include "clustering/distance.h"

using namespace clustering;

TEST(KMeansTest, BasicFit) {
    // Create simple 2D data with 3 clusters
    Matrix X(100, 2);

    // Cluster 1: around (0, 0)
    for (size_t i = 0; i < 30; ++i) {
        X[i][0] = static_cast<float>(i) / 30.0f;
        X[i][1] = static_cast<float>(i) / 30.0f;
    }

    // Cluster 2: around (5, 5)
    for (size_t i = 30; i < 60; ++i) {
        X[i][0] = 5.0f + static_cast<float>(i - 30) / 30.0f;
        X[i][1] = 5.0f + static_cast<float>(i - 30) / 30.0f;
    }

    // Cluster 3: around (10, 10)
    for (size_t i = 60; i < 100; ++i) {
        X[i][0] = 10.0f + static_cast<float>(i - 60) / 40.0f;
        X[i][1] = 10.0f + static_cast<float>(i - 60) / 40.0f;
    }

    KMeans kmeans(3);
    kmeans.fit(X);

    // Check results
    EXPECT_EQ(kmeans.labels().size(), 100u);
    EXPECT_EQ(kmeans.centroids().rows(), 3u);
    EXPECT_EQ(kmeans.centroids().cols(), 2u);
    EXPECT_GT(kmeans.n_iter(), 0u);
}

TEST(KMeansTest, Predict) {
    Matrix X(50, 2);
    for (size_t i = 0; i < 25; ++i) {
        X[i][0] = 0.0f;
        X[i][1] = 0.0f;
    }
    for (size_t i = 25; i < 50; ++i) {
        X[i][0] = 10.0f;
        X[i][1] = 10.0f;
    }

    KMeans kmeans(2);
    kmeans.fit(X);

    // Predict new points
    Matrix new_points(2, 2);
    new_points[0][0] = 0.1f;
    new_points[0][1] = 0.1f;
    new_points[1][0] = 9.9f;
    new_points[1][1] = 9.9f;

    Vector predictions = kmeans.predict(new_points);
    EXPECT_EQ(predictions.size(), 2u);
    EXPECT_NE(predictions[0], predictions[1]);  // Different clusters
}

TEST(KMeansTest, EmptyInput) {
    Matrix X(0, 2);
    KMeans kmeans(3);

    EXPECT_THROW(kmeans.fit(X), std::runtime_error);
}

TEST(KMeansTest, KExceedsPoints) {
    Matrix X(5, 2);
    KMeans kmeans(10);

    EXPECT_THROW(kmeans.fit(X), std::runtime_error);
}

TEST(KMeansTest, SingleCluster) {
    Matrix X(50, 2);
    for (size_t i = 0; i < 50; ++i) {
        X[i][0] = static_cast<float>(i);
        X[i][1] = static_cast<float>(i);
    }

    KMeans kmeans(1);
    kmeans.fit(X);

    EXPECT_EQ(kmeans.labels().size(), 50u);
    EXPECT_EQ(kmeans.centroids().rows(), 1u);

    // All labels should be 0
    for (size_t i = 0; i < 50; ++i) {
        EXPECT_EQ(kmeans.labels()[i], 0.0f);
    }
}

TEST(DistanceTest, NaiveVsAVX2) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    float dist_naive = l2_distance_naive(a, b, 8);
    float dist_avx2 = l2_distance_avx2(a, b, 8);

    EXPECT_NEAR(dist_naive, dist_avx2, 1e-5f);
}

TEST(DistanceTest, ZeroDistance) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {1.0f, 2.0f, 3.0f};

    float dist = l2_distance_avx2(a, b, 3);
    EXPECT_NEAR(dist, 0.0f, 1e-6f);
}

TEST(DistanceTest, KnownDistance) {
    float a[] = {0.0f, 0.0f};
    float b[] = {3.0f, 4.0f};

    float dist = l2_distance_avx2(a, b, 2);
    EXPECT_NEAR(dist, 5.0f, 1e-5f);
}

// ============================================================================
// MiniBatchKMeans tests (bug #1 regression: learning rate decay across
// partial_fit() calls; fix: local iteration counter).
// ============================================================================

#include "clustering/mini_batch.h"

TEST(MiniBatchKMeans, PartialFitWithoutFitDelegates) {
    // partial_fit() on an unfitted model must behave like fit().
    Matrix X(20, 2);
    for (size_t i = 0; i < 10; ++i) { X[i][0] = 0.0f; X[i][1] = 0.0f; }
    for (size_t i = 10; i < 20; ++i) { X[i][0] = 10.0f; X[i][1] = 10.0f; }
    MiniBatchKMeans mb(2, 5);
    mb.partial_fit(X);
    EXPECT_EQ(mb.labels().size(), 20u);
    // Two well-separated blobs: both clusters should be found.
    EXPECT_EQ(mb.centroids().rows(), 2u);
    // Each centroid must sit near SOME blob (label order not guaranteed).
    auto near_blob = [](float x, float y) {
        float da = x * x + y * y;                       // blob A at (0,0)
        float db = (x - 10) * (x - 10) + (y - 10) * (y - 10);  // blob B at (10,10)
        return std::min(da, db);
    };
    EXPECT_LT(near_blob(mb.centroids()[0][0], mb.centroids()[0][1]), 4.0f);
    EXPECT_LT(near_blob(mb.centroids()[1][0], mb.centroids()[1][1]), 4.0f);
}

TEST(MiniBatchKMeans, MultiplePartialFitsMoveCentroids) {
    // Repeated partial_fit() with shifted data must keep moving (the old bug
    // froze learning rate at n_iter()==0, making later batches no-ops).
    Matrix a(30, 2), b(30, 2);
    for (size_t i = 0; i < 30; ++i) { a[i][0] = 1.0f; a[i][1] = 1.0f; b[i][0] = 9.0f; b[i][1] = 9.0f; }
    MiniBatchKMeans mb(1, 10);
    mb.partial_fit(a);
    float first_x = mb.centroids()[0][0];
    mb.partial_fit(b);
    float second_x = mb.centroids()[0][0];
    mb.partial_fit(b);
    float third_x = mb.centroids()[0][0];
    EXPECT_LT(first_x, 5.0f) << "initially near blob A";
    EXPECT_GT(second_x, first_x) << "moves toward blob B";
    EXPECT_GT(third_x, 8.0f) << "converges on blob B despite decaying learning rate";
}
