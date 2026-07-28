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
