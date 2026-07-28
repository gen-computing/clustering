#include <gtest/gtest.h>
#include "clustering/versioning.h"
#include "clustering/drift.h"
#include "clustering/feature_store.h"
#include <filesystem>
#include <cmath>

using namespace clustering;

class OperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "/tmp/clustering_test_" + std::to_string(getpid());
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }

    std::string test_dir_;
};

TEST_F(OperationsTest, VersionManagerSaveLoad) {
    VersionManager vm;
    vm.set_storage_path(test_dir_);

    // Create test data
    Matrix centroids(3, 2);
    centroids[0][0] = 1.0f; centroids[0][1] = 2.0f;
    centroids[1][0] = 3.0f; centroids[1][1] = 4.0f;
    centroids[2][0] = 5.0f; centroids[2][1] = 6.0f;

    Vector labels(10);
    for (size_t i = 0; i < 10; ++i) {
        labels[i] = static_cast<float>(i % 3);
    }

    // Save
    size_t version_id = vm.save(centroids, labels);
    EXPECT_GT(version_id, 0u);

    // Load
    Matrix loaded_centroids;
    Vector loaded_labels;
    bool success = vm.load(version_id, loaded_centroids, loaded_labels);

    EXPECT_TRUE(success);
    EXPECT_EQ(loaded_centroids.rows(), 3u);
    EXPECT_EQ(loaded_centroids.cols(), 2u);
    EXPECT_EQ(loaded_labels.size(), 10u);

    // Verify data
    EXPECT_FLOAT_EQ(loaded_centroids[0][0], 1.0f);
    EXPECT_FLOAT_EQ(loaded_centroids[1][0], 3.0f);
    EXPECT_FLOAT_EQ(loaded_centroids[2][0], 5.0f);
}

TEST_F(OperationsTest, VersionManagerListVersions) {
    VersionManager vm;
    vm.set_storage_path(test_dir_);

    EXPECT_FALSE(vm.has_versions());

    // Save multiple versions
    Matrix centroids(2, 2);
    Vector labels(5);

    vm.save(centroids, labels);
    vm.save(centroids, labels);
    vm.save(centroids, labels);

    EXPECT_TRUE(vm.has_versions());
    EXPECT_EQ(vm.list_versions().size(), 3u);
    EXPECT_EQ(vm.latest_version(), 3u);
}

TEST_F(OperationsTest, VersionManagerLoadNonexistent) {
    VersionManager vm;
    vm.set_storage_path(test_dir_);

    Matrix centroids;
    Vector labels;
    bool success = vm.load(999, centroids, labels);

    EXPECT_FALSE(success);
}

TEST(DriftDetectorTest, BasicCheck) {
    DriftDetector detector;
    detector.set_threshold(0.1f);

    // Create simple data
    Matrix X(100, 2);
    Vector labels(100);
    Matrix centroids(3, 2);

    // Initialize with some pattern
    for (size_t i = 0; i < 100; ++i) {
        X[i][0] = static_cast<float>(i % 3) * 5.0f;
        X[i][1] = static_cast<float>(i % 3) * 5.0f;
        labels[i] = static_cast<float>(i % 3);
    }

    centroids[0][0] = 0.0f; centroids[0][1] = 0.0f;
    centroids[1][0] = 5.0f; centroids[1][1] = 5.0f;
    centroids[2][0] = 10.0f; centroids[2][1] = 10.0f;

    DriftMetrics metrics = detector.check(X, labels, centroids);

    EXPECT_GE(metrics.silhouette_score, -1.0f);
    EXPECT_LE(metrics.silhouette_score, 1.0f);
    EXPECT_GE(metrics.davies_bouldin_index, 0.0f);
    EXPECT_GE(metrics.calinski_harabasz_score, 0.0f);
    EXPECT_GE(metrics.cluster_stability, 0.0f);
    EXPECT_LE(metrics.cluster_stability, 1.0f);
}

TEST(DriftDetectorTest, Threshold) {
    DriftDetector detector;
    detector.set_threshold(0.01f);
    detector.set_window_size(5);

    Matrix X(50, 2);
    Vector labels(50);
    Matrix centroids(2, 2);

    centroids[0][0] = 0.0f; centroids[0][1] = 0.0f;
    centroids[1][0] = 10.0f; centroids[1][1] = 10.0f;

    // Run multiple checks
    for (int round = 0; round < 10; ++round) {
        for (size_t i = 0; i < 50; ++i) {
            X[i][0] = static_cast<float>(i % 2) * 10.0f + round * 0.1f;
            X[i][1] = static_cast<float>(i % 2) * 10.0f + round * 0.1f;
            labels[i] = static_cast<float>(i % 2);
        }

        DriftMetrics metrics = detector.check(X, labels, centroids);
        // Should not drift with gradual change
        EXPECT_FALSE(metrics.drift_detected);
    }
}

TEST_F(OperationsTest, PreprocessNormalize) {
    FeatureStore store;
    store.set_cache_path(test_dir_);

    Matrix X(10, 2);
    for (size_t i = 0; i < 10; ++i) {
        X[i][0] = static_cast<float>(i);
        X[i][1] = static_cast<float>(i * 2);
    }

    Matrix normalized = store.preprocess(X, "normalize");

    EXPECT_EQ(normalized.rows(), 10u);
    EXPECT_EQ(normalized.cols(), 2u);

    // Check that values are normalized
    for (size_t j = 0; j < 2; ++j) {
        float norm = 0.0f;
        for (size_t i = 0; i < 10; ++i) {
            norm += normalized[i][j] * normalized[i][j];
        }
        norm = std::sqrt(norm);
        EXPECT_NEAR(norm, 1.0f, 1e-5f);
    }
}

TEST_F(OperationsTest, PreprocessCache) {
    FeatureStore store;
    store.set_cache_path(test_dir_);

    Matrix X(5, 2);
    X[0][0] = 1.0f; X[0][1] = 2.0f;
    X[1][0] = 3.0f; X[1][1] = 4.0f;
    X[2][0] = 5.0f; X[2][1] = 6.0f;
    X[3][0] = 7.0f; X[3][1] = 8.0f;
    X[4][0] = 9.0f; X[4][1] = 10.0f;

    // First call - cache miss, but preprocess caches result
    Matrix result1 = store.preprocess(X, "standardize");
    std::string key = "5_2_standardize";
    // After preprocess, cache should exist
    EXPECT_TRUE(store.cache_hit(key));

    // Second call - cache hit
    Matrix result2 = store.preprocess(X, "standardize");
    EXPECT_TRUE(store.cache_hit(key));

    // Results should be identical
    EXPECT_EQ(result1.rows(), result2.rows());
    EXPECT_EQ(result1.cols(), result2.cols());
}
