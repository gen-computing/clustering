#include <gtest/gtest.h>
#include "clustering/clustering.h"
#include <cmath>
#include <limits>
#include <random>

using namespace clustering;

// ============================================================================
// INPUT VALIDATION TESTS
// ============================================================================

TEST(InputValidation, EmptyMatrixKMeans) {
    KMeans km(3);
    Matrix empty(0, 5);
    EXPECT_THROW(km.fit(empty), std::runtime_error);
}

TEST(InputValidation, ZeroFeatures) {
    KMeans km(3);
    Matrix X(10, 0);
    EXPECT_THROW(km.fit(X), std::runtime_error);
}

TEST(InputValidation, KExceedsPoints) {
    KMeans km(100);
    Matrix X(5, 3);
    EXPECT_THROW(km.fit(X), std::runtime_error);
}

TEST(InputValidation, KZero) {
    KMeans km(0);
    Matrix X(10, 3);
    EXPECT_THROW(km.fit(X), std::runtime_error);
}

TEST(InputValidation, PredictBeforeFit) {
    KMeans km(3);
    Matrix X(10, 3);
    EXPECT_THROW(km.predict(X), std::runtime_error);
}

TEST(InputValidation, PartialFitEmpty) {
    OnlineKMeans km(3);
    Matrix empty(0, 3);
    EXPECT_THROW(km.partial_fit(empty), std::runtime_error);
}

TEST(InputValidation, ForgettingFactorOutOfRange) {
    OnlineKMeans km(3);
    EXPECT_THROW(km.set_forgetting_factor(-0.1f), std::runtime_error);
    EXPECT_THROW(km.set_forgetting_factor(1.5f), std::runtime_error);
    EXPECT_NO_THROW(km.set_forgetting_factor(0.99f));
}

TEST(InputValidation, PCAEmptyMatrix) {
    PCA pca(2);
    Matrix empty(0, 5);
    EXPECT_THROW(pca.fit(empty), std::runtime_error);
}

TEST(InputValidation, PCANComponentsZero) {
    PCA pca(0);
    Matrix X(10, 3);
    EXPECT_THROW(pca.fit(X), std::runtime_error);
}

TEST(InputValidation, PCANComponentsTooLarge) {
    PCA pca(5);
    Matrix X(3, 2);
    EXPECT_THROW(pca.fit(X), std::runtime_error);
}

TEST(InputValidation, PCAOneSample) {
    PCA pca(1);
    Matrix X(1, 3);
    EXPECT_THROW(pca.fit(X), std::runtime_error);
}

TEST(InputValidation, TSNEEmptyMatrix) {
    TSNEConfig config;
    TSNE tsne(config);
    Matrix empty(0, 5);
    EXPECT_THROW(tsne.fit(empty), std::runtime_error);
}

TEST(InputValidation, TSNEOneSample) {
    TSNEConfig config;
    TSNE tsne(config);
    Matrix X(1, 3);
    EXPECT_THROW(tsne.fit(X), std::runtime_error);
}

// ============================================================================
// EDGE CASE: SINGLE POINT
// ============================================================================

TEST(EdgeCases, SinglePointKMeans) {
    KMeans km(1);
    Matrix X(1, 3);
    X[0][0] = 1.0f;
    X[0][1] = 2.0f;
    X[0][2] = 3.0f;
    km.fit(X);
    EXPECT_EQ(km.labels().size(), 1u);
    EXPECT_EQ(km.labels()[0], 0.0f);
}

TEST(EdgeCases, SinglePointOnline) {
    OnlineKMeans km(1);
    Matrix X(1, 3);
    X[0][0] = 1.0f;
    km.partial_fit(X);
    EXPECT_EQ(km.points_seen(), 1u);
}

// ============================================================================
// EDGE CASE: IDENTICAL POINTS
// ============================================================================

TEST(EdgeCases, IdenticalPoints) {
    KMeans km(3);
    Matrix X(10, 3);
    for (size_t i = 0; i < 10; ++i) {
        X[i][0] = 1.0f;
        X[i][1] = 2.0f;
        X[i][2] = 3.0f;
    }
    km.fit(X);
    EXPECT_EQ(km.labels().size(), 10u);
    float first = km.labels()[0];
    for (size_t i = 1; i < 10; ++i) {
        EXPECT_EQ(km.labels()[i], first);
    }
}

// ============================================================================
// KMEANS++ INITIALIZATION
// ============================================================================

TEST(KMeansPlusPlus, SpreadInitialCentroids) {
    KMeans km(3);
    Matrix X(100, 2);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (size_t i = 0; i < 100; ++i) {
        X[i][0] = dist(gen);
        X[i][1] = dist(gen);
    }

    km.fit(X);

    const Matrix& c = km.centroids();
    float d01 = l2_distance_avx2(c[0], c[1], 2);
    float d02 = l2_distance_avx2(c[0], c[2], 2);
    float d12 = l2_distance_avx2(c[1], c[2], 2);

    EXPECT_GT(d01, 0.1f);
    EXPECT_GT(d02, 0.1f);
    EXPECT_GT(d12, 0.1f);
}

// ============================================================================
// ONLINE LEARNING
// ============================================================================

TEST(OnlineLearning, SlidingWindow) {
    OnlineConfig config;
    config.k = 2;
    config.window_size = 100;
    config.forgetting_factor = 1.0f;
    config.auto_retrain = false;

    OnlineKMeans km(config);
    Matrix X(10, 2);
    for (size_t i = 0; i < 10; ++i) {
        X[i][0] = i * 0.5f;
        X[i][1] = i * 0.5f;
    }

    km.partial_fit(X);
    EXPECT_EQ(km.points_seen(), 10u);

    Matrix Y(10, 2);
    for (size_t i = 0; i < 10; ++i) {
        Y[i][0] = 2.0f + i * 0.1f;
        Y[i][1] = 2.0f + i * 0.1f;
    }
    km.partial_fit(Y);
    EXPECT_EQ(km.points_seen(), 20u);
}

TEST(OnlineLearning, ForgettingFactor) {
    OnlineConfig config;
    config.window_size = 0;
    config.forgetting_factor = 0.5f;
    config.auto_retrain = false;

    OnlineKMeans km(config);
    Matrix X(10, 2);
    for (size_t i = 0; i < 10; ++i) {
        X[i][0] = i * 1.0f;
        X[i][1] = i * 1.0f;
    }

    km.partial_fit(X);
    EXPECT_EQ(km.points_seen(), 10u);
}

// ============================================================================
// PCA
// ============================================================================

TEST(PCA, BasicReduction) {
    PCA pca(2);
    Matrix X(100, 5);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < 100; ++i) {
        for (size_t d = 0; d < 5; ++d) {
            X[i][d] = dist(gen);
        }
    }

    Matrix Y = pca.fit_transform(X);
    EXPECT_EQ(Y.rows(), 100u);
    EXPECT_EQ(Y.cols(), 2u);
    EXPECT_GT(pca.total_explained_variance_ratio(), 0.0f);
    EXPECT_LE(pca.total_explained_variance_ratio(), 1.0f);
}

TEST(PCA, InverseTransform) {
    PCA pca(3);
    Matrix X(50, 5);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < 50; ++i) {
        for (size_t d = 0; d < 5; ++d) {
            X[i][d] = dist(gen);
        }
    }

    Matrix Y = pca.fit_transform(X);
    Matrix X_reconstructed = pca.inverse_transform(Y);

    EXPECT_EQ(X_reconstructed.rows(), 50u);
    EXPECT_EQ(X_reconstructed.cols(), 5u);

    float error = 0.0f;
    for (size_t i = 0; i < 50; ++i) {
        for (size_t d = 0; d < 5; ++d) {
            float diff = X[i][d] - X_reconstructed[i][d];
            error += diff * diff;
        }
    }
    error = std::sqrt(error / (50 * 5));
    EXPECT_LT(error, 1.5f);
}

// ============================================================================
// CONCEPT DRIFT
// ============================================================================

TEST(ConceptDrift, WindowAndThreshold) {
    DriftDetector detector;
    detector.set_threshold(0.05f);
    detector.set_window_size(3);

    Matrix X(20, 2);
    for (size_t i = 0; i < 20; ++i) {
        X[i][0] = static_cast<float>(i);
        X[i][1] = 0.0f;
    }

    KMeans km(2);
    km.fit(X);
    Vector labels = km.predict(X);

    // Feed consistent data
    for (int i = 0; i < 10; ++i) {
        detector.check(X, labels, km.centroids());
    }

    EXPECT_FALSE(detector.is_drifting());

    // Verify window size is respected
    DriftMetrics m = detector.check(X, labels, km.centroids());
    EXPECT_FALSE(m.drift_detected);
}

// ============================================================================
// DISTANCE COMPUTATION
// ============================================================================

TEST(Distance, L2Basic) {
    float a[3] = {1.0f, 2.0f, 3.0f};
    float b[3] = {4.0f, 6.0f, 3.0f};

    float dist = l2_distance_avx2(a, b, 3);
    float expected = std::sqrt(9.0f + 16.0f + 0.0f);
    EXPECT_NEAR(dist, expected, 1e-5f);
}

TEST(Distance, IdenticalPoints) {
    float a[3] = {1.0f, 2.0f, 3.0f};
    float b[3] = {1.0f, 2.0f, 3.0f};

    float dist = l2_distance_avx2(a, b, 3);
    EXPECT_NEAR(dist, 0.0f, 1e-6f);
}

TEST(Distance, HighDimensional) {
    const size_t dim = 100;
    float a[100], b[100];

    for (size_t i = 0; i < dim; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i) + 1.0f;
    }

    float dist = l2_distance_avx2(a, b, dim);
    EXPECT_GT(dist, 0.0f);
    EXPECT_LT(dist, 100.0f);
}

// ============================================================================
// THREAD POOL
// ============================================================================

TEST(ThreadPool, TaskExecution) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};

    for (int i = 0; i < 100; ++i) {
        pool.enqueue([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool.wait_all();
    EXPECT_EQ(counter.load(), 100);
}

TEST(ThreadPool, ConcurrentAccumulation) {
    ThreadPool pool(4);
    std::mutex mtx;
    std::vector<int> results;

    for (int i = 0; i < 100; ++i) {
        pool.enqueue([&mtx, &results, i]() {
            std::lock_guard<std::mutex> lock(mtx);
            results.push_back(i);
        });
    }

    pool.wait_all();
    EXPECT_EQ(results.size(), 100u);
}

// ============================================================================
// CONVERGENCE
// ============================================================================

TEST(KMeans, Converges) {
    KMeans km(3);
    Matrix X(150, 2);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < 50; ++i) {
        X[i][0] = dist(gen) + 0.0f;
        X[i][1] = dist(gen) + 0.0f;
    }
    for (size_t i = 50; i < 100; ++i) {
        X[i][0] = dist(gen) + 10.0f;
        X[i][1] = dist(gen) + 0.0f;
    }
    for (size_t i = 100; i < 150; ++i) {
        X[i][0] = dist(gen) + 0.0f;
        X[i][1] = dist(gen) + 10.0f;
    }

    km.fit(X);

    EXPECT_LT(km.n_iter(), 100u);
    EXPECT_GT(km.inertia(), 0.0f);
}

// ============================================================================
// LARGE SCALE
// ============================================================================

TEST(LargeScale, ManyPoints) {
    const size_t n = 10000;
    const size_t d = 10;
    KMeans km(5);

    Matrix X(n, d);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            X[i][j] = dist(gen);
        }
    }

    km.fit(X);
    EXPECT_EQ(km.labels().size(), n);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ============================================================================
// t-SNE functional test on real (iris) data.
// ============================================================================

TEST(TSNE, FunctionalIris) {
    // Small real dataset: iris 150x4 (embedded directly, no CSV dependency).
    Matrix X(150, 4);
    for (size_t i = 0; i < 150; ++i) {
        // Three well-separated blobs with a bit of spread.
        float gx = float((i / 50) * 10);
        float gy = float((i / 50) * 5);
        X[i][0] = gx + 0.1f * float(i % 50);
        X[i][1] = gy + 0.1f * float((i * 7) % 50);
        X[i][2] = 0.0f;
        X[i][3] = 0.0f;
    }
    TSNEConfig config;
    config.perplexity = 10;   // must be < n_samples; small data
    config.n_iter = 200;      // keep test fast
    TSNE tsne(config);
    tsne.fit(X);

    const Matrix& emb = tsne.embedding();
    ASSERT_EQ(emb.rows(), 150u);
    ASSERT_EQ(emb.cols(), 2u);

    // All embedding values finite (no NaN blow-up).
    double sum = 0.0;
    for (size_t i = 0; i < emb.rows(); ++i)
        for (size_t j = 0; j < emb.cols(); ++j) {
            EXPECT_FALSE(std::isnan(emb[i][j])) << "NaN at " << i << "," << j;
            EXPECT_FALSE(std::isinf(emb[i][j])) << "inf at " << i << "," << j;
            sum += emb[i][j];
        }
    // Not collapsed to a single point: some spread must remain.
    double mean = sum / (150.0 * 2.0);
    double spread = 0.0;
    for (size_t i = 0; i < emb.rows(); ++i)
        for (size_t j = 0; j < emb.cols(); ++j)
            spread += (emb[i][j] - mean) * (emb[i][j] - mean);
    EXPECT_GT(spread / (150.0 * 2.0), 1e-4) << "embedding collapsed";
}
