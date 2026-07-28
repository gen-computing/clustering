#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "clustering/clustering.h"

using namespace clustering;

Matrix load_csv(const std::string& path, int max_rows = -1) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line)) {
        if (max_rows > 0 && static_cast<int>(data.size()) >= max_rows) break;

        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stof(cell));
            } catch (...) {
                break;  // Skip non-numeric (e.g., class labels)
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    if (data.empty()) throw std::runtime_error("Empty dataset");

    size_t n = data.size();
    size_t d = data[0].size();
    Matrix X(n, d);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            X[i][j] = data[i][j];
        }
    }

    return X;
}

struct BenchResult {
    std::string name;
    double fit_ms;
    double predict_ms;
    float inertia;
    size_t iterations;
};

BenchResult benchmark_kmeans(const Matrix& X, size_t k, const std::string& name) {
    BenchResult result;
    result.name = name;

    KMeans km(k);

    auto t1 = std::chrono::high_resolution_clock::now();
    km.fit(X);
    auto t2 = std::chrono::high_resolution_clock::now();
    result.fit_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result.iterations = km.n_iter();
    result.inertia = km.inertia();

    auto t3 = std::chrono::high_resolution_clock::now();
    km.predict(X);
    auto t4 = std::chrono::high_resolution_clock::now();
    result.predict_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    return result;
}

BenchResult benchmark_minibatch(const Matrix& X, size_t k, const std::string& name) {
    BenchResult result;
    result.name = name;

    MiniBatchKMeans km(k, 100);

    auto t1 = std::chrono::high_resolution_clock::now();
    km.fit(X);
    auto t2 = std::chrono::high_resolution_clock::now();
    result.fit_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result.iterations = km.n_iter();
    result.inertia = km.inertia();

    auto t3 = std::chrono::high_resolution_clock::now();
    km.predict(X);
    auto t4 = std::chrono::high_resolution_clock::now();
    result.predict_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    return result;
}

BenchResult benchmark_online(const Matrix& X, size_t k, size_t batch_size, const std::string& name) {
    BenchResult result;
    result.name = name;

    OnlineConfig config;
    config.k = k;
    config.auto_retrain = false;

    OnlineKMeans km(config);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < X.rows(); i += batch_size) {
        size_t end = std::min(i + batch_size, X.rows());
        size_t n = end - i;

        Matrix batch(n, X.cols());
        for (size_t j = 0; j < n; ++j) {
            for (size_t d = 0; d < X.cols(); ++d) {
                batch[j][d] = X[i + j][d];
            }
        }
        km.partial_fit(batch);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    result.fit_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result.iterations = 1;
    result.inertia = 0;

    auto t3 = std::chrono::high_resolution_clock::now();
    km.predict(X);
    auto t4 = std::chrono::high_resolution_clock::now();
    result.predict_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    return result;
}

void print_result(const BenchResult& r) {
    std::cout << std::left << std::setw(25) << r.name
              << std::right << std::setw(10) << std::fixed << std::setprecision(1) << r.fit_ms << " ms"
              << std::setw(10) << r.predict_ms << " ms"
              << std::setw(10) << std::setprecision(0) << r.inertia
              << std::setw(8) << r.iterations << " iter" << std::endl;
}

void print_header() {
    std::cout << std::left << std::setw(25) << "Algorithm"
              << std::right << std::setw(14) << "Fit Time"
              << std::setw(14) << "Predict Time"
              << std::setw(14) << "Inertia"
              << std::setw(12) << "Iters" << std::endl;
    std::cout << std::string(79, '-') << std::endl;
}

void benchmark_dataset(const std::string& path, const std::string& name, size_t k) {
    std::cout << "\n=== " << name << " (k=" << k << ") ===" << std::endl;

    Matrix X = load_csv(path);
    std::cout << "Loaded: " << X.rows() << " samples x " << X.cols() << " features\n" << std::endl;

    print_header();

    auto r1 = benchmark_kmeans(X, k, "KMeans");
    print_result(r1);

    auto r2 = benchmark_minibatch(X, k, "MiniBatchKMeans");
    print_result(r2);

    auto r3 = benchmark_online(X, k, 100, "OnlineKMeans(batch=100)");
    print_result(r3);

    // Speedup
    double speedup = r1.fit_ms / r3.fit_ms;
    std::cout << "\nMiniBatch vs KMeans speedup: " << std::fixed << std::setprecision(2) << r1.fit_ms / r2.fit_ms << "x" << std::endl;
    std::cout << "Online vs KMeans speedup: " << speedup << "x" << std::endl;
}

void benchmark_pca(const Matrix& X, const std::string& name) {
    std::cout << "\n=== PCA: " << name << " ===" << std::endl;
    std::cout << "Input: " << X.rows() << " x " << X.cols() << std::endl;

    for (size_t n_comp : {2, 5, 10}) {
        if (n_comp >= X.cols()) continue;

        PCA pca(n_comp);

        auto t1 = std::chrono::high_resolution_clock::now();
        Matrix Y = pca.fit_transform(X);
        auto t2 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        float var = pca.total_explained_variance_ratio() * 100.0f;

        std::cout << "  " << X.cols() << "d -> " << n_comp << "d: "
                  << std::fixed << std::setprecision(1) << ms << "ms, "
                  << std::setprecision(1) << var << "% variance retained" << std::endl;
    }
}

void benchmark_drift(const Matrix& X, size_t k) {
    std::cout << "\n=== Drift Detection ===" << std::endl;

    KMeans km(k);
    km.fit(X);
    Vector labels = km.predict(X);

    DriftDetector detector;
    detector.set_threshold(0.1f);
    detector.set_window_size(5);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        detector.check(X, labels, km.centroids());
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "10 drift checks: " << std::fixed << std::setprecision(1) << ms << "ms" << std::endl;
    std::cout << "Per check: " << ms / 10.0 << "ms" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Clustering Engine Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;

    std::string data_dir = "data/";

    // Iris (150 x 4, k=3)
    benchmark_dataset(data_dir + "iris.csv", "Iris (150x4)", 3);

    // Wine (178 x 13, k=3)
    benchmark_dataset(data_dir + "wine.csv", "Wine (178x13)", 3);

    // Synthetic (10000 x 32, k=10)
    benchmark_dataset(data_dir + "synthetic_10k_32d.csv", "Synthetic 10k (10000x32)", 10);

    // PCA benchmark
    Matrix synthetic = load_csv(data_dir + "synthetic_10k_32d.csv");
    benchmark_pca(synthetic, "Synthetic 10k");

    // Drift detection benchmark (use smaller subset)
    Matrix iris = load_csv(data_dir + "iris.csv");
    benchmark_drift(iris, 3);

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Done." << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
