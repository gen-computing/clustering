// ============================================================================
// stress_bench.cpp -- Stress benchmark: how big a dataset can the engine handle?
//
// Generates a synthetic CSV, streams it into a Matrix backed by the
// disk-spill cache (small RAM cap), runs KMeans on it, and reports:
//   - peak RSS (resident set) vs dataset size vs RAM cap
//   - CSV streaming throughput
//   - sequential scan throughput
//   - KMeans fit time + inertia
//
// Usage: stress_bench [rows] [cols] [ram_cap_mb] [k]
//   default: 1000000 rows, 32 cols, 512 MiB cap, k=8
// ============================================================================

#include <clustering/matrix.h>
#include <clustering/kmeans.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>

using namespace clustering;
namespace fs = std::filesystem;

static size_t peak_rss_kb() {
    // Linux: /proc/self/status VmHWM (peak resident set, kB).
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0)
            return (size_t)std::stoull(line.substr(7));
    }
    return 0;
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Stream rows of synthetic blobs into a CSV file. O(1) memory.
static void generate_csv(const std::string& path, size_t rows, size_t cols) {
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::ofstream out(path);
    for (size_t j = 0; j < cols; ++j)
        out << (j == 0 ? "" : ",") << "f" << j;
    out << "\n";
    const size_t blobs = 4;
    for (size_t i = 0; i < rows; ++i) {
        size_t b = rng() % blobs;
        float offset = float(b * 8);
        for (size_t j = 0; j < cols; ++j) {
            if (j) out << ",";
            out << (offset + noise(rng));
        }
        out << "\n";
    }
    out.close();
}

int main(int argc, char** argv) {
    size_t rows = argc > 1 ? std::stoull(argv[1]) : 1000000;
    size_t cols = argc > 2 ? std::stoull(argv[2]) : 32;
    size_t cap_mb = argc > 3 ? std::stoull(argv[3]) : 512;
    size_t k = argc > 4 ? std::stoull(argv[4]) : 8;

    set_matrix_ram_cap(cap_mb * 1024 * 1024);

    double t0 = now_s();
    double before = (double)peak_rss_kb() / 1024.0;

    std::string csv = fs::temp_directory_path() / "stress_bench.csv";
    generate_csv(csv, rows, cols);
    std::cout << "generated " << csv << " (" << rows << "x" << cols
              << ", " << (rows * cols * 4 / 1048576.0) << " MiB raw) in "
              << (now_s() - t0) << " s\n";

    // ---- Stream CSV into a Matrix under the RAM cap ----
    // Or, with mode "direct", fill the matrix in place (skips CSV I/O) so
    // huge datasets can be tested without the CSV generation bottleneck.
    Matrix m;
    if (argc > 5 && std::string(argv[5]) == "direct") {
        t0 = now_s();
        m.resize(rows, cols);
        std::mt19937 rng(12345);
        std::normal_distribution<float> noise(0.0f, 1.0f);
        for (size_t i = 0; i < rows; ++i) {
            float* row = m[i];
            size_t b = rng() % 4;
            float offset = float(b * 8);
            for (size_t j = 0; j < cols; ++j) row[j] = offset + noise(rng);
        }
        std::cout << "filled matrix directly in " << (now_s() - t0)
                  << " s (" << (rows / (now_s() - t0) / 1000000.0) << " M rows/s)\n";
    } else {
        {
            std::ifstream in(csv);
            std::string line;
            std::getline(in, line);  // header
            m.resize(rows, cols);
            t0 = now_s();
            size_t loaded = 0;
            while (std::getline(in, line) && loaded < rows) {
                float* row = m[loaded];
                size_t j = 0;
                size_t pos = 0;
                while (j < cols && pos < line.size()) {
                    size_t end = line.find(',', pos);
                    if (end == std::string::npos) end = line.size();
                    row[j++] = std::stof(line.substr(pos, end - pos));
                    pos = end + 1;
                }
                ++loaded;
            }
            m.resize(loaded, cols);
            std::cout << "streamed " << loaded << " rows into matrix in "
                      << (now_s() - t0) << " s ("
                      << (loaded / (now_s() - t0) / 1000000.0) << " M rows/s)\n";
        }
        std::filesystem::remove(csv);
    }
    std::cout << "matrix " << m.rows() << "x" << m.cols()
              << " disk_backed=" << (m.is_disk_backed() ? "yes" : "no")
              << " disk_bytes=" << m.disk_bytes_used() / 1048576.0 << " MiB"
              << " ram_cache=" << m.ram_bytes_used() / 1048576.0 << " MiB\n";

    // ---- Sequential scan (row-major, LRU-friendly, read-only path) ----
    t0 = now_s();
    double sum = 0;
    const Matrix& cm = m;
    for (size_t i = 0; i < cm.rows(); ++i) {
        const float* row = cm[i];
        for (size_t j = 0; j < cm.cols(); ++j) sum += row[j];
    }
    double scan_s = now_s() - t0;
    std::cout << "sequential scan: " << scan_s << " s ("
              << (cm.rows() / scan_s / 1000000.0) << " M rows/s), sum=" << sum << "\n";

    // ---- KMeans on the (possibly disk-backed) matrix ----
    KMeansConfig cfg;
    cfg.k = k;
    cfg.max_iter = argc > 6 ? std::stoull(argv[6]) : 10;
    cfg.tol = 0.0f;  // force max_iter passes for a stable runtime comparison
    cfg.max_threads = 1;  // disk-backed matrices serialize on the cache mutex
    KMeans km(cfg);
    t0 = now_s();
    km.fit(m);
    double fit_s = now_s() - t0;
    std::cout << "KMeans k=" << k << " fit: " << fit_s << " s, inertia="
              << km.inertia() << "\n";

    double peak = (double)peak_rss_kb() / 1024.0;
    std::cout << "\nRESULT: dataset=" << (rows * cols * 4 / 1048576.0) << " MiB"
              << " ram_cap=" << cap_mb << " MiB"
              << " peak_rss_before=" << before << " MiB"
              << " peak_rss_during=" << peak << " MiB"
              << " (delta=" << (peak - before) << " MiB)\n";
    return 0;
}
