// ============================================================================
// Logging system -- Compile-time configurable debug/release logging.
//
// COMPILE-TIME DEBUG LOGGING:
//   cmake -DCLUSTERING_DEBUG=ON  (or define CLUSTERING_DEBUG before including)
//   Enables: function entry/exit, algorithm timing, distance dimensions,
//            iteration counts, memory allocations, thread pool activity.
//
// RELEASE LOGGING (default):
//   Only: algorithm start/completion, validation errors, final metrics.
//
// USAGE:
//   LOG_DEBUG("Distance matrix: %zu x %zu", n, k);   // only in debug builds
//   LOG_INFO("KMeans converged in %zu iterations", n); // always active
//   LOG_ERROR("Empty input matrix");                    // always active
//
// Output goes to stderr (fprintf). Thread-safe via mutex on LOG_DEBUG.
// ============================================================================

#pragma once

#include <cstdio>
#include <ctime>
#include <chrono>
#include <mutex>
#include <string>

namespace clustering {

// Global mutext for thread-safe logging (LOG_DEBUG only).
inline std::mutex& log_mutex() {
    static std::mutex m;
    return m;
}

// Timestamp string for log prefixes.
inline std::string log_timestamp() {
    time_t now = time(nullptr);
    char buf[32];
    strftime(buf, sizeof(buf), "%H:%M:%S", localtime(&now));
    return buf;
}

} // namespace clustering

// ============================================================================
// LOG MACROS
//
// LOG_DEBUG: Compiles to nothing unless CLUSTERING_DEBUG is defined.
//   Use for: per-iteration diagnostics, SIMD vector dumps, memory tracking.
// LOG_INFO:  Always active. Use for: algorithm milestones, file I/O, user actions.
// LOG_ERROR: Always active. Use for: input validation failures, file errors.
// ============================================================================

#ifdef CLUSTERING_DEBUG
    #define LOG_DEBUG(fmt, ...) \
        do { \
            std::lock_guard<std::mutex> _lck(::clustering::log_mutex()); \
            fprintf(stderr, "[DEBUG %s] " fmt "\n", \
                    ::clustering::log_timestamp().c_str(), ##__VA_ARGS__); \
        } while(0)
#else
    #define LOG_DEBUG(fmt, ...) ((void)0)
#endif

#define LOG_INFO(fmt, ...) \
    fprintf(stderr, "[INFO %s] " fmt "\n", \
            ::clustering::log_timestamp().c_str(), ##__VA_ARGS__)

#define LOG_ERROR(fmt, ...) \
    fprintf(stderr, "[ERROR %s] " fmt "\n", \
            ::clustering::log_timestamp().c_str(), ##__VA_ARGS__)
