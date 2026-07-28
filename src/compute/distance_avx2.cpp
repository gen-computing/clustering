// ============================================================================
// Distance computation -- L2 (Euclidean) distance with SIMD optimization.
//
// Distance computation is the HOT PATH of clustering. During training,
// compute_distance_matrix() is called every iteration, computing
// n_points * k_clusters distances. For 100k points and k=10, that's
// 1 million distances per iteration, each involving d floating-point ops.
//
// SIMD (Single Instruction Multiple Data) processes multiple floats at once
// using CPU vector registers, giving 8x (AVX2) or 16x (AVX512) throughput
// for the arithmetic operations.
//
// REGISTERS USED:
//   AVX2: 256-bit YMM registers (8 floats)
//   AVX512: 512-bit ZMM registers (16 floats)
// ============================================================================

#include "clustering/distance.h"
#include "clustering/thread_pool.h"
#include <cmath>          // std::sqrt
#include <algorithm>      // std::min
#include <immintrin.h>    // Intel intrinsics: __m256, _mm256_loadu_ps, etc.
                          // These are the "building blocks" of SIMD programming.
                          // Each intrinsic maps to a single CPU instruction.

namespace clustering {

// ============================================================================
// l2_distance_naive -- SCALAR (SIMPLE) EUCLIDEAN DISTANCE
//
// Computes sqrt(sum((a[i] - b[i])²)) one float at a time.
// Used as a CORRECTNESS REFERENCE: the SIMD versions should produce
// the same results within floating-point precision.
//
// Algorithm: Pythagorean theorem in d dimensions.
//   diff_i = a[i] - b[i]       // difference along dimension i
//   sum += diff_i * diff_i      // add squared difference
//   return sqrt(sum)             // square root of total
// ============================================================================

float l2_distance_naive(const float* a, const float* b, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// ============================================================================
// l2_distance_avx2 -- 8-FLOATS-AT-ONCE EUCLIDEAN DISTANCE
//
// Intel AVX2 (Advanced Vector Extensions 2) provides 256-bit registers
// that hold 8 float values. Operations on these registers process all
// 8 values simultaneously.
//
// Key intrinsics used:
//   _mm256_loadu_ps(ptr):   Load 8 floats from memory (unaligned)
//   _mm256_sub_ps(a, b):    Subtract 8 floats pairwise
//   _mm256_fmadd_ps(a,b,c): a*b + c -- Fused Multiply-Add, 1 instruction!
//                           (instead of 2: multiply then add)
//   _mm256_setzero_ps():    Create register with 8 zeros
//
// For dimensions not divisible by 8: scalar fallback handles the remainder.
// ============================================================================

float l2_distance_avx2(const float* a, const float* b, size_t d) {
    // Initialize the accumulator register to zeros.
    // This register will hold 8 partial sums in parallel.
    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;

    // ---- MAIN LOOP: process 8 floats at a time ----
    // i+8 <= d means there are at least 8 elements left.
    for (; i + 8 <= d; i += 8) {
        // Load 8 consecutive floats from each array.
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);

        // Subtract: diff = a - b
        __m256 diff = _mm256_sub_ps(va, vb);

        // Accumulate squared differences
#ifdef __FMA__
        // FMA (Fused Multiply-Add): sum = sum + diff * diff
        // Single CPU instruction, lower rounding error.
        sum = _mm256_fmadd_ps(diff, diff, sum);
#else
        // Fallback: multiply then add (two instructions)
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
#endif
    }

    // ---- HORIZONTAL SUM: combine 8 partial sums into 1 ----
    // AVX2 doesn't have a single "horizontal add" instruction for 8 floats.
    // We must reduce step by step:

    // Step 1: Extract the upper 128 bits (4 floats) from the 256-bit register.
    //         hi = [sum[4], sum[5], sum[6], sum[7]]
    __m128 hi = _mm256_extractf128_ps(sum, 1);

    // Step 2: Get the lower 128 bits.
    //         lo = [sum[0], sum[1], sum[2], sum[3]]
    __m128 lo = _mm256_castps256_ps128(sum);

    // Step 3: Add upper and lower halves -> 4 float sums.
    __m128 s = _mm_add_ps(hi, lo);

    // Step 4: Horizontal add pairs (2 rounds to reduce 4 -> 1).
    //         hadd: s = [s0+s1, s2+s3, s0+s1, s2+s3]
    s = _mm_hadd_ps(s, s);
    //         hadd again: s = [s0+s1+s2+s3, ..., ...]
    s = _mm_hadd_ps(s, s);

    // Step 5: Extract the final scalar result (first element).
    float result;
    _mm_store_ss(&result, s);

    // ---- REMAINDER: handle any leftover elements (0-7 of them) ----
    // This runs only if d was not a multiple of 8.
    for (; i < d; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    // Final square root gives the Euclidean distance.
    return std::sqrt(result);
}

// ============================================================================
// l2_distance_avx512 -- 16-FLOATS-AT-ONCE (NEWEST CPUs only)
//
// AVX-512 extends the vector width to 512 bits (16 floats).
// Available on: Intel Skylake-X (2017+), Ice Lake (2019+), AMD Zen 4 (2022+).
//
// Falls back to AVX2 if the CPU doesn't support AVX-512 (compile-time check).
// The #ifdef __AVX512F__ checks if the compiler was told to target a CPU
// with AVX-512 support (via -mavx512f flag).
// ============================================================================

float l2_distance_avx512(const float* a, const float* b, size_t d) {
#ifdef __AVX512F__
    // Initialize 512-bit accumulator (16 floats = 16 partial sums).
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;

    // Process 16 floats at a time.
    for (; i + 16 <= d; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    // AVX-512 has a built-in horizontal reduction!
    float result = _mm512_reduce_add_ps(sum);

    // Remainder with AVX2 fallback (process 8 at a time).
    for (; i + 8 <= d; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 s256 = _mm256_mul_ps(diff, diff);  // No FMA intrinsic for 256-bit alone

        // Horizontal sum for 256-bit (same as AVX2 path).
        __m128 hi = _mm256_extractf128_ps(s256, 1);
        __m128 lo = _mm256_castps256_ps128(s256);
        __m128 s = _mm_add_ps(hi, lo);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float temp;
        _mm_store_ss(&temp, s);
        result += temp;
    }

    // Scalar remainder.
    for (; i < d; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return std::sqrt(result);
#else
    // AVX-512 not available at compile time -> use AVX2.
    return l2_distance_avx2(a, b, d);
#endif
}

// ============================================================================
// compute_distances_to_centroids -- ONE POINT vs ALL CENTROIDS
//
// For a single query point, compute its distance to each of the k centroids.
// Fills the `distances` array (must have room for k floats).
// This is O(k * d) with AVX2 acceleration per distance.
// ============================================================================

void compute_distances_to_centroids(
    const float* point,
    const Matrix& centroids,
    float* distances
) {
    // For each centroid c, compute L2 distance from `point` to `centroids[c]`.
    // centroids.rows() = k (number of clusters).
    // centroids.cols() = d (number of features).
    for (size_t c = 0; c < centroids.rows(); ++c) {
        distances[c] = l2_distance_avx2(point, centroids[c], centroids.cols());
    }
}

// ============================================================================
// compute_distance_matrix -- ALL POINTS vs ALL CENTROIDS (PARALLEL)
//
// This is the heavy-lifting function called by KMeans every iteration.
// Computes n * k distances. For large n, uses the ThreadPool to split
// work across CPU cores.
//
// SINGLE-THREADED PATH: when max_threads <= 1 or n < 1000 (thread overhead
// isn't worth it for small workloads).
//
// MULTI-THREADED PATH: splits n rows into chunks, each processed by a
// different thread. Each thread calls compute_distances_to_centroids()
// for its chunk of points.
// ============================================================================

void compute_distance_matrix(
    const Matrix& X,
    const Matrix& centroids,
    Matrix& distances,
    size_t max_threads
) {
    size_t n = X.rows();         // Number of data points
    size_t k = centroids.rows(); // Number of clusters
    size_t d = X.cols();         // Number of features

    // Ensure the distances matrix is the right size.
    distances.resize(n, k);

    // ---- DECISION: Single-threaded or parallel? ----
    if (max_threads <= 1 || n < 1000) {
        // Single-threaded: simple double loop.
        // For small n, thread creation/synchronization overhead exceeds benefit.
        for (size_t i = 0; i < n; ++i) {
            compute_distances_to_centroids(X[i], centroids, distances[i]);
        }
    } else {
        // Multi-threaded: split work into chunks.
        // Get the global singleton thread pool (4-8 threads by default).

        ThreadPool& pool = ThreadPool::global();

        // Calculate chunk size: each thread gets approximately n/max_threads rows.
        // Example: n=10000, max_threads=4 -> chunk_size=2500 rows per thread.
        size_t chunk_size = (n + max_threads - 1) / max_threads;

        // Futures: each holds the "promise" that a thread's work will complete.
        // We collect them to wait for all threads before returning.
        std::vector<std::future<void>> futures;

        for (size_t t = 0; t < max_threads && t * chunk_size < n; ++t) {
            size_t start = t * chunk_size;          // First row this thread handles
            size_t end = std::min(start + chunk_size, n);  // One past last row

            // Enqueue a task that computes distances for rows [start, end).
            // The lambda captures variables by reference (&). This is safe because
            // we wait for all futures before the function returns (references stay valid).
            futures.push_back(pool.enqueue([&X, &centroids, &distances, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    compute_distances_to_centroids(X[i], centroids, distances[i]);
                }
            }));
        }

        // WAIT for all threads to finish before returning.
        // Without this, the caller might read `distances` while threads are
        // still writing to it -> data race -> incorrect results.
        for (auto& f : futures) {
            f.get();  // Blocks until this specific thread's task is done
        }
    }
}

// ============================================================================
// nearest_centroid -- FIND CLOSEST CENTROID TO A SINGLE POINT
//
// Returns: index (0 to k-1) of the centroid with smallest Euclidean distance.
// Used by partial_fit() and OnlineKMeans for incremental updates.
// ============================================================================

size_t nearest_centroid(
    const float* point,
    const Matrix& centroids
) {
    // Start with centroid 0 as the "best so far".
    size_t best = 0;
    float best_dist = l2_distance_avx2(point, centroids[0], centroids.cols());

    // Check all other centroids.
    for (size_t c = 1; c < centroids.rows(); ++c) {
        float dist = l2_distance_avx2(point, centroids[c], centroids.cols());
        if (dist < best_dist) {
            best_dist = dist;
            best = c;  // Found a closer centroid
        }
    }

    return best;
}

} // namespace clustering
