#pragma once

#include "matrix.h"
#include <cstddef>

namespace clustering {

// ============================================================================
// DISTANCE FUNCTIONS
//
// Distance functions measure how "far apart" two points are in feature space.
// The most common is Euclidean (L2) distance -- the straight-line distance.
//
// These functions are CRITICAL for performance. Distance computation happens
// millions of times during clustering (every point vs every centroid, every
// iteration). Optimizing them with SIMD (Single Instruction Multiple Data)
// instructions gives massive speedups.
// ============================================================================

// ----- NAIVE (SCALAR) DISTANCE -----

// l2_distance_naive: Simple, readable Euclidean distance.
//   Computes sqrt((a[0]-b[0])² + (a[1]-b[1])² + ... + (a[d-1]-b[d-1])²)
//   one float at a time. Used as a REFERENCE implementation to verify
//   the optimized versions produce the same results.
//
// Parameters:
//   a, b -- pointer to float arrays (two points in feature space)
//   d    -- number of dimensions (length of each array)
//
// Returns: The Euclidean (straight-line) distance between a and b.
float l2_distance_naive(const float* a, const float* b, size_t d);

// ----- AVX2 OPTIMIZED DISTANCE (256-bit SIMD) -----

// l2_distance_avx2: Fast Euclidean distance using AVX2 CPU instructions.
//   Processes 8 floats at once (instead of 1) using 256-bit registers.
//   AVX2 is available on most CPUs from 2013 onwards (Intel Haswell, AMD Excavator+).
//
// HOW SIMD WORKS (simplified):
//   Normal:   result = a1-b1; result += a2-b2; result += a3-b3; ... (8 steps)
//   AVX2:     load 8 values at once into a 256-bit register,
//             subtract, multiply, add -- all in ONE instruction each.
//
//   The FMA instruction (Fused Multiply-Add) does: diff*diff + sum
//   in a single CPU cycle, avoiding extra rounding error from two separate ops.
//
//   If d is not a multiple of 8, the remaining (<8) elements are processed
//   one at a time (scalar fallback).
float l2_distance_avx2(const float* a, const float* b, size_t d);

// ----- AVX512 OPTIMIZED DISTANCE (512-bit SIMD, newer CPUs) -----

// l2_distance_avx512: Even faster version using AVX-512 instructions.
//   Processes 16 floats at once. Only available on newer CPUs
//   (Intel Skylake-X 2017+, AMD Zen 4 2022+).
//
//   Falls back to AVX2 if the CPU doesn't support AVX-512 (compile-time check).
float l2_distance_avx512(const float* a, const float* b, size_t d);

// ----- BATCH DISTANCE COMPUTATION -----

// compute_distances_to_centroids: Find distances from ONE point to ALL centroids.
//   Fills the `distances` array with Euclidean distance from `point` to each centroid.
//
// Parameters:
//   point     -- the query point (array of `centroids.cols()` floats)
//   centroids -- k rows of centroid coordinates
//   distances -- output array (must have room for k floats), filled by this function
void compute_distances_to_centroids(
    const float* point,
    const Matrix& centroids,
    float* distances
);

// compute_distance_matrix: Compute ALL distances from ALL points to ALL centroids.
//   This is the heavy-duty operation that KMeans runs every iteration.
//   Uses ThreadPool for parallel execution when max_threads > 1 and n >= 1000.
//
//   For small datasets (<1000 points), runs single-threaded (thread overhead > benefit).
//
// Parameters:
//   X           -- the data points (n rows x d columns)
//   centroids   -- the cluster centers (k rows x d columns)
//   distances   -- output matrix (n rows x k columns), filled by this function
//   max_threads -- 0 = auto, 1 = single-threaded, N = use N threads
void compute_distance_matrix(
    const Matrix& X,
    const Matrix& centroids,
    Matrix& distances,
    size_t max_threads = 0
);

// nearest_centroid: Find which centroid is closest to a given point.
//   Equivalent to calling compute_distances_to_centroids and finding argmin.
//
// Parameters:
//   point     -- the query point
//   centroids -- the cluster centers
//
// Returns: Index of the nearest centroid (0 to centroids.rows() - 1).
size_t nearest_centroid(
    const float* point,
    const Matrix& centroids
);

} // namespace clustering
