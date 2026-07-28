# Debug Plan — Clustering Engine

## Critical Bugs (7)

| # | File:Line | Issue | Fix |
|---|-----------|-------|-----|
| 1 | `src/core/mini_batch.cpp:70` | Learning rate never decays after initial `fit()`. `n_iter()` stuck at 0 for subsequent `partial_fit()` calls. Algorithm unstable. | Track local iteration counter in MiniBatchKMeans. |
| 2 | `src/core/dbscan.cpp:155` | `predict()` inner loop always finds first clustered point then breaks with `found=false`. Returns 0 (noise) for every point. | Rewrite predict: for new point, find nearest core/border point in training data, return its label. Store training data during fit(). |
| 3 | `src/dimensionality/tsne.cpp:425` | `transform()` computes distance between `embedding_[i]` and `embedding_[j]` — ignores input `X` entirely. Placeholder logic. | Store original training X during fit(). Use kNN in original space to assign new points to nearest embedding position. |
| 4 | `src/app/main.cpp:386` | `algos[3]` accessed but array has 3 elements (indices 0-2). OOB → undefined behavior. | Expand array to 4 elements or gate DBSCAN from export. |
| 5 | `src/app/main.cpp:174` | `run_clustering_async` hardcodes `cfg.max_iter=300, cfg.max_threads=4` ignoring `g.max_iter` and `g.max_threads`. | Use user settings from `g`. |
| 6 | `src/app/main.cpp:1003` | FBO resize deletes old FBO/texture but not RBO. RBO leaked every resize. | Delete RBO before recreating FBO. |
| 7 | `src/compute/distance_avx2.cpp` | `_mm256_fmadd_ps` used without `#ifdef __FMA__` guard. Runtime segfault on non-FMA CPUs. | Add FMA compile-time check with scalar fallback. |

## Medium Bugs (8)

| # | File:Line | Issue | Fix |
|---|-----------|-------|-----|
| 8 | `src/core/kmeans.cpp` | Roulette-wheel selection can fall through without picking centroid when all distances ≈ 0. | Add fallback: pick random centroid when total ≈ 0. |
| 9 | `src/operational/evaluation.cpp` | DriftDetector history accumulates across different k evaluations. Contaminates results. | Create fresh DriftDetector per k iteration. |
| 10 | `src/compute/thread_pool.cpp:192` | `ThreadPool::global()` — no synchronization on lazy init. Double-init race. | Use `std::call_once` with `std::once_flag`. |
| 11 | `src/operational/versioning.cpp` | No validation of file-read data (k, d, n). Corrupted .bin → OOM or UB. | Add bounds checking after reading metadata. |
| 12 | `src/app/main.cpp:198` | `g.labels`/`g.centroids` written under `result_mutex` but read from main thread without mutex. | Acquire mutex before reading in render loop. |
| 13 | `src/app/main.cpp:235` | `compare_history` pushes from background thread without mutex. | Guard with mutex. |
| 14 | `src/gui/csv_importer.cpp` | `cancel_` and `loading_` not `std::atomic<bool>`. Data race. | Make atomic. |
| 15 | `src/gui/preprocess_pipeline.cpp:62` | Redo for transforms marks cells NaN instead of recomputing values. | Store transform params, reapply on redo. |

## Code Smells (7)

| # | File | Issue | Fix |
|---|------|-------|-----|
| 16 | `renderer.cpp` | ~200 lines duplicated between `render_frame()` and `render_to_fbo()`. | Extract shared `render_impl(int fb_w, int fb_h)`. |
| 17 | `column_stats.cpp:27` | Static return `empty` — thread-unsafe. | Return by value. |
| 18 | `column_stats.cpp:66` | Numerically unstable variance formula. | Use Welford's online algorithm. |
| 19 | `dbscan.cpp` | Double `#include "clustering/logging.h"`. | Remove duplicate. |
| 20 | `renderer.cpp` | Double `#include "clustering/renderer.h"`. | Remove duplicate. |
| 21 | `online.cpp` | Per-point `std::vector<float>` allocation in sliding window. | Use pre-allocated ring buffer. |
| 22 | `preprocess_pipeline.cpp` | `std` namespace variable shadowed. | Rename variable. |
| 23 | `tsne.cpp:50` | `config_.n_components` mutated during `fit()`. | Use local variable. |

## Test Gaps

| Missing Coverage |
|------------------|
| MiniBatchKMeans — no test at all |
| Concept drift detection when drift actually happens |
| DBSCAN predict() correctness |
| t-SNE functional test with real data |
| PCA high-dim (>32) AVX2 edge cases |
| Thread pool pool(1) and pool(0) |
| Export functions (PNG/CSV/report) |
| Undo/redo after row/column drops |
