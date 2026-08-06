# Debug Plan — Clustering Engine

## Status: All 15 original bugs fixed. New infrastructure + 1 open item.

## Critical Bugs (7) — ALL FIXED

| # | File:Line | Issue | Fix | Status |
|---|-----------|-------|-----|--------|
| 1 | `src/core/mini_batch.cpp:70` | Learning rate never decays after initial `fit()`. `n_iter()` stuck at 0 for subsequent `partial_fit()` calls. | Track local iteration counter in MiniBatchKMeans. | Fixed |
| 2 | `src/core/dbscan.cpp:155` | `predict()` inner loop always finds first clustered point then breaks with `found=false`. Returns 0 (noise) for every point. | Rewrite predict: nearest core/border point in training data. | Fixed |
| 3 | `src/dimensionality/tsne.cpp:425` | `transform()` computes distance between `embedding_[i]` and `embedding_[j]` — ignores input `X` entirely. | Store original X; kNN in original space maps new points to embedding. | Fixed |
| 4 | `src/app/main.cpp:386` | `algos[3]` OOB (array has 3 elements). | Expand array to 4. | Fixed |
| 5 | `src/app/main.cpp:174` | `run_clustering_async` hardcodes max_iter/max_threads ignoring `g`. | Use user settings from `g`. | Fixed |
| 6 | `src/app/main.cpp:1003` | RBO leaked on every FBO resize. | Delete RBO before recreating FBO. | Fixed |
| 7 | `src/compute/distance_avx2.cpp` | `_mm256_fmadd_ps` without `#ifdef __FMA__` guard. | FMA compile-time check + scalar fallback. | Fixed |

## Medium Bugs (8) — ALL FIXED

| # | File:Line | Issue | Fix | Status |
|---|-----------|-------|-----|--------|
| 8 | `src/core/kmeans.cpp` | Roulette-wheel selection can fall through when all distances ≈ 0. | Random-centroid fallback. | Fixed |
| 9 | `src/operational/evaluation.cpp` | DriftDetector history accumulates across k evaluations. | Fresh DriftDetector per k iteration. | Fixed |
| 10 | `src/compute/thread_pool.cpp:192` | `ThreadPool::global()` double-init race. | `std::call_once`. | Fixed |
| 11 | `src/operational/versioning.cpp` | No validation of read metadata (k, d, n). | Bounds checking after read. | Fixed |
| 12 | `src/app/main.cpp:198` | `g.labels`/`g.centroids` read without mutex. | Acquire mutex in render loop. | Fixed |
| 13 | `src/app/main.cpp:235` | `compare_history` pushed from bg thread without mutex. | Guard with mutex. | Fixed |
| 14 | `src/gui/csv_importer.cpp` | `cancel_`/`loading_` not atomic. | Made atomic. | Fixed |
| 15 | `src/gui/preprocess_pipeline.cpp:62` | Redo marks cells NaN instead of recomputing. | `PreprocessAction.new_values` stored; undo→old, redo→new; row-drop redo re-drops. | Fixed (session 2) |

## Code Smells (7)

| # | File | Issue | Fix | Status |
|---|------|-------|-----|--------|
| 16 | `renderer.cpp` | ~200 lines duplicated between `render_frame()` and `render_to_fbo()`. | Extract shared `render_impl()`. | OPEN |
| 17 | `column_stats.cpp:27` | Static return `empty` — thread-unsafe. | Return pointer (`const ColumnStats*`, nullptr OOB). | Fixed (session 2) |
| 18 | `column_stats.cpp:66` | Unstable variance formula. | Welford's online algorithm. | Fixed (session 2) |
| 19 | `dbscan.cpp` | Duplicate `#include "clustering/logging.h"`. | Removed. | Fixed |
| 20 | `renderer.cpp` | Duplicate renderer.h include. | Removed. | Fixed |
| 21 | `online.cpp` | Per-point vector allocation in sliding window. | Pre-allocated ring buffer. | Fixed |
| 22 | `preprocess_pipeline.cpp` | `std` namespace shadowed. | Renamed. | Fixed |
| 23 | `tsne.cpp:50` | `config_.n_components` mutated during fit. | Local variable. | Fixed |

## Session 2 — Large-Data Infrastructure (NEW)

| Change | What | Status |
|--------|------|--------|
| Matrix disk backend | Matrices past global RAM cap (default 512 MiB, `set_matrix_ram_cap()`) spill to temp file; LRU row-block cache (~2 MiB blocks), mutex-guarded, dirty write-back, `data()` throws when disk-backed. | Done |
| Read/write path split | Const `operator[]` = read-only (no write-back); mutable marks block dirty. Halves IO for read-only algorithm passes. | Done |
| CSV streaming | `CSVImporter` pre-allocates, parses row-by-row, per-column NaN counts, trims, drops >90% NaN columns; Excel-style col names past Z. `DataTable::set_data(Matrix&&)` move overload. | Done |
| Renderer big-data path | Matrices >128 MiB referenced not copied per frame (caller must keep alive). | Done |
| Stress benchmark | `benchmarks/stress_bench.cpp` (`stress_bench` target): CSV gen or direct fill, RSS via VmHWM, scan + KMeans fit on disk-backed data. | Done |
| Column stats disk-safe | `data()` → `operator[]` (disk-backed safe). | Done |

## Verified Stress Numbers (5400rpm HDD box)

| Dataset | RAM cap | disk_backed | peak RSS | result |
|---------|---------|-------------|----------|--------|
| 200k×32 (24 MiB) | 512 MiB | no | +32 MiB | KMeans k=8/50it 5.6s |
| 2M×32 (244 MiB) | 128 MiB | yes | +197 MiB | KMeans k=8/10it 38.9s |
| 10M×32 (1.22 GB) | 256 MiB | yes | TBD | slow on HDD; read-only path fix halves IO |

## Test Gaps

| Missing Coverage | Status |
|------------------|--------|
| MiniBatchKMeans — no test at all | Done — 2 tests (delegate fit, learning-rate decay regression) |
| Concept drift detection when drift actually happens | Done — DetectsRealDrift |
| DBSCAN predict() correctness | Done — PredictAssignsNearPoints |
| t-SNE functional test with real data | Done — TSNE.FunctionalIris (finite, non-collapsed) |
| PCA high-dim (>32) AVX2 edge cases | Done — PCA.HighDimension64 |
| Thread pool pool(1) and pool(0) | Done — DistanceParity.ThreadCountsAgree (threads 0/1/4 parity) |
| Export functions (PNG/CSV/report) | OPEN — GUI/GLFW-bound, needs GUI test target |
| Undo/redo after row/column drops | OPEN — GUI-bound (DataTable), needs GUI test target |
| Golden values vs scikit-learn (real data) | Done — 8 tests, iris/wine/synthetic vs sklearn 1.9 (inertia, cluster sizes, DBSCAN, PCA variance) |
| Matrix disk backend | Done — 7 tests; 102/102 total |

## DBSCAN Bug Found by Golden Tests (Fixed)

Wine dataset (14 features) exposed that the KD-tree was hardcoded to
`KD_DIM=3`: all neighborhood searches ran in the first-3-dimensions
projection, so 4+ dimensional data produced wrong clusters (wine: 3 fake
clusters vs sklearn's correct 0). Fixed via nanoflann runtime dimension
(`DIM=-1`), which also fixes `estimate_epsilon`'s stack overflow
(`query_point[3]` filled for `d > 3`). Regression tests:
DBSCAN.HighDimNoTruncation, RealDataTest.DBSCANWineBothNoise.
