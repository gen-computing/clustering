# Clustering Engine - Getting Started Guide

## Recommended Reading Order

### If you are new to this codebase, read in this order:

#### Layer 1: Foundation (start here)
| Order | File | Why read it |
|-------|------|-------------|
| 1 | `include/clustering/matrix.h` | Fundamental data type. Every other class uses Matrix. Understand row-major storage, operator[], resize. |
| 2 | `include/clustering/distance.h` | Distance functions. Core math used by every algorithm. See AVX2 SIMD optimization. |
| 3 | `src/compute/distance_avx2.cpp` | Implementation of AVX2 distance. Understand `__m256`, `_mm256_fmadd_ps`, horizontal sum. |
| 4 | `src/compute/thread_pool.cpp` | Parallel execution. Understand worker loop, condition variable, task queue. |

#### Layer 2: Core Algorithms
| Order | File | Why read it |
|-------|------|-------------|
| 5 | `include/clustering/kmeans.h` | KMeans class interface. Read header first to understand API. |
| 6 | `src/core/kmeans.cpp` | KMeans implementation: KMeans++ init, E-step, M-step, convergence check. |
| 7 | `include/clustering/online.h` | OnlineKMeans extends KMeans. Understand streaming concepts. |
| 8 | `src/core/online.cpp` | Sliding window, forgetting factor, drift-triggered retrain. |
| 9 | `include/clustering/dbscan.h` | DBSCAN interface. Density-based, no k needed. |
| 10 | `src/core/dbscan.cpp` | Region query, expand cluster, core/border/noise classification. |
| 11 | `include/clustering/mini_batch.h` + `.cpp` | Random batch sampling KMeans (simple, short files). |

#### Layer 3: Dimensionality Reduction
| Order | File | Why read it |
|-------|------|-------------|
| 12 | `include/clustering/pca.h` | PCA interface. Understand components, explained variance. |
| 13 | `src/dimensionality/pca.cpp` | Eigendecomposition via Eigen/BLAS, transform/inverse. |
| 14 | `include/clustering/tsne.h` | t-SNE interface. Perplexity, KL divergence. |
| 15 | `src/dimensionality/tsne.cpp` | Binary search for sigma, gradient descent with momentum. |

#### Layer 4: Operational Tools
| Order | File | Why read it |
|-------|------|-------------|
| 16 | `include/clustering/drift.h` | DriftDetector + DriftMetrics struct. |
| 17 | `src/operational/drift.cpp` | Silhouette, Davies-Bouldin, Calinski-Harabasz, cluster stability math. |
| 18 | `include/clustering/evaluation.h` | ClusterEvaluator: elbow, silhouette analysis for optimal k. |
| 19 | `src/operational/evaluation.cpp` | Run KMeans for k=2..N, collect metrics, find best k. |
| 20 | `include/clustering/versioning.h` + `.cpp` | Binary save/load of centroids and labels. |
| 21 | `include/clustering/feature_store.h` + `.cpp` | Preprocessing normalization + disk cache. |

#### Layer 5: GUI Data Layer
| Order | File | Why read it |
|-------|------|-------------|
| 22 | `src/gui/data_table.h` + `.cpp` | DataTable: Matrix + column names + missing mask + undo stack. |
| 23 | `src/gui/preprocess_pipeline.h` + `.cpp` | Diff-based undo/redo for column operations. |
| 24 | `src/gui/csv_importer.h` + `.cpp` | CSV parsing with type detection and missing handling. |
| 25 | `src/gui/column_stats.h` + `.cpp` | Cached per-column statistics with histogram sampling. |
| 26 | `src/gui/missing_handler.h` + `.cpp` | 8 strategies for missing data (drop, impute, interpolate). |

#### Layer 6: Visualization
| Order | File | Why read it |
|-------|------|-------------|
| 27 | `include/clustering/renderer.h` | OpenGL renderer API. Camera controls, FBO embedding. |
| 28 | `src/visualization/mat4.cpp` | 4x4 matrix math for 3D: perspective, lookAt, multiply. |
| 29 | `src/visualization/shaders.cpp` | GLSL vertex/fragment shader source code + compilation. |
| 30 | `src/visualization/text.cpp` | Bitmap font rendering for text overlay. |
| 31 | `src/visualization/renderer.cpp` | Main render loop: camera, draw axes, points, centroids, text. |

#### Layer 7: Integration
| Order | File | Why read it |
|-------|------|-------------|
| 32 | `examples/imgui_bench.cpp` | Full GUI application. Menu bar, panels, CSV loading, clustering, evaluation, export. |
| 33 | `include/clustering/clustering.h` | Umbrella header. See what's included. |
| 34 | `CMakeLists.txt` | Build system. Library sources, executable targets, dependencies. |

#### Layer 8: Documentation
| Order | File | Why read it |
|-------|------|-------------|
| 35 | `docs/ARCHITECTURE.md` | System data flow, module dependency graph, memory layout, threading. |
| 36 | `docs/SEQUENCE.md` | 6 sequence diagrams showing runtime behavior of key workflows. |
| 37 | `docs/MODULES.md` | 9 module deep-dives with Mermaid diagrams and algorithm explanations. |
| 38 | `README.md` | Overview, benchmarks, setup, API quick reference. |
| 39 | `IMPLEMENTATION_PLAN.md` | Current status, what's built, what's planned next. |

---

## Quick Orientation: What Does What

### Algorithms (what computes the clustering)
```
src/core/kmeans.cpp       → Standard KMeans (batch, KMeans++ init)
src/core/mini_batch.cpp   → Fast KMeans with random subsets
src/core/online.cpp       → Streaming KMeans with sliding window
src/core/dbscan.cpp       → Density-based clustering (no k needed)
src/dimensionality/pca.cpp → Dimensionality reduction via Eigen/BLAS
src/dimensionality/tsne.cpp→ 2D/3D visualization embedding
```

### Operational (what supports the algorithms)
```
src/operational/drift.cpp       → Quality metrics + concept drift detection
src/operational/evaluation.cpp  → Optimal k detection (elbow/silhouette/DB)
src/operational/versioning.cpp  → Save/load model versions
src/operational/feature_store.cpp → Preprocessing with caching
src/operational/rollback.cpp    → Version rollback utilities
src/compute/distance_avx2.cpp   → AVX2 SIMD distance computation
src/compute/thread_pool.cpp     → Parallel task execution
```

### GUI (what the user interacts with)
```
examples/imgui_bench.cpp        → Main application (menu, panels, layout)
src/gui/data_table.cpp          → Data storage + undo stack
src/gui/csv_importer.cpp        → CSV file loading
src/gui/column_stats.cpp        → Per-column statistics
src/gui/preprocess_pipeline.cpp → Preprocessing operations + undo/redo
src/gui/missing_handler.cpp     → Missing value strategies
src/visualization/renderer.cpp  → OpenGL 3D rendering
```

### Headers (what declares the API)
```
include/clustering/matrix.h      → Matrix + Vector data types
include/clustering/kmeans.h      → KMeans + MiniBatch + Online
include/clustering/dbscan.h      → DBSCAN
include/clustering/pca.h         → PCA
include/clustering/tsne.h        → t-SNE
include/clustering/distance.h    → Distance functions
include/clustering/thread_pool.h → Thread pool
include/clustering/drift.h       → Drift detection
include/clustering/evaluation.h  → Cluster evaluation
include/clustering/renderer.h    → OpenGL renderer
```

---

## Key Patterns Across the Codebase

### Pattern 1: Constructor Overloading
```cpp
// Every major class has two constructors:
KMeans(size_t k);                    // Simple: just specify k
KMeans(const KMeansConfig& config);  // Full: pass config struct
```
Applies to: KMeans, OnlineKMeans, DBSCAN, Renderer, TSNE

### Pattern 2: Input Validation
```cpp
// Every fit() method starts with:
if (X.rows() == 0) throw std::runtime_error("Empty input matrix");
if (X.cols() == 0) throw std::runtime_error("No features");
```
Applies to: KMeans, PCA, DBSCAN, OnlineKMeans

### Pattern 3: Fitted Guard
```cpp
// predict() and other post-fit methods check:
if (!fitted_) throw std::runtime_error("Model not fitted");
```
Applies to: KMeans, DBSCAN, PCA

### Pattern 4: Row-Major Pointer Access
```cpp
// operator[] returns pointer to row start:
float* operator[](size_t row) { return data_.data() + row * cols_; }
// Usage: m[row][col] (two pointer dereferences)
```
Applies to: Matrix (fundamental, used everywhere)

### Pattern 5: Global State for GLFW Callbacks
```cpp
// renderer.cpp uses global g_impl because GLFW callbacks
// are C function pointers without 'this' parameter:
static RendererImpl g_impl;
static void cb_key(GLFWwindow* w, int key, ...) {
    auto& r = g_impl;  // Access state from C callback
}
```
Applies to: renderer.cpp only

---

## Build and Run

```bash
# Build everything
cmake -B build -DCMAKE_BUILD_TYPE=Release
make -C build -j4 imgui_bench

# Run the GUI
./build/imgui_bench

# Run tests
make -C build tests && ./build/tests

# Run console benchmark
./build/benchmark
```

---

## Debugging Tips

### Enable verbose logging
```cpp
// In your source file, before includes:
#define CLUSTERING_DEBUG 1
```
Then watch stderr for: algorithm timing, distance dimensions, iteration counts, centroid deltas.

### Debug a specific algorithm
```cpp
// In examples/imgui_bench.cpp, in run_clustering():
fprintf(stderr, "X: %zu x %zu, k=%d\n", X.rows(), X.cols(), g.k);
// Output shows data dimensions and parameters before crash.
```

### Run a minimal test
```bash
# Create test.cpp:
#include "clustering/clustering.h"
using namespace clustering;
int main() {
    Matrix X(10, 2);
    for(size_t i=0;i<10;i++){ X[i][0]=i; X[i][1]=i*2; }
    KMeans km(2); km.fit(X);
    printf("OK: inertia=%.2f\n", km.inertia());
}
# g++ -std=c++20 -I include -L build -lclustering test.cpp -o test
```
