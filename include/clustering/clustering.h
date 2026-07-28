#pragma once

// ============================================================================
// Clustering Engine - Umbrella Header (one include to get EVERYTHING)
//
// #include "clustering/clustering.h" gives you access to:
//   - Matrix & Vector (data structures)
//   - KMeans, MiniBatchKMeans, OnlineKMeans (clustering algorithms)
//   - PCA, t-SNE (dimensionality reduction)
//   - DriftDetector, VersionManager, FeatureStore (operational tools)
//   - ThreadPool (parallel computation)
//   - Distance functions (AVX2-optimized math)
//   - Renderer (OpenGL 3D visualization)
//
// Alternatively, include individual headers if you only need one component.
// ============================================================================

#include "clustering/matrix.h"
#include "clustering/kmeans.h"
#include "clustering/thread_pool.h"
#include "clustering/distance.h"
#include "clustering/versioning.h"
#include "clustering/drift.h"
#include "clustering/renderer.h"
#include "clustering/online.h"
#include "clustering/mini_batch.h"
#include "clustering/pca.h"
#include "clustering/tsne.h"
#include "clustering/feature_store.h"
#include "clustering/dbscan.h"
#include "clustering/evaluation.h"
