# Clustering Engine - Sequence Diagrams

## 1. CSV Import Flow

```mermaid
sequenceDiagram
    actor User
    participant Menu as Menu Bar
    participant Dialog as tinyfd
    participant Importer as CSVImporter
    participant DT as DataTable
    participant Stats as ColumnStatsCache
    participant GUI as Data Table View

    User->>Menu: Click "Open CSV"
    Menu->>Dialog: tinyfd_openFileDialog()
    Dialog-->>Menu: file path
    Menu->>Importer: load(path)
    Importer->>Importer: count_lines()
    Importer->>Dialog: progress callback
    loop For each line
        Importer->>Importer: parse row, detect types
        Importer->>Importer: handle missing tokens
    end
    Importer->>DT: set_data(mat, col_names)
    DT->>DT: scan NaN in data_
    DT->>DT: build missing_ mask
    Importer-->>Menu: CSVLoadResult{success}
    Menu->>Stats: set_data(&DT)
    Stats->>Stats: invalidate all
    Menu-->>User: Show row/col count
    GUI->>DT: render table with NaN highlight
```

## 2. KMeans Clustering Flow

```mermaid
sequenceDiagram
    actor User
    participant Panel as Clustering Panel
    participant KM as KMeans
    participant Dist as compute_distance_matrix
    participant AVX as l2_distance_avx2
    participant TP as ThreadPool
    participant Renderer as OpenGL Renderer
    participant VP as 3D Viewport

    User->>Panel: Set k=5, click "Run Clustering"
    Panel->>KM: fit(X)
    KM->>KM: validate input (X not empty, k <= n)
    KM->>KM: initialize_centroids() - KMeans++

    loop Until convergence (max 300 iter)
        KM->>Dist: compute_distance_matrix(X, centroids)
        Dist->>TP: enqueue per-chunk tasks
        loop For each chunk (parallel)
            TP->>AVX: compute_distances_to_centroids()
            AVX->>AVX: _mm256_loadu_ps / _mm256_fmadd_ps
            AVX-->>TP: distance chunk
        end
        TP-->>Dist: distances_ matrix filled
        KM->>KM: assign_clusters() - argmin per row
        KM->>KM: update_centroids() - mean per cluster
        KM->>KM: check_convergence() - inertia delta
    end

    KM->>KM: compute cluster sizes
    KM-->>Panel: labels, centroids, inertia, n_iter
    Panel->>Renderer: set_data(X, labels, centroids)
    Panel->>Renderer: set_metrics(inertia, n_iter)
    Panel->>VP: render_to_fbo() via FBO texture
    VP-->>User: Show clustered points + metrics
```

## 3. DBSCAN Flow

```mermaid
sequenceDiagram
    actor User
    participant Panel as Clustering Panel
    participant DB as DBSCAN
    participant AVX as l2_distance_avx2
    participant Renderer as OpenGL Renderer

    User->>Panel: Set eps=0.5, min_pts=5, DBSCAN, Run
    Panel->>DB: fit(X)

    loop For each unvisited point i
        DB->>DB: visited_[i] = true
        DB->>AVX: region_query(X, i, neighbors)
        AVX->>AVX: l2_distance_avx2(i, j) for all j
        AVX-->>DB: neighbors within epsilon
        alt neighbors >= min_pts
            DB->>DB: cluster_id++
            DB->>DB: expand_cluster(X, i, cluster_id)
            loop For each seed point
                DB->>AVX: region_query(X, seed)
                DB->>DB: add new neighbors to seeds
                DB->>DB: assign to cluster if unclassified
            end
        else neighbors < min_pts
            DB->>DB: mark as noise (-2)
        end
    end

    DB->>DB: relabel noise (-2) to cluster 0
    DB-->>Panel: labels, n_clusters, n_noise
    DB->>DB: compute dummy centroids from cluster means
    Panel->>Renderer: set_data(X, labels, dummy centroids)
    Renderer-->>User: Show clusters + noise in gray
```

## 4. Preprocessing with Undo/Redo

```mermaid
sequenceDiagram
    actor User
    participant Panel as Preprocessing Panel
    participant Pipeline as PreprocessPipeline
    participant DT as DataTable
    participant Stats as ColumnStatsCache

    User->>Panel: Select "Standardize", click "Apply"
    Panel->>Pipeline: standardize_column(col)
    Pipeline->>DT: read values, compute mean/std
    Pipeline->>Pipeline: create PreprocessAction
    loop For each cell
        Pipeline->>Pipeline: cells.push({row, col})
        Pipeline->>Pipeline: old_values.push(original)
        Pipeline->>DT: data_[row][col] = (v - mean) / std
    end
    Pipeline->>Pipeline: apply(action) -> history_.push_back
    Pipeline-->>Panel: action stored in history
    Panel->>Stats: invalidate()
    Stats->>Stats: recompute on next access

    User->>Panel: Click "Undo"
    Panel->>Pipeline: undo()
    Pipeline->>Pipeline: current_-- (move back)
    Pipeline->>Pipeline: get action = history_[current_]
    loop For each changed cell
        Pipeline->>DT: fill_value(row, col, old_values[k])
    end
    Pipeline-->>Panel: data restored
    Panel->>Stats: invalidate()

    User->>Panel: Click "Undo All"
    Panel->>Pipeline: while can_undo() undo()
    Pipeline-->>Panel: all operations reversed
```

## 5. Cluster Evaluation Flow

```mermaid
sequenceDiagram
    actor User
    participant Panel as Evaluation Panel
    participant Eval as ClusterEvaluator
    participant KM as KMeans
    participant Drift as DriftDetector
    participant GUI as GUI Plots

    User->>Panel: Set k range 2-15, click "Run Evaluation"
    Panel->>Eval: evaluate(X, min_k=2, max_k=15)

    loop For k = 2 to 15
        Eval->>KM: new KMeans(k).fit(X)
        KM-->>Eval: labels, centroids, inertia
        Eval->>Drift: check(X, labels, centroids)
        Drift->>Drift: compute_silhouette()
        Drift->>Drift: compute_davies_bouldin()
        Drift->>Drift: compute_calinski_harabasz()
        Drift-->>Eval: DriftMetrics
        Eval->>Eval: EvalResult{k, inertia, silhouette, DB, CH}
    end

    Eval->>Eval: best_k_silhouette(results)
    Eval-->>Panel: vector<EvalResult>, best_k
    GUI->>GUI: PlotLines("inertia", ...) - elbow curve
    GUI->>GUI: PlotLines("silhouette", ...) - peak marker
    GUI->>GUI: PlotLines("davies_bouldin", ...) - dip marker
    GUI-->>User: "Best k = N (by silhouette)"
    User->>Panel: Click "Use this k"
    Panel->>Panel: set k = best_k, algo = KMeans
```

## 6. 3D Viewport Render Flow

```mermaid
sequenceDiagram
    participant MainLoop as Main Loop (60 FPS)
    participant VP as ViewportChild
    participant FBO as OpenGL FBO
    participant GL as Renderer
    participant Shader as GPU Shaders
    participant Image as ImGui::Image

    MainLoop->>VP: BeginChild("ViewportChild")
    VP->>VP: GetContentRegionAvail() -> size
    alt size changed
        VP->>FBO: glDelete + create_fbo(new_w, new_h)
    end

    VP->>GL: render_to_fbo(fbo, w, h)
    GL->>FBO: glBindFramebuffer(fbo)
    GL->>FBO: glViewport, glClear

    GL->>GL: compute bounding box from pts
    GL->>GL: compute camera pos from rot_x/rot_y/zoom
    GL->>GL: build MVP matrix (frustum * lookAt)

    Note over GL,Shader: Draw coordinate axes (R/G/B lines)
    GL->>Shader: glUseProgram(line_prog)
    GL->>Shader: upload uMVP uniform
    GL->>Shader: glDrawArrays(GL_LINES, 0, 6)

    Note over GL,Shader: Draw data points (colored circles)
    GL->>Shader: glUseProgram(point_prog)
    GL->>Shader: upload uMVP + uSize
    GL->>FBO: glBufferData(positions + colors)
    GL->>Shader: glDrawArrays(GL_POINTS, 0, n_pts)

    Note over GL,Shader: Draw centroids (large white circles)
    GL->>Shader: upload uSize * 2.5
    GL->>FBO: glBufferData(centroid positions, white)
    GL->>Shader: glDrawArrays(GL_POINTS, 0, n_clust)

    GL->>FBO: glBindFramebuffer(0)

    VP->>Image: ImGui::Image(fbo_texture)
    Image-->>User: Rendered 3D view

    alt User drags mouse on viewport
        VP->>GL: rotate_view(dx, dy)
        GL->>GL: rot_y += dx, rot_x += dy
    else User scrolls
        VP->>GL: zoom_view(factor)
        GL->>GL: zoom *= factor
    else User presses R
        VP->>GL: reset_view()
        GL->>GL: rot_x=0.4, rot_y=0.5, zoom=1.0
    end
```
