#include "clustering/clustering.h"
#include "gui/data_table.h"
#include "gui/column_stats.h"
#include "gui/csv_importer.h"
#include "gui/preprocess_pipeline.h"
#include "gui/missing_handler.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#define TINYFILEDIALOGS_IMPLEMENTATION
#include "tinyfiledialogs.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <thread>
#include <atomic>
#include <cstring>
#include <chrono>

using namespace clustering;

// Global state
struct AppState {
    DataTable table;
    ColumnStatsCache stats;
    CSVImporter importer;
    MissingHandler missing_handler;
    bool data_loaded = false;
    bool loading = false;
    bool has_header = true;
    float load_progress = 0.0f;

    KMeans* kmeans = nullptr;
    DBSCAN* dbscan_obj = nullptr;
    int selected_algo = 0;
    int k = 5;
    int max_iter = 300;
    float tol = 1e-4f;
    int max_threads = 4;
    int batch_size = 100;
    int window_size = 1000;
    float forgetting_factor = 0.99f;
    bool online_auto_retrain = false;
    float drift_threshold = 0.1f;
    int retrain_interval = 1000;

    // DBSCAN params
    float dbscan_eps = 0.5f;
    int dbscan_min_pts = 5;

    bool realtime_viz = true;
    std::atomic<bool> clustering_running{false};
    bool clustering_done = false;
    float inertia = 0;
    int n_iter = 0;
    Vector labels;
    Matrix centroids;
    std::vector<size_t> cluster_sizes;
    std::vector<int> selected_cols;  // Columns selected for clustering (int for ImGui)
    size_t n_noise = 0;

    // Evaluation
    bool eval_done = false;
    std::vector<EvalResult> eval_results;
    int eval_min_k = 2;
    int eval_max_k = 15;
    int eval_best_k = 0;
    bool eval_running = false;
    std::thread eval_thread;
    std::thread cluster_thread;
    std::atomic<int> eval_progress_k{0};

    // Current cluster evaluation
    float current_silhouette = 0;
    float current_db = 0;
    float current_ch = 0;
    bool current_eval_done = false;

    // Comparison history
    struct CompareEntry {
        std::string algo;
        int k;
        float silhouette;
        float db;
        float ch;
        float inertia;
    };
    std::vector<CompareEntry> compare_history;

    std::mutex result_mutex;

    // For PCA pre-reduction
    bool use_pca = false;
    int pca_components = 10;
    Matrix reduced_data;
    Vector pca_var_ratio;
    float pca_total_var = 0;

    // Status bar
    std::string status_text;
    float status_time = 0;

    // Selected column for stats
    int selected_col = 0;

    // Preprocessing UI
    int preprocess_op = 0;
    int missing_strategy_idx = 2; // Median impute
    bool apply_to_all_columns = true;

    // 3D viewport
    GLuint fbo = 0;
    GLuint fbo_texture = 0;
    GLuint fbo_rbo = 0;
    int fbo_width = 800;
    int fbo_height = 600;
    Renderer* renderer_obj = nullptr;
    bool renderer_ready = false;

    // Export
    char export_path[256] = "clusters.png";
    int export_width = 1920;
    int export_height = 1080;

    // Data table view
    int sort_col = -1;
    bool sort_ascending = true;
    std::vector<size_t> sorted_indices;

    bool show_table = true;
    bool show_viewport = true;
} g;

static void create_fbo(int w, int h) {
    glGenFramebuffers(1, &g.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g.fbo);
    glGenTextures(1, &g.fbo_texture);
    glBindTexture(GL_TEXTURE_2D, g.fbo_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g.fbo_texture, 0);
    glGenRenderbuffers(1, &g.fbo_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, g.fbo_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, g.fbo_rbo);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    g.fbo_width = w;
    g.fbo_height = h;
}

static Matrix extract_selected_cols(const Matrix& src, const std::vector<int>& sel) {
    std::vector<size_t> cols;
    for (size_t j = 0; j < sel.size(); ++j)
        if (sel[j]) cols.push_back(j);
    if (cols.empty() || cols.size() == src.cols()) return src;
    Matrix out(src.rows(), cols.size());
    for (size_t i = 0; i < src.rows(); ++i)
        for (size_t j = 0; j < cols.size(); ++j)
            out[i][j] = src[i][cols[j]];
    return out;
}

static void run_clustering_async() {
    if (!g.data_loaded || g.table.rows() == 0) return;
    if (g.cluster_thread.joinable()) g.cluster_thread.join();
    g.clustering_done = false;
    g.clustering_running = true;
    g.cluster_thread = std::thread([]() {
        g.status_text = "Clustering...";
        g.status_time = 999.0f;
        const Matrix& raw = g.use_pca && g.reduced_data.rows() > 0 ? g.reduced_data : g.table.data();
        const Matrix X = extract_selected_cols(raw, g.selected_cols);
        delete g.kmeans; g.kmeans = nullptr;

        // Local copies for thread-safe computation
        Vector loc_labels; Matrix loc_centroids;
        float loc_inertia = 0; int loc_iter = 0; size_t loc_k = 0; size_t loc_noise = 0;
        int selected = g.selected_algo;

        if (selected == 0) {
            KMeansConfig cfg; cfg.k = (size_t)g.k; cfg.max_iter = g.max_iter; cfg.max_threads = g.max_threads;
            if (g.realtime_viz) {
                cfg.iter_callback = [](size_t iter, const Matrix& centroids, const Vector& labels) -> bool {
                    {
                        std::lock_guard<std::mutex> lk(g.result_mutex);
                        g.centroids = centroids;
                        g.labels = labels;
                        g.n_iter = (int)iter;
                    }
                    g.status_text = "KMeans iter " + std::to_string(iter) + "/" + std::to_string(g.max_iter);
                    g.status_time = 999.0f;
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    return false;
                };
            }
            auto* km = new KMeans(cfg); km->fit(X);
            loc_labels = km->labels(); loc_centroids = km->centroids();
            loc_inertia = km->inertia(); loc_iter = (int)km->n_iter();
            g.kmeans = km; loc_k = g.k;
        } else if (selected == 1) {
            auto* km = new MiniBatchKMeans(g.k, g.batch_size); km->fit(X);
            loc_labels = km->labels(); loc_centroids = km->centroids();
            loc_inertia = km->inertia(); loc_iter = (int)km->n_iter();
            g.kmeans = km; loc_k = g.k;
        } else if (selected == 2) {
            OnlineConfig cfg;
            cfg.k = g.k;
            cfg.window_size = (size_t)g.window_size;
            cfg.forgetting_factor = g.forgetting_factor;
            cfg.auto_retrain = g.online_auto_retrain;
            cfg.drift_threshold = g.drift_threshold;
            cfg.retrain_interval = (size_t)g.retrain_interval;

            // Add batch-level animation if realtime viz enabled
            if (g.realtime_viz) {
                cfg.iter_callback = [](size_t iter, const Matrix& centroids, const Vector& labels) -> bool {
                    {
                        std::lock_guard<std::mutex> lk(g.result_mutex);
                        g.centroids = centroids;
                        g.labels = labels;
                        g.n_iter = (int)iter;
                    }
                    g.status_text = "OnlineKMeans update " + std::to_string(iter);
                    g.status_time = 999.0f;
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    return false;
                };
            }

            auto* km = new OnlineKMeans(cfg); km->fit(X);
            loc_labels = km->labels(); loc_centroids = km->centroids();
            loc_inertia = km->inertia(); loc_iter = (int)km->n_iter();
            g.kmeans = km; loc_k = g.k;
        } else if (selected == 3) {
            DBSCANConfig dbcfg; dbcfg.epsilon = g.dbscan_eps; dbcfg.min_pts = (size_t)g.dbscan_min_pts;
            DBSCAN db(dbcfg); db.fit(X);
            loc_labels = db.labels(); loc_k = db.n_clusters(); loc_noise = db.n_noise();
        }

        // Auto-evaluate BEFORE mutex block
        Matrix ecent = loc_centroids;
        if (ecent.rows() == 0 && loc_k > 0) {
            ecent.resize(loc_k + 1, X.cols()); ecent.fill(0);
            std::vector<size_t> ec(loc_k + 1, 0);
            for (size_t i = 0; i < loc_labels.size(); ++i) {
                size_t lbl = (size_t)loc_labels[i];
                if (lbl <= loc_k) { for (size_t d = 0; d < X.cols(); ++d) ecent[lbl][d] += X[i][d]; ec[lbl]++; }
            }
            for (size_t c = 0; c <= loc_k; ++c) if (ec[c] > 0) for (size_t d = 0; d < X.cols(); ++d) ecent[c][d] /= (float)ec[c];
        }
        DriftDetector det;
        DriftMetrics dm = det.check(X, loc_labels, ecent.rows() > 0 ? ecent : loc_centroids);

        // Copy to global state under mutex
        {
            std::lock_guard<std::mutex> lk(g.result_mutex);
            g.labels = std::move(loc_labels);
            g.centroids = std::move(loc_centroids);
            g.inertia = loc_inertia;
            g.n_iter = loc_iter;
            g.k = (int)loc_k;
            g.n_noise = loc_noise;
            g.cluster_sizes.clear();
            size_t nc = selected == 3 ? loc_k + 1 : std::max(loc_k, (size_t)1);
            g.cluster_sizes.resize(nc, 0);
            for (size_t i = 0; i < g.labels.size(); ++i) {
                size_t lbl = (size_t)g.labels[i];
                if (lbl < g.cluster_sizes.size()) g.cluster_sizes[lbl]++;
            }
            g.clustering_done = true;
            g.current_silhouette = dm.silhouette_score;
            g.current_db = dm.davies_bouldin_index;
            g.current_ch = dm.calinski_harabasz_score;
            g.current_eval_done = true;
            static const char* algo_names[] = {"KMeans","MiniBatch","Online","DBSCAN"};
            g.compare_history.push_back({algo_names[selected], (int)loc_k, g.current_silhouette, g.current_db, g.current_ch, loc_inertia});
            if (g.compare_history.size() > 20) g.compare_history.erase(g.compare_history.begin());
        }
        g.clustering_running = false;

        g.status_text = "Clustering done: inertia=" + std::to_string(loc_inertia).substr(0,8);
        g.status_time = 5.0f;
    });
}

static void open_csv() {
    const char* path = tinyfd_openFileDialog("Open CSV", "", 1, nullptr, "CSV files", 0);
    if (!path) return;
    fprintf(stderr, "[csv] Loading: %s\n", path);

    CSVLoadResult res = g.importer.load(path, g.has_header, ',', -1,
        [](size_t loaded, size_t total) {
            g.load_progress = total > 0 ? (float)loaded / total : 0.0f;
        });

    if (res.success) {
        fprintf(stderr, "[csv] Loaded %zu rows, %zu cols\n", res.table.rows(), res.table.cols());
        g.table = std::move(res.table);
        g.stats.set_data(&g.table);
        g.stats.invalidate();
        g.data_loaded = true;
        g.clustering_done = false;
        g.reduced_data = Matrix();
        g.sorted_indices.clear();
        g.sort_col = -1;
        delete g.kmeans; g.kmeans = nullptr;
        g.labels = Vector(); g.centroids = Matrix(); g.cluster_sizes.clear();
        if (g.renderer_ready) {
            g.renderer_obj->set_data(Matrix(), Vector(), Matrix());
            g.renderer_obj->set_metrics(0, 0);
            g.renderer_obj->reset_view();
        }
        g.status_text = "Loaded: " + std::to_string(res.table.rows()) + " rows x " + std::to_string(res.table.cols()) + " cols";
        g.status_time = 3.0f;
    } else {
        fprintf(stderr, "[csv] Load failed: %s\n", res.error.c_str());
        g.status_text = "Load failed: " + res.error;
        g.status_time = 5.0f;
    }
}

static void export_labels_csv() {
    const char* path = tinyfd_saveFileDialog("Export Labels", "labels.csv", 0, nullptr, nullptr);
    if (!path) return;
    FILE* f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "id,cluster\n");
    for (size_t i = 0; i < g.labels.size(); ++i)
        fprintf(f, "%zu,%.0f\n", i, g.labels[i]);
    fclose(f);
}

static void export_centroids_csv() {
    const char* path = tinyfd_saveFileDialog("Export Centroids", "centroids.csv", 0, nullptr, nullptr);
    if (!path) return;
    FILE* f = fopen(path, "w");
    if (!f) return;
    for (size_t i = 0; i < g.centroids.cols(); ++i) {
        if (i > 0) fprintf(f, ",");
        fprintf(f, "dim_%zu", i);
    }
    fprintf(f, "\n");
    for (size_t i = 0; i < g.centroids.rows(); ++i) {
        for (size_t j = 0; j < g.centroids.cols(); ++j) {
            if (j > 0) fprintf(f, ",");
            fprintf(f, "%f", g.centroids[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

static void export_png() {
    const char* path = tinyfd_saveFileDialog("Export PNG", "clusters.png", 1, nullptr, nullptr);
    if (!path) return;
    int w = g.export_width, h = g.export_height;

    // Render viewport to offscreen FBO at export resolution
    GLuint exp_fbo, exp_tex, exp_rbo;
    glGenFramebuffers(1, &exp_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, exp_fbo);
    glGenTextures(1, &exp_tex);
    glBindTexture(GL_TEXTURE_2D, exp_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, exp_tex, 0);
    glGenRenderbuffers(1, &exp_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, exp_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, exp_rbo);

    g.renderer_obj->render_to_fbo(exp_fbo, w, h);
    glFinish();

    // Read pixels (OpenGL reads bottom-up, we need top-down for PNG)
    std::vector<unsigned char> pixels(w * h * 3);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // Flip vertically: OpenGL origin is bottom-left, image origin is top-left
    std::vector<unsigned char> flipped(w * h * 3);
    for (int y = 0; y < h; ++y)
        memcpy(&flipped[y * w * 3], &pixels[(h - 1 - y) * w * 3], w * 3);

    stbi_write_png(path, w, h, 3, flipped.data(), w * 3);

    glDeleteFramebuffers(1, &exp_fbo);
    glDeleteTextures(1, &exp_tex);
    glDeleteRenderbuffers(1, &exp_rbo);
    g.status_text = "PNG exported: " + std::string(path) + " (" + std::to_string(w) + "x" + std::to_string(h) + ")";
    g.status_time = 3.0f;
}

static void export_report() {
    const char* path = tinyfd_saveFileDialog("Export Report", "clustering_report.txt", 0, nullptr, nullptr);
    if (!path) return;
    FILE* f = fopen(path, "w");
    if (!f) return;

    fprintf(f, "========================================\n");
    fprintf(f, "  Clustering Engine - Analysis Report\n");
    fprintf(f, "========================================\n\n");

    fprintf(f, "Dataset: %zu rows x %zu columns\n", g.table.rows(), g.table.cols());
    fprintf(f, "Missing cells: %zu\n", g.table.missing_total());
    fprintf(f, "Columns: ");
    for (size_t j = 0; j < g.table.cols(); ++j)
        fprintf(f, "%s%s", g.table.column_names()[j].c_str(), j+1 < g.table.cols() ? ", " : "\n");

    // Preprocessing history
    const auto& hist = g.table.pipeline().history();
    fprintf(f, "\n--- Preprocessing (%zu operations) ---\n", hist.size());
    for (size_t i = 0; i < hist.size(); ++i)
        fprintf(f, "  %zu. %s\n", i+1, hist[i].description.c_str());

    // PCA
    if (g.reduced_data.rows() > 0) {
        fprintf(f, "\n--- PCA ---\n");
        fprintf(f, "Components: %d\n", g.pca_components);
        fprintf(f, "Total variance retained: %.1f%%\n", g.pca_total_var * 100);
        for (size_t i = 0; i < g.pca_var_ratio.size(); ++i)
            fprintf(f, "  C%zu: %.1f%%\n", i+1, g.pca_var_ratio[i] * 100);
    }

    // Clustering
    if (g.clustering_done) {
        fprintf(f, "\n--- Clustering ---\n");
        static const char* algos[] = {"KMeans","MiniBatchKMeans","OnlineKMeans","DBSCAN"};
        fprintf(f, "Algorithm: %s\n", algos[g.selected_algo]);
        fprintf(f, "k = %d\n", g.k);
        fprintf(f, "Inertia: %.4f\n", g.inertia);
        fprintf(f, "Iterations: %d\n", g.n_iter);
        fprintf(f, "Cluster sizes:\n");
        for (size_t c = 0; c < g.cluster_sizes.size(); ++c)
            fprintf(f, "  Cluster %zu: %zu points (%.1f%%)\n", c, g.cluster_sizes[c],
                    100.0f * g.cluster_sizes[c] / g.labels.size());
        fprintf(f, "\nCentroids (%zu x %zu):\n", g.centroids.rows(), g.centroids.cols());
        for (size_t i = 0; i < g.centroids.rows(); ++i) {
            fprintf(f, "  C%zu: ", i);
            for (size_t j = 0; j < g.centroids.cols(); ++j) {
                if (j > 0) fprintf(f, ", ");
                fprintf(f, "%.4f", g.centroids[i][j]);
            }
            fprintf(f, "\n");
        }
    }

    fprintf(f, "\n========================================\n");
    fprintf(f, "  End of Report\n");
    fprintf(f, "========================================\n");
    fclose(f);
    g.status_text = "Report exported: " + std::string(path);
    g.status_time = 3.0f;
}

static void apply_preprocess() {
    if (!g.data_loaded) return;
    static const char* ops[] = {"Normalize", "Standardize", "MinMax Scale", "Log Transform", "Clip Outliers"};

    if (g.apply_to_all_columns) {
        for (size_t j = 0; j < g.table.cols(); ++j) {
            switch (g.preprocess_op) {
                case 0: g.table.pipeline().normalize_column(j); break;
                case 1: g.table.pipeline().standardize_column(j); break;
                case 2: g.table.pipeline().minmax_scale_column(j); break;
                case 3: g.table.pipeline().log_transform_column(j); break;
                case 4: g.table.pipeline().clip_outliers_column(j); break;
            }
        }
        g.status_text = std::string(ops[g.preprocess_op]) + " applied to all " + std::to_string(g.table.cols()) + " columns";
    } else {
        size_t col = (size_t)g.selected_col;
        switch (g.preprocess_op) {
            case 0: g.table.pipeline().normalize_column(col); break;
            case 1: g.table.pipeline().standardize_column(col); break;
            case 2: g.table.pipeline().minmax_scale_column(col); break;
            case 3: g.table.pipeline().log_transform_column(col); break;
            case 4: g.table.pipeline().clip_outliers_column(col); break;
        }
        g.status_text = std::string(ops[g.preprocess_op]) + " applied to column " + g.table.column_names()[col];
    }
    g.status_time = 3.0f;
    g.stats.invalidate();
    g.clustering_done = false;
}

static void apply_missing() {
    if (!g.data_loaded) return;
    auto strategies = MissingHandler::all_strategies();
    MissingHandlerConfig cfg;
    cfg.strategy = strategies[g.missing_strategy_idx];
    g.missing_handler.set_config(cfg);
    if (g.apply_to_all_columns) {
        g.missing_handler.apply(g.table);
    } else {
        g.missing_handler.apply_to_column(g.table, (size_t)g.selected_col);
    }
    g.stats.invalidate();
    g.clustering_done = false;
    auto strats = MissingHandler::all_strategies();
    g.status_text = std::string("Missing data: ") + MissingHandler::strategy_name(strats[g.missing_strategy_idx]);
    g.status_time = 3.0f;
}

static void undo_all() {
    if (!g.data_loaded) return;
    size_t total = g.table.pipeline().history().size();
    while (g.table.pipeline().can_undo()) g.table.pipeline().undo();
    g.stats.invalidate();
    g.clustering_done = false;
    g.status_text = "Undone all " + std::to_string(total) + " operations";
    g.status_time = 3.0f;
}

static void run_pca() {
    if (!g.data_loaded || g.pca_components < 2) return;
    g.clustering_done = false;
    g.current_eval_done = false;
    if (g.renderer_ready) { g.renderer_obj->set_data(Matrix(), Vector(), Matrix()); g.renderer_obj->set_metrics(0,0); }
    PCA pca(g.pca_components);
    g.reduced_data = pca.fit_transform(g.table.data());
    g.pca_var_ratio = pca.explained_variance_ratio();
    g.pca_total_var = pca.total_explained_variance_ratio();
    g.status_text = "PCA: " + std::to_string(g.pca_components) + " components, " +
                    std::to_string((int)(g.pca_total_var * 100)) + "% variance retained";
    g.status_time = 5.0f;
}

static void undo_pca() {
    g.reduced_data = Matrix();
    g.pca_var_ratio = Vector();
    g.pca_total_var = 0;
    g.use_pca = false;
    g.clustering_done = false;
    g.status_text = "PCA undone, restored original data";
    g.status_time = 3.0f;
}

static void run_evaluation() {
    if (!g.data_loaded) return;
    if (g.eval_thread.joinable()) g.eval_thread.join();
    g.eval_running = true;
    g.eval_done = false;
    g.eval_results.clear();
    g.eval_progress_k = 0;

    int min_k = g.eval_min_k, max_k = g.eval_max_k;
    bool use_pca = g.use_pca;
    Matrix reduced = g.reduced_data;
    Matrix table_data = g.table.data();

    g.eval_thread = std::thread([min_k, max_k, use_pca, reduced, table_data]() {
        const Matrix& X = use_pca && reduced.rows() > 0 ? reduced : table_data;
        g.eval_results.clear();
        g.eval_progress_k = (int)min_k;

        for (int k = min_k; k <= max_k; ++k) {
            g.eval_progress_k = k;
            g.status_text = "Evaluating k=" + std::to_string(k) + "/" + std::to_string(max_k) + "...";
            g.status_time = 999.0f;

            KMeansConfig cfg;
            cfg.k = (size_t)k;
            cfg.max_iter = 100;
            cfg.max_threads = 4;
            KMeans km(cfg);
            km.fit(X);

            EvalResult r;
            r.k = k;
            r.inertia = km.inertia();

            const size_t SILHOUETTE_MAX_N = 1000;
            DriftDetector detector;
            if (X.rows() <= SILHOUETTE_MAX_N) {
                DriftMetrics dm = detector.check(X, km.labels(), km.centroids());
                r.silhouette_score = dm.silhouette_score;
                r.davies_bouldin = dm.davies_bouldin_index;
                r.calinski_harabasz = dm.calinski_harabasz_score;
            } else {
                size_t sn = SILHOUETTE_MAX_N;
                size_t step = X.rows() / sn;
                Matrix Xs(sn, X.cols());
                Vector labels_s(sn);
                for (size_t i = 0; i < sn; ++i) {
                    size_t idx = i * step;
                    for (size_t d = 0; d < X.cols(); ++d) Xs[i][d] = X[idx][d];
                    labels_s[i] = km.labels()[idx];
                }
                DriftMetrics dm = detector.check(Xs, labels_s, km.centroids());
                r.silhouette_score = dm.silhouette_score;
                r.davies_bouldin = dm.davies_bouldin_index;
                r.calinski_harabasz = dm.calinski_harabasz_score;
            }
            g.eval_results.push_back(r);
        }

        ClusterEvaluator eval;
        g.eval_best_k = (int)eval.best_k_silhouette(g.eval_results);
        g.eval_done = true;
        g.eval_running = false;
        g.status_text = "Evaluation done. Best k=" + std::to_string(g.eval_best_k) + " (by silhouette)";
        g.status_time = 5.0f;
    });
}

static void evaluate_current() {
    if (!g.clustering_done || g.labels.size() == 0) return;
    const Matrix& X = g.use_pca && g.reduced_data.rows() > 0 ? g.reduced_data : g.table.data();
    Matrix centroids = g.centroids;
    if (centroids.rows() == 0) {
        centroids.resize((size_t)g.k + 1, X.cols()); centroids.fill(0);
        std::vector<size_t> counts((size_t)g.k + 1, 0);
        for (size_t i = 0; i < X.rows(); ++i) {
            size_t lbl = (size_t)g.labels[i];
            if (lbl <= (size_t)g.k) {
                for (size_t d = 0; d < X.cols(); ++d) centroids[lbl][d] += X[i][d];
                counts[lbl]++;
            }
        }
        for (size_t c = 0; c <= (size_t)g.k; ++c)
            if (counts[c] > 0)
                for (size_t d = 0; d < X.cols(); ++d) centroids[c][d] /= (float)counts[c];
    }
    DriftDetector det;
    DriftMetrics dm = det.check(X, g.labels, centroids);
    g.current_silhouette = dm.silhouette_score;
    g.current_db = dm.davies_bouldin_index;
    g.current_ch = dm.calinski_harabasz_score;
    g.current_eval_done = true;
    static const char* algos[] = {"KMeans","MiniBatch","Online","DBSCAN"};
    g.compare_history.push_back({algos[g.selected_algo], g.k,
        g.current_silhouette, g.current_db, g.current_ch, g.inertia});
    if (g.compare_history.size() > 20) g.compare_history.erase(g.compare_history.begin());
    g.status_text = "Evaluated: Sil=" + std::to_string(g.current_silhouette).substr(0,5)
                  + " DB=" + std::to_string(g.current_db).substr(0,5);
    g.status_time = 5.0f;
}

int main() {
    fprintf(stderr, "[init] Starting...\n");
    if (!glfwInit()) { fprintf(stderr, "[FATAL] glfwInit failed\n"); return 1; }
    fprintf(stderr, "[init] GLFW OK\n");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 8);

    GLFWwindow* window = glfwCreateWindow(1600, 1000, "Clustering Engine - Interactive Tool", nullptr, nullptr);
    if (!window) { fprintf(stderr, "[FATAL] glfwCreateWindow failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    fprintf(stderr, "[init] Window created, GL context ready\n");
    glewExperimental = GL_TRUE;
    GLenum glew_err = glewInit();
    if (glew_err != GLEW_OK) { fprintf(stderr, "[FATAL] glewInit: %s\n", glewGetErrorString(glew_err)); return 1; }
    fprintf(stderr, "[init] GLEW OK, OpenGL %s\n", glGetString(GL_VERSION));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    fprintf(stderr, "[init] ImGui context created\n");
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    fprintf(stderr, "[init] ImGui backends initialized\n");

    g.renderer_obj = new Renderer(RendererConfig());
    g.renderer_ready = g.renderer_obj->init_headless();
    fprintf(stderr, "[init] Renderer headless: %s\n", g.renderer_ready ? "OK" : "FAILED");
    create_fbo(800, 600);
    fprintf(stderr, "[init] FBO created (%dx%d)\n", 800, 600);
    fprintf(stderr, "[init] Entering main loop\n");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // === TOP MENU BAR ===
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Open CSV...", "Ctrl+O")) open_csv();
                ImGui::Separator();
                if (ImGui::BeginMenu("Export")) {
                    if (ImGui::MenuItem("Labels CSV", nullptr, false, g.clustering_done)) export_labels_csv();
                    if (ImGui::MenuItem("Centroids CSV", nullptr, false, g.clustering_done)) export_centroids_csv();
                    if (ImGui::MenuItem("PNG Screenshot", nullptr, false, g.clustering_done)) export_png();
                    ImGui::Separator();
                    if (ImGui::MenuItem("Full Report (.txt)", nullptr, false, g.data_loaded)) export_report();
                    ImGui::EndMenu();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Exit", "Alt+F4")) glfwSetWindowShouldClose(window, GLFW_TRUE);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Edit")) {
                if (ImGui::MenuItem("Undo", "Ctrl+Z", false, g.data_loaded && g.table.pipeline().can_undo())) {
                    g.table.pipeline().undo(); g.stats.invalidate();
                }
                if (ImGui::MenuItem("Redo", "Ctrl+Y", false, g.data_loaded && g.table.pipeline().can_redo())) {
                    g.table.pipeline().redo(); g.stats.invalidate();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Help")) {
                if (ImGui::MenuItem("About")) ImGui::OpenPopup("About");
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        // About popup
        if (ImGui::BeginPopupModal("About", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Clustering Engine - Interactive Tool v1.0");
            ImGui::Text("High-performance C++ clustering with OpenGL visualization.");
            ImGui::Separator();
            ImGui::Text("Controls: Left-drag = rotate, Scroll = zoom, R = reset");
            if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        // === TOP TAB BAR ===
        static int active_tab = 0;
        ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetFrameHeight()));
        ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, 28));
        ImGui::Begin("##tabbar", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::Button("Preprocess", ImVec2(120, 22))) active_tab = 0;
        ImGui::SameLine();
        if (ImGui::Button("Cluster & Evaluate", ImVec2(150, 22))) active_tab = 1;
        ImGui::SameLine();
        ImGui::TextDisabled("| Data Table + 3D Viewport on right");
        ImGui::End();

        float tab_h = 28.0f;
        float top_offset = ImGui::GetFrameHeight() + tab_h;

        // === MAIN WINDOW (fills screen below tab bar) ===
        ImGui::SetNextWindowPos(ImVec2(0, top_offset));
        ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x,
                                        ImGui::GetIO().DisplaySize.y - top_offset));
        ImGui::Begin("Main", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoBringToFrontOnFocus);

        // --- LEFT SIDEBAR (resizable child) ---
        static float left_w = 350;
        ImGui::BeginChild("Sidebar", ImVec2(left_w, 0), ImGuiChildFlags_ResizeX, ImGuiWindowFlags_AlwaysVerticalScrollbar);

        if (active_tab == 0) {
        // IMPORT
        if (ImGui::CollapsingHeader("Import", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Header row", &g.has_header);
            if (ImGui::Button("Open CSV File", ImVec2(-1, 30))) open_csv();
            ImGui::Text("Data: %s", g.data_loaded ? "Loaded" : "None");
            if (g.data_loaded) {
                ImGui::Text("Rows: %zu  Cols: %zu", g.table.rows(), g.table.cols());
                ImGui::Text("Missing: %zu cells", g.table.missing_total());
            }
            if (g.loading) {
                ImGui::ProgressBar(g.load_progress, ImVec2(-1, 0));
                ImGui::Text("%.0f%%", g.load_progress * 100);
            }
            if (g.data_loaded && ImGui::Button("Edit Column Names", ImVec2(-1, 0))) {
                ImGui::OpenPopup("EditColumns");
            }
            if (ImGui::BeginPopupModal("EditColumns", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                static char col_names_buf[4096] = {};
                if (ImGui::IsWindowAppearing()) {
                    std::string names;
                    for (size_t j = 0; j < g.table.cols(); ++j) {
                        if (j > 0) names += ", ";
                        names += g.table.column_names()[j];
                    }
                    strncpy(col_names_buf, names.c_str(), sizeof(col_names_buf)-1);
                }
                ImGui::Text("Enter comma-separated column names:");
                ImGui::InputTextMultiline("##colnames", col_names_buf, sizeof(col_names_buf), ImVec2(400, 60));
                if (ImGui::Button("Apply")) {
                    std::string s(col_names_buf);
                    std::vector<std::string> new_names;
                    size_t pos = 0;
                    while (pos < s.size()) {
                        size_t comma = s.find(',', pos);
                        std::string name = s.substr(pos, comma - pos);
                        while (!name.empty() && name.front() == ' ') name.erase(0,1);
                        while (!name.empty() && name.back() == ' ') name.pop_back();
                        if (!name.empty()) new_names.push_back(name);
                        if (comma == std::string::npos) break;
                        pos = comma + 1;
                    }
                        if (!new_names.empty() && new_names.size() <= g.table.cols()) {
                        auto& cnames = g.table.column_names_mut();
                        for (size_t j = 0; j < g.table.cols(); ++j) {
                            if (j < new_names.size())
                                cnames[j] = new_names[j];
                            else
                                cnames[j] = "Col_" + std::to_string(j);
                        }
                        ImGui::CloseCurrentPopup();
                        g.status_text = "Column names updated";
                        g.status_time = 2.0f;
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
            }
        }

        // COLUMN STATS
        if (ImGui::CollapsingHeader("Column Stats", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (g.data_loaded && g.table.cols() > 0) {
                size_t ncols = g.table.cols();
                if (g.selected_col >= (int)ncols) g.selected_col = 0;
                std::string preview;
                for (size_t j = 0; j < ncols; ++j) {
                    if (j > 0) preview += '\0';
                    preview += g.table.column_names()[j];
                }
                ImGui::Combo("Column", &g.selected_col, preview.c_str());
                size_t col = (size_t)g.selected_col;
                const ColumnStats& s = g.stats.get(col);
                ImGui::Text("Count:   %zu", s.count);
                ImGui::Text("Missing: %zu (%.1f%%)", s.missing_count, s.count > 0 ? 100.0f*s.missing_count/s.count : 0.0);
                ImGui::Text("Mean:    %.4f", s.mean);
                ImGui::Text("Median:  %.4f", s.median);
                ImGui::Text("Std Dev: %.4f", s.std_dev);
                ImGui::Text("Min:     %.4f", s.min_val);
                ImGui::Text("Max:     %.4f", s.max_val);
                ImGui::Text("Q1:      %.4f", s.q1);
                ImGui::Text("Q3:      %.4f", s.q3);
                auto hd = g.stats.sample_for_histogram(col);
                if (!hd.empty()) ImGui::PlotHistogram("##hist", hd.data(), hd.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(-1, 50));
                ImGui::Separator();
                if (ImGui::Button("Rename Column", ImVec2(-1, 0))) {
                    ImGui::OpenPopup("RenameColumn");
                }
                if (ImGui::BeginPopupModal("RenameColumn", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                    static char rename_buf[128] = {};
                    if (ImGui::IsWindowAppearing()) {
                        strncpy(rename_buf, g.table.column_names()[col].c_str(), sizeof(rename_buf)-1);
                    }
                    ImGui::Text("New name for '%s':", g.table.column_names()[col].c_str());
                    ImGui::InputText("##newname", rename_buf, sizeof(rename_buf));
                    if (ImGui::Button("Apply")) {
                        g.table.column_names_mut()[col] = std::string(rename_buf);
                        ImGui::CloseCurrentPopup();
                        g.status_text = "Renamed to " + std::string(rename_buf);
                        g.status_time = 2.0f;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
                    ImGui::EndPopup();
                }
                if (ImGui::Button("Remove Column", ImVec2(-1, 0))) {
                    g.table.remove_column(col);
                    g.stats.invalidate();
                    g.clustering_done = false;
                    if (g.selected_col >= (int)g.table.cols()) g.selected_col = 0;
                    g.status_text = "Removed column";
                    g.status_time = 2.0f;
                }
            } else { ImGui::TextDisabled("No data"); }
        }

        // PREPROCESSING
        if (ImGui::CollapsingHeader("Preprocessing")) {
            if (g.data_loaded) {
                static const char* ops[] = {"Normalize","Standardize","MinMax Scale","Log Transform","Clip Outliers"};
                static const char* op_tips[] = {
                    "Scale each column so its L2 norm = 1 (sqrt of sum of squares). Makes all features comparable in magnitude.",
                    "Center each column to mean=0 and scale to standard deviation=1 (z-score). Most common ML preprocessing.",
                    "Rescale each column to range [0,1]. Formula: (x - min) / (max - min). Preserves distribution shape.",
                    "Apply natural logarithm to each value. Useful for skewed data (e.g., income, population). Clamps to 1e-10.",
                    "Cap values below 1st percentile and above 99th percentile to reduce outlier influence."
                };
                ImGui::Combo("Operation", &g.preprocess_op, ops, IM_ARRAYSIZE(ops));
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("%s", op_tips[g.preprocess_op]);
                ImGui::Checkbox("All columns", &g.apply_to_all_columns);
                if (ImGui::Button("Apply", ImVec2(-1, 22))) apply_preprocess();
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Apply the selected operation to %s", g.apply_to_all_columns ? "all numeric columns" : "the currently selected column only");

                ImGui::Separator();
                ImGui::Text("Missing Data:");
                auto strats = MissingHandler::all_strategies();
                std::string sn;
                for (auto s : strats) { if(!sn.empty()) sn+='\0'; sn+=MissingHandler::strategy_name(s); }
                ImGui::Combo("##mstrat", &g.missing_strategy_idx, sn.c_str());
                static const char* strat_tips[] = {
                    "Remove all rows that contain at least one missing (NaN) value.",
                    "Remove columns where the fraction of missing values exceeds the threshold (default 40%).",
                    "Replace missing values with the column mean. Good for normally distributed numeric data.",
                    "Replace missing values with the column median. Robust to outliers and skewed distributions.",
                    "Replace missing values with the most frequent value in the column.",
                    "Replace missing values with a constant number (default 0.0). Use for domain-specific sentinels.",
                    "Replace missing value with the previous valid value in the same column. Use for time series.",
                    "Linearly interpolate missing values between the nearest valid neighbors. Use for smooth trends."
                };
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", strat_tips[g.missing_strategy_idx]);
                if (ImGui::Button("Fill Missing", ImVec2(-1, 22))) apply_missing();
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Apply the selected missing-data strategy to all numeric columns");
                ImGui::Separator();
                if (ImGui::Button("Undo", ImVec2(55,0))){g.table.pipeline().undo();g.stats.invalidate();}
                ImGui::SameLine(); if(ImGui::Button("Redo", ImVec2(55,0))){g.table.pipeline().redo();g.stats.invalidate();}
                ImGui::SameLine(); if(ImGui::Button("Undo All", ImVec2(70,0))) undo_all();
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Revert all preprocessing and missing-data operations, restoring original data");
                ImGui::SameLine(); ImGui::Text("(%zu/%zu)", g.table.pipeline().current(), g.table.pipeline().history().size());
                ImGui::Separator();
                ImGui::Text("Data Operations:");
                if (ImGui::Button("Drop Rows with Missing", ImVec2(-1, 0))) {
                    g.table.drop_rows_with_missing();
                    g.stats.invalidate();
                    g.clustering_done = false;
                    g.status_text = "Dropped rows with missing values";
                    g.status_time = 2.0f;
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove all rows that contain at least one NaN value. Current: %zu rows", g.table.rows());
                static int remove_row_idx = 0;
                ImGui::SetNextItemWidth(80);
                ImGui::InputInt("Row #", &remove_row_idx, 0, 0);
                ImGui::SameLine();
                if (ImGui::Button("Remove Row", ImVec2(-1, 0))) {
                    if (remove_row_idx >= 0 && (size_t)remove_row_idx < g.table.rows()) {
                        g.table.remove_row((size_t)remove_row_idx);
                        g.stats.invalidate();
                        g.clustering_done = false;
                        g.status_text = "Removed row " + std::to_string(remove_row_idx);
                        g.status_time = 2.0f;
                    }
                }
                const auto& hist = g.table.pipeline().history();
                if (!hist.empty()) {
                    ImGui::BeginChild("HistList", ImVec2(0, ImGui::GetTextLineHeight() * 9), ImGuiChildFlags_Borders);
                    for (size_t i=0;i<hist.size();++i) {
                        bool cur=(i==g.table.pipeline().current()-1);
                        ImGui::TextColored(cur?ImVec4(0.3f,1,0.3f,1):ImVec4(0.5f,0.5f,0.5f,1),"%s %s",cur?">":" ",hist[i].description.c_str());
                    }
                    ImGui::EndChild();
                }
            } else { ImGui::TextDisabled("No data"); }
        }
        } // end tab 0 (Preprocess)

        if (active_tab == 1) {
        if (ImGui::CollapsingHeader("Find Optimal k", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (g.data_loaded) {
                ImGui::TextDisabled("Sweep KMeans to find optimal cluster count");
                ImGui::SliderInt("##kmin", &g.eval_min_k, 2, 10);
                ImGui::SameLine(); ImGui::Text("k Min");
                ImGui::SliderInt("##kmax", &g.eval_max_k, std::max(3,g.eval_min_k+1), 30);
                ImGui::SameLine(); ImGui::Text("k Max");
                ImGui::BeginDisabled(g.eval_running);
                if (ImGui::Button("Run Sweep", ImVec2(-1, 28))) run_evaluation();
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Run KMeans for k=%d..%d. Computes inertia (elbow method), silhouette, and Davies-Bouldin for each k.", g.eval_min_k, g.eval_max_k);
                if (g.eval_running) {
                    ImGui::Text("Running k=%d/%d...", g.eval_progress_k.load(), g.eval_max_k);
                    ImGui::ProgressBar((float)(g.eval_progress_k.load() - g.eval_min_k + 1) / (float)(g.eval_max_k - g.eval_min_k + 1));
                }
                if (g.eval_done && !g.eval_results.empty()) {
                    ClusterEvaluator eval;
                    size_t k_elbow = eval.best_k_elbow(g.eval_results);
                    size_t k_sil = eval.best_k_silhouette(g.eval_results);
                    size_t k_db = eval.best_k_db(g.eval_results);
                    ImGui::Text("Best k: %zu (elbow)  |  %zu (silhouette)  |  %zu (DB)", k_elbow, k_sil, k_db);
                    if (ImGui::Button(("Use k=" + std::to_string(k_elbow) + " (elbow)").c_str())) { g.k = (int)k_elbow; g.selected_algo = 0; }
                    ImGui::SameLine();
                    if (ImGui::Button("Clear")) { g.eval_done = false; g.eval_results.clear(); }
                    std::vector<float> inertias, silhouettes, db_scores;
                    for (auto& r : g.eval_results) { inertias.push_back(r.inertia); silhouettes.push_back(r.silhouette_score); db_scores.push_back(r.davies_bouldin); }
                    ImGui::Text("Inertia (elbow method):"); if (ImGui::IsItemHovered()) ImGui::SetTooltip("Lower = tighter clusters. Look for the 'elbow' where the curve bends. The k at the bend is the optimal cluster count.");
                    ImGui::PlotLines("##e_inertia", inertias.data(), inertias.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(-1, 40));
                    ImGui::Text("Silhouette (higher=better):"); if (ImGui::IsItemHovered()) ImGui::SetTooltip("Range [-1,1]. Peak = optimal k. Measures how well-separated clusters are.");
                    ImGui::PlotLines("##e_sil", silhouettes.data(), silhouettes.size(), 0, nullptr, -1.0f, 1.0f, ImVec2(-1, 40));
                    ImGui::Text("Davies-Bouldin (lower=better):"); if (ImGui::IsItemHovered()) ImGui::SetTooltip("Lower = better separated clusters. Dip (minimum) = optimal k. Measures average cluster similarity.");
                    ImGui::PlotLines("##e_db", db_scores.data(), db_scores.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(-1, 40));
                }
            } else { ImGui::TextDisabled("No data"); }
        }

        if (ImGui::CollapsingHeader("Clustering", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (g.data_loaded) {
                // Column selector
                if (ImGui::TreeNodeEx("Select Columns", ImGuiTreeNodeFlags_DefaultOpen)) {
                    size_t nsel = 0;
                    for (auto v : g.selected_cols) if (v) nsel++;
                    ImGui::Text("%zu / %zu columns selected", nsel, g.selected_cols.size());
                    ImGui::Separator();
                    for (size_t j = 0; j < g.table.cols() && j < g.selected_cols.size(); ++j) {
                        bool sel = g.selected_cols[j] != 0;
                        char label[64];
                        snprintf(label, sizeof(label), "##col%zu", j);
                        if (ImGui::Checkbox(label, &sel)) {
                            g.selected_cols[j] = sel ? 1 : 0;
                        }
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.9f, 1.0f), "%s", g.table.column_names()[j].c_str());
                    }
                    ImGui::Separator();
                    if (ImGui::SmallButton("Select All")) {
                        for (auto& v : g.selected_cols) v = 1;
                    }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Deselect All")) {
                        for (auto& v : g.selected_cols) v = 0;
                    }
                    ImGui::TreePop();
                }
                ImGui::Separator();

                static const char* algos[] = {"KMeans","MiniBatchKMeans","OnlineKMeans","DBSCAN"};
                static int prev_algo = 0;
                ImGui::Combo("Algorithm", &g.selected_algo, algos, IM_ARRAYSIZE(algos));
                if (g.selected_algo != prev_algo && g.clustering_done) {
                    g.clustering_done = false;
                    g.current_eval_done = false;
                    if (g.renderer_ready) { g.renderer_obj->set_data(Matrix(), Vector(), Matrix()); g.renderer_obj->set_metrics(0,0); }
                }
                prev_algo = g.selected_algo;
                if (g.selected_algo==3) {
                    ImGui::SliderFloat("Epsilon", &g.dbscan_eps, 0.01f, 200.0f, "%.3f");
                    ImGui::SliderInt("Min Points", &g.dbscan_min_pts, 2, 50);
                    if (ImGui::Button("Auto-Estimate Epsilon", ImVec2(-1, 0))) {
                        float ae = DBSCAN::estimate_epsilon(g.table.data(), (size_t)g.dbscan_min_pts, 500);
                        g.dbscan_eps = ae > 0 ? ae : g.dbscan_eps;
                        g.status_text = "Auto epsilon = " + std::to_string(g.dbscan_eps).substr(0,6);
                        g.status_time = 3.0f;
                    }
                } else {
                    ImGui::SliderInt("Clusters", &g.k, 1, std::min(50, (int)g.table.rows()-1));
                    if (g.selected_algo==1) ImGui::SliderInt("Batch Size", &g.batch_size, 10, 1000);
                    if (g.selected_algo==2) {
                        ImGui::SliderInt("Window", &g.window_size, 100, 10000);
                        ImGui::SliderFloat("Forget", &g.forgetting_factor, 0.8f, 1.0f);
                        ImGui::Checkbox("Auto Retrain on Drift", &g.online_auto_retrain);
                        if (g.online_auto_retrain) {
                            ImGui::SliderFloat("Drift Threshold", &g.drift_threshold, 0.01f, 0.5f);
                            ImGui::SliderInt("Check Interval", &g.retrain_interval, 100, 5000);
                        }
                    }
                }
                ImGui::Checkbox("PCA pre-reduce", &g.use_pca);
                if (g.use_pca) {
                    ImGui::SliderInt("Components", &g.pca_components, 2, std::min(50,(int)g.table.cols()));
                    if (ImGui::Button("Run PCA")) run_pca();
                    if (g.reduced_data.rows() > 0) {
                        ImGui::TextColored(ImVec4(0.3f,1,0.3f,1), "PCA: %zu x %d, %.0f%% var", g.reduced_data.rows(), g.pca_components, g.pca_total_var*100);
                        if (ImGui::Button("Undo PCA")) undo_pca();
                    } else ImGui::TextDisabled("Run PCA first");
                }
                static bool realtime_viz = true;
                ImGui::Checkbox("Animate iterations", &g.realtime_viz);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show centroids moving and points reassigning in real time during KMeans training. Adds ~50ms per iteration for visibility.");
                ImGui::BeginDisabled(g.eval_running || g.clustering_running);
                if (ImGui::Button("Run Clustering", ImVec2(-1, 30))) run_clustering_async();
                ImGui::EndDisabled();
                if (g.clustering_running) {
                    ImGui::TextDisabled("Running...");
                    ImGui::ProgressBar(-1.0f, ImVec2(-1, 0));
                }
                ImGui::Separator();
                if (ImGui::Button("Reset All", ImVec2(-1, 24))) {
                    g.clustering_done = false;
                    g.current_eval_done = false;
                    g.labels = Vector(); g.centroids = Matrix(); g.cluster_sizes.clear();
                    g.inertia = 0; g.n_iter = 0; g.n_noise = 0;
                    g.current_silhouette = 0; g.current_db = 0; g.current_ch = 0;
                    if (g.renderer_ready) { g.renderer_obj->set_data(Matrix(), Vector(), Matrix()); g.renderer_obj->set_metrics(0,0); g.renderer_obj->reset_view(); }
                }
                if (g.clustering_done) {
                    if (g.selected_algo==3)
                        ImGui::Text("Clusters: %d | Noise: %zu", g.k, g.n_noise);
                    else
                        ImGui::Text("Inertia: %.4f | Iter: %d", g.inertia, g.n_iter);
                    ImGui::Text("Silhouette: %.4f", g.current_silhouette);
                    ImGui::Text("Davies-Bouldin: %.4f", g.current_db);
                    for (size_t c=0;c<g.cluster_sizes.size();++c)
                        if (g.selected_algo!=3||c>0)
                            ImGui::Text(g.selected_algo==3&&c==0?"  Noise: %zu pts":"  C%zu: %zu pts",
                                       g.selected_algo==3&&c==0?g.cluster_sizes[c]:c, g.cluster_sizes[c]);
                }
            } else { ImGui::TextDisabled("No data"); }
        }
        } // end tab 1 (Cluster & Evaluate)

        left_w = ImGui::GetWindowWidth(); // track resize
        ImGui::EndChild();
        ImGui::SameLine();

        // --- RIGHT PANEL (table + viewport) ---
        ImGui::BeginChild("RightPanel", ImVec2(0, 0));

        if (active_tab == 0) { // Data table only in Preprocess tab
            float table_h_frac = g.show_viewport ? 0.5f : 1.0f;
            ImGui::BeginChild("TableChild", ImVec2(0, -ImGui::GetContentRegionAvail().y * (1.0f - table_h_frac)));
            if (g.data_loaded && g.table.rows() > 0) {
                size_t n = g.table.rows(), d = g.table.cols();
                if (ImGui::BeginTable("datatable", (int)d+1, ImGuiTableFlags_Borders|ImGuiTableFlags_RowBg|ImGuiTableFlags_ScrollX|ImGuiTableFlags_ScrollY|ImGuiTableFlags_Sortable|ImGuiTableFlags_Resizable)) {
                    ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 45);
                    for (size_t j=0;j<d;++j) ImGui::TableSetupColumn(g.table.column_names()[j].c_str());
                    ImGui::TableHeadersRow();
                    if (ImGuiTableSortSpecs* sorts = ImGui::TableGetSortSpecs()) {
                        if (sorts->SpecsDirty) {
                            g.sort_col = sorts->Specs->ColumnUserID;
                            g.sort_ascending = sorts->Specs->SortDirection == ImGuiSortDirection_Ascending;
                            if (g.sort_col>=0) {
                                g.sorted_indices.resize(n); for(size_t i=0;i<n;++i)g.sorted_indices[i]=i;
                                int scol=g.sort_col-1;
                                if(scol>=0&&(size_t)scol<d) std::sort(g.sorted_indices.begin(), g.sorted_indices.end(),[&](size_t a,size_t b){float va=g.table.data()[a][scol],vb=g.table.data()[b][scol];return g.sort_ascending?va<vb:va>vb;});
                            }
                            sorts->SpecsDirty=false;
                        }
                    }
                    ImGuiListClipper clipper; clipper.Begin((int)n);
                    while(clipper.Step()){for(int i=clipper.DisplayStart;i<clipper.DisplayEnd;++i){size_t ri=g.sorted_indices.empty()?(size_t)i:g.sorted_indices[i];ImGui::TableNextRow();ImGui::TableNextColumn();ImGui::Text("%d",i+1);for(size_t j=0;j<d;++j){ImGui::TableNextColumn();if(g.table.is_missing(ri,j))ImGui::TextColored(ImVec4(1,0.3f,0.3f,1),"NaN");else ImGui::Text("%.4f",g.table.data()[ri][j]);}}}
                    ImGui::EndTable();
                }
            } else { ImGui::TextDisabled("Open a CSV file to view data (File > Open CSV)"); }
            ImGui::EndChild();
        }

        if (active_tab == 1) { // 3D viewport only in Cluster tab
            ImGui::BeginChild("ViewportChild", ImVec2(0, 0));
            if (g.data_loaded && g.renderer_ready) {
                ImVec2 vp_size = ImGui::GetContentRegionAvail();
                int vp_w = (int)vp_size.x, vp_h = (int)std::max(vp_size.y, 50.0f);
                if (vp_w > 0 && vp_h > 0) {
                    if (vp_w != g.fbo_width || vp_h != g.fbo_height) {
                        glDeleteFramebuffers(1, &g.fbo); glDeleteTextures(1, &g.fbo_texture);
                        glDeleteRenderbuffers(1, &g.fbo_rbo);
                        create_fbo(vp_w, vp_h);
                    }
                    if (!g.clustering_done) {
                        const Matrix& X = g.use_pca && g.reduced_data.rows() > 0 ? g.reduced_data : g.table.data();
                        bool has_intermediate = false;
                        {
                            std::lock_guard<std::mutex> lk(g.result_mutex);
                            has_intermediate = g.centroids.rows() > 0 && g.labels.size() > 0;
                        }
                        if (has_intermediate && g.realtime_viz) {
                            std::lock_guard<std::mutex> lk(g.result_mutex);
                            g.renderer_obj->set_data(X, g.labels, g.centroids);
                            g.renderer_obj->set_metrics(0, g.n_iter);
                        } else {
                            Vector neutral_lbl(X.rows()); neutral_lbl.fill(-1.0f);
                            Matrix neutral_cnt(1, X.cols()); neutral_cnt.fill(0);
                            g.renderer_obj->set_data(X, neutral_lbl, neutral_cnt);
                            g.renderer_obj->set_metrics(0, 0);
                        }
                    } else {
                        // Set clustered data from main thread (avoids data race with bg thread)
                        const Matrix& X = g.use_pca && g.reduced_data.rows() > 0 ? g.reduced_data : g.table.data();
                        {
                            std::lock_guard<std::mutex> lk(g.result_mutex);
                            g.renderer_obj->set_data(X, g.labels, g.centroids.rows() > 0 ? g.centroids : Matrix(1, X.cols()));
                            g.renderer_obj->set_metrics(g.inertia, g.n_iter);
                        }
                    }
                    g.renderer_obj->render_to_fbo(g.fbo, vp_w, vp_h);
                    ImVec2 img_pos = ImGui::GetCursorScreenPos();
                    ImGui::Image((ImTextureID)(intptr_t)g.fbo_texture, ImVec2((float)vp_w, (float)vp_h), ImVec2(0,1), ImVec2(1,0));

                    // ImGui text overlay on viewport (replaces OpenGL bitmap font)
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    float tx = img_pos.x + 10, ty = img_pos.y + 8;
                    dl->AddText(ImVec2(tx, ty), IM_COL32(50,150,255,255), "CLUSTERING ENGINE"); ty += 18;
                    dl->AddText(ImVec2(tx, ty), IM_COL32(220,220,240,200), ("Points: " + std::to_string(g.clustering_done ? (g.use_pca && g.reduced_data.rows()>0 ? g.reduced_data.rows() : g.table.rows()) : g.table.rows())).c_str()); ty += 16;
                    if (g.clustering_done) {
                        dl->AddText(ImVec2(tx, ty), IM_COL32(220,220,240,200), ("Clusters: " + std::to_string(g.selected_algo==3 ? g.k : g.k)).c_str()); ty += 16;
                        if (g.selected_algo != 3) {
                            char buf[64]; snprintf(buf, sizeof(buf), "Inertia: %.1f", g.inertia); dl->AddText(ImVec2(tx, ty), IM_COL32(220,220,240,200), buf); ty += 16;
                            snprintf(buf, sizeof(buf), "Iterations: %d", g.n_iter); dl->AddText(ImVec2(tx, ty), IM_COL32(220,220,240,200), buf); ty += 16;
                        }
                        if (g.current_eval_done) {
                            char buf[64]; snprintf(buf, sizeof(buf), "Silhouette: %.4f", g.current_silhouette); dl->AddText(ImVec2(tx, ty), IM_COL32(100,255,100,220), buf); ty += 16;
                            snprintf(buf, sizeof(buf), "Davies-Bouldin: %.4f", g.current_db); dl->AddText(ImVec2(tx, ty), IM_COL32(255,200,100,220), buf); ty += 16;
                        }
                        // Per-cluster sizes
                        for (size_t c=0;c<g.cluster_sizes.size()&&c<10;++c) {
                            if (g.selected_algo==3 && c==0) continue; // skip noise
                            char buf[64]; snprintf(buf, sizeof(buf), "  C%zu: %zu pts", c, g.cluster_sizes[c]);
                            dl->AddText(ImVec2(tx, ty), IM_COL32(200,200,220,180), buf); ty += 14;
                        }
                    } else {
                        dl->AddText(ImVec2(tx, ty), IM_COL32(180,180,200,160), "Unclustered data (run clustering)"); ty += 16;
                    }

                    if (ImGui::IsItemHovered()) {
                        ImGuiIO& io = ImGui::GetIO();
                        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
                            g.renderer_obj->rotate_view(io.MouseDelta.x * 0.008f, io.MouseDelta.y * 0.008f);
                        if (io.MouseWheel != 0)
                            g.renderer_obj->zoom_view(1.0f + io.MouseWheel * 0.1f);
                        if (ImGui::IsKeyPressed(ImGuiKey_R))
                            g.renderer_obj->reset_view();
                    }
                    // Per-centroid labels (project 3D to 2D)
                    if (g.clustering_done && g.centroids.rows() > 0) {
                        static const float cpal[][3] = {{0.95f,0.3f,0.3f},{0.3f,0.65f,0.95f},{0.3f,0.9f,0.4f},{0.95f,0.85f,0.25f},{0.9f,0.35f,0.9f},{0.25f,0.9f,0.9f},{0.95f,0.6f,0.25f},{0.65f,0.35f,0.95f},{0.55f,0.85f,0.3f},{0.35f,0.35f,0.95f}};
                        for (size_t c = 0; c < (size_t)g.centroids.rows() && c < (size_t)g.cluster_sizes.size(); ++c) {
                            float sx, sy;
                            float cz = g.centroids.cols() > 2 ? g.centroids[c][2] : 0.0f;
                            if (g.renderer_obj->project_to_screen(g.centroids[c][0], g.centroids[c][1], cz, vp_w, vp_h, &sx, &sy)) {
                                float px = img_pos.x + sx, py = img_pos.y + sy;
                                int ci = (int)c % 10;
                                dl->AddCircleFilled(ImVec2(px, py), 5.0f, IM_COL32((int)(cpal[ci][0]*255),(int)(cpal[ci][1]*255),(int)(cpal[ci][2]*255),255));
                                char lbuf[64];
                                size_t sz = c < g.cluster_sizes.size() ? g.cluster_sizes[c] : 0;
                                snprintf(lbuf, sizeof(lbuf), "C%zu: %zu", c, sz);
                                dl->AddText(ImVec2(px + 8, py - 6), IM_COL32(255,255,255,220), lbuf);
                            }
                        }
                    }
                }
            } else {
                ImGui::TextDisabled("Load a CSV file to begin (File > Open CSV)");
            }
            if (g.renderer_ready) {
                ImGui::Spacing();
                if (ImGui::Button("Reset View (R)")) g.renderer_obj->reset_view();
                ImGui::SameLine();
                ImGui::TextDisabled("Drag=rotate | Scroll=zoom | R=reset");
            }
            ImGui::EndChild();
        }

        // Comparison history (right panel, below viewport)
        if (active_tab == 1 && !g.compare_history.empty()) {
            ImGui::Separator();
            ImGui::Text("Comparison History:");
            ImGui::SameLine();
            if (ImGui::Button("Clear")) g.compare_history.clear();
            if (ImGui::BeginTable("chist", 5, ImGuiTableFlags_Borders|ImGuiTableFlags_RowBg|ImGuiTableFlags_ScrollY, ImVec2(0,80))) {
                ImGui::TableSetupColumn("Algo"); ImGui::TableSetupColumn("k"); ImGui::TableSetupColumn("Silhouette"); ImGui::TableSetupColumn("DB"); ImGui::TableSetupColumn("Inertia");
                ImGui::TableHeadersRow();
                for (auto& e : g.compare_history) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::Text("%s", e.algo.c_str());
                    ImGui::TableNextColumn(); ImGui::Text("%d", e.k);
                    ImGui::TableNextColumn(); ImGui::Text("%.4f", e.silhouette);
                    ImGui::TableNextColumn(); ImGui::Text("%.4f", e.db);
                    ImGui::TableNextColumn(); ImGui::Text("%.1f", e.inertia);
                }
                ImGui::EndTable();
            }
        }

        ImGui::EndChild(); // RightPanel
        ImGui::End();      // Main

        // === STATUS BAR ===
        float status_h = ImGui::GetFrameHeight() * 2 + 8;
        ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetIO().DisplaySize.y - status_h));
        ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, status_h));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 4));
        ImGui::Begin("##status", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoSavedSettings);

        static const char* algo_names[] = {"KMeans","MiniBatch","Online","DBSCAN"};

        if (g.clustering_running) {
            if (g.selected_algo == 2) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Online  k=%d  W=%d  F=%.2f",
                    g.k, g.window_size, g.forgetting_factor);
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "|  Iter: %d  |  Running...", g.n_iter);
            } else {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "%s  k=%d  %s",
                    algo_names[g.selected_algo], g.k, g.use_pca ? "PCA" : "raw");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "|  Iter: %d  |  Running...", g.n_iter);
            }
        } else if (g.clustering_done) {
            if (g.selected_algo == 2) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Online  k=%d  W=%d  F=%.2f  %s",
                    g.k, g.window_size, g.forgetting_factor,
                    g.online_auto_retrain ? "Auto-Retrain" : "Manual");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                    "|  Inertia: %.2f  Iter: %d  Sil: %.4f  DB: %.4f",
                    g.inertia, g.n_iter, g.current_silhouette, g.current_db);
            } else {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "%s  k=%d", algo_names[g.selected_algo], g.k);
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                    "|  Inertia: %.2f  Iter: %d  Sil: %.4f  DB: %.4f",
                    g.inertia, g.n_iter, g.current_silhouette, g.current_db);
            }
        } else if (!g.status_text.empty()) {
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "%s", g.status_text.c_str());
            g.status_time -= ImGui::GetIO().DeltaTime;
            if (g.status_time <= 0) g.status_text.clear();
        } else if (g.data_loaded) {
            size_t nsel = 0;
            for (auto v : g.selected_cols) if (v) nsel++;
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "%zu rows  %zu cols  |  %zu selected for clustering",
                g.table.rows(), g.table.cols(), nsel);
        }
        ImGui::PopStyleVar();
        ImGui::End();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    delete g.kmeans;
    delete g.dbscan_obj;
    delete g.renderer_obj;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
