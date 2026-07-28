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

using namespace clustering;

// Global state
struct AppState {
    DataTable table;
    ColumnStatsCache stats;
    CSVImporter importer;
    MissingHandler missing_handler;
    bool data_loaded = false;
    bool loading = false;
    float load_progress = 0.0f;

    KMeans* kmeans = nullptr;
    int selected_algo = 0;
    int k = 5;
    int max_iter = 300;
    float tol = 1e-4f;
    int max_threads = 4;
    int batch_size = 100;
    int window_size = 1000;
    float forgetting_factor = 0.99f;

    // DBSCAN params
    float dbscan_eps = 0.5f;
    int dbscan_min_pts = 5;

    bool clustering_done = false;
    float inertia = 0;
    int n_iter = 0;
    Vector labels;
    Matrix centroids;
    std::vector<size_t> cluster_sizes;
    size_t n_noise = 0;

    // Evaluation
    bool eval_done = false;
    std::vector<EvalResult> eval_results;
    int eval_min_k = 2;
    int eval_max_k = 15;
    int eval_best_k = 0;
    bool eval_running = false;

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
    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    g.fbo_width = w;
    g.fbo_height = h;
}

static void run_clustering() {
    if (!g.data_loaded || g.table.rows() == 0) return;
    fprintf(stderr, "[cluster] Running %s, k=%d, rows=%zu\n",
            g.selected_algo == 0 ? "KMeans" : (g.selected_algo == 1 ? "MiniBatch" : "Online"),
            g.k, g.table.rows());

    const Matrix& X = g.use_pca && g.reduced_data.rows() > 0 ? g.reduced_data : g.table.data();

    delete g.kmeans;
    g.kmeans = nullptr;

    if (g.selected_algo == 0) {
        auto* km = new KMeans(g.k);
        km->fit(X);
        g.labels = km->labels();
        g.centroids = km->centroids();
        g.inertia = km->inertia();
        g.n_iter = km->n_iter();
        g.kmeans = km;
        g.n_noise = 0;
    } else if (g.selected_algo == 1) {
        auto* km = new MiniBatchKMeans(g.k, g.batch_size);
        km->fit(X);
        g.labels = km->labels();
        g.centroids = km->centroids();
        g.inertia = km->inertia();
        g.n_iter = km->n_iter();
        g.kmeans = km;
        g.n_noise = 0;
    } else if (g.selected_algo == 2) {
        OnlineConfig cfg;
        cfg.k = g.k;
        cfg.window_size = g.window_size;
        cfg.forgetting_factor = g.forgetting_factor;
        cfg.auto_retrain = false;
        auto* km = new OnlineKMeans(cfg);
        km->fit(X);
        g.labels = km->labels();
        g.centroids = km->centroids();
        g.inertia = km->inertia();
        g.n_iter = km->n_iter();
        g.kmeans = km;
        g.n_noise = 0;
    } else if (g.selected_algo == 3) {
        DBSCANConfig dbcfg;
        dbcfg.epsilon = g.dbscan_eps;
        dbcfg.min_pts = (size_t)g.dbscan_min_pts;
        auto* db = new DBSCAN(dbcfg);
        db->fit(X);
        g.labels = db->labels();
        g.centroids = Matrix();  // DBSCAN has no centroids
        g.inertia = 0;
        g.n_iter = 0;
        g.k = db->n_clusters();
        g.n_noise = db->n_noise();
        delete g.kmeans; g.kmeans = nullptr;
        g.kmeans = (KMeans*)db;  // hack: store as KMeans* for cleanup
        g.cluster_sizes.resize(g.k + 1);  // +1 for noise cluster 0
    } else {
        OnlineConfig cfg;
        cfg.k = g.k;
        cfg.window_size = g.window_size;
        cfg.forgetting_factor = g.forgetting_factor;
        cfg.auto_retrain = false;
        auto* km = new OnlineKMeans(cfg);
        km->fit(X);
        g.labels = km->labels();
        g.centroids = km->centroids();
        g.inertia = km->inertia();
        g.n_iter = km->n_iter();
        g.kmeans = km;
    }

    g.cluster_sizes.resize(g.k > 0 ? (g.selected_algo == 3 ? (size_t)g.k + 1 : (size_t)g.k) : 1, 0);
    for (size_t i = 0; i < g.labels.size(); ++i) {
        size_t lbl = (size_t)g.labels[i];
        if (lbl < g.cluster_sizes.size()) g.cluster_sizes[lbl]++;
    }

    // Update renderer
    if (g.selected_algo == 3 && g.centroids.rows() == 0) {
        // DBSCAN: create dummy centroids from cluster means
        Matrix dummy_c(g.k, X.cols());
        dummy_c.fill(0);
        std::vector<size_t> counts(g.k + 1, 0);
        for (size_t i = 0; i < X.rows(); ++i) {
            size_t lbl = (size_t)g.labels[i];
            if (lbl > 0 && lbl <= (size_t)g.k) {
                for (size_t d = 0; d < X.cols(); ++d) dummy_c[lbl-1][d] += X[i][d];
                counts[lbl]++;
            }
        }
        for (size_t c = 0; c < (size_t)g.k; ++c)
            if (counts[c+1] > 0)
                for (size_t d = 0; d < X.cols(); ++d) dummy_c[c][d] /= (float)counts[c+1];
        g.renderer_obj->set_data(X, g.labels, dummy_c);
    } else {
        g.renderer_obj->set_data(X, g.labels, g.centroids);
    }
    g.renderer_obj->set_metrics(g.inertia, g.n_iter);
    g.clustering_done = true;
    fprintf(stderr, "[cluster] Done: inertia=%.4f, iter=%d\n", g.inertia, g.n_iter);
    g.status_text = "Clustering done: inertia=" + std::to_string(g.inertia).substr(0,8) + ", iterations=" + std::to_string(g.n_iter);
    g.status_time = 5.0f;
}

static void open_csv() {
    const char* path = tinyfd_openFileDialog("Open CSV", "", 1, nullptr, "CSV files", 0);
    if (!path) return;
    fprintf(stderr, "[csv] Loading: %s\n", path);
    g.loading = true;
    g.load_progress = 0.0f;

    CSVLoadResult res = g.importer.load(path, true, ',', -1,
        [](size_t loaded, size_t total) {
            g.load_progress = total > 0 ? (float)loaded / total : 0.0f;
        });

    g.loading = false;
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
        // Clear old clustering state
        delete g.kmeans; g.kmeans = nullptr;
        g.labels = Vector(); g.centroids = Matrix(); g.cluster_sizes.clear();
        if (g.renderer_ready) {
            g.renderer_obj->set_data(Matrix(), Vector(), Matrix());
            g.renderer_obj->set_metrics(0, 0);
            g.renderer_obj->reset_view();
        }
    } else {
        fprintf(stderr, "[csv] Load failed: %s\n", res.error.c_str());
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
        static const char* algos[] = {"KMeans","MiniBatchKMeans","OnlineKMeans"};
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
    g.eval_running = true;
    g.status_text = "Evaluating k=" + std::to_string(g.eval_min_k) + " to " + std::to_string(g.eval_max_k) + "...";
    g.status_time = 2.0f;

    const Matrix& X = g.use_pca && g.reduced_data.rows() > 0 ? g.reduced_data : g.table.data();
    ClusterEvaluator eval;
    g.eval_results = eval.evaluate(X, (size_t)g.eval_min_k, (size_t)g.eval_max_k, 100, 4);
    g.eval_best_k = (int)eval.best_k_silhouette(g.eval_results);
    g.eval_done = true;
    g.eval_running = false;
    g.status_text = "Evaluation done. Best k=" + std::to_string(g.eval_best_k) + " (by silhouette)";
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
                if (ImGui::MenuItem("Export Labels CSV", nullptr, false, g.clustering_done)) export_labels_csv();
                if (ImGui::MenuItem("Export Centroids CSV", nullptr, false, g.clustering_done)) export_centroids_csv();
                if (ImGui::MenuItem("Export PNG Screenshot", nullptr, false, g.clustering_done)) export_png();
                ImGui::Separator();
                if (ImGui::MenuItem("Export Full Report", nullptr, false, g.data_loaded)) export_report();
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
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Show Data Table", nullptr, &g.show_table);
                ImGui::MenuItem("Show 3D Viewport", nullptr, &g.show_viewport);
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

        // === MAIN WINDOW (fills screen, resizable) ===
        ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetFrameHeight()));
        ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x,
                                        ImGui::GetIO().DisplaySize.y - ImGui::GetFrameHeight()));
        ImGui::Begin("Main", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoBringToFrontOnFocus);

        // --- LEFT SIDEBAR (resizable child) ---
        static float left_w = 350;
        ImGui::BeginChild("Sidebar", ImVec2(left_w, 0), ImGuiChildFlags_ResizeX);

        // IMPORT
        if (ImGui::CollapsingHeader("Import", ImGuiTreeNodeFlags_DefaultOpen)) {
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

        // CLUSTERING
        if (ImGui::CollapsingHeader("Clustering", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (g.data_loaded) {
                static const char* algos[] = {"KMeans","MiniBatchKMeans","OnlineKMeans","DBSCAN"};
                ImGui::Combo("Algorithm", &g.selected_algo, algos, IM_ARRAYSIZE(algos));
                if (ImGui::IsItemHovered()) {
                    if (g.selected_algo==3) ImGui::SetTooltip("Density-based. Finds arbitrary shapes. No k needed. Tune epsilon and min_pts.");
                    else if (g.selected_algo==2) ImGui::SetTooltip("Streaming with sliding window + forgetting factor.");
                    else if (g.selected_algo==1) ImGui::SetTooltip("Fast KMeans using random mini-batches.");
                    else ImGui::SetTooltip("Standard KMeans with KMeans++ init.");
                }
                if (g.selected_algo==3) {
                    ImGui::SliderFloat("Epsilon", &g.dbscan_eps, 0.01f, 5.0f, "%.3f");
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max distance to be neighbors. Larger = fewer clusters.");
                    ImGui::SliderInt("Min Points", &g.dbscan_min_pts, 2, 50);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Min points for dense region. Higher = more noise.");
                } else {
                    ImGui::SliderInt("k", &g.k, 1, 50);
                    if (g.selected_algo==1) ImGui::SliderInt("Batch Size", &g.batch_size, 10, 1000);
                    if (g.selected_algo==2){ImGui::SliderInt("Window", &g.window_size, 100, 10000); ImGui::SliderFloat("Forget", &g.forgetting_factor, 0.8f, 1.0f);}
                }
                ImGui::Checkbox("PCA pre-reduce", &g.use_pca);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reduce dimensionality before clustering. 'Run Clustering' will use cached PCA data if available.");
                if (g.use_pca) {
                    ImGui::SliderInt("Components", &g.pca_components, 2, std::min(50,(int)g.table.cols()));
                    if (ImGui::Button("Run PCA")) run_pca();
                    if (g.reduced_data.rows() > 0) {
                        ImGui::TextColored(ImVec4(0.3f,1,0.3f,1), "PCA ready: %zu x %d", g.reduced_data.rows(), g.pca_components);
                        ImGui::Text("Variance retained: %.1f%%", g.pca_total_var * 100);
                        if (g.pca_var_ratio.size() > 0) {
                            for (size_t i = 0; i < std::min(g.pca_var_ratio.size(), size_t(5)); ++i)
                                ImGui::Text("  C%zu: %.1f%%", i+1, g.pca_var_ratio[i]*100);
                            if (g.pca_var_ratio.size() > 5) ImGui::TextDisabled("  ... %zu more", g.pca_var_ratio.size()-5);
                        }
                        if (ImGui::Button("Undo PCA")) undo_pca();
                    } else { ImGui::TextDisabled("Click 'Run PCA' first"); }
                }
                if (ImGui::Button("Run Clustering", ImVec2(-1, 35))) run_clustering();
                if (g.clustering_done) {
                    ImGui::Separator();
                    if (g.selected_algo==3) {
                        ImGui::Text("Clusters found: %d", g.k);
                        ImGui::Text("Noise points: %zu", g.n_noise);
                        for (size_t c=0;c<g.cluster_sizes.size();++c) {
                            if (c==0) ImGui::Text("  Noise: %zu pts", g.cluster_sizes[c]);
                            else ImGui::Text("  C%zu: %zu pts", c, g.cluster_sizes[c]);
                        }
                    } else {
                        ImGui::Text("Inertia: %.4f", g.inertia);
                        ImGui::Text("Iterations: %d", g.n_iter);
                        for (size_t c=0;c<g.cluster_sizes.size();++c) ImGui::Text("  C%zu: %zu pts", c, g.cluster_sizes[c]);
                    }
                    ImGui::Separator();
                    if (ImGui::Button("Export Report", ImVec2(-1, 0))) export_report();
                    if (ImGui::Button("Export Labels CSV", ImVec2(-1, 0))) export_labels_csv();
                    if (ImGui::Button("Export PNG (viewport)", ImVec2(-1, 0))) export_png();
                    ImGui::InputInt("PNG W", &g.export_width); ImGui::InputInt("PNG H", &g.export_height);
                }
            } else { ImGui::TextDisabled("No data"); }
        }

        // EVALUATION
        if (ImGui::CollapsingHeader("Evaluate k (Quality)")) {
            if (g.data_loaded) {
                ImGui::SliderInt("k Min", &g.eval_min_k, 2, 10);
                ImGui::SliderInt("k Max", &g.eval_max_k, std::max(3,g.eval_min_k+1), 30);
                if (ImGui::Button("Run Evaluation", ImVec2(-1, 30))) run_evaluation();
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Test multiple k values. Shows inertia (elbow), silhouette, Davies-Bouldin plots.");
                if (g.eval_running) { ImGui::Text("Evaluating..."); }
                if (g.eval_done && !g.eval_results.empty()) {
                    ImGui::TextColored(ImVec4(0.3f,1,0.3f,1), "Best k = %d (by silhouette)", g.eval_best_k);
                    if (ImGui::Button("Use this k")) { g.k = g.eval_best_k; g.selected_algo = 0; }
                    ImGui::SameLine();
                    if (ImGui::Button("Clear")) { g.eval_done = false; g.eval_results.clear(); }
                    std::vector<float> inertias, silhouettes, db_scores;
                    for (auto& r : g.eval_results) {
                        inertias.push_back(r.inertia);
                        silhouettes.push_back(r.silhouette_score);
                        db_scores.push_back(r.davies_bouldin);
                    }
                    ImGui::Text("Inertia (elbow method):");
                    ImGui::PlotLines("##inertia", inertias.data(), inertias.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(-1, 55));
                    ImGui::Text("Silhouette (peak = optimal):");
                    ImGui::PlotLines("##sil", silhouettes.data(), silhouettes.size(), 0, nullptr, -1.0f, 1.0f, ImVec2(-1, 55));
                    ImGui::Text("Davies-Bouldin (dip = optimal):");
                    ImGui::PlotLines("##db", db_scores.data(), db_scores.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(-1, 55));
                    ImGui::TextDisabled("k=%d..%d  |  Best: silhouette=%d", g.eval_min_k, g.eval_max_k, g.eval_best_k);
                }
            } else { ImGui::TextDisabled("No data loaded"); }
        }

        left_w = ImGui::GetWindowWidth(); // track resize
        ImGui::EndChild();
        ImGui::SameLine();

        // --- RIGHT PANEL (table + viewport) ---
        ImGui::BeginChild("RightPanel", ImVec2(0, 0));

        if (g.show_table) {
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

        if (g.show_viewport) {
            ImGui::BeginChild("ViewportChild", ImVec2(0, 0));
            if (g.clustering_done && g.renderer_ready) {
                ImVec2 vp_size = ImGui::GetContentRegionAvail();
                int vp_w = (int)vp_size.x, vp_h = (int)std::max(vp_size.y, 50.0f);
                if (vp_w > 0 && vp_h > 0) {
                    if (vp_w != g.fbo_width || vp_h != g.fbo_height) {
                        glDeleteFramebuffers(1, &g.fbo); glDeleteTextures(1, &g.fbo_texture);
                        create_fbo(vp_w, vp_h);
                    }
                    g.renderer_obj->render_to_fbo(g.fbo, vp_w, vp_h);
                    ImGui::Image((ImTextureID)(intptr_t)g.fbo_texture, ImVec2((float)vp_w, (float)vp_h), ImVec2(0,1), ImVec2(1,0));

                    if (ImGui::IsItemHovered()) {
                        ImGuiIO& io = ImGui::GetIO();
                        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
                            g.renderer_obj->rotate_view(io.MouseDelta.x * 0.008f, io.MouseDelta.y * 0.008f);
                        if (io.MouseWheel != 0)
                            g.renderer_obj->zoom_view(1.0f + io.MouseWheel * 0.1f);
                        if (ImGui::IsKeyPressed(ImGuiKey_R))
                            g.renderer_obj->reset_view();
                    }
                }
            } else {
                ImGui::TextDisabled("Load data and run clustering to see 3D visualization");
            }
            // Reset button always visible
            if (g.renderer_ready) {
                ImGui::Spacing();
                if (ImGui::Button("Reset View (R)")) g.renderer_obj->reset_view();
                ImGui::SameLine();
                ImGui::TextDisabled("Drag=rotate | Scroll=zoom | R=reset");
            }
            ImGui::EndChild();
        }

        ImGui::EndChild(); // RightPanel
        ImGui::End();      // Main

        // === STATUS BAR ===
        if (!g.status_text.empty()) {
            float status_h = ImGui::GetFrameHeight();
            ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetIO().DisplaySize.y - status_h));
            ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, status_h));
            ImGui::Begin("##status", nullptr,
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoSavedSettings);
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "  %s", g.status_text.c_str());
            g.status_time -= ImGui::GetIO().DeltaTime;
            if (g.status_time <= 0) g.status_text.clear();
            ImGui::End();
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    delete g.kmeans;
    delete g.renderer_obj;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
