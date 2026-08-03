#pragma once

#include "clustering/clustering.h"
#include "gui/data_table.h"
#include "gui/column_stats.h"
#include "gui/csv_importer.h"
#include "gui/preprocess_pipeline.h"
#include "gui/missing_handler.h"
#include <imgui.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <string>

namespace clustering_app {

using clustering::Matrix;
using clustering::Vector;
using clustering::KMeans;
using clustering::KMeansConfig;
using clustering::MiniBatchKMeans;
using clustering::OnlineKMeans;
using clustering::OnlineConfig;
using clustering::DBSCAN;
using clustering::DBSCANConfig;
using clustering::PCA;
using clustering::DriftDetector;
using clustering::DriftMetrics;
using clustering::ClusterEvaluator;
using clustering::EvalResult;
using clustering::DataTable;
using clustering::ColumnStatsCache;
using clustering::CSVImporter;
using clustering::CSVLoadResult;
using clustering::MissingHandler;
using clustering::ColumnStats;
using clustering::Renderer;
using clustering::RendererConfig;
using clustering::TSNE;
using clustering::TSNEConfig;

// ============================================================================
// AppState — All global state for the interactive clustering tool.
// ============================================================================
struct AppState {
    // --- Data ---
    DataTable table;
    ColumnStatsCache stats;
    CSVImporter importer;
    MissingHandler missing_handler;
    bool data_loaded = false;
    bool loading = false;
    bool has_header = true;
    float load_progress = 0.0f;

    // --- Clustering params ---
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

    // --- DBSCAN params ---
    float dbscan_eps = 0.5f;
    int dbscan_min_pts = 5;

    // --- DBSCAN sweep ---
    float eval_eps_min = 0.1f;
    float eval_eps_max = 10.0f;
    int eval_eps_steps = 10;

    // --- Clustering state ---
    bool realtime_viz = true;
    std::atomic<bool> clustering_running{false};
    bool clustering_done = false;
    float inertia = 0;
    int n_iter = 0;
    Vector labels;
    Matrix centroids;
    std::vector<size_t> cluster_sizes;
    std::vector<int> selected_cols;
    size_t n_noise = 0;

    // --- Evaluation ---
    bool eval_done = false;
    std::vector<EvalResult> eval_results;
    int eval_min_k = 2;
    int eval_max_k = 15;
    int eval_best_k = 0;
    bool eval_running = false;
    std::thread eval_thread;
    std::thread cluster_thread;
    std::atomic<int> eval_progress_k{0};

    // --- Metrics ---
    float current_silhouette = 0;
    float current_db = 0;
    float current_ch = 0;
    bool current_eval_done = false;

    // --- Compare history ---
    struct CompareEntry {
        std::string algo;
        int k;
        float silhouette;
        float db;
        float ch;
        float inertia;
    };
    std::vector<CompareEntry> compare_history;

    // --- Synchronization ---
    std::mutex result_mutex;

    // --- PCA ---
    bool use_pca = false;
    int pca_components = 10;
    Matrix reduced_data;
    Vector pca_var_ratio;
    float pca_total_var = 0;

    // --- t-SNE ---
    int tsne_perplexity = 30;
    float tsne_lr = 200.0f;
    int tsne_iter = 1000;
    int tsne_seed = -1;
    Matrix tsne_embedding;
    bool tsne_done = false;

    // --- Viewport ---
    unsigned int fbo = 0;
    unsigned int fbo_texture = 0;
    unsigned int fbo_rbo = 0;
    int fbo_width = 800;
    int fbo_height = 600;
    Renderer* renderer_obj = nullptr;
    bool renderer_ready = false;

    // --- Table ---
    int sort_col = -1;
    bool sort_ascending = true;
    std::vector<size_t> sorted_indices;
    bool show_viewport = true;

    // --- UI ---
    int selected_col = 0;
    int preprocess_op = 0;
    int missing_strategy_idx = 2;
    bool apply_to_all_columns = true;

    // --- Status ---
    std::string status_text;
    float status_time = 0;
};

// ============================================================================
// UI Panel Functions — each renders one section of the interface.
// ============================================================================

// Menu bar (File/Edit/Help) + About popup
void render_menu_bar(AppState& g);

// Tab 0: Import panel
void render_import(AppState& g);

// Tab 0: Column Stats panel
void render_column_stats(AppState& g);

// Tab 1: Preprocessing panel
void render_preprocessing(AppState& g);

// Tab 1: Dimensionality Reduction panel (PCA + t-SNE)
void render_dimred(AppState& g);

// Tab 2: Clustering panels
void render_select_columns(AppState& g);
void render_select_algorithm(AppState& g);
void render_algo_params(AppState& g);
void render_clustering(AppState& g);

// Tab 2: Find Optimal k panel
void render_find_optimal_k(AppState& g);

// Tab 2: Find Optimal DBSCAN params panel
void render_find_optimal_dbscan(AppState& g);

// Tab 0: Data table (right panel)
void render_data_table(AppState& g);

// Tab 1: 3D viewport (right panel)
void render_viewport(AppState& g);

// Tab 1: Comparison history
void render_compare_history(AppState& g, int active_tab);

// Status bar
void render_status_bar(AppState& g);

// ============================================================================
// Action Functions — business logic triggered by UI.
// ============================================================================

void open_csv(AppState& g);
void export_labels_csv(const AppState& g);
void export_centroids_csv(const AppState& g);
void export_preprocessed_csv(const AppState& g);
void export_png(const AppState& g);
void export_report(const AppState& g);
void apply_preprocess(AppState& g);
void apply_missing(AppState& g);
void undo_all(AppState& g);
void run_pca(AppState& g);
void undo_pca(AppState& g);
void run_clustering_async(AppState& g);
void run_evaluation(AppState& g);
void evaluate_current(AppState& g);
void create_fbo(AppState& g, int w, int h);
Matrix extract_selected_cols(const Matrix& src, const std::vector<int>& sel);

} // namespace clustering_app
