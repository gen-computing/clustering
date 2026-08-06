// ============================================================================
// ui_actions.cpp — Business logic functions triggered by UI buttons.
// ============================================================================

#include "ui.h"
#include "clustering/distance.h"
#include "clustering/logging.h"
#include "gui/exports.h"
#include <tinyfiledialogs.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <GL/glew.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>

namespace clustering_app {

void set_status(const AppState& g, const std::string& msg, float seconds) {
    std::lock_guard<std::mutex> lk(g.status_mutex);
    g.status_text = msg;
    g.status_time = seconds;
}

void render_section_header(const char* title) {
    if (g_font_header) ImGui::PushFont(g_font_header);
    ImGui::TextColored(ImVec4(0.9f, 0.9f, 1.0f, 1.0f), "%s", title);
    if (g_font_header) ImGui::PopFont();
    ImGui::Separator();
    ImGui::Spacing();
}

void reset_dimred(AppState& g) {
    g.use_pca = false;
    g.reduced_data = Matrix();
    g.pca_var_ratio = Vector();
    g.pca_total_var = 0;
    g.tsne_embedding = Matrix();
    g.tsne_done = false;
    g.cluster_source = 0;
    g.clustering_done = false;
    g.labels = Vector(); g.centroids = Matrix(); g.cluster_sizes.clear();
    g.n_noise = 0; g.n_iter = 0; g.inertia = 0;
    g.current_eval_done = false;
    if (g.renderer_ready) {
        g.renderer_obj->set_data(Matrix(), Vector(), Matrix());
        g.renderer_obj->set_metrics(0, 0);
    }
    set_status(g, "Dimensionality reduction cleared", 3.0f);
}

void run_tsne_async(AppState& g) {
    if (!g.data_loaded || g.tsne_running.load()) return;
    if (g.tsne_thread.joinable()) g.tsne_thread.join();

    size_t max_n = std::min((size_t)5000, g.table.rows());
    const Matrix base = extract_selected_cols(g.table.data(), g.selected_cols);
    size_t cols = base.cols();
    Matrix X(max_n, cols);
    {
        std::lock_guard<std::mutex> lk(g.result_mutex);
        for (size_t i = 0; i < max_n; ++i)
            for (size_t j = 0; j < cols; ++j)
                X[i][j] = base[i][j];
    }
    clustering::TSNEConfig cfg;
    cfg.perplexity = g.tsne_perplexity;
    cfg.learning_rate = g.tsne_lr;
    cfg.n_iter = (size_t)g.tsne_iter;
    cfg.random_seed = g.tsne_seed;

    g.tsne_running = true;
    set_status(g, "Running t-SNE... (this can take a while)", 999.0f);

    g.tsne_thread = std::thread([&g, X = std::move(X), cfg]() {
        try {
            clustering::TSNE tsne(cfg);
            tsne.fit(X);
            Matrix emb = tsne.embedding();
            float kl = tsne.kl_divergence();
            { std::lock_guard<std::mutex> lk(g.result_mutex);
              g.tsne_embedding = std::move(emb);
              g.tsne_done = true;
              g.cluster_source = 2; }  // clustering now consumes the embedding
            char buf[128]; snprintf(buf, sizeof(buf), "t-SNE done  |  KL: %.4f  |  %zu x 2", kl, emb.rows());
            set_status(g, buf, 6.0f);
        } catch (const std::exception& e) {
            set_status(g, std::string("t-SNE failed: ") + e.what(), 6.0f);
        }
        g.tsne_running = false;
    });
}

Matrix extract_selected_cols(const Matrix& src, const std::vector<int>& sel) {
    std::vector<size_t> cols;
    for (size_t j = 0; j < sel.size() && j < src.cols(); ++j)
        if (sel[j]) cols.push_back(j);
    if (cols.empty() || cols.size() == src.cols()) return src;
    Matrix out(src.rows(), cols.size());
    for (size_t i = 0; i < src.rows(); ++i)
        for (size_t j = 0; j < cols.size(); ++j)
            out[i][j] = src[i][cols[j]];
    return out;
}

void create_fbo(AppState& g, int w, int h) {
    unsigned int fbo, tex, rbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    g.fbo = fbo;
    g.fbo_texture = tex;
    g.fbo_rbo = rbo;
    g.fbo_width = w;
    g.fbo_height = h;
}

void open_csv(AppState& g) {
    const char* path = tinyfd_openFileDialog("Open CSV", "", 1, nullptr, "CSV files", 0);
    fprintf(stderr, "[open_csv] dialog returned: %s\n", path ? path : "(null)");
    if (!path) return;

    CSVLoadResult res = g.importer.load(path, g.has_header, ',', -1,
        [](size_t loaded, size_t total) {});
    fprintf(stderr, "[open_csv] load: success=%d rows=%zu cols=%zu error='%s'\n",
        (int)res.success, res.table.rows(), res.table.cols(), res.error.c_str());
    if (res.success) {
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
        g.selected_cols.assign(g.table.cols(), 1);
        g.cluster_source = 0;

        // Large datasets: OnlineKMeans learns incrementally (no full re-cluster),
        // so it is the sensible default for big files. User can still switch.
        const size_t ONLINE_THRESHOLD = 50000;
        if (g.table.rows() > ONLINE_THRESHOLD && g.selected_algo == 0) {
            g.selected_algo = 2;
            set_status(g, "Large dataset (" + std::to_string(g.table.rows()) +
                " rows): OnlineKMeans pre-selected for streaming learning", 8.0f);
        }
        if (g.renderer_ready) {
            g.renderer_obj->set_data(Matrix(), Vector(), Matrix());
            g.renderer_obj->set_metrics(0, 0);
            g.renderer_obj->reset_view();
        }
        set_status(g, "Loaded: " + std::to_string(res.table.rows()) + " rows x " + std::to_string(res.table.cols()) + " cols", 3.0f);
    } else {
        set_status(g, "Load failed: " + res.error, 8.0f);
        fprintf(stderr, "[open_csv] load failed: %s\n", res.error.c_str());
    }
}
void export_labels_csv(const AppState& g) {
    const char* path = tinyfd_saveFileDialog("Export Labels", "labels.csv", 0, nullptr, nullptr);
    if (!path) return;
    if (!clustering::export_labels_csv(g.table, g.labels, path))
        set_status(g, "Export failed: could not write file", 5.0f);
}

void export_centroids_csv(const AppState& g) {
    const char* path = tinyfd_saveFileDialog("Export Centroids", "centroids.csv", 0, nullptr, nullptr);
    if (!path) return;
    if (!clustering::export_centroids_csv(g.centroids, path))
        set_status(g, "Export failed: could not write file", 5.0f);
}

void export_preprocessed_csv(const AppState& g) {
    const char* path = tinyfd_saveFileDialog("Export Preprocessed Data", "preprocessed.csv", 0, nullptr, nullptr);
    if (!path) return;
    if (!clustering::export_preprocessed_csv(g.table, path))
        set_status(g, "Export failed: could not write file", 5.0f);
}

void export_png(const AppState& g) {
    const char* path = tinyfd_saveFileDialog("Export PNG", "screenshot.png", 0, nullptr, nullptr);
    if (!path || !g.renderer_ready) return;

    int w = 1920, h = 1080;
    unsigned int exp_fbo, exp_tex, exp_rbo;
    glGenFramebuffers(1, &exp_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, exp_fbo);
    glGenTextures(1, &exp_tex);
    glBindTexture(GL_TEXTURE_2D, exp_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, exp_tex, 0);
    glGenRenderbuffers(1, &exp_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, exp_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, exp_rbo);

    const_cast<AppState&>(g).renderer_obj->render_to_fbo(exp_fbo, w, h);

    std::vector<unsigned char> pixels(w * h * 3);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // Flip vertically
    std::vector<unsigned char> flipped(w * h * 3);
    for (int y = 0; y < h; ++y)
        memcpy(&flipped[y * w * 3], &pixels[(h - 1 - y) * w * 3], w * 3);

    stbi_write_png(path, w, h, 3, flipped.data(), w * 3);

    glDeleteFramebuffers(1, &exp_fbo);
    glDeleteTextures(1, &exp_tex);
    glDeleteRenderbuffers(1, &exp_rbo);
}

void export_report(const AppState& g) {
    const char* path = tinyfd_saveFileDialog("Export Report", "report.txt", 0, nullptr, nullptr);
    if (!path) return;
    if (!clustering::export_report(g.table, g.selected_cols, g.selected_algo, g.k,
                                   g.inertia, g.n_iter, g.current_silhouette,
                                   g.current_db, g.clustering_done, path))
        set_status(g, "Export failed: could not write file", 5.0f);
}

void apply_preprocess(AppState& g) {
    if (!g.data_loaded) return;
    const char* ops[] = {"Normalize","Standardize","MinMax Scale","Log Transform","Clip Outliers"};
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
    } else {
        size_t col = (size_t)g.selected_col;
        if (col >= g.table.cols()) return;
        switch (g.preprocess_op) {
            case 0: g.table.pipeline().normalize_column(col); break;
            case 1: g.table.pipeline().standardize_column(col); break;
            case 2: g.table.pipeline().minmax_scale_column(col); break;
            case 3: g.table.pipeline().log_transform_column(col); break;
            case 4: g.table.pipeline().clip_outliers_column(col); break;
        }
    }
    g.stats.invalidate();
    g.clustering_done = false;
    set_status(g, "Applied " + std::string(ops[g.preprocess_op]), 2.0f);
}

void apply_missing(AppState& g) {
    if (!g.data_loaded) return;
    g.missing_handler.apply(g.table);
    g.stats.invalidate();
    g.clustering_done = false;
    set_status(g, "Applied missing strategy", 2.0f);
}

void undo_all(AppState& g) {
    while (g.table.pipeline().can_undo())
        g.table.pipeline().undo();
    g.stats.invalidate();
    g.clustering_done = false;
}

void run_pca(AppState& g) {
    if (!g.data_loaded) return;
    const Matrix X = extract_selected_cols(g.table.data(), g.selected_cols);
    // Clamp components to valid range
    size_t max_comp = std::min(X.rows(), X.cols());
    if (max_comp < 2) { set_status(g, "PCA needs at least 2 samples and 2 features", 3.0f); return; }
    int n_comp = std::min(g.pca_components, (int)max_comp);
    PCA pca((size_t)n_comp);
    g.reduced_data = pca.fit_transform(X);
    g.pca_var_ratio = pca.explained_variance_ratio();
    g.pca_total_var = pca.total_explained_variance_ratio();
    g.use_pca = true;
    g.clustering_done = false;
    g.cluster_source = 1;  // clustering now consumes PCA-reduced data
    set_status(g, "PCA: " + std::to_string(g.reduced_data.rows()) + " x " + std::to_string(n_comp) + ", " + std::to_string((int)(g.pca_total_var * 100)) + "% var", 3.0f);
}

void undo_pca(AppState& g) {
    g.use_pca = false;
    g.reduced_data = Matrix();
    g.clustering_done = false;
    if (g.cluster_source == 1) g.cluster_source = 0;
}

void run_clustering_async(AppState& g) {
    if (!g.data_loaded || g.table.rows() == 0) return;
    if (g.cluster_thread.joinable()) g.cluster_thread.join();
    g.cancel_requested = false;
    g.clustering_done = false;
    g.clustering_running = true;
    g.cluster_thread = std::thread([&g]() {
        try {
        set_status(g, "Clustering...", 999.0f);
        const Matrix* rawp = &g.table.data();
        if (g.cluster_source == 1 && g.reduced_data.rows() > 0) rawp = &g.reduced_data;
        else if (g.cluster_source == 2 && g.tsne_embedding.rows() > 0) rawp = &g.tsne_embedding;
        const Matrix X = g.cluster_source == 0
            ? extract_selected_cols(*rawp, g.selected_cols) : *rawp;
        delete g.kmeans; g.kmeans = nullptr;

        Vector loc_labels; Matrix loc_centroids;
        float loc_inertia = 0; int loc_iter = 0; size_t loc_k = 0, loc_noise = 0;
        int selected = g.selected_algo;
        bool stopped = false;

        if (selected == 0) {
            KMeansConfig cfg; cfg.k = (size_t)g.k; cfg.max_iter = g.max_iter; cfg.max_threads = g.max_threads;
            if (g.realtime_viz) {
                cfg.iter_callback = [&g](size_t iter, const Matrix& centroids, const Vector& labels) -> bool {
                    { std::lock_guard<std::mutex> lk(g.result_mutex);
                      g.centroids = centroids; g.labels = labels; g.n_iter = (int)iter; }
                    set_status(g, "KMeans iter " + std::to_string(iter), 999.0f);
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    return g.cancel_requested.load();
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
            OnlineConfig cfg; cfg.k = g.k; cfg.window_size = (size_t)g.window_size;
            cfg.forgetting_factor = g.forgetting_factor;
            cfg.auto_retrain = g.online_auto_retrain;
            cfg.drift_threshold = g.drift_threshold;
            cfg.retrain_interval = (size_t)g.retrain_interval;
            if (g.realtime_viz) {
                cfg.iter_callback = [&g](size_t iter, const Matrix& centroids, const Vector& labels) -> bool {
                    { std::lock_guard<std::mutex> lk(g.result_mutex);
                      g.centroids = centroids; g.labels = labels; g.n_iter = (int)iter; }
                    set_status(g, "Online update " + std::to_string(iter), 999.0f);
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    return g.cancel_requested.load();
                };
            }
            auto* km = new OnlineKMeans(cfg); km->fit(X);
            loc_labels = km->labels(); loc_centroids = km->centroids();
            loc_inertia = km->inertia(); loc_iter = (int)km->n_iter();
            g.kmeans = km; loc_k = g.k;
        } else if (selected == 3) {
            DBSCANConfig dbcfg; dbcfg.epsilon = g.dbscan_eps; dbcfg.min_pts = (size_t)g.dbscan_min_pts;
            dbcfg.standardize = true;  // scale-free eps; robust on raw-feature data
            DBSCAN db(dbcfg); db.fit(X);
            loc_labels = db.labels(); loc_k = db.n_clusters(); loc_noise = db.n_noise();
        }
        stopped = g.cancel_requested.load();

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

        if (!stopped) {
            { std::lock_guard<std::mutex> lk(g.result_mutex);
                g.labels = std::move(loc_labels);
                g.centroids = std::move(loc_centroids);
                g.inertia = loc_inertia; g.n_iter = loc_iter;
                g.k = (int)loc_k; g.n_noise = loc_noise;
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
                g.drift_detected = dm.drift_detected;
                g.current_eval_done = true;
                static const char* algo_names[] = {"KMeans","MiniBatch","Online","DBSCAN"};
                g.compare_history.push_back({algo_names[selected], (int)loc_k, g.current_silhouette, g.current_db, g.current_ch, loc_inertia});
                if (g.compare_history.size() > 20) g.compare_history.erase(g.compare_history.begin());
            }
            set_status(g, "Done: inertia=" + std::to_string(loc_inertia).substr(0, 8), 5.0f);
        } else {
            set_status(g, "Clustering stopped by user", 4.0f);
        }
        g.clustering_running = false;
        } catch (const std::exception& e) {
            set_status(g, std::string("Clustering failed: ") + e.what(), 8.0f);
            g.clustering_running = false;
        } catch (...) {
            set_status(g, "Clustering failed: unknown error", 8.0f);
            g.clustering_running = false;
        }
    });
}

void run_evaluation(AppState& g) {
    if (!g.data_loaded || g.table.rows() == 0) return;
    if (g.eval_thread.joinable()) g.eval_thread.join();
    g.cancel_requested = false;
    g.eval_running = true;
    g.eval_done = false;
    g.eval_results.clear();

    int min_k = g.eval_min_k, max_k = g.eval_max_k;
    int src = g.cluster_source;
    Matrix reduced = g.reduced_data;
    Matrix table_data = g.table.data();
    std::vector<int> sel = g.selected_cols;

    g.eval_thread = std::thread([&g, min_k, max_k, src, reduced, table_data, sel]() {
        try {
        const Matrix X = src == 1 && reduced.rows() > 0 ? reduced
                      : src == 2 && g.tsne_embedding.rows() > 0 ? g.tsne_embedding
                      : extract_selected_cols(table_data, sel);
        std::vector<EvalResult> results;
        results.reserve((size_t)(max_k - min_k + 1));
        g.eval_progress_k = (int)min_k;

        for (int k = min_k; k <= max_k; ++k) {
            if (g.cancel_requested.load()) break;
            g.eval_progress_k = k;
            set_status(g, "Evaluating k=" + std::to_string(k) + "/" + std::to_string(max_k) + "...", 999.0f);

            KMeansConfig cfg; cfg.k = (size_t)k; cfg.max_iter = 100; cfg.max_threads = 4;
            KMeans km(cfg); km.fit(X);

            EvalResult r; r.k = k; r.inertia = km.inertia();

            const size_t MAX_N = 1000;
            DriftDetector det;
            if (X.rows() <= MAX_N) {
                DriftMetrics dm = det.check(X, km.labels(), km.centroids());
                r.silhouette_score = dm.silhouette_score;
                r.davies_bouldin = dm.davies_bouldin_index;
                r.calinski_harabasz = dm.calinski_harabasz_score;
            } else {
                size_t sn = MAX_N, step = X.rows() / sn;
                Matrix Xs(sn, X.cols()); Vector labels_s(sn);
                for (size_t i = 0; i < sn; ++i) { size_t idx = i * step;
                    for (size_t d = 0; d < X.cols(); ++d) Xs[i][d] = X[idx][d];
                    labels_s[i] = km.labels()[idx]; }
                DriftMetrics dm = det.check(Xs, labels_s, km.centroids());
                r.silhouette_score = dm.silhouette_score;
                r.davies_bouldin = dm.davies_bouldin_index;
                r.calinski_harabasz = dm.calinski_harabasz_score;
            }
            results.push_back(r);
        }

        { std::lock_guard<std::mutex> lk(g.result_mutex);
            g.eval_results = std::move(results); }

        if (g.cancel_requested.load()) {
            set_status(g, "Evaluation stopped by user", 4.0f);
        } else {
            ClusterEvaluator eval;
            g.eval_best_k = (int)eval.best_k_silhouette(g.eval_results);
            set_status(g, "Best k=" + std::to_string(g.eval_best_k), 5.0f);
        }
        g.eval_done = true;
        g.eval_running = false;
        } catch (const std::exception& e) {
            set_status(g, std::string("Evaluation failed: ") + e.what(), 8.0f);
            g.eval_running = false;
        }
    });
}

void evaluate_current(AppState& g) {
    if (!g.clustering_done || g.labels.size() == 0) return;
    const Matrix X = g.cluster_source == 1 && g.reduced_data.rows() > 0 ? g.reduced_data
                   : g.cluster_source == 2 && g.tsne_embedding.rows() > 0 ? g.tsne_embedding
                   : extract_selected_cols(g.table.data(), g.selected_cols);
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
            if (counts[c] > 0) for (size_t d = 0; d < X.cols(); ++d) centroids[c][d] /= (float)counts[c];
    }
    DriftDetector det;
    clustering::DriftMetrics dm = det.check(X, g.labels, centroids);
    g.current_silhouette = dm.silhouette_score;
    g.current_db = dm.davies_bouldin_index;
    g.current_ch = dm.calinski_harabasz_score;
    g.current_eval_done = true;
}

} // namespace clustering_app
