// ============================================================================
// ui_cluster.cpp — Tab 1 panels: Find Optimal k, Clustering.
// ============================================================================

#include "ui.h"
#include "clustering/evaluation.h"
#include "clustering/dbscan.h"
#include <imgui.h>

namespace clustering_app {

void render_find_optimal_k(AppState& g) {
    if (ImGui::CollapsingHeader("Find Optimal k", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (g.data_loaded) {
            ImGui::TextDisabled("Sweep KMeans to find optimal cluster count");
            ImGui::SliderInt("##kmin", &g.eval_min_k, 2, 10);
            ImGui::SameLine(); ImGui::Text("k Min");
            ImGui::SliderInt("##kmax", &g.eval_max_k, std::max(3, g.eval_min_k + 1), 30);
            ImGui::SameLine(); ImGui::Text("k Max");

            ImGui::BeginDisabled(g.eval_running);
            if (ImGui::Button("Run Sweep", ImVec2(-1, 28))) run_evaluation(g);
            ImGui::EndDisabled();

            if (g.eval_running) {
                ImGui::Text("Running k=%d/%d...", g.eval_progress_k.load(), g.eval_max_k);
                ImGui::ProgressBar((float)(g.eval_progress_k.load() - g.eval_min_k + 1) / (float)(g.eval_max_k - g.eval_min_k + 1));
            }

            if (g.eval_done && !g.eval_results.empty()) {
                ClusterEvaluator eval;
                size_t k_elbow = eval.best_k_elbow(g.eval_results);
                size_t k_sil = eval.best_k_silhouette(g.eval_results);
                size_t k_db = eval.best_k_db(g.eval_results);
                ImGui::Text("Best k: %zu (elbow) | %zu (sil) | %zu (DB)", k_elbow, k_sil, k_db);

                if (ImGui::Button(("Use k=" + std::to_string(k_elbow) + " (elbow)").c_str())) {
                    g.k = (int)k_elbow; g.selected_algo = 0;
                }
                ImGui::SameLine();
                if (ImGui::Button("Clear")) { g.eval_done = false; g.eval_results.clear(); }

                std::vector<float> inertias, silhouettes, db_scores;
                for (auto& r : g.eval_results) {
                    inertias.push_back(r.inertia);
                    silhouettes.push_back(r.silhouette_score);
                    db_scores.push_back(r.davies_bouldin);
                }
                ImGui::Text("Inertia (elbow):");
                ImGui::PlotLines("##e_inertia", inertias.data(), inertias.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(-1, 40));
                ImGui::Text("Silhouette (higher=better):");
                ImGui::PlotLines("##e_sil", silhouettes.data(), silhouettes.size(), 0, nullptr, -1.0f, 1.0f, ImVec2(-1, 40));
                ImGui::Text("Davies-Bouldin (lower=better):");
                ImGui::PlotLines("##e_db", db_scores.data(), db_scores.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(-1, 40));
            }
        }
    }
}

void render_clustering(AppState& g) {
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
                    char label[64]; snprintf(label, sizeof(label), "##col%zu", j);
                    if (ImGui::Checkbox(label, &sel)) g.selected_cols[j] = sel ? 1 : 0;
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.9f, 1.0f), "%s", g.table.column_names()[j].c_str());
                }
                ImGui::Separator();
                if (ImGui::SmallButton("Select All")) for (auto& v : g.selected_cols) v = 1;
                ImGui::SameLine();
                if (ImGui::SmallButton("Deselect All")) for (auto& v : g.selected_cols) v = 0;
                ImGui::TreePop();
            }
            ImGui::Separator();

            // Algorithm
            static const char* algos[] = {"KMeans", "MiniBatchKMeans", "OnlineKMeans", "DBSCAN"};
            static int prev_algo = 0;
            ImGui::Combo("Algorithm", &g.selected_algo, algos, IM_ARRAYSIZE(algos));
            if (g.selected_algo != prev_algo && g.clustering_done) {
                g.clustering_done = false; g.current_eval_done = false;
                if (g.renderer_ready) { g.renderer_obj->set_data(Matrix(), Vector(), Matrix()); g.renderer_obj->set_metrics(0, 0); }
            }
            prev_algo = g.selected_algo;

            // Algorithm-specific params
            if (g.selected_algo == 3) {
                ImGui::SliderFloat("Epsilon", &g.dbscan_eps, 0.01f, 200.0f, "%.3f");
                ImGui::SliderInt("Min Points", &g.dbscan_min_pts, 2, 50);
                if (ImGui::Button("Auto-Estimate Epsilon", ImVec2(-1, 0))) {
                    float ae = DBSCAN::estimate_epsilon(g.table.data(), (size_t)g.dbscan_min_pts, 500);
                    g.dbscan_eps = ae > 0 ? ae : g.dbscan_eps;
                    g.status_text = "Auto epsilon = " + std::to_string(g.dbscan_eps).substr(0, 6);
                    g.status_time = 3.0f;
                }
            } else {
                ImGui::SliderInt("Clusters", &g.k, 1, std::min(50, (int)g.table.rows() - 1));
                if (g.selected_algo == 1) ImGui::SliderInt("Batch Size", &g.batch_size, 10, 1000);
                if (g.selected_algo == 2) {
                    ImGui::SliderInt("Window", &g.window_size, 100, 10000);
                    ImGui::SliderFloat("Forget", &g.forgetting_factor, 0.8f, 1.0f);
                    ImGui::Checkbox("Auto Retrain on Drift", &g.online_auto_retrain);
                    if (g.online_auto_retrain) {
                        ImGui::SliderFloat("Drift Threshold", &g.drift_threshold, 0.01f, 0.5f);
                        ImGui::SliderInt("Check Interval", &g.retrain_interval, 100, 5000);
                    }
                }
            }

            // PCA
            ImGui::Checkbox("PCA pre-reduce", &g.use_pca);
            if (g.use_pca) {
                ImGui::SliderInt("Components", &g.pca_components, 2, std::min(50, (int)g.table.cols()));
                if (ImGui::Button("Run PCA")) run_pca(g);
                if (g.reduced_data.rows() > 0) {
                    ImGui::TextColored(ImVec4(0.3f, 1, 0.3f, 1), "PCA: %zu x %d, %.0f%% var",
                        g.reduced_data.rows(), g.pca_components, g.pca_total_var * 100);
                    if (ImGui::Button("Undo PCA")) undo_pca(g);
                }
            }

            // Animate
            ImGui::Checkbox("Animate iterations", &g.realtime_viz);

            // Run button
            ImGui::BeginDisabled(g.eval_running || g.clustering_running);
            if (ImGui::Button("Run Clustering", ImVec2(-1, 30))) run_clustering_async(g);
            ImGui::EndDisabled();

            if (g.clustering_running) {
                ImGui::TextDisabled("Running...");
                ImGui::ProgressBar(-1.0f, ImVec2(-1, 0));
            }

            // Reset
            ImGui::Separator();
            if (ImGui::Button("Reset All", ImVec2(-1, 24))) {
                g.clustering_done = false; g.current_eval_done = false;
                g.labels = Vector(); g.centroids = Matrix(); g.cluster_sizes.clear();
                g.inertia = 0; g.n_iter = 0; g.n_noise = 0;
                g.current_silhouette = 0; g.current_db = 0; g.current_ch = 0;
                if (g.renderer_ready) { g.renderer_obj->set_data(Matrix(), Vector(), Matrix()); g.renderer_obj->set_metrics(0, 0); g.renderer_obj->reset_view(); }
            }

            // Results
            if (g.clustering_done) {
                if (g.selected_algo == 3)
                    ImGui::Text("Clusters: %d | Noise: %zu", g.k, g.n_noise);
                else
                    ImGui::Text("Inertia: %.4f | Iter: %d", g.inertia, g.n_iter);
                ImGui::Text("Silhouette: %.4f", g.current_silhouette);
                ImGui::Text("Davies-Bouldin: %.4f", g.current_db);
                for (size_t c = 0; c < g.cluster_sizes.size(); ++c)
                    if (g.selected_algo != 3 || c > 0)
                        ImGui::Text(g.selected_algo == 3 && c == 0 ? "  Noise: %zu pts" : "  C%zu: %zu pts",
                            g.selected_algo == 3 && c == 0 ? g.cluster_sizes[c] : c, g.cluster_sizes[c]);
            }
        }
    }
}

} // namespace clustering_app
