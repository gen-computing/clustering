// ============================================================================
// ui_status.cpp — Status bar at bottom of screen.
// ============================================================================

#include "ui.h"
#include <imgui.h>
#include <chrono>

namespace clustering_app {

// Track clustering start time for ETA
static std::chrono::steady_clock::time_point cluster_start_time;
static bool cluster_timer_active = false;

void render_status_bar(AppState& g) {
    float status_h = ImGui::GetFrameHeight() * 2 + 8;
    ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetIO().DisplaySize.y - status_h));
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, status_h));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 4));
    ImGui::Begin("##status", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings);

    static const char* algo_names[] = {"KMeans", "MiniBatch", "Online", "DBSCAN"};

    // Track clustering start time
    if (g.clustering_running && !cluster_timer_active) {
        cluster_start_time = std::chrono::steady_clock::now();
        cluster_timer_active = true;
    }
    if (!g.clustering_running) {
        cluster_timer_active = false;
    }

    if (g.clustering_running) {
        // Show algorithm, params, progress bar, and time estimate
        float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - cluster_start_time).count();

        if (g.selected_algo == 2) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Online  k=%d  W=%d  F=%.2f",
                g.k, g.window_size, g.forgetting_factor);
        } else {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "%s  k=%d  %s",
                algo_names[g.selected_algo], g.k, g.use_pca ? "PCA" : "raw");
        }
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "|  Iter: %d  |  %.1fs", g.n_iter, elapsed);

        // Progress bar
        float progress = (g.n_iter > 0) ? std::min(1.0f, (float)g.n_iter / (float)g.max_iter) : -1.0f;
        ImGui::ProgressBar(progress, ImVec2(-1, 0));

    } else if (g.clustering_done) {
        float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - cluster_start_time).count();
        if (g.selected_algo == 2) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Online  k=%d  W=%d  F=%.2f  %s",
                g.k, g.window_size, g.forgetting_factor, g.online_auto_retrain ? "Auto" : "Manual");
            ImGui::SameLine();
        } else {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "%s  k=%d", algo_names[g.selected_algo], g.k);
            ImGui::SameLine();
        }
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
            "|  Inertia: %.2f  Iter: %d  %.1fs  Sil: %.4f  DB: %.4f",
            g.inertia, g.n_iter, elapsed, g.current_silhouette, g.current_db);

    } else if (g.reduced_data.rows() > 0) {
        // PCA result
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "PCA: %zu x %d  Var: %.1f%%",
            g.reduced_data.rows(), g.pca_components, g.pca_total_var * 100);
        ImGui::SameLine();
        if (g.tsne_done) {
            ImGui::TextColored(ImVec4(0.8f, 0.4f, 1.0f, 1.0f), "  |  t-SNE: %zu x 2",
                g.tsne_embedding.rows());
        }
    } else if (g.tsne_done) {
        ImGui::TextColored(ImVec4(0.8f, 0.4f, 1.0f, 1.0f), "t-SNE: %zu x 2  KL: %.4f",
            g.tsne_embedding.rows(), 0.0f);

    } else if (!g.status_text.empty()) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "%s", g.status_text.c_str());
        g.status_time -= ImGui::GetIO().DeltaTime;
        if (g.status_time <= 0) g.status_text.clear();
    } else if (g.data_loaded) {
        size_t nsel = 0;
        for (auto v : g.selected_cols) if (v) nsel++;
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "%zu rows  %zu cols  |  %zu selected",
            g.table.rows(), g.table.cols(), nsel);
    }

    ImGui::PopStyleVar();
    ImGui::End();
}

} // namespace clustering_app
