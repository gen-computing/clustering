// ============================================================================
// ui_status.cpp — Status bar at bottom of screen.
// ============================================================================

#include "ui.h"
#include <imgui.h>

namespace clustering_app {

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
                g.k, g.window_size, g.forgetting_factor, g.online_auto_retrain ? "Auto" : "Manual");
            ImGui::SameLine();
        } else {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "%s  k=%d", algo_names[g.selected_algo], g.k);
            ImGui::SameLine();
        }
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
            "|  Inertia: %.2f  Iter: %d  Sil: %.4f  DB: %.4f",
            g.inertia, g.n_iter, g.current_silhouette, g.current_db);
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
