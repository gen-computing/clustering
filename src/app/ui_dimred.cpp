// ============================================================================
// ui_dimred.cpp — Tab 1: Dimensionality Reduction (PCA + t-SNE).
// ============================================================================

#include "ui.h"
#include "clustering/distance.h"
#include <imgui.h>

namespace clustering_app {

static char tsne_status[128] = "";

void render_pca(AppState& g) {
    render_section_header("PCA");
    if (g.data_loaded) {
        ImGui::TextDisabled("Linear dimensionality reduction");
        if (g_font_big) ImGui::PushFont(g_font_big);
        ImGui::SliderInt("Components", &g.pca_components, 2, std::min(50, (int)g.table.cols()));
        if (g_font_big) ImGui::PopFont();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of dimensions to keep. Max = min(rows, cols).");

        ImGui::BeginDisabled(g.clustering_running || g.tsne_running);
        if (ImGui::Button("Run PCA", ImVec2(-1, 0))) run_pca(g);
        ImGui::EndDisabled();

        if (g.reduced_data.rows() > 0) {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Result: %zu x %d",
                g.reduced_data.rows(), g.pca_components);
            ImGui::Text("Variance retained: %.1f%%", g.pca_total_var * 100);
            if (g.pca_var_ratio.size() > 0) {
                ImGui::TextDisabled("Per-component variance:");
                for (int i = 0; i < std::min(10, (int)g.pca_var_ratio.size()); ++i) {
                    ImGui::Text("  C%d: %.1f%%", i + 1, g.pca_var_ratio[i] * 100);
                }
            }
            if (ImGui::Button("Undo PCA", ImVec2(-1, 0))) undo_pca(g);
        }
    }
}

void render_tsne(AppState& g) {
    render_section_header("t-SNE");
    if (g.data_loaded) {
        ImGui::TextDisabled("Non-linear visualization embedding");
        if (g_font_big) ImGui::PushFont(g_font_big);
        ImGui::SliderInt("Perplexity", &g.tsne_perplexity, 5, std::min(50, (int)g.table.rows() - 1));
        if (g_font_big) ImGui::PopFont();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How many neighbors to consider. Typical: 5-50.");
        ImGui::SliderFloat("Learning Rate", &g.tsne_lr, 10.0f, 1000.0f, "%.0f");
        ImGui::SliderInt("Iterations", &g.tsne_iter, 250, 2000);
        ImGui::SliderInt("Seed", &g.tsne_seed, -1, 100);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("-1 = random, >=0 = reproducible");

        ImGui::BeginDisabled(g.clustering_running || g.tsne_running);
        if (ImGui::Button("Run t-SNE", ImVec2(-1, 0))) run_tsne_async(g);
        ImGui::EndDisabled();

        if (g.tsne_running) {
            ImGui::TextColored(ImVec4(0.8f, 0.4f, 1.0f, 1.0f), "Running t-SNE in background...");
        }

        if (g.tsne_done && g.tsne_embedding.rows() > 0) {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Embedding: %zu x 2",
                g.tsne_embedding.rows());
        }
    }
}

void render_dimred(AppState& g) {
    render_section_header("Dimensionality Reduction");

    if (g.data_loaded) {
        // Which data feeds clustering + the viewport.
        static const char* src_names[] = {"Original (preprocessed) data", "PCA-reduced data", "t-SNE embedding"};
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Clustering input: %s", src_names[g.cluster_source]);
        ImGui::TextDisabled("  Change it in the Cluster & Evaluate tab (Input data)");
        ImGui::Separator();
    }

    render_pca(g);
    render_tsne(g);

    if (g.data_loaded && (g.reduced_data.rows() > 0 || g.tsne_done)) {
        ImGui::Separator();
        if (ImGui::Button("Reset Dimensionality Reduction", ImVec2(-1, 28))) reset_dimred(g);
        ImGui::TextDisabled("Clears PCA + t-SNE; viewport returns to original data");
    }
}

} // namespace clustering_app
