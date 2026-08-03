// ============================================================================
// ui_dimred.cpp — Tab 1: Dimensionality Reduction (PCA + t-SNE).
// ============================================================================

#include "ui.h"
#include "clustering/distance.h"
#include <imgui.h>

namespace clustering_app {

static char tsne_status[128] = "";

void render_pca(AppState& g) {
    if (ImGui::CollapsingHeader("PCA", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (g.data_loaded) {
            ImGui::TextDisabled("Linear dimensionality reduction");
            ImGui::SliderInt("Components", &g.pca_components, 2, std::min(50, (int)g.table.cols()));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of dimensions to keep. Max = min(rows, cols).");

            ImGui::BeginDisabled(g.clustering_running);
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
}

void render_tsne(AppState& g) {
    if (ImGui::CollapsingHeader("t-SNE", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (g.data_loaded) {
            ImGui::TextDisabled("Non-linear visualization embedding");
            ImGui::SliderInt("Perplexity", &g.tsne_perplexity, 5, std::min(50, (int)g.table.rows() - 1));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("How many neighbors to consider. Typical: 5-50.");
            ImGui::SliderFloat("Learning Rate", &g.tsne_lr, 10.0f, 1000.0f, "%.0f");
            ImGui::SliderInt("Iterations", &g.tsne_iter, 250, 2000);
            ImGui::SliderInt("Seed", &g.tsne_seed, -1, 100);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("-1 = random, >=0 = reproducible");

            ImGui::BeginDisabled(g.clustering_running);
            if (ImGui::Button("Run t-SNE", ImVec2(-1, 0))) {
                g.status_text = "Running t-SNE...";
                g.status_time = 999.0f;
                // Run t-SNE synchronously (blocking but simple)
                size_t max_n = std::min((size_t)5000, g.table.rows());
                Matrix X(max_n, g.table.cols());
                for (size_t i = 0; i < max_n; ++i)
                    for (size_t j = 0; j < g.table.cols(); ++j)
                        X[i][j] = g.table.data()[i][j];

                clustering::TSNEConfig cfg;
                cfg.perplexity = g.tsne_perplexity;
                cfg.learning_rate = g.tsne_lr;
                cfg.n_iter = g.tsne_iter;
                cfg.random_seed = g.tsne_seed;

                clustering::TSNE tsne(cfg);
                tsne.fit(X);
                g.tsne_embedding = tsne.embedding();
                g.tsne_done = true;
                snprintf(tsne_status, sizeof(tsne_status), "KL: %.4f", tsne.kl_divergence());
                g.status_text = "t-SNE done";
                g.status_time = 3.0f;
            }
            ImGui::EndDisabled();

            if (g.tsne_done) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Embedding: %zu x 2",
                    g.tsne_embedding.rows());
                ImGui::Text("%s", tsne_status);
            }
        }
    }
}

void render_dimred(AppState& g) {
    render_pca(g);
    render_tsne(g);
}

} // namespace clustering_app
