// ============================================================================
// ui_viewport.cpp — Tab 1: 3D OpenGL viewport with FBO rendering.
// ============================================================================

#include "ui.h"
#include <imgui.h>
#include <GL/glew.h>

namespace clustering_app {

void render_viewport(AppState& g) {
    ImGui::BeginChild("ViewportChild", ImVec2(0, 0));
    if (g.data_loaded && g.renderer_ready) {
        ImVec2 vp_size = ImGui::GetContentRegionAvail();
        int vp_w = (int)vp_size.x, vp_h = (int)std::max(vp_size.y, 50.0f);
        if (vp_w > 0 && vp_h > 0) {
            // Resize FBO if needed
            if (vp_w != g.fbo_width || vp_h != g.fbo_height) {
                glDeleteFramebuffers(1, &g.fbo);
                glDeleteTextures(1, &g.fbo_texture);
                glDeleteRenderbuffers(1, &g.fbo_rbo);
                create_fbo(g, vp_w, vp_h);
            }

            // Set data - handle PCA/t-SNE reduced data
            if (g.reduced_data.rows() > 0) {
                // PCA reduced data
                Vector neutral(g.reduced_data.rows()); neutral.fill(-1.0f);
                Matrix cnt(1, g.reduced_data.cols());
                g.renderer_obj->set_data(g.reduced_data, neutral, cnt);
                g.renderer_obj->set_metrics(0, 0);
            } else if (g.tsne_embedding.rows() > 0) {
                // t-SNE embedding (2D, pad to 3D for rendering)
                Matrix padded(g.tsne_embedding.rows(), 3);
                for (size_t i = 0; i < g.tsne_embedding.rows(); ++i) {
                    padded[i][0] = g.tsne_embedding[i][0];
                    padded[i][1] = g.tsne_embedding[i][1];
                    padded[i][2] = 0.0f;
                }
                Vector neutral(padded.rows()); neutral.fill(-1.0f);
                Matrix cnt(1, 3);
                g.renderer_obj->set_data(padded, neutral, cnt);
                g.renderer_obj->set_metrics(0, 0);
            } else if (!g.clustering_done) {
                // No reduced data, show original with neutral labels
                const Matrix& X = g.table.data();
                Vector neutral(X.rows()); neutral.fill(-1.0f);
                Matrix cnt(1, X.cols());
                g.renderer_obj->set_data(X, neutral, cnt);
                g.renderer_obj->set_metrics(0, 0);
            } else {
                // Clustering done
                { std::lock_guard<std::mutex> lk(g.result_mutex);
                  g.renderer_obj->set_data(g.table.data(), g.labels, g.centroids.rows() > 0 ? g.centroids : Matrix(1, g.table.cols()));
                  g.renderer_obj->set_metrics(g.inertia, g.n_iter); }
            }

            // Render
            g.renderer_obj->render_to_fbo(g.fbo, vp_w, vp_h);
            ImVec2 img_pos = ImGui::GetCursorScreenPos();
            ImGui::Image((ImTextureID)(intptr_t)g.fbo_texture, ImVec2((float)vp_w, (float)vp_h), ImVec2(0, 1), ImVec2(1, 0));

            // Text overlay
            ImDrawList* dl = ImGui::GetWindowDrawList();
            float tx = img_pos.x + 10, ty = img_pos.y + 8;
            dl->AddText(ImVec2(tx, ty), IM_COL32(50, 150, 255, 255), "CLUSTERING ENGINE"); ty += 18;

            size_t npts = g.clustering_done ?
                (g.use_pca && g.reduced_data.rows() > 0 ? g.reduced_data.rows() : g.table.rows()) :
                g.table.rows();
            dl->AddText(ImVec2(tx, ty), IM_COL32(220, 220, 240, 200),
                ("Points: " + std::to_string(npts)).c_str()); ty += 16;

            if (g.clustering_done) {
                dl->AddText(ImVec2(tx, ty), IM_COL32(220, 220, 240, 200),
                    ("Clusters: " + std::to_string(g.k)).c_str()); ty += 16;
                if (g.selected_algo != 3) {
                    char buf[64]; snprintf(buf, sizeof(buf), "Inertia: %.1f", g.inertia);
                    dl->AddText(ImVec2(tx, ty), IM_COL32(220, 220, 240, 200), buf); ty += 16;
                    snprintf(buf, sizeof(buf), "Iterations: %d", g.n_iter);
                    dl->AddText(ImVec2(tx, ty), IM_COL32(220, 220, 240, 200), buf); ty += 16;
                }
                if (g.current_eval_done) {
                    char buf[64]; snprintf(buf, sizeof(buf), "Silhouette: %.4f", g.current_silhouette);
                    dl->AddText(ImVec2(tx, ty), IM_COL32(100, 255, 100, 220), buf); ty += 16;
                    snprintf(buf, sizeof(buf), "Davies-Bouldin: %.4f", g.current_db);
                    dl->AddText(ImVec2(tx, ty), IM_COL32(255, 200, 100, 220), buf); ty += 16;
                }
                for (size_t c = 0; c < g.cluster_sizes.size() && c < 10; ++c) {
                    if (g.selected_algo == 3 && c == 0) continue;
                    char buf[64]; snprintf(buf, sizeof(buf), "  C%zu: %zu pts", c, g.cluster_sizes[c]);
                    dl->AddText(ImVec2(tx, ty), IM_COL32(200, 200, 220, 180), buf); ty += 14;
                }
            } else {
                dl->AddText(ImVec2(tx, ty), IM_COL32(180, 180, 200, 160), "Unclustered data (run clustering)"); ty += 16;
            }

            // Mouse interaction
            if (ImGui::IsItemHovered()) {
                ImGuiIO& io = ImGui::GetIO();
                if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
                    g.renderer_obj->rotate_view(io.MouseDelta.x * 0.008f, io.MouseDelta.y * 0.008f);
                if (io.MouseWheel != 0)
                    g.renderer_obj->zoom_view(1.0f + io.MouseWheel * 0.1f);
                if (ImGui::IsKeyPressed(ImGuiKey_R))
                    g.renderer_obj->reset_view();
            }

            // Centroid labels
            static const float cpal[][3] = {{0.95f,0.3f,0.3f},{0.3f,0.65f,0.95f},{0.3f,0.9f,0.4f},{0.95f,0.85f,0.25f},{0.9f,0.35f,0.9f},{0.25f,0.9f,0.9f},{0.95f,0.6f,0.25f},{0.65f,0.35f,0.95f},{0.55f,0.85f,0.3f},{0.35f,0.35f,0.95f}};
            if (g.clustering_done && g.centroids.rows() > 0) {
                for (size_t c = 0; c < (size_t)g.centroids.rows() && c < (size_t)g.cluster_sizes.size(); ++c) {
                    float sx, sy;
                    float cz = g.centroids.cols() > 2 ? g.centroids[c][2] : 0.0f;
                    if (g.renderer_obj->project_to_screen(g.centroids[c][0], g.centroids[c][1], cz, vp_w, vp_h, &sx, &sy)) {
                        float px = img_pos.x + sx, py = img_pos.y + sy;
                        int ci = (int)c % 10;
                        dl->AddCircleFilled(ImVec2(px, py), 5.0f,
                            IM_COL32((int)(cpal[ci][0] * 255), (int)(cpal[ci][1] * 255), (int)(cpal[ci][2] * 255), 255));
                        char lbuf[64]; size_t sz = c < g.cluster_sizes.size() ? g.cluster_sizes[c] : 0;
                        snprintf(lbuf, sizeof(lbuf), "C%zu: %zu", c, sz);
                        dl->AddText(ImVec2(px + 8, py - 6), IM_COL32(255, 255, 255, 220), lbuf);
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

void render_compare_history(AppState& g, int active_tab) {
    if (active_tab == 1 && !g.compare_history.empty()) {
        ImGui::Separator();
        ImGui::Text("Comparison History:");
        ImGui::SameLine();
        if (ImGui::Button("Clear")) g.compare_history.clear();
        if (ImGui::BeginTable("chist", 5,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
            ImVec2(0, 80))) {
            ImGui::TableSetupColumn("Algo");
            ImGui::TableSetupColumn("k");
            ImGui::TableSetupColumn("Silhouette");
            ImGui::TableSetupColumn("DB");
            ImGui::TableSetupColumn("Inertia");
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
}

} // namespace clustering_app
