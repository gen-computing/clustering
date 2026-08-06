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

            // Set data. Displays the SAME source clustering consumes
            // (cluster_source), so labels always align with the points.
            // Source 0 additionally honors the selected-columns mask so the
            // plot replots whenever the selection changes.
            Matrix padded;
            Matrix selected_disp;
            const Matrix* disp = nullptr;
            size_t disp_rows = 0;
            if (g.cluster_source == 2 && g.tsne_done && g.tsne_embedding.rows() > 0) {
                padded.resize(g.tsne_embedding.rows(), 3);
                for (size_t i = 0; i < g.tsne_embedding.rows(); ++i) {
                    padded[i][0] = g.tsne_embedding[i][0];
                    padded[i][1] = g.tsne_embedding[i][1];
                    padded[i][2] = 0.0f;
                }
                disp = &padded;
                disp_rows = g.tsne_embedding.rows();
            } else if (g.cluster_source == 1 && g.reduced_data.rows() > 0) {
                disp = &g.reduced_data;
                disp_rows = g.reduced_data.rows();
            } else {
                selected_disp = extract_selected_cols(g.table.data(), g.selected_cols);
                if (selected_disp.cols() == 1) {
                    // Renderer reads [i][0..1]; pad single-column selection.
                    Matrix pad(selected_disp.rows(), 2);
                    for (size_t i = 0; i < selected_disp.rows(); ++i) {
                        pad[i][0] = selected_disp[i][0];
                        pad[i][1] = 0.0f;
                    }
                    selected_disp = std::move(pad);
                }
                disp = &selected_disp;
                disp_rows = selected_disp.rows();
            }

            Vector show_labels;
            Matrix show_centroids;
            bool labels_valid = g.clustering_done && g.labels.size() == disp_rows;
            if (labels_valid) {
                { std::lock_guard<std::mutex> lk(g.result_mutex);
                    show_labels = g.labels;
                    show_centroids = g.centroids; }
                if (g.cluster_source == 0 && show_centroids.rows() > 0 &&
                    show_centroids.cols() != disp->cols()) {
                    // Clustering ran on more columns than currently selected:
                    // project centroids into the displayed subspace too.
                    show_centroids = extract_selected_cols(show_centroids, g.selected_cols);
                }
                if (g.selected_algo == 3) {
                    // DBSCAN: label 0 = noise. Renderer colors label >= 0, so
                    // map noise to -1 (gray); clusters 1..k keep their colors.
                    for (size_t i = 0; i < show_labels.size(); ++i)
                        if (show_labels[i] < 1.0f) show_labels[i] = -1.0f;
                    show_centroids = Matrix();  // centroids meaningless in DBSCAN
                }
                g.renderer_obj->set_data(*disp, show_labels,
                    show_centroids.rows() > 0 ? show_centroids : Matrix(1, disp->cols()));
                g.renderer_obj->set_metrics(g.inertia, g.n_iter);
            } else {
                Vector neutral(disp_rows); neutral.fill(-1.0f);
                Matrix cnt(1, disp->cols());
                g.renderer_obj->set_data(*disp, neutral, cnt);
                g.renderer_obj->set_metrics(0, 0);
            }

            // Render
            g.renderer_obj->render_to_fbo(g.fbo, vp_w, vp_h);
            ImVec2 img_pos = ImGui::GetCursorScreenPos();
            ImGui::Image((ImTextureID)(intptr_t)g.fbo_texture, ImVec2((float)vp_w, (float)vp_h), ImVec2(0, 1), ImVec2(1, 0));

            // Text overlay
            ImDrawList* dl = ImGui::GetWindowDrawList();
            float tx = img_pos.x + 10, ty = img_pos.y + 8;
            dl->AddText(ImVec2(tx, ty), IM_COL32(50, 150, 255, 255), "CLUSTERING ENGINE"); ty += 18;

            size_t npts = g.clustering_done ? disp_rows : g.table.rows();
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
            if (g.clustering_done && show_centroids.rows() > 0) {
                for (size_t c = 0; c < (size_t)show_centroids.rows() && c < (size_t)g.cluster_sizes.size(); ++c) {
                    float sx, sy;
                    float cz = show_centroids.cols() > 2 ? show_centroids[c][2] : 0.0f;
                    if (g.renderer_obj->project_to_screen(show_centroids[c][0], show_centroids[c][1], cz, vp_w, vp_h, &sx, &sy)) {
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
