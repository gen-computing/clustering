// ============================================================================
// ui_menu.cpp — Menu bar and About popup.
// ============================================================================

#include "ui.h"
#include <imgui.h>
#include <GLFW/glfw3.h>

namespace clustering_app {

static bool show_about = false;

void render_menu_bar(AppState& g) {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open CSV", "Ctrl+O")) open_csv(g);
            ImGui::Separator();
            if (ImGui::MenuItem("Export Labels", nullptr, false, g.clustering_done)) export_labels_csv(g);
            if (ImGui::MenuItem("Export Centroids", nullptr, false, g.clustering_done)) export_centroids_csv(g);
            if (ImGui::MenuItem("Export PNG", nullptr, false, g.renderer_ready)) export_png(g);
            if (ImGui::MenuItem("Export Report", nullptr, false, g.data_loaded)) export_report(g);
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) glfwSetWindowShouldClose(glfwGetCurrentContext(), true);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit")) {
            if (ImGui::MenuItem("Undo Preprocessing", "Ctrl+Z")) { g.table.pipeline().undo(); g.stats.invalidate(); }
            if (ImGui::MenuItem("Redo Preprocessing", "Ctrl+Y")) { g.table.pipeline().redo(); g.stats.invalidate(); }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("About")) show_about = true;
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (show_about) {
        ImGui::OpenPopup("About Clustering Engine");
        show_about = false;
    }
    if (ImGui::BeginPopupModal("About Clustering Engine", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Clustering Engine v1.0.0");
        ImGui::Separator();
        ImGui::Text("High-performance C++ clustering with AVX2 SIMD");
        ImGui::Text("Interactive GUI with 3D OpenGL visualization");
        ImGui::Separator();
        ImGui::Text("Algorithms: KMeans, MiniBatch, Online, DBSCAN");
        ImGui::Text("Features: PCA, t-SNE, Drift Detection");
        ImGui::Separator();
        ImGui::Text("License: MIT");
        if (ImGui::Button("OK")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

} // namespace clustering_app
