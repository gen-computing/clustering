// ============================================================================
// main.cpp — GLFW/ImGui/OpenGL application shell.
// All UI logic is in ui_*.cpp files. This file only handles:
//   1. Window/context creation
//   2. ImGui initialization
//   3. Main render loop (calls panel functions)
//   4. Cleanup
// ============================================================================

#include "ui.h"
#include "clustering/logging.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>

using namespace clustering_app;

namespace clustering_app {
ImFont* g_font_big = nullptr;
ImFont* g_font_header = nullptr;
}

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Clustering Engine", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { fprintf(stderr, "Failed to init GLEW\n"); return 1; }

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    // Large fonts: DejaVu Sans from system (fallback: scaled default bitmap font).
    const char* font_paths[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    };
    ImFont* font_loaded = nullptr;
    for (const char* fp : font_paths) {
        font_loaded = io.Fonts->AddFontFromFileTTF(fp, 15.0f);
        if (font_loaded) break;
    }
    if (!font_loaded) font_loaded = io.Fonts->AddFontDefault();
    g_font_big = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18.0f);
    if (!g_font_big) { ImFontConfig cbig; cbig.SizePixels = 18.0f; g_font_big = io.Fonts->AddFontDefault(&cbig); }
    g_font_header = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26.0f);
    if (!g_font_header) { ImFontConfig chdr; chdr.SizePixels = 26.0f; g_font_header = io.Fonts->AddFontDefault(&chdr); }

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Application state
    AppState g;

    // Initialize renderer
    RendererConfig rcfg;
    rcfg.width = 800; rcfg.height = 600;
    rcfg.point_size = 6.0f; rcfg.show_centroids = true; rcfg.show_axes = true;
    g.renderer_obj = new Renderer(rcfg);
    g.renderer_ready = g.renderer_obj->init_headless();

    // Create FBO
    create_fbo(g, 800, 600);

    // Initialize data
    g.stats.set_data(&g.table);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Menu bar
        render_menu_bar(g);

        // Tab bar (leave room at the bottom for the status bar)
        static int active_tab = 0;
        float status_h = ImGui::GetFrameHeight() * 2 + 8;
        ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetFrameHeight()));
        ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, io.DisplaySize.y - ImGui::GetFrameHeight() - status_h),
            ImGuiCond_Always);
        ImGui::Begin("##Main", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoBringToFrontOnFocus);

        if (ImGui::BeginTabBar("##Tabs")) {
            ImGui::PushFont(g_font_big);
            if (ImGui::BeginTabItem("1. Preprocess")) { active_tab = 0; ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("2. Dimension Reduction")) { active_tab = 1; ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("3. Cluster & Evaluate")) { active_tab = 2; ImGui::EndTabItem(); }
            ImGui::PopFont();
            ImGui::EndTabBar();
        }

        // Left sidebar + Right panel
        static float left_w = 350;
        ImGui::BeginChild("Sidebar", ImVec2(left_w, 0), ImGuiChildFlags_ResizeX, ImGuiWindowFlags_AlwaysVerticalScrollbar);

        if (active_tab == 0) {
            render_import(g);
            render_column_stats(g);
            render_preprocessing(g);
        } else if (active_tab == 1) {
            render_dimred(g);
        } else {
            render_select_columns(g);
            render_select_algorithm(g);
            render_algo_params(g);
            if (g.selected_algo == 3) {
                render_find_optimal_dbscan(g);
            } else {
                render_find_optimal_k(g);
            }
            render_clustering(g);
        }

        left_w = ImGui::GetWindowWidth();
        ImGui::EndChild();
        ImGui::SameLine();

        // Right panel
        ImGui::BeginChild("RightPanel", ImVec2(0, 0));
        if (active_tab == 0) {
            render_data_table(g);
        } else if (active_tab == 1) {
            // Dimensionality Reduction: show viewport if we have reduced data, else data table
            if (g.reduced_data.rows() > 0 || g.tsne_embedding.rows() > 0) {
                render_viewport(g);
            } else {
                render_data_table(g);
            }
        } else {
            render_viewport(g);
            render_compare_history(g, active_tab);
        }
        ImGui::EndChild();

        ImGui::End(); // Main

        // Status bar
        render_status_bar(g, active_tab);

        // Render
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    if (g.cluster_thread.joinable()) g.cluster_thread.join();
    if (g.eval_thread.joinable()) g.eval_thread.join();
    if (g.tsne_thread.joinable()) g.tsne_thread.join();
    delete g.kmeans;
    delete g.renderer_obj;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
