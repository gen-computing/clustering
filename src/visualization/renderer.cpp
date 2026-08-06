// ============================================================================
// OpenGL Renderer implementation -- 3D visualization of clustering results.
//
// This renderer opens a window using GLFW and draws:
//   - Data points as colored circles (one color per cluster, 10-color palette)
//   - Centroids as larger white circles
//   - 3D coordinate axes (red=X, green=Y, blue=Z)
//   - Text overlay showing cluster info (custom bitmap font)
//
// GRAPHICS PIPELINE (how each frame is drawn):
//   1. CPU: Build vertex buffers (position + color for each dot).
//   2. CPU: Compute camera matrix (model-view-projection) from mouse state.
//   3. GPU: Vertex shader transforms 3D positions to screen coordinates.
//   4. GPU: Fragment shader draws soft circles (smoothstep for anti-aliasing).
//   5. GPU: Overlay text using orthographic projection.
//
// CAMERA (orbital model):
//   rotation_x: vertical angle (mouse up/down drag).
//   rotation_y: horizontal angle (mouse left/right drag).
//   zoom: camera distance from center (scroll wheel).
//   Camera orbits around the bounding box center of the data.
//
// COORDINATE SYSTEMS:
//   World (3D):    data points in their original feature space.
//   View:          transformed by camera position and orientation.
//   Projection:    perspective projection (things farther away look smaller).
//   Screen (2D):   final pixel coordinates on the monitor.
//
// SHADERS (small programs that run on the GPU):
//   Vertex shader: runs once per vertex, outputs transformed position.
//   Fragment shader: runs once per pixel, outputs color.
//   GLSL (OpenGL Shading Language): C-like syntax, compiled at runtime.
// ============================================================================

#include "clustering/renderer.h"
#include "mat4.h"
#include "shaders.h"
#include "text.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>

namespace clustering {

// RendererImpl -- Global renderer state (C-style singleton).
//
// Why global? The GLFW callbacks (key, mouse, scroll) are C function pointers
// with no `this` parameter. They need access to renderer state. Using a global
// variable is the simplest way to bridge GLFW's C API with C++.
//
// This means only ONE renderer can exist at a time. For multiple windows,
// you'd need a more sophisticated approach.
// ============================================================================

struct RendererImpl {
    // ---- GLFW/OpenGL handles ----
    GLFWwindow* window = nullptr;     // The actual window
    GLuint point_prog = 0;            // Shader program for points
    GLuint text_prog = 0;             // Shader program for text (unused)
    GLuint line_prog = 0;             // Shader program for lines

    // ---- OpenGL buffer handles ----
    // VAO (Vertex Array Object): stores attribute layout (which buffer, data format).
    // VBO (Vertex Buffer Object): stores actual vertex data on GPU.
    GLuint vao_pts = 0, vbo_pts = 0;       // Points
    GLuint vao_cent = 0, vbo_cent = 0;     // Centroids
    GLuint vao_ax = 0, vbo_ax = 0;         // Axes
    GLuint vao_txt = 0, vbo_txt = 0;       // Text overlay

    // ---- Camera state (modified by mouse input) ----
    float rot_x = 0.4f;      // Vertical rotation (radians). 0.4 = ~23 degrees.
    float rot_y = 0.5f;      // Horizontal rotation (radians). 0.5 = ~29 degrees.
    float zoom = 1.0f;       // Zoom factor (1.0 = default distance)
    double mx = 0, my = 0;   // Previous mouse position (for drag delta)
    bool pressed = false;    // Is left mouse button held?

    // ---- Configuration ----
    RendererConfig cfg;      // User-provided settings

    // ---- Data to visualize ----
    // Points may be referenced (not copied) when large, to avoid per-frame
    // copies of disk-backed matrices. The caller must keep referenced data
    // alive for as long as the renderer shows it.
    const Matrix* pts = nullptr;      // Data points (owned or borrowed)
    std::shared_ptr<Matrix> owned_pts; // Owns points when copied (small inputs)
    Vector lbls;             // Cluster labels (copied, small)
    Matrix cent;             // Centroids (copied, small)
    size_t n_pts = 0;        // Number of data points
    size_t n_clust = 0;      // Number of clusters

    bool inited = false;     // Has init() been called?

    // ---- Metrics overlay ----
    float inertia = 0;       // Clustering inertia
    int iterations = 0;      // Number of KMeans iterations
    float fps = 0;           // Frames per second
    double last_time = 0;    // Time of last FPS update
    int frame_count = 0;     // Frames since last FPS update
};

// The global singleton renderer state.
static RendererImpl g_impl;

// ============================================================================
// GLFW CALLBACKS -- Input event handlers called by GLFW.
//
// These are C-style static/global functions. GLFW calls them when the user
// presses a key, moves the mouse, etc. Each callback receives a pointer to
// the GLFWwindow, from which we access g_impl (the global renderer state).
// ============================================================================

// Key callback: handles ESC (close) and R (reset view).
static void cb_key(GLFWwindow* w, int key, int sc, int act, int mod) {
    (void)sc; (void)mod;  // Unused parameters -- silence compiler warnings
    auto& r = g_impl;     // Reference to global renderer state

    if (act == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            // Tell GLFW the window should close. The main loop will detect this
            // on the next glfwWindowShouldClose() check and exit cleanly.
            glfwSetWindowShouldClose(w, GLFW_TRUE);
        }
        if (key == GLFW_KEY_R) {
            // Reset camera to default viewing angle.
            r.rot_x = 0.4f;
            r.rot_y = 0.5f;
            r.zoom = 1.0f;
        }
    }
}

// Mouse button callback: tracks left button press/release for rotation.
static void cb_mbtn(GLFWwindow* w, int btn, int act, int mod) {
    (void)mod;
    auto& r = g_impl;

    if (btn == GLFW_MOUSE_BUTTON_LEFT) {
        r.pressed = (act == GLFW_PRESS);  // Track press state
        if (act == GLFW_PRESS) {
            // Record initial mouse position to compute deltas.
            glfwGetCursorPos(w, &r.mx, &r.my);
        }
    }
}

// Cursor position callback: handles mouse drag for camera rotation.
// Called continuously while mouse moves (not just when button is held).
// We check r.pressed to only rotate when left button is held.
static void cb_mpos(GLFWwindow* w, double x, double y) {
    auto& r = g_impl;
    if (!r.pressed) return;

    // Delta from last mouse position.
    // x - r.mx: horizontal movement (rotate around Y axis).
    // y - r.my: vertical movement (rotate around X axis).
    // 0.008 is sensitivity: how fast the view rotates per pixel of mouse movement.
    r.rot_y += (x - r.mx) * 0.008f;
    r.rot_x += (y - r.my) * 0.008f;

    // Clamp vertical rotation to prevent the camera from flipping.
    // Without clamping, rotating past ±90° would invert the view.
    r.rot_x = std::max(-1.5f, std::min(1.5f, r.rot_x));  // ~±86 degrees

    // Update previous position for next delta calculation.
    r.mx = x;
    r.my = y;
}

// Scroll callback: handles zoom in/out.
// yoffset > 0 = scroll up (zoom in), yoffset < 0 = scroll down (zoom out).
static void cb_scroll(GLFWwindow* w, double xo, double yo) {
    (void)xo;  // Horizontal scroll is unused (most mice don't have it)
    auto& r = g_impl;

    // Adjust zoom: each scroll tick changes by 15%.
    r.zoom *= (1.0f + (float)yo * 0.15f);

    // Clamp zoom to prevent going inside the data or too far out.
    r.zoom = std::max(0.001f, std::min(500.0f, r.zoom));
}

// ============================================================================
// RENDERER CLASS -- Public API (thin wrappers around g_impl)
// ============================================================================

Renderer::Renderer() {}
Renderer::Renderer(const RendererConfig& c) { g_impl.cfg = c; }
Renderer::~Renderer() { shutdown(); }

// ============================================================================
// init() -- CREATE WINDOW AND SET UP OPENGL
//
// Steps:
//   1. Initialize GLFW library (global state).
//   2. Configure window hints (OpenGL version, samples for anti-aliasing).
//   3. Create the window.
//   4. Set up OpenGL context (make it current).
//   5. Initialize GLEW (loads OpenGL function pointers).
//   6. Set up GLFW callbacks (key, mouse, scroll).
//   7. Enable features: point size programmability, alpha blending.
//   8. Create shader programs (compile + link GLSL code).
//   9. Create VAOs and VBOs for points, centroids, axes, text.
//
// Returns: true on success, false if any step fails (window creation, etc).
// ============================================================================

bool Renderer::init() {
    auto& r = g_impl;
    if (r.inited) return true;  // Already initialized -- nothing to do

    // ---- Step 1: Initialize GLFW ----
    if (!glfwInit()) return false;

    // ---- Step 2: Set window hints (must be BEFORE creating window) ----
    glfwWindowHint(GLFW_SAMPLES, 8);                              // 8x MSAA anti-aliasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);                // OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Core profile (no deprecated functions)

    // ---- Step 3: Create window ----
    r.window = glfwCreateWindow(r.cfg.width, r.cfg.height, r.cfg.title.c_str(), nullptr, nullptr);
    if (!r.window) {
        glfwTerminate();
        return false;
    }

    // ---- Step 4: Make OpenGL context current ----
    // All subsequent OpenGL calls apply to this window's context.
    glfwMakeContextCurrent(r.window);

    // Enable VSync: limit frame rate to monitor refresh rate (usually 60 FPS).
    // Prevents screen tearing and saves CPU/GPU power.
    glfwSwapInterval(1);

    // ---- Step 5: Set up input callbacks ----
    // glfwSetWindowUserPointer stores a pointer we can retrieve in callbacks.
    // Here we store the Renderer* for potential future use (not currently used,
    // since we use global state instead).
    glfwSetWindowUserPointer(r.window, this);
    glfwSetKeyCallback(r.window, cb_key);
    glfwSetMouseButtonCallback(r.window, cb_mbtn);
    glfwSetCursorPosCallback(r.window, cb_mpos);
    glfwSetScrollCallback(r.window, cb_scroll);

    // ---- Step 6: Initialize GLEW ----
    // GLEW must be initialized AFTER an OpenGL context is current.
    // glewExperimental = GL_TRUE enables modern OpenGL features.
    glewExperimental = GL_TRUE;
    glewInit();  // Loads all OpenGL function pointers

    // ---- Step 7: Enable OpenGL features ----
    // GL_PROGRAM_POINT_SIZE: allow setting gl_PointSize in the vertex shader.
    // Without this, point size is fixed and can't be changed per-draw.
    glEnable(GL_PROGRAM_POINT_SIZE);

    // GL_BLEND: enable alpha blending (transparency).
    // Needed for anti-aliased circles and text overlay.
    glEnable(GL_BLEND);

    // Blend function: source alpha * source color + (1 - source alpha) * destination color.
    // This is the standard "over" compositing: transparent pixels don't overwrite background.
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ---- Step 8: Create shader programs ----
    r.point_prog = create_program(point_vs, point_fs);
    r.line_prog  = create_program(line_vs, line_fs);

    // ---- Step 9: Create VAOs and VBOs ----
    // VAO (Vertex Array Object): stores the configuration of vertex attributes.
    // VBO (Vertex Buffer Object): a block of GPU memory for vertex data.
    //
    // We create separate VAO/VBO pairs for points, centroids, axes, and text
    // because each has different data and is drawn separately.
    glGenVertexArrays(1, &r.vao_pts);   glGenBuffers(1, &r.vbo_pts);
    glGenVertexArrays(1, &r.vao_cent);  glGenBuffers(1, &r.vbo_cent);
    glGenVertexArrays(1, &r.vao_ax);    glGenBuffers(1, &r.vbo_ax);
    glGenVertexArrays(1, &r.vao_txt);   glGenBuffers(1, &r.vbo_txt);

    r.inited = true;
    r.last_time = glfwGetTime();  // Initialize FPS timer
    return true;
}

// ============================================================================
// init_headless() -- INITIALIZE GL RESOURCES WITHOUT CREATING WINDOW
//
// Used when embedding the renderer inside an existing OpenGL context
// (e.g., ImGui window via FBO). Skips GLFW window creation.
// The caller must have a current OpenGL 3.3+ context.
// ============================================================================

bool Renderer::init_headless() {
    auto& r = g_impl;
    if (r.inited) return true;

    // ---- Enable OpenGL features ----
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ---- Create shader programs ----
    r.point_prog = create_program(point_vs, point_fs);
    r.line_prog  = create_program(line_vs, line_fs);

    // ---- Create VAOs and VBOs ----
    glGenVertexArrays(1, &r.vao_pts);   glGenBuffers(1, &r.vbo_pts);
    glGenVertexArrays(1, &r.vao_cent);  glGenBuffers(1, &r.vbo_cent);
    glGenVertexArrays(1, &r.vao_ax);    glGenBuffers(1, &r.vbo_ax);
    glGenVertexArrays(1, &r.vao_txt);   glGenBuffers(1, &r.vbo_txt);

    r.inited = true;
    return true;
}

// ============================================================================
// shutdown() -- DESTROY WINDOW AND CLEAN UP
// ============================================================================

void Renderer::shutdown() {
    auto& r = g_impl;
    if (r.window) {
        glfwDestroyWindow(r.window);
        r.window = nullptr;
        // Only terminate GLFW if this renderer created the window. In
        // headless (embedded/ImGui) mode the host owns the GLFW lifetime.
        glfwTerminate();
    }
    r.inited = false;
}

// ============================================================================
// set_data() -- PROVIDE DATA FOR VISUALIZATION
// ============================================================================

void Renderer::set_data(const Matrix& p, const Vector& l, const Matrix& c) {
    auto& r = g_impl;
    // Copy small inputs (callers may pass temporaries); borrow large ones.
    const size_t ref_threshold = 128u * 1024u * 1024u; // 128 MiB
    if (p.rows() * p.cols() * sizeof(float) > ref_threshold) {
        r.owned_pts.reset();
        r.pts = &p;
    } else {
        r.owned_pts = std::make_shared<Matrix>(p);
        r.pts = r.owned_pts.get();
    }
    r.lbls = l;
    r.cent = c;
    r.n_pts = p.rows();
    r.n_clust = c.rows();
}

// ============================================================================
// set_metrics() -- SET OVERLAY TEXT VALUES
// ============================================================================

void Renderer::set_metrics(float inert, int iters) {
    g_impl.inertia = inert;
    g_impl.iterations = iters;
}

// ============================================================================
// run() -- MAIN RENDER LOOP (BLOCKS UNTIL WINDOW CLOSES)
//
// The "game loop" pattern:
//   1. Check if window should close (ESC key, X button).
//   2. Render one frame (compute matrices, upload data, draw).
//   3. Swap front/back buffers (double buffering: draw to back, show front).
//   4. Poll for new input events (mouse, keyboard).
//   5. Go to step 1.
//
// This runs at VSync rate (typically 60 FPS) because glfwSwapInterval(1) limits it.
// ============================================================================

void Renderer::run() {
    auto& r = g_impl;
    if (!r.inited && !init()) return;  // Auto-init if not already initialized

    while (!glfwWindowShouldClose(r.window)) {
        render_frame();                    // Draw everything
        glfwSwapBuffers(r.window);        // Show the frame (swap front/back buffers)
        glfwPollEvents();                 // Process input events (keyboard, mouse)
    }
}

// ============================================================================
// render_frame() -- DRAW ONE COMPLETE FRAME
//
// Called every frame (~60 times/sec). Steps:
//   1. Update FPS counter.
//   2. Clear the screen (background color).
//   3. Compute bounding box of data (for camera positioning).
//   4. Compute camera position from rotation angles + zoom.
//   5. Build MVP (Model-View-Projection) matrix.
//   6. Draw coordinate axes (3 lines: X=red, Y=green, Z=blue).
//   7. Draw data points (colored by cluster).
//   8. Draw centroids (larger white points).
//   9. Draw text overlay (engine name, counts, inertia, FPS).
// ============================================================================

void Renderer::render_frame() {
    auto& r = g_impl;

    // ---- FPS Counter ----
    // Count frames, update FPS every 1 second.
    r.frame_count++;
    double now = glfwGetTime();
    if (now - r.last_time >= 1.0) {
        r.fps = r.frame_count / (now - r.last_time);
        r.frame_count = 0;
        r.last_time = now;
    }

    // ---- Get framebuffer size ----
    // On high-DPI displays, framebuffer size may differ from window size.
    int fbw, fbh;
    glfwGetFramebufferSize(r.window, &fbw, &fbh);

    // Skip rendering if window is minimized (zero size).
    if (fbw == 0 || fbh == 0) return;

    // ---- Viewport and clear ----
    // glViewport: map OpenGL's [-1,1] coordinate range to the actual pixel area.
    glViewport(0, 0, fbw, fbh);

    // Set clear color: dark blue-gray background (0.08, 0.09, 0.12).
    glClearColor(0.08f, 0.09f, 0.12f, 1.0f);

    // Clear both color buffer (pixels) and depth buffer (Z-ordering).
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    render_impl(fbw, fbh, true);
}

// ============================================================================
// render_to_fbo() -- RENDER TO FRAMEBUFFER OBJECT (for ImGui embedding)
//
// Binds the given FBO, sets viewport to the given size, renders one frame,
// then restores the default framebuffer. The FBO's color attachment texture
// can then be passed to ImGui::Image() for embedded viewport display.
// ============================================================================

void Renderer::render_to_fbo(unsigned int fbo, int width, int height) {
    auto& r = g_impl;
    if (!r.inited) return;

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, width, height);
    glClearColor(0.08f, 0.09f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    render_impl(width, height, false);

    // Restore default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::render_impl(int width, int height, bool with_fps) {
    auto& r = g_impl;

    float aspect = (float)width / height;

    float cx = 0, cy = 0, cz = 0, max_range = 1.0f;
    if (r.n_pts > 0) {
        float minx = (*r.pts)[0][0], maxx = minx, miny = (*r.pts)[0][1], maxy = miny;
        float minz = 0, maxz = 0;
        if (r.pts->cols() > 2) { minz = (*r.pts)[0][2]; maxz = minz; }
        for (size_t i = 0; i < r.n_pts; i++) {
            if ((*r.pts)[i][0] < minx) minx = (*r.pts)[i][0]; if ((*r.pts)[i][0] > maxx) maxx = (*r.pts)[i][0];
            if ((*r.pts)[i][1] < miny) miny = (*r.pts)[i][1]; if ((*r.pts)[i][1] > maxy) maxy = (*r.pts)[i][1];
            if (r.pts->cols() > 2) { if ((*r.pts)[i][2] < minz) minz = (*r.pts)[i][2]; if ((*r.pts)[i][2] > maxz) maxz = (*r.pts)[i][2]; }
        }
        cx = (minx + maxx) / 2; cy = (miny + maxy) / 2; cz = (minz + maxz) / 2;
        max_range = std::max({maxx - minx, maxy - miny, maxz - minz}) / 2.0f;
        if (max_range < 0.1f) max_range = 1.0f;
    }

    float dist = max_range * 3.0f / r.zoom;
    float ex = cx + dist * std::sin(r.rot_y) * std::cos(r.rot_x);
    float ey = cy + dist * std::sin(r.rot_x);
    float ez = cz + dist * std::cos(r.rot_y) * std::cos(r.rot_x);

    Mat4 proj3d = Mat4::perspective(45.0f * 3.14159f / 180.0f, aspect, dist * 0.01f, dist * 20.0f);
    Mat4 view3d = Mat4::lookAt(ex, ey, ez, cx, cy, cz, 0, 1, 0);
    Mat4 mvp3d = proj3d * view3d;

    // Axes
    {
        float al = max_range * 1.2f;
        float ax[] = {
            cx,cy,cz, 0.8,0.2,0.2, cx+al,cy,cz, 0.8,0.2,0.2,
            cx,cy,cz, 0.2,0.8,0.2, cx,cy+al,cz, 0.2,0.8,0.2,
            cx,cy,cz, 0.2,0.2,0.8, cx,cy,cz+al, 0.2,0.2,0.8,
        };
        glUseProgram(r.line_prog);
        glUniformMatrix4fv(glGetUniformLocation(r.line_prog, "uMVP"), 1, GL_FALSE, mvp3d.m);
        glBindVertexArray(r.vao_ax);
        glBindBuffer(GL_ARRAY_BUFFER, r.vbo_ax);
        glBufferData(GL_ARRAY_BUFFER, sizeof(ax), ax, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);
        glLineWidth(2.0f);
        glDrawArrays(GL_LINES, 0, 6);
    }

    // Points
    glUseProgram(r.point_prog);
    glUniformMatrix4fv(glGetUniformLocation(r.point_prog, "uMVP"), 1, GL_FALSE, mvp3d.m);
    glUniform1f(glGetUniformLocation(r.point_prog, "uSize"), r.cfg.point_size);

    if (r.n_pts > 0) {
        std::vector<float> vd(r.n_pts * 6);
        for (size_t i = 0; i < r.n_pts; i++) {
            float gray = 0.45f;
            float cr = gray, cg = gray, cb = gray;
            if (r.lbls[i] >= 0) {
                size_t c = (size_t)r.lbls[i] % 10;
                cr = palette[c][0]; cg = palette[c][1]; cb = palette[c][2];
            }
            vd[i * 6 + 0] = (*r.pts)[i][0]; vd[i * 6 + 1] = (*r.pts)[i][1];
            vd[i * 6 + 2] = (r.pts->cols() > 2) ? (*r.pts)[i][2] : 0.0f;
            vd[i * 6 + 3] = cr; vd[i * 6 + 4] = cg; vd[i * 6 + 5] = cb;
        }
        glBindVertexArray(r.vao_pts);
        glBindBuffer(GL_ARRAY_BUFFER, r.vbo_pts);
        glBufferData(GL_ARRAY_BUFFER, vd.size() * sizeof(float), vd.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);
        glDrawArrays(GL_POINTS, 0, r.n_pts);
    }

    // Centroids
    if (r.n_clust > 0 && r.cfg.show_centroids) {
        std::vector<float> cd(r.n_clust * 6);
        for (size_t i = 0; i < r.n_clust; i++) {
            cd[i * 6 + 0] = r.cent[i][0]; cd[i * 6 + 1] = r.cent[i][1];
            cd[i * 6 + 2] = (r.cent.cols() > 2) ? r.cent[i][2] : 0.0f;
            cd[i * 6 + 3] = 1.0f; cd[i * 6 + 4] = 1.0f; cd[i * 6 + 5] = 1.0f;
        }
        glUniform1f(glGetUniformLocation(r.point_prog, "uSize"), r.cfg.point_size * 2.5f);
        glBindVertexArray(r.vao_cent);
        glBindBuffer(GL_ARRAY_BUFFER, r.vbo_cent);
        glBufferData(GL_ARRAY_BUFFER, cd.size() * sizeof(float), cd.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);
        glDrawArrays(GL_POINTS, 0, r.n_clust);
    }

    // Text overlay
    glDisable(GL_DEPTH_TEST);
    {
        float sx = 16.0f, sy = (float)height - 30.0f, scale = 1.8f;
        float char_w = 7.0f * scale, char_h = 10.0f * scale, sp = 2.0f * scale;
        std::vector<std::string> lines = {
            "CLUSTERING ENGINE",
            "Points: " + std::to_string(r.n_pts),
            "Clusters: " + std::to_string(r.n_clust),
            "Iterations: " + std::to_string(r.iterations)
        };
        { std::ostringstream oss; oss << std::fixed << std::setprecision(1) << r.inertia; lines.push_back("Inertia: " + oss.str()); }
        if (with_fps) {
            std::ostringstream oss; oss << std::fixed << std::setprecision(0) << r.fps << " FPS";
            lines.push_back(oss.str());
        }

        Mat4 ortho = Mat4::ortho(0, (float)width, 0, (float)height, -1, 1);
        glUseProgram(r.point_prog);
        glUniformMatrix4fv(glGetUniformLocation(r.point_prog,"uMVP"), 1, GL_FALSE, ortho.m);
        glUniform1f(glGetUniformLocation(r.point_prog,"uSize"), 3.0f * scale);

        for (size_t line = 0; line < lines.size(); line++) {
            float lx = sx, ly = sy - line * (char_h + sp);
            float cr = 0.9f, cg = 0.9f, cb = 0.95f;
            if (line == 0) { cr = 0.3f; cg = 0.7f; cb = 1.0f; }
            std::vector<float> verts;
            for (char ch : lines[line]) {
                auto bitmap = get_char_bitmap(ch);
                for (auto& [bx, by] : bitmap) {
                    float px = lx + bx * scale, py = ly + by * scale;
                    float q[] = {
                        px, py, 0,0, cr,cg,cb,
                        px+scale, py, 1,0, cr,cg,cb,
                        px+scale, py+scale, 1,1, cr,cg,cb,
                        px, py, 0,0, cr,cg,cb,
                        px+scale, py+scale, 1,1, cr,cg,cb,
                        px, py+scale, 0,1, cr,cg,cb,
                    };
                    verts.insert(verts.end(), q, q+42);
                }
                lx += char_w + sp;
            }
            if (!verts.empty()) {
                glBindVertexArray(r.vao_txt);
                glBindBuffer(GL_ARRAY_BUFFER, r.vbo_txt);
                glBufferData(GL_ARRAY_BUFFER, verts.size()*sizeof(float), verts.data(), GL_DYNAMIC_DRAW);
                glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,7*sizeof(float),(void*)0); glEnableVertexAttribArray(0);
                glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,7*sizeof(float),(void*)(2*sizeof(float))); glEnableVertexAttribArray(1);
                glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,7*sizeof(float),(void*)(4*sizeof(float))); glEnableVertexAttribArray(2);
                glDrawArrays(GL_TRIANGLES, 0, verts.size()/7);
            }
        }
    }
    glEnable(GL_DEPTH_TEST);

    glBindVertexArray(0);
}

// ============================================================================
// rotate_view() / zoom_view() -- CAMERA CONTROL FROM EXTERNAL INPUT
//
// Used when embedding the renderer in ImGui. The viewport panel forwards
// mouse drag and scroll events through these methods.
// ============================================================================

void Renderer::rotate_view(float dx, float dy) {
    auto& r = g_impl;
    r.rot_y += dx;
    r.rot_x += dy;
    r.rot_x = std::max(-1.5f, std::min(1.5f, r.rot_x));
}

void Renderer::zoom_view(float factor) {
    auto& r = g_impl;
    r.zoom *= factor;
    r.zoom = std::max(0.001f, std::min(500.0f, r.zoom));
}

void Renderer::reset_view() {
    auto& r = g_impl;
    r.rot_x = 0.4f;
    r.rot_y = 0.5f;
    r.zoom = 1.0f;
}

bool Renderer::project_to_screen(float wx, float wy, float wz, int vp_w, int vp_h, float* sx, float* sy) {
    auto& r = g_impl;
    if (!r.inited || r.n_pts == 0) return false;

    float cx = 0, cy = 0, cz = 0, max_range = 1.0f;
    if (r.n_pts > 0) {
        float minx = (*r.pts)[0][0], maxx = minx, miny = (*r.pts)[0][1], maxy = miny;
        float minz = 0, maxz = 0;
        if (r.pts->cols() > 2) { minz = (*r.pts)[0][2]; maxz = minz; }
        for (size_t i = 0; i < r.n_pts; i++) {
            if ((*r.pts)[i][0] < minx) minx = (*r.pts)[i][0]; if ((*r.pts)[i][0] > maxx) maxx = (*r.pts)[i][0];
            if ((*r.pts)[i][1] < miny) miny = (*r.pts)[i][1]; if ((*r.pts)[i][1] > maxy) maxy = (*r.pts)[i][1];
            if (r.pts->cols() > 2) { if ((*r.pts)[i][2] < minz) minz = (*r.pts)[i][2]; if ((*r.pts)[i][2] > maxz) maxz = (*r.pts)[i][2]; }
        }
        cx = (minx + maxx) / 2; cy = (miny + maxy) / 2; cz = (minz + maxz) / 2;
        max_range = std::max({maxx - minx, maxy - miny, maxz - minz}) / 2.0f;
        if (max_range < 0.1f) max_range = 1.0f;
    }

    float dist = max_range * 3.0f / r.zoom;
    float ex = cx + dist * std::sin(r.rot_y) * std::cos(r.rot_x);
    float ey = cy + dist * std::sin(r.rot_x);
    float ez = cz + dist * std::cos(r.rot_y) * std::cos(r.rot_x);

    float aspect = (float)vp_w / vp_h;
    Mat4 proj = Mat4::perspective(45.0f * 3.14159f / 180.0f, aspect, dist * 0.01f, dist * 20.0f);
    Mat4 view = Mat4::lookAt(ex, ey, ez, cx, cy, cz, 0, 1, 0);
    Mat4 mvp = proj * view;

    // Transform world point through MVP
    float clip_x = mvp.m[0]*wx + mvp.m[4]*wy + mvp.m[8]*wz + mvp.m[12];
    float clip_y = mvp.m[1]*wx + mvp.m[5]*wy + mvp.m[9]*wz + mvp.m[13];
    float clip_z = mvp.m[2]*wx + mvp.m[6]*wy + mvp.m[10]*wz + mvp.m[14];
    float clip_w = mvp.m[3]*wx + mvp.m[7]*wy + mvp.m[11]*wz + mvp.m[15];

    if (clip_w <= 0 || clip_z < -clip_w || clip_z > clip_w) return false;

    float ndc_x = clip_x / clip_w;
    float ndc_y = clip_y / clip_w;
    *sx = (ndc_x + 1.0f) * 0.5f * (float)vp_w;
    *sy = (1.0f - ndc_y) * 0.5f * (float)vp_h;
    return true;
}

// ============================================================================
// STATIC CALLBACK WRAPPERS (forward to global C-style callbacks)
//
// These bridge the C++ Renderer member function interface to the global
// C-style callbacks that GLFW requires. Since GLFW doesn't know about C++
// member functions, we use the global g_impl state and these static wrappers.
// ============================================================================

void Renderer::key_callback(GLFWwindow* w, int k, int s, int a, int m) {
    cb_key(w, k, s, a, m);
}

void Renderer::mouse_button_callback(GLFWwindow* w, int b, int a, int m) {
    cb_mbtn(w, b, a, m);
}

void Renderer::cursor_pos_callback(GLFWwindow* w, double x, double y) {
    cb_mpos(w, x, y);
}

void Renderer::scroll_callback(GLFWwindow* w, double x, double y) {
    cb_scroll(w, x, y);
}

} // namespace clustering
