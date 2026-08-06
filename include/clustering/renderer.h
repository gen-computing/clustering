#pragma once

#include "matrix.h"
#include <string>

struct GLFWwindow;  // Forward declaration: we only use pointer, so we don't
                    // need the full GLFW header here. This speeds up compilation.

namespace clustering {

// ============================================================================
// RendererConfig -- Settings for the OpenGL 3D visualization window.
// ============================================================================
struct RendererConfig {
    int width = 1280;              // Window width in pixels
    int height = 720;              // Window height in pixels
    std::string title = "Clustering Engine"; // Window title bar text

    float point_size = 8.0f;      // How big each data point is drawn (in pixels).
                                   // Larger = easier to see, but more overlap.

    bool show_centroids = true;   // Draw cluster centers as larger white dots?

    bool show_axes = true;        // Draw the 3D coordinate axes (X=red, Y=green, Z=blue)?
};

// ============================================================================
// Renderer -- OpenGL 3.3 3D visualization of clustering results.
//
// WHAT IT DOES:
//   Opens a window and renders your data points as colored 3D dots.
//   Each cluster gets a different color. Centroids shown as larger white dots.
//   Text overlay shows cluster count, inertia, iterations, FPS.
//
// CONTROLS (when the window is open):
//   - Left mouse drag: Rotate the 3D view (orbital camera).
//   - Mouse scroll: Zoom in/out.
//   - R key: Reset view to default angle and zoom.
//   - ESC key: Close the window.
//
// USAGE:
//   1. Create a Renderer with a RendererConfig.
//   2. Call init() to open the window and set up OpenGL.
//   3. Call set_data() with your points, labels, and centroids.
//   4. Call set_metrics() to show inertia and iteration count.
//   5. Call run() -- this BLOCKS until the user closes the window.
//      (run() is the main render loop: draw frame, swap buffers, poll events, repeat).
//
// LIMITATIONS:
//   - Uses a GLOBAL renderer state (not thread-safe, only one window at a time).
//   - Requires a display/GPU with OpenGL 3.3+ support.
//   - No save/export functionality beyond what's rendered.
//   - Text rendering is custom bitmap font (limited character set).
// ============================================================================
class Renderer {
public:
    Renderer();
    explicit Renderer(const RendererConfig& config);
    ~Renderer();

    // Prevent copying. A renderer owns a GLFW window, which can't be copied.
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // init: Create the GLFW window and OpenGL context.
    // Sets up shaders, vertex buffers, viewport, blending, callbacks.
    // Returns: true on success, false if window/OpenGL creation failed.
    // Must be called before set_data() or run().
    bool init();

    // init_headless: Initialize only OpenGL resources (shaders, VAOs, VBOs)
    // without creating a GLFW window. Use this when embedding the renderer
    // inside an existing OpenGL context (e.g., ImGui window).
    // The caller must already have a current OpenGL context.
    // Returns true on success.
    bool init_headless();

    // shutdown: Destroy the window and terminate GLFW.
    // Called automatically by the destructor.
    void shutdown();

    // set_data: Give the renderer data to visualize.
    // Parameters:
    //   points    -- data to display (each row = one 3D point, first 3 columns used).
    //                If data has only 2 columns, Z is set to 0.
    //   labels    -- which cluster each point belongs to (determines color).
    //   centroids -- cluster centers (shown as larger white dots).
    // Small point matrices are copied. Matrices larger than 128 MiB are
    // referenced (not copied) to avoid per-frame copies of disk-backed data;
    // the caller must keep such points alive while the renderer shows them.
    void set_data(const Matrix& points, const Vector& labels, const Matrix& centroids);

    // set_metrics: Set the numbers shown in the text overlay.
    void set_metrics(float inertia, int iterations);

    // run: Enter the main render loop (blocks until window is closed).
    // Draws one frame, swaps buffers, processes input events, repeats.
    // Returns when the user closes the window or presses ESC.
    void run();

    // render_to_fbo: Render one frame to a framebuffer object instead of the
    // default window framebuffer. Used for embedding the 3D viewport inside
    // an ImGui window. The FBO must have a color attachment (texture) bound.
    // After calling, the FBO's color attachment contains the rendered frame.
    // Parameters: fbo (OpenGL FBO handle), width, height in pixels.
    void render_to_fbo(unsigned int fbo, int width, int height);

    // rotate_view: Rotate camera by delta angles. For forwarding ImGui mouse drag.
    void rotate_view(float dx, float dy);

    // zoom_view: Adjust zoom by a multiplicative factor. For forwarding scroll.
    void zoom_view(float factor);

    // reset_view: Reset camera to default rotation and zoom.
    void reset_view();

    // project_to_screen: Project 3D world point to 2D screen coordinates
    // using current camera state. Returns false if point is behind camera.
    // sx, sy: output screen pixel coordinates (relative to viewport origin).
    bool project_to_screen(float wx, float wy, float wz, int vp_w, int vp_h, float* sx, float* sy);

    // ----- STATIC CALLBACKS (called by GLFW when input events occur) -----
    // These are static because GLFW doesn't know about C++ member functions.
    // They cast the window's user pointer back to Renderer* to access state.

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

private:
    // render_frame: Draw ONE frame (called repeatedly in the run() loop).
    //   1. Compute FPS counter.
    //   2. Clear the screen (dark background).
    //   3. Compute camera position from rotation angles and zoom.
    //   4. Draw 3D coordinate axes (red X, green Y, blue Z).
    //   5. Draw data points as colored circles.
    //   6. Draw centroids as larger white circles.
    //   7. Draw text overlay (engine name, point count, clusters, inertia, FPS).
    void render_frame();

    // render_impl: Shared scene drawing (axes, points, centroids, text overlay)
    // used by both render_frame() and render_to_fbo(). Viewport/clear/framebuffer
    // handling stays in the callers; this draws into whatever is bound.
    void render_impl(int width, int height, bool with_fps);

    RendererConfig config_;    // Window and display settings

    GLFWwindow* window_;       // The GLFW window handle (nullptr if not initialized)

    // ----- OPENGL OBJECTS (GPU-side handles) -----

    unsigned int shader_program_;   // Combined vertex+fragment shader for points
    unsigned int vao_points_;       // Vertex Array Object -- stores buffer config for points
    unsigned int vbo_points_;       // Vertex Buffer Object -- stores point position/color data
    unsigned int vao_centroids_;    // VAO for centroid dots
    unsigned int vbo_centroids_;    // VBO for centroid data
    unsigned int vao_axes_;         // VAO for axis lines
    unsigned int vbo_axes_;         // VBO for axis line data

    // ----- CAMERA STATE -----

    float rotation_x_;              // Vertical rotation angle (radians)
    float rotation_y_;              // Horizontal rotation angle (radians)
    float zoom_;                    // Zoom factor (1.0 = default distance)
    double last_mouse_x_;           // Previous mouse X for drag delta calculation
    double last_mouse_y_;           // Previous mouse Y for drag delta calculation
    bool mouse_pressed_;            // Is left mouse button currently held?

    // ----- DATA TO DISPLAY -----

    Matrix points_;                 // The data points
    Vector labels_;                 // Cluster assignments (which color per point)
    Matrix centroids_;              // Cluster centers
    size_t num_points_;             // Cached count for faster access
    size_t num_clusters_;           // Cached count for faster access
    bool initialized_;              // Has init() been called successfully?
};

} // namespace clustering
