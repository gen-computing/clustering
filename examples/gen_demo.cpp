// gen_demo.cpp — Generate demo GIF frames
#include "clustering/clustering.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <random>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <vector>

using namespace clustering;

static void save_ppm(const char* path, const unsigned char* rgb, int w, int h) {
    FILE* f = fopen(path, "wb");
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int y = h - 1; y >= 0; y--)
        fwrite(rgb + y * w * 3, 1, w * 3, f);
    fclose(f);
}

int main() {
    mkdir("docs/screenshots", 0755);
    const int W = 960, H = 540;
    std::vector<unsigned char> pixels(W * H * 3);

    // Init GLFW + OpenGL context
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(W, H, "demo", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glewInit();

    // Create FBO
    unsigned int fbo, tex, rbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, W, H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Init renderer headless
    RendererConfig cfg;
    cfg.width = W; cfg.height = H;
    cfg.point_size = 5.0f;
    cfg.show_centroids = true;
    cfg.show_axes = true;
    Renderer renderer(cfg);
    renderer.init_headless();

    int frame = 0;

    auto capture = [&](const char* label) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        renderer.render_to_fbo(fbo, W, H);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        char path[128];
        snprintf(path, sizeof(path), "docs/screenshots/frame_%03d.ppm", frame++);
        save_ppm(path, pixels.data(), W, H);
        fprintf(stderr, "  %s -> %s\n", label, path);
    };

    // Scene 1: unclustered gray (10 frames)
    fprintf(stderr, "Scene 1: unclustered\n");
    {
        int n = 400;
        Matrix X(n, 3);
        std::mt19937 gen(42);
        std::normal_distribution<float> noise(0.0f, 1.0f);
        for (int i = 0; i < n; i++) { X[i][0] = noise(gen)*3; X[i][1] = noise(gen)*3; X[i][2] = noise(gen)*3; }
        Vector neutral(n); neutral.fill(-1.0f);
        Matrix cnt(1, 3);
        renderer.set_data(X, neutral, cnt);
        renderer.set_metrics(0, 0);
        for (int f = 0; f < 10; f++) {
            renderer.rotate_view(0.03f, 0.0f);
            capture("gray");
        }
    }

    // Scene 2: KMeans iterations (15 frames)
    fprintf(stderr, "Scene 2: KMeans\n");
    {
        int n_per = 60, k = 5, n = n_per * k;
        Matrix X(n, 3);
        std::mt19937 gen(42);
        std::normal_distribution<float> noise(0.0f, 0.5f);
        float C[][3] = {{-4,3,0},{4,3,0},{0,-4,0},{-3,-2,2},{3,-2,2}};
        for (int c = 0; c < k; c++)
            for (int i = 0; i < n_per; i++) { int idx = c*n_per+i; X[idx][0]=C[c][0]+noise(gen); X[idx][1]=C[c][1]+noise(gen); X[idx][2]=C[c][2]+noise(gen); }

        KMeansConfig km_cfg; km_cfg.k = k; km_cfg.max_iter = 15; km_cfg.max_threads = 1;
        km_cfg.iter_callback = [&](size_t iter, const Matrix& centroids, const Vector& labels) -> bool {
            renderer.set_data(X, labels, centroids);
            renderer.set_metrics(0, (int)iter);
            renderer.rotate_view(0.02f, 0.01f);
            capture("kmeans");
            return false;
        };
        KMeans km(km_cfg); km.fit(X);
    }

    // Scene 3: final rotating (10 frames)
    fprintf(stderr, "Scene 3: final\n");
    {
        int n_per = 60, k = 5, n = n_per * k;
        Matrix X(n, 3);
        std::mt19937 gen(42);
        std::normal_distribution<float> noise(0.0f, 0.5f);
        float C[][3] = {{-4,3,0},{4,3,0},{0,-4,0},{-3,-2,2},{3,-2,2}};
        for (int c = 0; c < k; c++)
            for (int i = 0; i < n_per; i++) { int idx = c*n_per+i; X[idx][0]=C[c][0]+noise(gen); X[idx][1]=C[c][1]+noise(gen); X[idx][2]=C[c][2]+noise(gen); }
        KMeans km(k); km.fit(X);
        renderer.set_data(X, km.labels(), km.centroids());
        renderer.set_metrics(km.inertia(), km.n_iter());
        for (int f = 0; f < 10; f++) {
            renderer.rotate_view(0.03f, 0.015f);
            capture("final");
        }
    }

    fprintf(stderr, "Generated %d frames\n", frame);
    fprintf(stderr, "ffmpeg -framerate 6 -i docs/screenshots/frame_%%03d.ppm -vf \"scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer\" -loop 0 docs/screenshots/demo.gif\n");

    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteRenderbuffers(1, &rbo);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
