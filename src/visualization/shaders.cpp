// ============================================================================
// shaders.cpp -- GLSL shader source code and OpenGL compilation utilities.
//
// All 6 shaders (point vertex/fragment, text vertex/fragment, line vertex/fragment)
// are stored as C-string literals using raw string syntax R"(...)".
// They are compiled at runtime and linked into GPU programs.
//
// SHADER DETAILS:
//
// point_vs (vertex): Takes 3D position (vec3 aPos) and color (vec3 aCol).
//   Transforms position by uMVP matrix. Passes color to fragment shader.
//   Sets gl_PointSize from uSize uniform. Output: gl_Position, gl_PointSize, vCol.
//
// point_fs (fragment): Uses gl_PointCoord (0,0=bottom-left to 1,1=top-right of the point).
//   Computes distance from center. Uses smoothstep(0.38, 0.5, dist) for
//   anti-aliased circle. Center is fully opaque (a=1), edges fade to transparent.
//
// line_vs/fss: Simpler shaders for axis lines. No alpha blending, solid color.
//
// text_vs/fss: 2D overlay shaders. Use orthographic projection (uProj).
//   Accept 2D positions + UV coordinates. Fragment creates soft disc per character dot.
//
// compile_shader(): Creates an OpenGL shader object, loads source, compiles.
//   Returns shader handle. Errors silently (use glGetShaderiv in debug).
//
// create_program(): Creates a GPU program from vertex+fragment shaders.
//   Attaches both, links, then deletes the individual shaders (they're
//   embedded in the program and no longer needed as standalone objects).
// ============================================================================

#include "shaders.h"
#include <GL/glew.h>

// ---- POINT SHADERS ----
// Used for drawing data points and centroids as colored circles.
// Vertex shader: position * MVP → clip space, pass color through.
// Fragment shader: smooth circle using distance from center.

const char* point_vs = R"(
#version 330 core
layout(location=0) in vec3 aPos;     // 3D world-space position from VBO attribute 0
layout(location=1) in vec3 aCol;     // RGB color from VBO attribute 1
uniform mat4 uMVP;                    // Model-View-Projection matrix (set per draw call)
uniform float uSize;                  // Point size in screen pixels
out vec3 vCol;                        // Output: pass color to fragment shader
void main(){
    gl_Position = uMVP * vec4(aPos, 1.0);  // World → clip space
    gl_PointSize = uSize;                   // Set point pixel size
    vCol = aCol;                            // Forward color
}
)";

const char* point_fs = R"(
#version 330 core
in vec3 vCol;                          // Color from vertex shader
out vec4 FragColor;                    // RGBA output to framebuffer
void main(){
    vec2 c = gl_PointCoord - vec2(0.5);  // Center coordinates: [-0.5, 0.5] range
    float r = length(c);                  // Distance from center (0=center, 0.5=corner)
    float a = 1.0 - smoothstep(0.38, 0.5, r);  // Anti-aliased circle mask
    FragColor = vec4(vCol, a);                 // Output with alpha
}
)";

// ---- TEXT SHADERS ----
// Used for the 2D text overlay on top of the 3D scene.
// Renders each character as a quad with a soft circular dot mask.

const char* text_vs = R"(
#version 330 core
layout(location=0) in vec2 aPos;      // 2D screen position (attribute 0)
layout(location=1) in vec2 aUV;       // UV for disc masking (attribute 1)
layout(location=2) in vec3 aCol;      // RGB color (attribute 2)
uniform mat4 uProj;                    // Orthographic projection matrix
out vec2 vUV;
out vec3 vCol;
void main(){
    gl_Position = uProj * vec4(aPos, 0.0, 1.0);  // Screen → clip space
    vUV = aUV;
    vCol = aCol;
}
)";

const char* text_fs = R"(
#version 330 core
in vec2 vUV;
in vec3 vCol;
out vec4 FragColor;
void main(){
    float d = length(vUV - vec2(0.5));           // Distance from center of quad
    float a = 1.0 - smoothstep(0.35, 0.5, d);   // Soft disc mask
    FragColor = vec4(vCol, a * 0.8);             // Slightly transparent
}
)";

// ---- LINE SHADERS ----
// Used for the 3D coordinate axes (X=red, Y=green, Z=blue).
// Simple solid-color rendering, no alpha or special effects.

const char* line_vs = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aCol;
uniform mat4 uMVP;                    // Model-View-Projection matrix
out vec3 vCol;
void main(){
    gl_Position = uMVP * vec4(aPos, 1.0);
    vCol = aCol;
}
)";

const char* line_fs = R"(
#version 330 core
in vec3 vCol;
out vec4 FragColor;
void main(){
    FragColor = vec4(vCol, 1.0);      // Fully opaque solid color
}
)";

// ============================================================================
// compile_shader() -- Compile a single shader from GLSL source.
//
// OpenGL flow:
//   glCreateShader(type)      → allocate shader object on GPU
//   glShaderSource(s, 1, &src)→ upload GLSL code to shader
//   glCompileShader(s)        → compile on GPU driver
//
// The shader is now compiled and can be attached to a program.
// Compilation errors are stored in the shader info log (not checked here).
// ============================================================================
unsigned int compile_shader(const char* src, unsigned int type) {
    unsigned int s = glCreateShader(type);          // Allocate GPU shader object
    glShaderSource(s, 1, &src, nullptr);             // Upload source code
    glCompileShader(s);                               // Compile on GPU
    return s;
}

// ============================================================================
// create_program() -- Link vertex + fragment shaders into a GPU program.
//
// OpenGL flow:
//   glCreateProgram() → allocate program object
//   glAttachShader(p, v) → attach compiled vertex shader
//   glAttachShader(p, f) → attach compiled fragment shader
//   glLinkProgram(p) → link into executable GPU binary
//   glDeleteShader(v/f) → free shader objects (now embedded in program)
//
// After linking, the program is ready to use with glUseProgram().
// The individual shaders are deleted because the program holds copies.
// ============================================================================
unsigned int create_program(const char* vs, const char* fs) {
    unsigned int v = compile_shader(vs, GL_VERTEX_SHADER);
    unsigned int f = compile_shader(fs, GL_FRAGMENT_SHADER);
    unsigned int p = glCreateProgram();    // Create program object
    glAttachShader(p, v);                   // Attach vertex shader
    glAttachShader(p, f);                   // Attach fragment shader
    glLinkProgram(p);                       // Link into GPU executable
    glDeleteShader(v);                      // Clean up (no longer needed)
    glDeleteShader(f);
    return p;
}
