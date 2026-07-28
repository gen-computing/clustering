// ============================================================================
// Shader source code + OpenGL compilation utilities.
//
// Shaders are small GPU programs written in GLSL (OpenGL Shading Language).
// They are compiled at runtime (not build time) by the GPU driver.
//
// SHADER TYPES USED:
//   Vertex shader: runs once per vertex. Transforms 3D position to screen.
//   Fragment shader: runs once per pixel. Outputs color (with alpha blending).
//
// RENDERING PIPELINE (what happens when we draw):
//   1. CPU uploads vertex data (positions, colors) to GPU via VBO.
//   2. GPU vertex shader processes each vertex (MVP transform, pass color).
//   3. GPU rasterizer converts triangles/points to fragments (pixels).
//   4. GPU fragment shader processes each fragment (color, alpha).
//   5. GPU writes result to the framebuffer (FBO or default framebuffer).
//
// SOFT CIRCLES: Points are rendered as squares by default. Our fragment
// shader uses gl_PointCoord (pixel position within the point [0,1]×[0,1])
// and smoothstep() to discard corners, creating a smooth circle with
// anti-aliased edges.
// ============================================================================

#pragma once

// Point shaders: draw data points and centroids as colored circles.
//   point_vs: transforms 3D position by MVP matrix, sets point size.
//   point_fs: draws soft circle using gl_PointCoord + smoothstep.
extern const char* point_vs;
extern const char* point_fs;

// Text shaders: draw 2D text overlay using orthographic projection.
//   text_vs: transforms 2D position by ortho projection matrix.
//   text_fs: draws soft disc for each character dot.
extern const char* text_vs;
extern const char* text_fs;

// Line shaders: draw 3D coordinate axes (red=X, green=Y, blue=Z).
//   line_vs: transforms 3D position by MVP matrix.
//   line_fs: outputs solid color (no alpha blending for lines).
extern const char* line_vs;
extern const char* line_fs;

// compile_shader(): Compile a single shader from GLSL source string.
//   src: GLSL source code (null-terminated C string).
//   type: GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
// Returns: OpenGL shader object handle (unsigned int). Caller must manage.
// Note: Does not check compile errors for brevity. Use glGetShaderiv + glGetShaderInfoLog in debug builds.
unsigned int compile_shader(const char* src, unsigned int type);

// create_program(): Create a linked GPU program from vertex + fragment shaders.
//   vs: vertex shader source code.
//   fs: fragment shader source code.
// Returns: OpenGL program object handle. Caller must glDeleteProgram when done.
// The input shaders are compiled, attached, linked, then deleted (they're embedded in the program).
unsigned int create_program(const char* vs, const char* fs);
