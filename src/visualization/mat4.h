// ============================================================================
// Mat4 -- 4x4 matrix math for 3D camera and projection transformations.
//
// Used by the OpenGL renderer to compute the Model-View-Projection (MVP)
// matrix that transforms 3D world coordinates to 2D screen coordinates.
//
// STORAGE: Column-major (OpenGL convention).
//   m[c*4 + r] = element at row r, column c.
//   This is the layout expected by glUniformMatrix4fv().
//
// WHY A CUSTOM MAT4 INSTEAD OF GLM?
//   Zero-dependency. We only need 5 operations. GLM is a 700+ header library.
//   This 80-line file is self-contained and readable.
//
// COORDINATE SYSTEMS IN THE RENDERING PIPELINE:
//   1. Model space  (data points in their original coordinates)
//   2. World space  (same as model here -- no separate model transform)
//   3. View space   (camera-relative coordinates, via lookAt matrix)
//   4. Clip space   (after perspective projection, before divide-by-w)
//   5. NDC space    (after divide-by-w, [-1,1] cube)
//   6. Screen space (after viewport transform, pixel coordinates)
//
// The MVP matrix = projection * view * model.
// We apply it in the vertex shader: gl_Position = uMVP * vec4(aPos, 1.0);
// ============================================================================

#pragma once
#include <cmath>

struct Mat4 {
    float m[16];  // 4x4 matrix stored column-major

    // identity(): Returns the identity matrix.
    // Diagonal = 1, everything else = 0. M * I = I * M = M.
    static Mat4 identity();

    // perspective(): Creates a perspective projection matrix.
    // Makes distant objects appear smaller (foreshortening).
    // Parameters:
    //   fov:    vertical field of view in radians. 45° = 0.785 rad is natural.
    //   aspect: width/height ratio. Prevents squishing on non-square windows.
    //   near:   near clipping plane. Objects closer than this are invisible.
    //   far:    far clipping plane. Objects further than this are invisible.
    static Mat4 perspective(float fov, float aspect, float near, float far);

    // ortho(): Creates an orthographic (parallel) projection matrix.
    // No perspective foreshortening. Used for 2D overlays (text, UI).
    // Maps the box [l,r]×[b,t]×[n,f] to the [-1,1]³ cube.
    static Mat4 ortho(float l, float r, float b, float t, float n, float f);

    // lookAt(): Creates a view matrix for an orbital camera.
    // Parameters:
    //   ex,ey,ez: camera position in world space (eye point).
    //   cx,cy,cz: point the camera is looking at (center/target).
    //   ux,uy,uz: up direction, usually (0,1,0) for Y-up.
    // Builds orthonormal basis: forward (eye→center), right (forward×up), new up (right×forward).
    static Mat4 lookAt(float ex, float ey, float ez, float cx, float cy, float cz, float ux, float uy, float uz);

    // operator*: Matrix multiplication (column-major × column-major).
    // Used to combine transformations: MVP = projection * view.
    Mat4 operator*(const Mat4& b) const;
};
