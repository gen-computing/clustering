// ============================================================================
// mat4.cpp -- Implementation of 4x4 column-major matrix operations.
// See mat4.h for the detailed API documentation and coordinate system explanation.
// ============================================================================

#include "mat4.h"
#include <cmath>

// ============================================================================
// identity() -- Returns the identity matrix (diagonal = 1, all else = 0).
//
// Layout (column-major, each row shows columns left-to-right):
//   [1 0 0 0]
//   [0 1 0 0]
//   [0 0 1 0]
//   [0 0 0 1]
//
// Stored as: m[0]=1, m[5]=1, m[10]=1, m[15]=1, all others 0.
// Used as the starting point for building transformation matrices.
// ============================================================================
Mat4 Mat4::identity() {
    Mat4 r = {};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

// ============================================================================
// perspective() -- Creates a perspective projection matrix.
//
// MATRIX DERIVATION (simplified):
//   X' = X * 1/(aspect * tan(fov/2))
//   Y' = Y * 1/tan(fov/2)
//   Z' = -(far+near)/(far-near) * Z - 2*far*near/(far-near)
//   W' = -Z  (so after divide-by-W: Z'' = Z'/W' gives normalized depth)
//
// The -1 at position [11] is the "perspective divide trigger":
// After the vertex shader outputs gl_Position, the GPU divides XYZ by W.
// Since W = -Z (in camera space), this makes distant objects smaller.
//
// tan(fov/2): relates the vertical field of view to the near-plane height.
// For fov=45° (0.785 rad): tan(0.393) ≈ 0.414.
// ============================================================================
Mat4 Mat4::perspective(float fov, float aspect, float near, float far) {
    Mat4 r = {};

    // Half the vertical FOV tangent.
    float t = std::tan(fov / 2.0f);

    // Scale X by 1/(aspect * t): wider screens have smaller X scale (prevents stretching).
    r.m[0] = 1.0f / (aspect * t);

    // Scale Y by 1/t: based purely on vertical FOV.
    r.m[5] = 1.0f / t;

    // Z mapping: remap [near, far] to [-1, 1] (OpenGL NDC).
    // Negative because OpenGL camera looks down the -Z axis.
    r.m[10] = -(far + near) / (far - near);

    // Perspective divide: W = -Z_camera. After GPU division, Z_ndc = Z_clip / W.
    r.m[11] = -1.0f;

    // Z offset for the mapping (the constant term in Z_clip = a*Z_camera + b).
    r.m[14] = -(2.0f * far * near) / (far - near);

    return r;
}

// ============================================================================
// ortho() -- Creates an orthographic (parallel) projection matrix.
//
// Unlike perspective, objects don't get smaller with distance.
// This is used for 2D UI overlays where screen position = pixel position.
//
// Maps the input box [l,r]×[b,t]×[n,f] to the [-1,1]³ NDC cube:
//   X_ndc = 2*(X_world - l)/(r - l) - 1
//   Y_ndc = 2*(Y_world - b)/(t - b) - 1
//   Z_ndc = -2*(Z_world - n)/(f - n) - 1  (negative for OpenGL depth)
//
// Example: ortho(0, 1920, 0, 1080, -1, 1) maps screen pixels to NDC.
// ============================================================================
Mat4 Mat4::ortho(float l, float r, float b, float t, float n, float f) {
    Mat4 m = identity();

    // Scale factors: map input range width/height to NDC width/height (both = 2).
    m.m[0] = 2.0f / (r - l);
    m.m[5] = 2.0f / (t - b);
    m.m[10] = -2.0f / (f - n);  // Negative for OpenGL depth convention

    // Translation: center the box. After scaling, we need to shift by -center.
    // combined: X_ndc = scale * X - scale * center = scale * X + translate
    m.m[12] = -(r + l) / (r - l);  // = -(r+l)/2 * 2/(r-l)
    m.m[13] = -(t + b) / (t - b);
    m.m[14] = -(f + n) / (f - n);

    return m;
}

// ============================================================================
// lookAt() -- Creates a view matrix for an orbital camera.
//
// Takes camera position and target point, builds an orthonormal basis.
//
// STEP-BY-STEP:
//   1. forward = normalize(center - eye)     : the look direction
//   2. right   = normalize(forward × up)     : camera's local X axis
//   3. new_up  = right × forward              : camera's local Y axis (corrected)
//
// The view matrix V = R * T where:
//   R = rotation (basis vectors as rows of the inverse)
//   T = translation by -eye
//
// Since R is orthonormal (basis vectors are unit and perpendicular),
// the inverse rotation = transpose. So: V = R^T * T(-eye)
//
// This means:
//   V[row0] = (right_x, up_x, -forward_x, -dot(right, eye))
//   V[row1] = (right_y, up_y, -forward_y, -dot(up, eye))
//   V[row2] = (right_z, up_z, -forward_z, -dot(-forward, eye))
//             = (right_z, up_z, -forward_z,  dot(forward, eye))
//
// Forward is negated because OpenGL's camera looks down -Z.
// ============================================================================
Mat4 Mat4::lookAt(float ex, float ey, float ez, float cx, float cy, float cz, float ux, float uy, float uz) {
    // ---- Forward vector (center - eye, normalized) ----
    float fx = cx - ex, fy = cy - ey, fz = cz - ez;
    float fl = std::sqrt(fx*fx + fy*fy + fz*fz);
    fx /= fl; fy /= fl; fz /= fl;

    // ---- Right vector (forward × up, normalized) ----
    // Cross product: s = f × u
    //   s.x = f.y*u.z - f.z*u.y
    //   s.y = f.z*u.x - f.x*u.z
    //   s.z = f.x*u.y - f.y*u.x
    float sx = fy*uz - fz*uy, sy = fz*ux - fx*uz, sz = fx*uy - fy*ux;
    float sl = std::sqrt(sx*sx + sy*sy + sz*sz);
    sx /= sl; sy /= sl; sz /= sl;

    // ---- Corrected up vector (right × forward) ----
    // Ensures all three axes are mutually perpendicular.
    float ux2 = sy*fz - sz*fy, uy2 = sz*fx - sx*fz, uz2 = sx*fy - sy*fx;

    // Build the view matrix (column-major).
    Mat4 r = identity();

    // Rotation part: basis vectors as rows of inverse = as columns of transpose.
    r.m[0]=sx;  r.m[4]=sy;  r.m[8]=sz;    // Right axis in world → column 0
    r.m[1]=ux2; r.m[5]=uy2; r.m[9]=uz2;   // Up axis in world → column 1
    r.m[2]=-fx; r.m[6]=-fy; r.m[10]=-fz;   // Forward axis (negated for OpenGL) → column 2

    // Translation part: -R^T * eye = -dot(basis, eye) for each basis vector.
    r.m[12] = -(sx*ex + sy*ey + sz*ez);     // -dot(right, eye)
    r.m[13] = -(ux2*ex + uy2*ey + uz2*ez);  // -dot(up, eye)
    r.m[14] = (fx*ex + fy*ey + fz*ez);      // dot(forward, eye) (negated twice → positive)

    return r;
}

// ============================================================================
// operator*() -- Column-major × column-major matrix multiplication.
//
// For column-major storage element at (row, col) = m[col*4 + row].
// result[col][row] = sum_k ( left[row][k] * right[k][col] )
//
// Element access pattern (the innermost loop):
//   left[k][row]  = left.m[k*4 + row]    ← column k, row 'row'
//   right[col][k] = right.m[col*4 + k]   ← column 'col', row k
//
// Result stored column-major: result.m[c*4 + row]
//
// Complexity: 4×4×4 = 64 multiply-add operations. Negligible.
// ============================================================================
Mat4 Mat4::operator*(const Mat4& b) const {
    Mat4 r = {};
    // Iterate over output columns (c = 0..3).
    for (int c = 0; c < 4; c++) {
        // Iterate over output rows (row = 0..3).
        for (int row = 0; row < 4; row++) {
            // Dot product of row 'row' of 'this' with column 'c' of 'b'.
            float s = 0;
            for (int k = 0; k < 4; k++)
                s += m[k*4 + row] * b.m[c*4 + k];
            // Store in column-major order.
            r.m[c*4 + row] = s;
        }
    }
    return r;
}
