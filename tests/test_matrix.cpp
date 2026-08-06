// ============================================================================
// test_matrix.cpp -- Matrix and Vector unit tests.
// Tests: construction, resize, fill, element access, row-major layout.
// ============================================================================

#include <gtest/gtest.h>
#include "clustering/matrix.h"
#include <cstdio>

using namespace clustering;

// Force disk-backed mode by setting a tiny RAM cap, then restore.
struct DiskCapGuard {
    size_t old;
    DiskCapGuard(size_t bytes) : old(matrix_ram_cap()) { set_matrix_ram_cap(bytes); }
    ~DiskCapGuard() { set_matrix_ram_cap(old); }
};

TEST(Matrix, DiskBackedSpill) {
    DiskCapGuard g(256 * 1024);  // 256 KiB cap
    Matrix m(2000, 100);         // 800 KiB -- must spill to disk
    EXPECT_TRUE(m.is_disk_backed());
    EXPECT_EQ(m.rows(), 2000u);
    EXPECT_EQ(m.cols(), 100u);
    EXPECT_EQ(m.disk_bytes_used(), 2000u * 100u * sizeof(float));
    EXPECT_LT(m.ram_bytes_used(), 256u * 1024u);
}

TEST(Matrix, DiskBackedReadWrite) {
    DiskCapGuard g(256 * 1024);
    Matrix m(2000, 64);  // 512 KiB
    ASSERT_TRUE(m.is_disk_backed());
    for (size_t i = 0; i < m.rows(); i++)
        for (size_t j = 0; j < m.cols(); j++)
            m[i][j] = float(i * 1000 + j);
    for (size_t i = 0; i < m.rows(); i++)
        for (size_t j = 0; j < m.cols(); j++)
            EXPECT_FLOAT_EQ(m[i][j], float(i * 1000 + j));
}

TEST(Matrix, DiskBackedDataThrows) {
    DiskCapGuard g(256 * 1024);
    Matrix m(2000, 64);
    ASSERT_TRUE(m.is_disk_backed());
    EXPECT_THROW(m.data(), std::runtime_error);
    const Matrix& cm = m;
    EXPECT_THROW(cm.data(), std::runtime_error);
}

TEST(Matrix, DiskBackedCopyAndMove) {
    DiskCapGuard g(256 * 1024);
    Matrix m(1500, 80);
    ASSERT_TRUE(m.is_disk_backed());
    for (size_t i = 0; i < m.rows(); i++) m[i][0] = float(i);

    Matrix copy = m;  // deep copy with own temp file
    ASSERT_TRUE(copy.is_disk_backed());
    EXPECT_NE(m.disk_bytes_used(), 0u);
    for (size_t i = 0; i < m.rows(); i++)
        EXPECT_FLOAT_EQ(copy[i][0], float(i));

    Matrix moved = std::move(copy);  // move transfers storage
    ASSERT_TRUE(moved.is_disk_backed());
    for (size_t i = 0; i < m.rows(); i++)
        EXPECT_FLOAT_EQ(moved[i][0], float(i));

    m.fill(42.0f);  // dirty write-back path
    for (size_t i = 0; i < m.rows(); i++)
        EXPECT_FLOAT_EQ(m[i][0], 42.0f);
}

TEST(Matrix, DiskBackedResize) {
    DiskCapGuard g(64 * 1024);  // 64 KiB cap
    Matrix m(1000, 32);         // 128 KiB
    ASSERT_TRUE(m.is_disk_backed());
    m[500][7] = 3.5f;
    m.resize(1200, 32);  // grow
    EXPECT_EQ(m.rows(), 1200u);
    EXPECT_FLOAT_EQ(m[500][7], 3.5f);
    m.resize(800, 32);   // shrink
    EXPECT_EQ(m.rows(), 800u);
    EXPECT_FLOAT_EQ(m[500][7], 3.5f);
    EXPECT_FLOAT_EQ(m[0][0], 0.0f);  // new/unwritten area zeroed
}

TEST(Matrix, DiskBackedPromoteDemote) {
    DiskCapGuard g(256 * 1024);
    Matrix m(10, 4);  // small, inline
    EXPECT_FALSE(m.is_disk_backed());
    m[3][1] = 9.0f;
    m.resize(100000, 4);  // 1.6 MiB -> promote
    EXPECT_TRUE(m.is_disk_backed());
    EXPECT_FLOAT_EQ(m[3][1], 9.0f);
    m.resize(10, 4);  // demote back to RAM
    EXPECT_FALSE(m.is_disk_backed());
    EXPECT_FLOAT_EQ(m[3][1], 9.0f);
}

TEST(Matrix, DiskBackedColumnChangeRebuild) {
    DiskCapGuard g(64 * 1024);  // 64 KiB cap
    Matrix m(500, 40);          // 80 KiB
    ASSERT_TRUE(m.is_disk_backed());
    for (size_t i = 0; i < m.rows(); i++) m[i][39] = float(i);
    m.resize(500, 30);  // column drop forces rebuild
    EXPECT_EQ(m.cols(), 30u);
    EXPECT_FLOAT_EQ(m[0][0], 0.0f);
}

TEST(Matrix, DefaultConstructor) {
    Matrix m;
    EXPECT_EQ(m.rows(), 0u);
    EXPECT_EQ(m.cols(), 0u);
    EXPECT_EQ(m.size(), 0u);
}

TEST(Matrix, SizeConstructor) {
    Matrix m(10, 5);
    EXPECT_EQ(m.rows(), 10u);
    EXPECT_EQ(m.cols(), 5u);
    EXPECT_EQ(m.size(), 50u);
    // Should be zero-initialized
    for (size_t i = 0; i < 50; i++)
        EXPECT_FLOAT_EQ(m.data()[i], 0.0f);
}

TEST(Matrix, DataConstructor) {
    float raw[] = {1, 2, 3, 4, 5, 6};
    Matrix m(2, 3, raw);
    EXPECT_EQ(m.rows(), 2u);
    EXPECT_EQ(m.cols(), 3u);
    EXPECT_FLOAT_EQ(m[0][0], 1.0f);
    EXPECT_FLOAT_EQ(m[0][1], 2.0f);
    EXPECT_FLOAT_EQ(m[0][2], 3.0f);
    EXPECT_FLOAT_EQ(m[1][0], 4.0f);
    EXPECT_FLOAT_EQ(m[1][1], 5.0f);
    EXPECT_FLOAT_EQ(m[1][2], 6.0f);
}

TEST(Matrix, ElementAccess) {
    Matrix m(3, 4);
    m[1][2] = 7.0f;
    EXPECT_FLOAT_EQ(m[1][2], 7.0f);
    m[2][3] = 9.0f;
    EXPECT_FLOAT_EQ(m[2][3], 9.0f);
    // Other elements should still be 0
    EXPECT_FLOAT_EQ(m[0][0], 0.0f);
    EXPECT_FLOAT_EQ(m[0][3], 0.0f);
}

TEST(Matrix, Resize) {
    Matrix m(2, 2);
    m[0][0] = 1.0f; m[0][1] = 2.0f;
    m[1][0] = 3.0f; m[1][1] = 4.0f;

    m.resize(3, 3); // Grow -- resize does NOT preserve data (fills with 0)
    EXPECT_EQ(m.rows(), 3u);
    EXPECT_EQ(m.cols(), 3u);
    EXPECT_FLOAT_EQ(m[2][2], 0.0f); // New elements zero

    m[0][0] = 99.0f; // Set a value after resize
    m.resize(1, 1);  // Shrink
    EXPECT_EQ(m.rows(), 1u);
    EXPECT_EQ(m.cols(), 1u);
    EXPECT_FLOAT_EQ(m[0][0], 99.0f); // First element preserved (vector::resize shrink keeps front elements)
}

TEST(Matrix, Fill) {
    Matrix m(5, 3);
    m.fill(42.0f);
    for (size_t i = 0; i < 5; i++)
        for (size_t j = 0; j < 3; j++)
            EXPECT_FLOAT_EQ(m[i][j], 42.0f);
}

TEST(Matrix, ConstAccess) {
    Matrix m(2, 2);
    m[0][0] = 5.0f;
    m[1][1] = 7.0f;
    const Matrix& cm = m;
    EXPECT_FLOAT_EQ(cm[0][0], 5.0f);
    EXPECT_FLOAT_EQ(cm[1][1], 7.0f);
    EXPECT_FLOAT_EQ(cm.data()[3], 7.0f); // Raw access
}

TEST(Matrix, RowMajorLayout) {
    // Verify row-major: m[0][0], m[0][1], m[1][0], m[1][1]
    float raw[] = {10, 20, 30, 40};
    Matrix m(2, 2, raw);
    EXPECT_FLOAT_EQ(m.data()[0], 10.0f); // row 0, col 0
    EXPECT_FLOAT_EQ(m.data()[1], 20.0f); // row 0, col 1
    EXPECT_FLOAT_EQ(m.data()[2], 30.0f); // row 1, col 0
    EXPECT_FLOAT_EQ(m.data()[3], 40.0f); // row 1, col 1
}

TEST(Vector, DefaultConstructor) {
    Vector v;
    EXPECT_EQ(v.size(), 0u);
}

TEST(Vector, SizeConstructor) {
    Vector v(100);
    EXPECT_EQ(v.size(), 100u);
    for (size_t i = 0; i < 100; i++)
        EXPECT_FLOAT_EQ(v[i], 0.0f);
}

TEST(Vector, DataConstructor) {
    float raw[] = {1, 2, 3, 4, 5};
    Vector v(5, raw);
    EXPECT_EQ(v.size(), 5u);
    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[4], 5.0f);
}

TEST(Vector, ElementAccess) {
    Vector v(3);
    v[1] = 99.0f;
    EXPECT_FLOAT_EQ(v[1], 99.0f);
}

TEST(Vector, Resize) {
    Vector v(5);
    v[0] = 1.0f; v[4] = 5.0f;
    v.resize(10);
    EXPECT_EQ(v.size(), 10u);
    EXPECT_FLOAT_EQ(v[9], 0.0f);
}

TEST(Vector, Fill) {
    Vector v(10);
    v.fill(-1.0f);
    for (size_t i = 0; i < 10; i++)
        EXPECT_FLOAT_EQ(v[i], -1.0f);
}

TEST(Vector, ConstAccess) {
    Vector v(3);
    v[0] = 10.0f; v[2] = 30.0f;
    const Vector& cv = v;
    EXPECT_FLOAT_EQ(cv[0], 10.0f);
    EXPECT_FLOAT_EQ(cv[1], 0.0f);
    EXPECT_FLOAT_EQ(cv[2], 30.0f);
    EXPECT_FLOAT_EQ(cv.data()[0], 10.0f);
}
