#pragma once
// #pragma once tells the compiler: "only include this file once per compilation unit".
// It prevents errors when two different .cpp files accidentally include the same header.
// Alternative older way: #ifndef / #define / #endif guards.

#include <vector>   // std::vector -- a dynamic array that grows/shrinks automatically.
                    // Stores elements contiguously in memory (like a C array but safer).
#include <cstddef>  // Defines size_t -- an unsigned integer type for sizes and counts.
                    // size_t can hold the maximum size of any object on the system.
#include <cstdint>  // Defines fixed-width integers (int32_t, uint64_t etc) if needed.

namespace clustering {
// A namespace is like a folder for names. Everything inside `clustering` is kept
// separate from other libraries, so name clashes (e.g., another library's `Matrix`)
// won't cause errors. Use `clustering::Matrix` to refer to this type.

// ============================================================================
// Matrix class -- A 2D grid of floating-point numbers (like a spreadsheet).
//
// A matrix is the fundamental data structure in machine learning.
// Each row = one data point (e.g., one customer, one image, one measurement).
// Each column = one feature (e.g., age, price, color intensity).
//
// Example: A matrix with 100 rows and 5 columns stores 100 data points,
// each described by 5 numbers (features).
//
// Storage: Internally stored as ONE flat array (row-major order).
// Row 0 first, then Row 1, then Row 2, etc.
// So element at row 2, column 3 is found at position: 2 * cols_ + 3
// The function operator[] hides this calculation from the user.
// ============================================================================
class Matrix {
public:
    // ----- CONSTRUCTORS -----
    // A constructor creates a new Matrix object. There are three ways:

    // 1. DEFAULT CONSTRUCTOR: Creates an empty matrix (0 rows, 0 columns).
    //    Use this when you don't know the size yet and will resize later.
    //    Example: Matrix m;         // empty
    //             m.resize(10, 5);  // now 10x5
    Matrix() : rows_(0), cols_(0), data_() {}

    // 2. SIZE CONSTRUCTOR: Creates a matrix with given dimensions, filled with zeros.
    //    Parameters:
    //      rows -- how many rows (number of data points)
    //      cols -- how many columns (number of features per point)
    //    The `:` after the function signature is the "initializer list".
    //    It sets member variables BEFORE the function body runs -- more efficient.
    //    data_(rows * cols, 0.0f) means: create a vector with (rows*cols) elements,
    //    each set to 0.0f.
    //    Example: Matrix m(100, 5);  // 100 points, 5 features each, all zeros
    Matrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

    // 3. DATA CONSTRUCTOR: Creates a matrix by COPYING values from an existing
    //    raw C array (float pointer). This is used to receive data from numpy/Python.
    //    Parameters:
    //      rows, cols -- dimensions
    //      data       -- pointer to raw float array (numpy buffer, C array, etc.)
    //    data_(data, data + rows * cols) copies from [data] up to
    //    [data + (rows*cols)] -- that's the start and end of the source array.
    //    CAUTION: This COPIES the data. The original array and matrix are independent.
    //    Example: float raw[] = {1,2,3,4,5,6};
    //             Matrix m(2, 3, raw);  // copies raw into a 2x3 matrix
    Matrix(size_t rows, size_t cols, float* data)
        : rows_(rows), cols_(cols), data_(data, data + rows * cols) {}

    // ----- SIZE ACCESSORS -----
    // These functions tell you the dimensions. `const` means they don't
    // modify the Matrix -- safe to call on const references.

    // Returns: number of rows (data points)
    size_t rows() const { return rows_; }

    // Returns: number of columns (features per point)
    size_t cols() const { return cols_; }

    // Returns: total number of elements (rows * cols)
    size_t size() const { return rows_ * cols_; }

    // ----- RAW DATA ACCESS -----
    // These return a pointer to the internal flat array.
    // Two versions: one for mutable access (you can change values),
    // one for const access (read-only).

    // MUTABLE: Returns pointer you can write to.
    // Example: Matrix m(2,2); m.data()[0] = 5.0f;  // set first element to 5
    float* data() { return data_.data(); }

    // CONST: Returns read-only pointer. Called automatically when the Matrix
    // is const. Example: const Matrix& cm = m; cm.data()[0]; // reading only
    const float* data() const { return data_.data(); }

    // ----- ELEMENT ACCESS (operator[]) -----
    // operator[] lets you use matrix like a 2D array: m[row][col]
    //
    // How it works:
    //   m[2] returns a pointer to the start of row 2.
    //   Then m[2][3] accesses column 3 of row 2.
    //
    // The calculation: data_.data() + row * cols_
    //   row * cols_ = how many elements to skip to reach the desired row
    //   + data_.data() = starting address of the whole array
    //
    // Example:
    //   Matrix m(3, 4);          // 3 rows, 4 columns
    //   m[1][2] = 7.0f;          // set row 1, column 2 to 7.0
    //   float val = m[0][3];     // read row 0, column 3

    // Mutable version: m[row] gives writable pointer
    float* operator[](size_t row) { return data_.data() + row * cols_; }

    // Const version: m[row] gives read-only pointer (for const Matrix objects)
    const float* operator[](size_t row) const { return data_.data() + row * cols_; }

    // ----- RESIZE -----
    // Changes the matrix dimensions. If the new size is larger, new elements
    // are filled with 0.0f. If smaller, elements beyond the new size are discarded.
    // Parameters: rows, cols -- new dimensions
    void resize(size_t rows, size_t cols) {
        rows_ = rows;
        cols_ = cols;
        data_.resize(rows * cols, 0.0f);  // std::vector::resize -- new elements = 0.0f
    }

    // ----- FILL -----
    // Sets every element in the matrix to the same value.
    // std::fill is a standard library function that writes `value` from
    // the beginning to the end of the vector.
    // Example: m.fill(1.0f);  // every element becomes 1.0
    void fill(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }

private:
    size_t rows_;              // number of rows (data points)
    size_t cols_;              // number of columns (features)
    std::vector<float> data_;  // flat storage: all elements in row-major order
                               // Row 0: [col0][col1]...[colN]
                               // Row 1: [col0][col1]...[colN]
                               // etc.
};

// ============================================================================
// Vector class -- A 1D list of floating-point numbers.
//
// Used for: cluster labels (which cluster each point belongs to),
//           explained variance, mean values, and other 1D data.
//
// Very similar to Matrix but only one dimension.
// Element at position i is accessed with v[i].
// ============================================================================
class Vector {
public:
    // ----- CONSTRUCTORS -----

    // Default: empty vector, size 0
    Vector() : size_(0), data_() {}

    // Create vector of given size, all zeros.
    // Example: Vector v(100);  // 100 elements, all 0.0f
    Vector(size_t size) : size_(size), data_(size, 0.0f) {}

    // Create vector by copying data from raw C array.
    // Example: float raw[] = {1,2,3}; Vector v(3, raw);
    Vector(size_t size, float* data) : size_(size), data_(data, data + size) {}

    // ----- SIZE -----
    size_t size() const { return size_; }

    // ----- RAW DATA ACCESS -----

    // Mutable pointer (can write)
    float* data() { return data_.data(); }

    // Const pointer (read-only)
    const float* data() const { return data_.data(); }

    // ----- ELEMENT ACCESS (operator[]) -----
    // v[5] = 3.0f;  -- sets element at index 5 to 3.0

    // Mutable: returns a reference (can assign to it)
    float& operator[](size_t i) { return data_[i]; }

    // Const: returns a copy (read-only)
    float operator[](size_t i) const { return data_[i]; }

    // ----- RESIZE -----
    // Changes size. New elements filled with 0.0f.
    void resize(size_t size) {
        size_ = size;
        data_.resize(size, 0.0f);
    }

    // ----- FILL -----
    // Sets all elements to the same value.
    void fill(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }

private:
    size_t size_;              // number of elements
    std::vector<float> data_;  // the actual values
};

} // namespace clustering
