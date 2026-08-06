#pragma once
// #pragma once tells the compiler: "only include this file once per compilation unit".
// It prevents errors when two different .cpp files accidentally include the same header.
// Alternative older way: #ifndef / #define / #endif guards.

#include <vector>   // std::vector -- a dynamic array that grows/shrinks automatically.
                    // Stores elements contiguously in memory (like a C array but safer).
#include <cstddef>  // Defines size_t -- an unsigned integer type for sizes and counts.
#include <cstdint>  // Defines fixed-width integers (int32_t, uint64_t etc) if needed.
#include <memory>   // std::unique_ptr
#include <mutex>    // std::mutex -- thread safety for the disk cache
#include <list>     // std::list -- LRU block list
#include <unordered_map>
#include <string>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace clustering {
// A namespace is like a folder for names. Everything inside `clustering` is kept
// separate from other libraries, so name clashes (e.g., another library's `Matrix`)
// won't cause errors. Use `clustering::Matrix` to refer to this type.

// ============================================================================
// Global RAM cap for automatic disk-backed Matrix storage.
//
// A Matrix whose byte size exceeds this cap transparently spills to a
// temporary file and keeps only a bounded LRU window of rows in RAM.
// This is what makes large datasets (millions of rows) usable on modest
// hardware: peak memory stays near the cap regardless of dataset size.
//
// Default: 512 MiB. Set before constructing large matrices.
// ============================================================================
inline size_t matrix_ram_cap_ = size_t(512) * 1024 * 1024;
inline void set_matrix_ram_cap(size_t bytes) { matrix_ram_cap_ = bytes; }
inline size_t matrix_ram_cap() { return matrix_ram_cap_; }

// ============================================================================
// Matrix class -- A 2D grid of floating-point numbers (like a spreadsheet).
//
// A matrix is the fundamental data structure in machine learning.
// Each row = one data point (e.g., one customer, one image, one measurement).
// Each column = one feature (e.g., age, price, color intensity).
//
// Storage: Small matrices stay in ONE flat array (row-major order).
// Large matrices (above the global RAM cap) are transparently backed by a
// temporary file with an LRU row-block cache. The m[row][col] API is
// identical in both modes, so all algorithms work unchanged on huge data.
// ============================================================================
class Matrix {
public:
    // ----- CONSTRUCTORS -----

    // 1. DEFAULT CONSTRUCTOR: Creates an empty matrix (0 rows, 0 columns).
    Matrix() : rows_(0), cols_(0) {}

    // 2. SIZE CONSTRUCTOR: Creates a matrix with given dimensions, filled with zeros.
    Matrix(size_t rows, size_t cols) : rows_(0), cols_(0) { resize(rows, cols); }

    // 3. DATA CONSTRUCTOR: Creates a matrix by COPYING values from an existing
    //    raw C array (float pointer). Used to receive data from numpy/Python.
    Matrix(size_t rows, size_t cols, float* data) : rows_(0), cols_(0) {
        resize(rows, cols);
        if (disk_) {
            for (size_t i = 0; i < rows; ++i)
                std::memcpy((*this)[i], data + i * cols, cols * sizeof(float));
        } else {
            data_.assign(data, data + rows * cols);
        }
    }

    // Copy: full byte copy (own temp file when disk-backed).
    Matrix(const Matrix& other) : rows_(0), cols_(0) { copy_from(other); }
    Matrix& operator=(const Matrix& other) {
        if (this != &other) copy_from(other);
        return *this;
    }

    // Move: transfer storage (no file copy).
    Matrix(Matrix&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_),
          data_(std::move(other.data_)), disk_(std::move(other.disk_)) {
        other.rows_ = 0; other.cols_ = 0;
    }
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            disk_.reset();
            rows_ = other.rows_; cols_ = other.cols_;
            data_ = std::move(other.data_);
            disk_ = std::move(other.disk_);
            other.rows_ = 0; other.cols_ = 0;
        }
        return *this;
    }

    ~Matrix() = default;  // disk_ destructor removes the temp file

    // ----- SIZE ACCESSORS -----
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }

    // ----- STORAGE MODE -----
    // True when this matrix spills to a temp file (its byte size exceeds the
    // global RAM cap). data() throws when this is true.
    bool is_disk_backed() const { return disk_ != nullptr; }

    // Resident cache bytes (disk mode only).
    size_t ram_bytes_used() const;
    // Bytes stored on disk (disk mode only).
    size_t disk_bytes_used() const;

    // ----- RAW DATA ACCESS -----
    // Returns nullptr when disk-backed (contiguous buffer not resident).
    // Callers must use operator[] instead when is_disk_backed() is true.
    // Direct flat-array access. Only valid when the matrix is inlined in RAM
    // (is_disk_backed() == false). Throws std::runtime_error when disk-backed,
    // because the data is not contiguous in memory.
    float* data() {
        if (disk_) throw std::runtime_error("Matrix is disk-backed; data() unavailable");
        return data_.data();
    }
    const float* data() const {
        if (disk_) throw std::runtime_error("Matrix is disk-backed; data() unavailable");
        return data_.data();
    }

    // ----- ELEMENT ACCESS (operator[]) -----
    // m[row] returns pointer to start of row; m[row][col] the element.
    // In disk mode the containing block is loaded into the LRU cache.
    float* operator[](size_t row);
    const float* operator[](size_t row) const;

    // ----- RESIZE -----
    void resize(size_t rows, size_t cols);

    // ----- FILL -----
    void fill(float value);

private:
    struct DiskBackend;

    void copy_from(const Matrix& other);

    size_t rows_ = 0;
    size_t cols_ = 0;
    std::vector<float> data_;
    std::unique_ptr<DiskBackend> disk_;
};

// ============================================================================
// DiskBackend -- LRU row-block cache over a temporary file.
//
// Rows are grouped into fixed-size blocks. A bounded window of blocks is
// resident; evicted blocks are written back (if dirty) and reloaded from
// the file on demand. Sequential row access is O(1) via a last-block
// fast path. All access is guarded by a mutex so a worker thread can
// read a matrix while the UI thread renders it.
// ============================================================================
struct Matrix::DiskBackend {
    struct Block {
        int64_t id = -1;
        std::vector<float> data;
        bool dirty = false;
    };

    size_t rows = 0;
    size_t cols = 0;
    size_t block_rows = 0;
    size_t cap_blocks = 0;
    std::string path;
    std::fstream file;
    mutable std::mutex mtx;
    std::list<Block> lru;
    std::unordered_map<int64_t, std::list<Block>::iterator> index;
    int64_t last_id = -1;
    std::list<Block>::iterator last_it;
    size_t ram_used = 0;

    DiskBackend(size_t r, size_t c) : rows(r), cols(c) {
        size_t block_bytes = 2 * 1024 * 1024;  // target ~2 MiB per block
        size_t row_bytes = cols * sizeof(float);
        block_rows = std::max<size_t>(32, block_bytes / std::max<size_t>(row_bytes, 1));
        block_rows = std::min<size_t>(block_rows, 65536);

        size_t cap = matrix_ram_cap();
        cap_blocks = std::max<size_t>(2, cap / std::max<size_t>(block_rows * row_bytes, 1));

        static size_t counter = 0;
#ifdef _WIN32
        int pid = _getpid();
#else
        int pid = getpid();
#endif
        path = (std::filesystem::temp_directory_path() /
                ("clustering_" + std::to_string((size_t)pid) + "_" +
                 std::to_string(counter++) + ".bin")).string();
        file.open(path, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
    }

    ~DiskBackend() {
        file.close();
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    size_t file_bytes() const { return rows * cols * sizeof(float); }
    size_t n_blocks() const {
        return rows == 0 ? 0 : (rows + block_rows - 1) / block_rows;
    }

    void resize(size_t r, size_t c) {
        if (c != cols) {
            // Column count changed: full rebuild (rare; e.g. column drop).
            std::vector<float> old_data(rows * cols);
            size_t old_rows = rows, old_cols = cols;
            for (size_t i = 0; i < old_rows; ++i) {
                const float* src = row_ptr(i, old_cols, false);
                std::copy(src, src + old_cols, old_data.data() + i * old_cols);
            }
            rows = r; cols = c;
            file.close();
            file.open(path, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
            lru.clear(); index.clear(); ram_used = 0; last_id = -1;
            size_t row_bytes = cols * sizeof(float);
            block_rows = std::max<size_t>(32, (2 * 1024 * 1024) / std::max<size_t>(row_bytes, 1));
            block_rows = std::min<size_t>(block_rows, 65536);
            cap_blocks = std::max<size_t>(2, matrix_ram_cap() / std::max<size_t>(block_rows * row_bytes, 1));
            size_t d = std::min(old_cols, cols);
            size_t nr = std::min(old_rows, rows);
            for (size_t i = 0; i < nr; ++i) {
                float* dst = row_ptr(i, cols, true);
                for (size_t j = 0; j < d; ++j) dst[j] = old_data[i * old_cols + j];
            }
            return;
        }
        if (r == rows) return;
        size_t new_blocks = r == 0 ? 0 : (r + block_rows - 1) / block_rows;
        rows = r;
        // Drop cache entries beyond the new extent.
        for (auto it = index.begin(); it != index.end();) {
            if ((size_t)it->first >= new_blocks) {
                ram_used -= it->second->data.size() * sizeof(float);
                lru.erase(it->second);
                it = index.erase(it);
            } else ++it;
        }
        if (last_id >= (int64_t)new_blocks) last_id = -1;
        // Truncate the file.
        file.seekp(rows * cols * sizeof(float));
        file.flush();
    }

    // Get pointer to a row, loading its block into the cache.
    // mark_dirty=true for the mutable path (caller may write through the
    // pointer); const reads pass false so read-only passes never write back.
    float* row_ptr(size_t row, size_t d, bool mark_dirty) {
        int64_t b = (int64_t)(row / block_rows);
        size_t off = row % block_rows;
        std::lock_guard<std::mutex> lk(mtx);
        if (b == last_id) {
            if (mark_dirty) last_it->dirty = true;
            return last_it->data.data() + off * d;
        }
        auto it = index.find(b);
        if (it != index.end()) {
            auto lit = it->second;
            lru.splice(lru.begin(), lru, lit);
            last_id = b; last_it = lru.begin();
            if (mark_dirty) last_it->dirty = true;
            return last_it->data.data() + off * d;
        }
        // Evict LRU if at capacity.
        while (!lru.empty() && lru.size() >= cap_blocks) {
            Block& victim = lru.back();
            if (victim.dirty) write_block(victim);
            ram_used -= victim.data.size() * sizeof(float);
            index.erase(victim.id);
            lru.pop_back();
        }
        // Load block from file.
        Block blk;
        blk.id = b;
        blk.data.resize(block_rows * d, 0.0f);
        size_t byte_off = (size_t)b * block_rows * d * sizeof(float);
        file.clear();
        file.seekg(byte_off);
        file.read(reinterpret_cast<char*>(blk.data.data()), blk.data.size() * sizeof(float));
        lru.push_front(std::move(blk));
        index[b] = lru.begin();
        ram_used += lru.begin()->data.size() * sizeof(float);
        last_id = b; last_it = lru.begin();
        if (mark_dirty) last_it->dirty = true;
        return last_it->data.data() + off * d;
    }

    void write_block(Block& blk) {
        size_t byte_off = (size_t)blk.id * block_rows * cols * sizeof(float);
        file.clear();
        file.seekp(byte_off);
        file.write(reinterpret_cast<const char*>(blk.data.data()), blk.data.size() * sizeof(float));
        blk.dirty = false;
    }

    // Write back every dirty resident block, so the temp file reflects all
    // data (needed before file-level copies, e.g. Matrix copy_from).
    void flush_all() {
        std::lock_guard<std::mutex> lk(mtx);
        for (auto& blk : lru)
            if (blk.dirty) write_block(blk);
        file.flush();
    }

    void fill(float value, size_t d) {
        std::lock_guard<std::mutex> lk(mtx);
        size_t nb = n_blocks();
        for (size_t b = 0; b < nb; ++b) {
            auto it = index.find((int64_t)b);
            std::list<Block>::iterator lit;
            bool fresh = false;
            if (it != index.end()) {
                lit = it->second;
                lru.splice(lru.begin(), lru, lit);
            } else {
                while (!lru.empty() && lru.size() >= cap_blocks) {
                    Block& victim = lru.back();
                    if (victim.dirty) write_block(victim);
                    ram_used -= victim.data.size() * sizeof(float);
                    index.erase(victim.id);
                    lru.pop_back();
                }
                Block blk;
                blk.id = (int64_t)b;
                blk.data.resize(block_rows * d, 0.0f);
                lru.push_front(std::move(blk));
                index[(int64_t)b] = lru.begin();
                ram_used += lru.begin()->data.size() * sizeof(float);
                lit = lru.begin();
                fresh = true;
            }
            size_t count = (b + 1) * block_rows <= rows ? block_rows : rows - b * block_rows;
            std::fill(lit->data.data(), lit->data.data() + count * d, value);
            lit->dirty = true;
            (void)fresh;
        }
        last_id = -1;
    }
};

// ============================================================================
// Matrix out-of-line implementations (must follow DiskBackend's definition).
// ============================================================================

inline size_t Matrix::ram_bytes_used() const {
    return disk_ ? disk_->ram_used : data_.size() * sizeof(float);
}

inline size_t Matrix::disk_bytes_used() const {
    return disk_ ? disk_->file_bytes() : 0;
}

inline float* Matrix::operator[](size_t row) {
    if (disk_) return disk_->row_ptr(row, cols_, true);
    return data_.data() + row * cols_;
}

inline const float* Matrix::operator[](size_t row) const {
    if (disk_) return const_cast<DiskBackend*>(disk_.get())->row_ptr(row, cols_, false);
    return data_.data() + row * cols_;
}

inline void Matrix::resize(size_t rows, size_t cols) {
    size_t bytes = rows * cols * sizeof(float);
    size_t cap = matrix_ram_cap();
    bool want_disk = rows > 0 && bytes > cap;

    if (want_disk && !disk_) {
        // Promote: move existing inline data into a new disk backend.
        auto old = std::move(data_);
        size_t old_rows = rows_, old_cols = cols_;
        disk_ = std::make_unique<DiskBackend>(rows, cols);
        if (!old.empty() && old_rows > 0 && old_cols > 0) {
            size_t n = std::min(old_rows, rows);
            size_t d = std::min(old_cols, cols);
            for (size_t i = 0; i < n; ++i) {
                float* dst = disk_->row_ptr(i, cols, true);
                for (size_t j = 0; j < d; ++j) dst[j] = old[i * old_cols + j];
            }
        }
    } else if (!want_disk && disk_) {
        // Demote: materialize everything back into RAM.
        std::vector<float> tmp(rows * cols, 0.0f);
        size_t d = std::min(cols, cols_);
        for (size_t i = 0; i < rows && i < rows_; ++i) {
            const float* src = disk_->row_ptr(i, cols_, false);
            for (size_t j = 0; j < d; ++j) tmp[i * cols + j] = src[j];
        }
        disk_.reset();
        data_ = std::move(tmp);
    }

    if (disk_) {
        disk_->resize(rows, cols);
    } else {
        data_.resize(rows * cols, 0.0f);
    }
    rows_ = rows; cols_ = cols;
}

inline void Matrix::fill(float value) {
    if (disk_) {
        disk_->fill(value, cols_);
    } else {
        std::fill(data_.begin(), data_.end(), value);
    }
}

inline void Matrix::copy_from(const Matrix& other) {
    if (other.disk_) {
        rows_ = other.rows_; cols_ = other.cols_;
        data_.clear(); data_.shrink_to_fit();
        disk_ = std::make_unique<DiskBackend>(other.rows_, other.cols_);
        // Byte-copy the temp file contents (all dirty blocks flushed first).
        other.disk_->flush_all();
        std::ifstream in(other.disk_->path, std::ios::binary);
        std::ofstream out(disk_->path, std::ios::binary | std::ios::trunc);
        out << in.rdbuf();
        out.close();
        disk_->file.open(disk_->path, std::ios::in | std::ios::out | std::ios::binary);
    } else {
        disk_.reset();
        rows_ = other.rows_; cols_ = other.cols_;
        data_ = other.data_;
    }
}

// ============================================================================
// Vector class -- A 1D list of floating-point numbers.
//
// Used for: cluster labels (which cluster each point belongs to),
//           explained variance, mean values, and other 1D data.
// Always held in RAM (labels are one float per point -- negligible).
// ============================================================================
class Vector {
public:
    // ----- CONSTRUCTORS -----

    // Default: empty vector, size 0
    Vector() : size_(0), data_() {}

    // Create vector of given size, all zeros.
    Vector(size_t size) : size_(size), data_(size, 0.0f) {}

    // Create vector by copying data from raw C array.
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
