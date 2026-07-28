#pragma once

#include "clustering/matrix.h"
#include <vector>
#include <string>
#include <deque>
#include <utility>
#include <cmath>
#include <memory>
#include <cstdint>

namespace clustering {

struct PreprocessAction {
    std::string description;
    std::vector<std::pair<size_t, size_t>> cells;
    std::vector<float> old_values;
    size_t original_rows;
    std::vector<float> removed_rows;
};

class PreprocessPipeline;

class DataTable {
public:
    DataTable();
    ~DataTable();

    DataTable(DataTable&& other) noexcept;
    DataTable& operator=(DataTable&& other) noexcept;
    DataTable(const DataTable&) = delete;
    DataTable& operator=(const DataTable&) = delete;

    void set_data(const Matrix& data, std::vector<std::string> col_names = {});
    const Matrix& data() const { return data_; }
    Matrix& data_mut() { return data_; }
    size_t rows() const { return data_.rows(); }
    size_t cols() const { return data_.cols(); }
    const std::vector<std::string>& column_names() const { return col_names_; }
    std::vector<std::string>& column_names_mut() { return col_names_; }

    bool is_missing(size_t row, size_t col) const;
    size_t missing_count(size_t col) const;
    size_t missing_total() const;
    size_t row_missing_count(size_t row) const;

    void set_missing(size_t row, size_t col, bool missing = true);
    void fill_value(size_t row, size_t col, float value);
    void fill_column(size_t col, float value);
    void remove_row(size_t row);
    void remove_column(size_t col);
    void drop_rows_with_missing();

    PreprocessPipeline& pipeline() { return *pipeline_; }
    const PreprocessPipeline& pipeline() const { return *pipeline_; }

private:
    friend class PreprocessPipeline;
    Matrix data_;
    std::vector<std::string> col_names_;
    std::vector<uint8_t> missing_; // bitmask: 1=missing
    std::unique_ptr<PreprocessPipeline> pipeline_;
};

} // namespace clustering
