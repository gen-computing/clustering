#include "data_table.h"
#include "preprocess_pipeline.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace clustering {

DataTable::DataTable() : pipeline_(std::make_unique<PreprocessPipeline>(this)) {}
DataTable::~DataTable() = default;

DataTable::DataTable(DataTable&& other) noexcept
    : data_(std::move(other.data_))
    , col_names_(std::move(other.col_names_))
    , missing_(std::move(other.missing_))
    , pipeline_(std::move(other.pipeline_))
{
    if (pipeline_) {
        // fixup: pipeline points to the old table, need to re-point
        pipeline_ = std::make_unique<PreprocessPipeline>(this);
        pipeline_->copy_history_from(*other.pipeline_);
    }
}

DataTable& DataTable::operator=(DataTable&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        col_names_ = std::move(other.col_names_);
        missing_ = std::move(other.missing_);
        pipeline_ = std::make_unique<PreprocessPipeline>(this);
        if (other.pipeline_) pipeline_->copy_history_from(*other.pipeline_);
    }
    return *this;
}

void DataTable::set_data(const Matrix& data, std::vector<std::string> col_names) {
    data_ = data;
    init_from_data(std::move(col_names));
}

void DataTable::set_data(Matrix&& data, std::vector<std::string> col_names) {
    data_ = std::move(data);
    init_from_data(std::move(col_names));
}

void DataTable::init_from_data(std::vector<std::string> col_names) {
    if (col_names.empty()) {
        col_names_.clear();
        for (size_t j = 0; j < data_.cols(); ++j) {
            // Excel-style letters: A..Z, AA, AB, ...
            std::string name;
            size_t n = j;
            while (true) {
                name.insert(name.begin(), char('A' + n % 26));
                if (n < 26) break;
                n = n / 26 - 1;
            }
            col_names_.push_back(std::move(name));
        }
    } else {
        col_names_ = std::move(col_names);
    }
    missing_.assign(data_.rows() * data_.cols(), 0);
    for (size_t i = 0; i < data_.rows(); ++i)
        for (size_t j = 0; j < data_.cols(); ++j)
            if (std::isnan(data_[i][j]))
                missing_[i * data_.cols() + j] = 1;
}

bool DataTable::is_missing(size_t row, size_t col) const {
    return missing_[row * cols() + col] != 0;
}

size_t DataTable::missing_count(size_t col) const {
    size_t count = 0;
    for (size_t i = 0; i < rows(); ++i)
        if (is_missing(i, col)) count++;
    return count;
}

size_t DataTable::missing_total() const {
    size_t count = 0;
    for (auto v : missing_) if (v) count++;
    return count;
}

size_t DataTable::row_missing_count(size_t row) const {
    size_t count = 0;
    for (size_t j = 0; j < cols(); ++j)
        if (is_missing(row, j)) count++;
    return count;
}

void DataTable::set_missing(size_t row, size_t col, bool missing) {
    missing_[row * cols() + col] = missing ? 1 : 0;
    if (missing) data_[row][col] = std::numeric_limits<float>::quiet_NaN();
    else if (std::isnan(data_[row][col])) data_[row][col] = 0.0f;
}

void DataTable::fill_value(size_t row, size_t col, float value) {
    data_[row][col] = value;
    missing_[row * cols() + col] = 0;
}

void DataTable::fill_column(size_t col, float value) {
    for (size_t i = 0; i < rows(); ++i) {
        if (is_missing(i, col)) {
            data_[i][col] = value;
            missing_[i * cols() + col] = 0;
        }
    }
}

void DataTable::remove_row(size_t row) {
    if (row >= rows()) return;
    size_t r = rows(), c = cols();
    Matrix new_data(r - 1, c);
    std::vector<uint8_t> new_missing((r - 1) * c);
    size_t dst = 0;
    for (size_t i = 0; i < r; ++i) {
        if (i == row) continue;
        for (size_t j = 0; j < c; ++j) {
            new_data[dst][j] = data_[i][j];
            new_missing[dst * c + j] = missing_[i * c + j];
        }
        dst++;
    }
    data_ = std::move(new_data);
    missing_ = std::move(new_missing);
}

void DataTable::remove_column(size_t col) {
    if (col >= cols()) return;
    size_t r = rows(), c = cols();
    Matrix new_data(r, c - 1);
    std::vector<uint8_t> new_missing(r * (c - 1));
    for (size_t i = 0; i < r; ++i) {
        size_t dst = 0;
        for (size_t j = 0; j < c; ++j) {
            if (j == col) continue;
            new_data[i][dst] = data_[i][j];
            new_missing[i * (c - 1) + dst] = missing_[i * c + j];
            dst++;
        }
    }
    data_ = std::move(new_data);
    missing_ = std::move(new_missing);
    if (col < col_names_.size())
        col_names_.erase(col_names_.begin() + col);
}

void DataTable::insert_column(size_t col, const std::vector<float>& values,
                              const std::vector<uint8_t>& missing, const std::string& name) {
    if (col > cols()) return;
    size_t r = rows(), c = cols();
    Matrix new_data(r, c + 1);
    std::vector<uint8_t> new_missing(r * (c + 1));
    for (size_t i = 0; i < r; ++i) {
        size_t dst = 0;
        for (size_t j = 0; j <= c; ++j) {
            if (j == col) {
                if (i < values.size()) {
                    new_data[i][dst] = values[i];
                    new_missing[i * (c + 1) + dst] = i < missing.size() ? missing[i] : 0;
                }
                dst++;
                continue;
            }
            size_t sj = j < col ? j : j - 1;
            new_data[i][dst] = data_[i][sj];
            new_missing[i * (c + 1) + dst] = missing_[i * c + sj];
            dst++;
        }
    }
    data_ = std::move(new_data);
    missing_ = std::move(new_missing);
    if (col <= col_names_.size())
        col_names_.insert(col_names_.begin() + col, name);
}

void DataTable::drop_rows_with_missing() {
    size_t r = rows(), c = cols();
    std::vector<size_t> keep;
    for (size_t i = 0; i < r; ++i)
        if (row_missing_count(i) == 0)
            keep.push_back(i);
    if (keep.size() == r) return;
    Matrix new_data(keep.size(), c);
    std::vector<uint8_t> new_missing(keep.size() * c);
    for (size_t i = 0; i < keep.size(); ++i) {
        size_t src = keep[i];
        for (size_t j = 0; j < c; ++j) {
            new_data[i][j] = data_[src][j];
            new_missing[i * c + j] = missing_[src * c + j];
        }
    }
    data_ = std::move(new_data);
    missing_ = std::move(new_missing);
}

} // namespace clustering
