#include "preprocess_pipeline.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace clustering {

PreprocessPipeline::PreprocessPipeline(DataTable* table)
    : table_(table), current_(0) {}

PreprocessPipeline::~PreprocessPipeline() = default;

void PreprocessPipeline::apply(PreprocessAction action) {
    if (current_ < history_.size())
        history_.resize(current_);
    history_.push_back(std::move(action));
    current_++;
}

bool PreprocessPipeline::undo() {
    if (!can_undo()) return false;
    current_--;
    const auto& action = history_[current_];
    if (action.removed_rows.empty()) {
        for (size_t k = 0; k < action.cells.size(); ++k) {
            auto [row, col] = action.cells[k];
            table_->fill_value(row, col, action.old_values[k]);
        }
    } else {
        size_t r = table_->rows(), c = table_->cols();
        size_t orig_r = action.original_rows;
        size_t new_r = orig_r + action.removed_rows.size() / c;
        Matrix new_data(new_r, c);
        std::vector<uint8_t> new_missing(new_r * c);
        size_t dst = 0, removed_idx = 0;
        for (size_t i = 0; i < new_r; ++i) {
            bool was_removed = false;
            for (size_t k = 0; k < action.cells.size(); ++k) {
                if (action.cells[k].first == i) { was_removed = true; break; }
            }
            if (!was_removed && dst < r) {
                for (size_t j = 0; j < c; ++j) {
                    new_data[i][j] = table_->data()[dst][j];
                    new_missing[i * c + j] = 0;
                }
                dst++;
            } else {
                for (size_t j = 0; j < c; ++j) {
                    new_data[i][j] = action.removed_rows[removed_idx * c + j];
                    new_missing[i * c + j] = 0;
                }
                removed_idx++;
            }
        }
        table_->data_ = std::move(new_data);
        table_->missing_ = std::move(new_missing);
    }
    return true;
}

bool PreprocessPipeline::redo() {
    if (!can_redo()) return false;
    // Redo: re-apply the action's forward effect.
    // For fill actions, set cells back to NaN (missing).
    // For transform actions, we'd need the result values -- not stored yet,
    // so redo just marks those cells as dirty.
    const auto& action = history_[current_];
    for (size_t k = 0; k < action.cells.size(); ++k) {
        auto [row, col] = action.cells[k];
        table_->data_[row][col] = std::numeric_limits<float>::quiet_NaN();
        table_->missing_[row * table_->cols() + col] = 1;
    }
    current_++;
    return true;
}

void PreprocessPipeline::normalize_column(size_t col) {
    size_t n = table_->rows();
    float norm = 0.0f;
    for (size_t i = 0; i < n; ++i)
        if (!table_->is_missing(i, col))
            norm += table_->data()[i][col] * table_->data()[i][col];
    norm = std::sqrt(norm);
    if (norm <= 0) return;

    PreprocessAction action;
    action.description = "Normalize column " + std::to_string(col);
    for (size_t i = 0; i < n; ++i) {
        if (!table_->is_missing(i, col)) {
            float& v = table_->data_mut()[i][col];
            action.cells.push_back({i, col});
            action.old_values.push_back(v);
            v /= norm;
        }
    }
    apply(std::move(action));
}

void PreprocessPipeline::standardize_column(size_t col) {
    size_t n = table_->rows();
    size_t valid = 0;
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!table_->is_missing(i, col)) {
            float v = table_->data()[i][col];
            sum += v; sum_sq += v * v;
            valid++;
        }
    }
    if (valid < 2) return;
    float mean = sum / valid;
    float col_std = std::sqrt(std::max(0.0, (sum_sq - sum * sum / valid) / (valid - 1)));
    if (col_std <= 0) return;

    PreprocessAction action;
    action.description = "Standardize column " + std::to_string(col);
    for (size_t i = 0; i < n; ++i) {
        if (!table_->is_missing(i, col)) {
            float& v = table_->data_mut()[i][col];
            action.cells.push_back({i, col});
            action.old_values.push_back(v);
            v = (v - mean) / col_std;
        }
    }
    apply(std::move(action));
}

void PreprocessPipeline::minmax_scale_column(size_t col) {
    size_t n = table_->rows();
    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < n; ++i) {
        if (!table_->is_missing(i, col)) {
            float v = table_->data()[i][col];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
    }
    float range = max_v - min_v;
    if (range <= 0) return;

    PreprocessAction action;
    action.description = "MinMax scale column " + std::to_string(col);
    for (size_t i = 0; i < n; ++i) {
        if (!table_->is_missing(i, col)) {
            float& v = table_->data_mut()[i][col];
            action.cells.push_back({i, col});
            action.old_values.push_back(v);
            v = (v - min_v) / range;
        }
    }
    apply(std::move(action));
}

void PreprocessPipeline::log_transform_column(size_t col) {
    size_t n = table_->rows();
    PreprocessAction action;
    action.description = "Log transform column " + std::to_string(col);
    for (size_t i = 0; i < n; ++i) {
        if (!table_->is_missing(i, col)) {
            float& v = table_->data_mut()[i][col];
            action.cells.push_back({i, col});
            action.old_values.push_back(v);
            v = std::log(std::max(v, 1e-10f));
        }
    }
    apply(std::move(action));
}

void PreprocessPipeline::clip_outliers_column(size_t col, float lower_pct, float upper_pct) {
    size_t n = table_->rows();
    std::vector<float> vals;
    for (size_t i = 0; i < n; ++i)
        if (!table_->is_missing(i, col))
            vals.push_back(table_->data()[i][col]);
    if (vals.empty()) return;
    std::sort(vals.begin(), vals.end());
    float lo = vals[std::min(vals.size() - 1, (size_t)(lower_pct / 100.0f * (vals.size() - 1)))];
    float hi = vals[std::min(vals.size() - 1, (size_t)(upper_pct / 100.0f * (vals.size() - 1)))];

    PreprocessAction action;
    action.description = "Clip outliers col " + std::to_string(col);
    for (size_t i = 0; i < n; ++i) {
        if (!table_->is_missing(i, col)) {
            float& v = table_->data_mut()[i][col];
            if (v < lo || v > hi) {
                action.cells.push_back({i, col});
                action.old_values.push_back(v);
                v = std::max(lo, std::min(hi, v));
            }
        }
    }
    if (!action.cells.empty()) apply(std::move(action));
}

void PreprocessPipeline::clear() {
    history_.clear();
    current_ = 0;
}

void PreprocessPipeline::copy_history_from(const PreprocessPipeline& other) {
    history_ = other.history_;
    current_ = other.current_;
}

} // namespace clustering
