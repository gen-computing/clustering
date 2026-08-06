#include "missing_handler.h"
#include "preprocess_pipeline.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <limits>

namespace clustering {

const char* MissingHandler::strategy_name(MissingStrategy s) {
    switch (s) {
        case MissingStrategy::DropRows:          return "Drop rows with missing";
        case MissingStrategy::DropColumns:       return "Drop sparse columns";
        case MissingStrategy::MeanImpute:        return "Mean imputation";
        case MissingStrategy::MedianImpute:      return "Median imputation";
        case MissingStrategy::ModeImpute:        return "Mode imputation";
        case MissingStrategy::ConstantFill:      return "Fill with constant";
        case MissingStrategy::ForwardFill:       return "Forward fill";
        case MissingStrategy::LinearInterpolate: return "Linear interpolation";
    }
    return "Unknown";
}

std::vector<MissingStrategy> MissingHandler::all_strategies() {
    return {
        MissingStrategy::DropRows, MissingStrategy::DropColumns,
        MissingStrategy::MeanImpute, MissingStrategy::MedianImpute,
        MissingStrategy::ModeImpute, MissingStrategy::ConstantFill,
        MissingStrategy::ForwardFill, MissingStrategy::LinearInterpolate
    };
}

void MissingHandler::apply(DataTable& table) {
    switch (config_.strategy) {
        case MissingStrategy::DropRows:          drop_rows(table); break;
        case MissingStrategy::DropColumns:       drop_columns(table); break;
        case MissingStrategy::MeanImpute:        mean_impute(table); break;
        case MissingStrategy::MedianImpute:      median_impute(table); break;
        case MissingStrategy::ModeImpute:        mode_impute(table); break;
        case MissingStrategy::ConstantFill:      constant_fill(table); break;
        case MissingStrategy::ForwardFill:       forward_fill(table); break;
        case MissingStrategy::LinearInterpolate: linear_interpolate(table); break;
    }
}

void MissingHandler::apply_to_column(DataTable& table, size_t col) {
    switch (config_.strategy) {
        case MissingStrategy::MeanImpute:        mean_impute_column(table, col); break;
        case MissingStrategy::MedianImpute:      median_impute_column(table, col); break;
        case MissingStrategy::ModeImpute:        mode_impute_column(table, col); break;
        case MissingStrategy::ConstantFill:      constant_fill_column(table, col); break;
        case MissingStrategy::ForwardFill:       forward_fill_column(table, col); break;
        case MissingStrategy::LinearInterpolate: linear_interpolate_column(table, col); break;
        default: apply(table); break;
    }
}

void MissingHandler::drop_rows(DataTable& table) {
    size_t orig_r = table.rows(), c = table.cols();
    std::vector<size_t> keep;
    for (size_t i = 0; i < orig_r; ++i)
        if (table.row_missing_count(i) == 0)
            keep.push_back(i);
    if (keep.size() == orig_r) return;

    std::vector<float> removed;
    for (size_t i = 0; i < orig_r; ++i) {
        if (table.row_missing_count(i) > 0) {
            for (size_t j = 0; j < c; ++j)
                removed.push_back(table.data()[i][j]);
        }
    }

    PreprocessAction action;
    action.description = "Drop " + std::to_string(orig_r - keep.size()) + " rows with missing values";
    action.original_rows = keep.size();
    action.removed_rows = std::move(removed);
    for (size_t i = 0; i < orig_r; ++i)
        if (table.row_missing_count(i) > 0)
            action.cells.push_back({i, 0});

    Matrix new_data(keep.size(), c);
    std::vector<uint8_t> new_missing(keep.size() * c);
    for (size_t i = 0; i < keep.size(); ++i) {
        for (size_t j = 0; j < c; ++j) {
            new_data[i][j] = table.data()[keep[i]][j];
        }
    }
    table = DataTable();
    table.set_data(new_data);
    table.pipeline().apply(std::move(action));
}

void MissingHandler::drop_columns(DataTable& table) {
    float threshold = config_.drop_column_threshold;
    size_t n = table.rows();
    std::vector<size_t> to_drop;
    for (size_t j = 0; j < table.cols(); ++j) {
        float missing_ratio = n > 0 ? (float)table.missing_count(j) / n : 0.0f;
        if (missing_ratio > threshold) to_drop.push_back(j);
    }
    for (auto it = to_drop.rbegin(); it != to_drop.rend(); ++it)
        table.remove_column(*it);
    PreprocessAction action;
    action.description = "Dropped " + std::to_string(to_drop.size()) + " sparse columns";
    table.pipeline().apply(std::move(action));
}

void MissingHandler::mean_impute(DataTable& table) {
    for (size_t j = 0; j < table.cols(); ++j)
        mean_impute_column(table, j);
}

void MissingHandler::mean_impute_column(DataTable& table, size_t col) {
    size_t n = table.rows();
    size_t valid = 0;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!table.is_missing(i, col)) {
            sum += table.data()[i][col];
            valid++;
        }
    }
    if (valid == 0) return;
    float mean = sum / valid;

    PreprocessAction action;
    action.description = "Mean impute column " + table.column_names()[col];
    for (size_t i = 0; i < n; ++i) {
        if (table.is_missing(i, col)) {
            action.cells.push_back({i, col});
            action.old_values.push_back(std::numeric_limits<float>::quiet_NaN());
            table.fill_value(i, col, mean);
            action.new_values.push_back(mean);
        }
    }
    if (!action.cells.empty()) table.pipeline().apply(std::move(action));
}

void MissingHandler::median_impute(DataTable& table) {
    for (size_t j = 0; j < table.cols(); ++j)
        median_impute_column(table, j);
}

void MissingHandler::median_impute_column(DataTable& table, size_t col) {
    size_t n = table.rows();
    std::vector<float> vals;
    for (size_t i = 0; i < n; ++i)
        if (!table.is_missing(i, col))
            vals.push_back(table.data()[i][col]);
    if (vals.empty()) return;
    std::sort(vals.begin(), vals.end());
    float median = vals[vals.size() / 2];

    PreprocessAction action;
    action.description = "Median impute column " + table.column_names()[col];
    for (size_t i = 0; i < n; ++i) {
        if (table.is_missing(i, col)) {
            action.cells.push_back({i, col});
            action.old_values.push_back(std::numeric_limits<float>::quiet_NaN());
            table.fill_value(i, col, median);
            action.new_values.push_back(median);
        }
    }
    if (!action.cells.empty()) table.pipeline().apply(std::move(action));
}

void MissingHandler::mode_impute(DataTable& table) {
    for (size_t j = 0; j < table.cols(); ++j)
        mode_impute_column(table, j);
}

void MissingHandler::mode_impute_column(DataTable& table, size_t col) {
    size_t n = table.rows();
    std::unordered_map<float, size_t> freq;
    for (size_t i = 0; i < n; ++i) {
        if (!table.is_missing(i, col))
            freq[table.data()[i][col]]++;
    }
    if (freq.empty()) return;
    float mode = freq.begin()->first;
    for (auto& [k, v] : freq)
        if (v > freq[mode]) mode = k;

    PreprocessAction action;
    action.description = "Mode impute column " + table.column_names()[col];
    for (size_t i = 0; i < n; ++i) {
        if (table.is_missing(i, col)) {
            action.cells.push_back({i, col});
            action.old_values.push_back(std::numeric_limits<float>::quiet_NaN());
            table.fill_value(i, col, mode);
            action.new_values.push_back(mode);
        }
    }
    if (!action.cells.empty()) table.pipeline().apply(std::move(action));
}

void MissingHandler::constant_fill(DataTable& table) {
    for (size_t j = 0; j < table.cols(); ++j)
        constant_fill_column(table, j);
}

void MissingHandler::constant_fill_column(DataTable& table, size_t col) {
    float val = config_.constant_value;
    size_t n = table.rows();
    PreprocessAction action;
    action.description = "Fill column " + table.column_names()[col] + " with " + std::to_string(val);
    for (size_t i = 0; i < n; ++i) {
        if (table.is_missing(i, col)) {
            action.cells.push_back({i, col});
            action.old_values.push_back(std::numeric_limits<float>::quiet_NaN());
            table.fill_value(i, col, val);
            action.new_values.push_back(val);
        }
    }
    if (!action.cells.empty()) table.pipeline().apply(std::move(action));
}

void MissingHandler::forward_fill(DataTable& table) {
    for (size_t j = 0; j < table.cols(); ++j)
        forward_fill_column(table, j);
}

void MissingHandler::forward_fill_column(DataTable& table, size_t col) {
    size_t n = table.rows();
    float last_valid = 0.0f;
    bool has_last = false;
    PreprocessAction action;
    action.description = "Forward fill column " + table.column_names()[col];
    for (size_t i = 0; i < n; ++i) {
        if (table.is_missing(i, col)) {
            if (has_last) {
                action.cells.push_back({i, col});
                action.old_values.push_back(std::numeric_limits<float>::quiet_NaN());
                table.fill_value(i, col, last_valid);
                action.new_values.push_back(last_valid);
            }
        } else {
            last_valid = table.data()[i][col];
            has_last = true;
        }
    }
    if (!action.cells.empty()) table.pipeline().apply(std::move(action));
}

void MissingHandler::linear_interpolate(DataTable& table) {
    for (size_t j = 0; j < table.cols(); ++j)
        linear_interpolate_column(table, j);
}

void MissingHandler::linear_interpolate_column(DataTable& table, size_t col) {
    size_t n = table.rows();
    PreprocessAction action;
    action.description = "Linear interpolate column " + table.column_names()[col];

    for (size_t i = 0; i < n; ++i) {
        if (!table.is_missing(i, col)) continue;
        size_t before = i, after = i;
        while (before > 0 && table.is_missing(before - 1, col)) before--;
        while (after + 1 < n && table.is_missing(after + 1, col)) after++;
        float v_before = (before > 0 && !table.is_missing(before - 1, col)) ? table.data()[before - 1][col] : 0;
        float v_after = (after + 1 < n && !table.is_missing(after + 1, col)) ? table.data()[after + 1][col] : 0;
        bool has_before = (before > 0 && !table.is_missing(before - 1, col));
        bool has_after = (after + 1 < n && !table.is_missing(after + 1, col));

        for (size_t k = before; k <= after; ++k) {
            if (has_before && has_after) {
                float t = float(k - (before - 1)) / float(after + 1 - (before - 1));
                float val = v_before + t * (v_after - v_before);
                action.cells.push_back({k, col});
                action.old_values.push_back(std::numeric_limits<float>::quiet_NaN());
                table.fill_value(k, col, val);
                action.new_values.push_back(val);
            } else if (has_before) {
                action.cells.push_back({k, col});
                action.old_values.push_back(std::numeric_limits<float>::quiet_NaN());
                table.fill_value(k, col, v_before);
                action.new_values.push_back(v_before);
            } else if (has_after) {
                action.cells.push_back({k, col});
                action.old_values.push_back(std::numeric_limits<float>::quiet_NaN());
                table.fill_value(k, col, v_after);
                action.new_values.push_back(v_after);
            }
        }
        i = after;
    }
    if (!action.cells.empty()) table.pipeline().apply(std::move(action));
}

} // namespace clustering
