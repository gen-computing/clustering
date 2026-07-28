#pragma once

#include "data_table.h"
#include <string>
#include <vector>

namespace clustering {

enum class MissingStrategy {
    DropRows,
    DropColumns,
    MeanImpute,
    MedianImpute,
    ModeImpute,
    ConstantFill,
    ForwardFill,
    LinearInterpolate
};

struct MissingHandlerConfig {
    MissingStrategy strategy = MissingStrategy::MedianImpute;
    float constant_value = 0.0f;
    float drop_column_threshold = 0.40f;
};

class MissingHandler {
public:
    MissingHandler() = default;

    void set_config(const MissingHandlerConfig& cfg) { config_ = cfg; }
    const MissingHandlerConfig& config() const { return config_; }

    void apply(DataTable& table);
    void apply_to_column(DataTable& table, size_t col);

    static const char* strategy_name(MissingStrategy s);
    static std::vector<MissingStrategy> all_strategies();

private:
    void drop_rows(DataTable& table);
    void drop_columns(DataTable& table);
    void mean_impute(DataTable& table);
    void median_impute(DataTable& table);
    void mode_impute(DataTable& table);
    void constant_fill(DataTable& table);
    void forward_fill(DataTable& table);
    void linear_interpolate(DataTable& table);

    void mean_impute_column(DataTable& table, size_t col);
    void median_impute_column(DataTable& table, size_t col);
    void mode_impute_column(DataTable& table, size_t col);
    void constant_fill_column(DataTable& table, size_t col);
    void forward_fill_column(DataTable& table, size_t col);
    void linear_interpolate_column(DataTable& table, size_t col);

    MissingHandlerConfig config_;
};

} // namespace clustering
