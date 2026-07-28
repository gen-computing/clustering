#pragma once

#include "data_table.h"
#include <vector>
#include <cstddef>
#include <string>

namespace clustering {

struct ColumnStats {
    size_t count;
    size_t missing_count;
    float mean;
    float median;
    float std_dev;
    float min_val;
    float max_val;
    float q1;
    float q3;
    bool computed;

    ColumnStats() : count(0), missing_count(0), mean(0), median(0),
        std_dev(0), min_val(0), max_val(0), q1(0), q3(0), computed(false) {}
};

class ColumnStatsCache {
public:
    ColumnStatsCache();
    void set_data(const DataTable* table);
    void invalidate();
    void invalidate_column(size_t col);
    const ColumnStats& get(size_t col);
    std::vector<float> sample_for_histogram(size_t col, size_t n_samples = 1000) const;

private:
    void compute(size_t col);
    const DataTable* table_;
    std::vector<ColumnStats> cache_;
};

} // namespace clustering
