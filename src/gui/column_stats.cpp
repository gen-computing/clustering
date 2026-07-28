#include "column_stats.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace clustering {

ColumnStatsCache::ColumnStatsCache() : table_(nullptr) {}

void ColumnStatsCache::set_data(const DataTable* table) {
    table_ = table;
    cache_.clear();
    if (table) cache_.resize(table->cols());
}

void ColumnStatsCache::invalidate() {
    for (auto& s : cache_) s.computed = false;
}

void ColumnStatsCache::invalidate_column(size_t col) {
    if (col < cache_.size()) cache_[col].computed = false;
}

const ColumnStats& ColumnStatsCache::get(size_t col) {
    if (col >= cache_.size()) {
        static ColumnStats empty;
        return empty;
    }
    if (!cache_[col].computed) compute(col);
    return cache_[col];
}

void ColumnStatsCache::compute(size_t col) {
    auto& s = cache_[col];
    s = ColumnStats();
    size_t n = table_->rows();
    if (n == 0) { s.computed = true; return; }

    s.count = n;
    s.missing_count = 0;
    double sum = 0.0, sum_sq = 0.0;
    s.min_val = std::numeric_limits<float>::max();
    s.max_val = std::numeric_limits<float>::lowest();

    std::vector<float> values;
    values.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        if (table_->is_missing(i, col)) {
            s.missing_count++;
            continue;
        }
        float v = table_->data()[i][col];
        values.push_back(v);
        sum += v;
        sum_sq += v * v;
        if (v < s.min_val) s.min_val = v;
        if (v > s.max_val) s.max_val = v;
    }

    size_t valid = values.size();
    if (valid == 0) { s.computed = true; return; }

    s.mean = static_cast<float>(sum / valid);
    s.std_dev = valid > 1 ? std::sqrt(std::max(0.0, (sum_sq - sum * sum / valid) / (valid - 1))) : 0.0f;

    std::sort(values.begin(), values.end());
    auto percentile = [&](float p) -> float {
        size_t idx = static_cast<size_t>(p / 100.0f * (valid - 1));
        return values[std::min(idx, valid - 1)];
    };
    s.median = percentile(50.0f);
    s.q1 = percentile(25.0f);
    s.q3 = percentile(75.0f);
    s.computed = true;
}

std::vector<float> ColumnStatsCache::sample_for_histogram(size_t col, size_t n_samples) const {
    std::vector<float> result;
    size_t n = table_->rows();
    if (n == 0) return result;

    if (n <= n_samples) {
        for (size_t i = 0; i < n; ++i)
            if (!table_->is_missing(i, col))
                result.push_back(table_->data()[i][col]);
    } else {
        std::mt19937 gen(42);
        size_t step = n / n_samples;
        for (size_t i = 0; i < n_samples; ++i) {
            size_t idx = i * step + (gen() % std::max(size_t(1), step));
            if (idx < n && !table_->is_missing(idx, col))
                result.push_back(table_->data()[idx][col]);
        }
    }
    return result;
}

} // namespace clustering
