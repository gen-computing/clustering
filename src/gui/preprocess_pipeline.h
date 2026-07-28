#pragma once

#include "data_table.h"
#include <vector>
#include <string>
#include <deque>
#include <utility>

namespace clustering {

class PreprocessPipeline {
public:
    PreprocessPipeline(DataTable* table);
    ~PreprocessPipeline();

    void apply(PreprocessAction action);
    bool undo();
    bool redo();
    bool can_undo() const { return current_ > 0; }
    bool can_redo() const { return current_ < history_.size(); }

    const std::vector<PreprocessAction>& history() const { return history_; }
    size_t current() const { return current_; }

    void normalize_column(size_t col);
    void standardize_column(size_t col);
    void minmax_scale_column(size_t col);
    void log_transform_column(size_t col);
    void clip_outliers_column(size_t col, float lower_pct = 1.0f, float upper_pct = 99.0f);

    void clear();

    void copy_history_from(const PreprocessPipeline& other);

private:
    void record_action(PreprocessAction action);
    DataTable* table_;
    std::vector<PreprocessAction> history_;
    size_t current_;
};

} // namespace clustering
