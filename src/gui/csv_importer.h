#pragma once

#include "data_table.h"
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <vector>

namespace clustering {

struct CSVLoadResult {
    DataTable table;
    bool success;
    std::string error;
    size_t total_rows;
    size_t loaded_rows;
};

using ProgressCallback = std::function<void(size_t loaded, size_t total)>;

class CSVImporter {
public:
    CSVImporter();
    ~CSVImporter();

    CSVLoadResult load(const std::string& path,
                       bool has_header = true,
                       char delimiter = ',',
                       int max_rows = -1,
                       ProgressCallback progress = {});

    void load_async(const std::string& path,
                    std::function<void(CSVLoadResult)> on_done,
                    bool has_header = true,
                    char delimiter = ',',
                    int max_rows = -1,
                    ProgressCallback progress = {});

    void cancel();
    bool is_loading() const { return loading_.load(); }
    float loading_progress() const { return progress_.load(); }

    static std::vector<std::string> supported_missing_tokens();

private:
    void detect_types(const std::vector<std::string>& row, std::vector<bool>& numeric);
    float parse_cell(const std::string& cell, bool& is_missing);
    size_t count_lines(const std::string& path);

    std::atomic<bool> cancel_;
    std::atomic<bool> loading_;
    std::atomic<float> progress_;
    std::thread worker_;
};

} // namespace clustering
