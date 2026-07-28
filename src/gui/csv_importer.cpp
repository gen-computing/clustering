#include "csv_importer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace clustering {

CSVImporter::CSVImporter() : cancel_(false), loading_(false), progress_(0.0f) {}
CSVImporter::~CSVImporter() { cancel(); }

void CSVImporter::cancel() {
    cancel_ = true;
    if (worker_.joinable()) worker_.join();
}

size_t CSVImporter::count_lines(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return 0;
    size_t n = 0;
    std::string line;
    while (std::getline(f, line)) n++;
    return n;
}

CSVLoadResult CSVImporter::load(const std::string& path,
                                 bool has_header, char delimiter,
                                 int max_rows, ProgressCallback progress) {
    CSVLoadResult result;
    result.success = false;
    result.total_rows = 0;
    result.loaded_rows = 0;

    std::ifstream file(path);
    if (!file.is_open()) {
        result.error = "Cannot open file: " + path;
        return result;
    }

    size_t total = count_lines(path);
    result.total_rows = total;
    loading_ = true;
    progress_ = 0.0f;

    std::string line;
    std::vector<std::vector<float>> rows;
    std::vector<std::string> col_names;
    std::vector<bool> numeric_cols;
    size_t line_num = 0;
    bool first = true;

    while (std::getline(file, line)) {
        if (cancel_) { result.error = "Cancelled"; loading_ = false; return result; }
        if (max_rows > 0 && (int)rows.size() >= max_rows) break;

        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, delimiter)) {
            while (!cell.empty() && cell.front() == '"' && cell.back() != '"') {
                std::string next;
                if (std::getline(ss, next, delimiter))
                    cell += delimiter + next;
                else break;
            }
            if (!cell.empty() && cell.front() == '"' && cell.back() == '"')
                cell = cell.substr(1, cell.size() - 2);
            cells.push_back(cell);
        }

        if (first && has_header) {
            col_names = cells;
            numeric_cols.resize(cells.size(), true);
            first = false;
            continue;
        }

        if (first) {
            numeric_cols.resize(cells.size(), true);
            for (size_t j = 0; j < cells.size(); ++j)
                col_names.push_back("Col_" + std::to_string(j));
            first = false;
        }

        detect_types(cells, numeric_cols);

        std::vector<float> row;
        for (size_t j = 0; j < cells.size() && j < numeric_cols.size(); ++j) {
            bool is_missing = false;
            float val = 0.0f;
            if (numeric_cols[j]) {
                val = parse_cell(cells[j], is_missing);
            } else {
                is_missing = true;
                val = std::numeric_limits<float>::quiet_NaN();
            }
            row.push_back(val);
        }
        rows.push_back(std::move(row));
        result.loaded_rows++;

        if (progress) progress(result.loaded_rows, total);
        line_num++;
    }

    loading_ = false;
    progress_ = 1.0f;

    if (rows.empty()) {
        result.error = "No data rows found";
        return result;
    }

    size_t n = rows.size();
    size_t d = rows[0].size();
    Matrix mat(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d && j < rows[i].size(); ++j)
            mat[i][j] = rows[i][j];

    // Drop columns where most values are NaN (text columns like labels/timestamps)
    // Threshold: >90% NaN → drop column
    std::vector<size_t> keep_cols;
    std::vector<std::string> keep_names;
    for (size_t j = 0; j < d; ++j) {
        size_t nan_count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(mat[i][j])) nan_count++;
        }
        if (nan_count * 10 < n * 9) {  // keep if <90% NaN
            keep_cols.push_back(j);
            if (j < col_names.size()) keep_names.push_back(col_names[j]);
        }
    }

    if (keep_cols.size() < d) {
        size_t new_d = keep_cols.size();
        Matrix filtered(n, new_d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < new_d; ++j)
                filtered[i][j] = mat[i][keep_cols[j]];
        mat = std::move(filtered);
        col_names = std::move(keep_names);
    }

    result.table.set_data(mat, col_names);
    result.success = true;
    return result;
}

void CSVImporter::load_async(const std::string& path,
                              std::function<void(CSVLoadResult)> on_done,
                              bool has_header, char delimiter,
                              int max_rows, ProgressCallback progress_cb) {
    cancel();
    cancel_ = false;
    worker_ = std::thread([this, path, on_done, has_header, delimiter, max_rows, progress_cb]() {
        auto result = load(path, has_header, delimiter, max_rows, progress_cb);
        if (on_done) on_done(std::move(result));
    });
}

void CSVImporter::detect_types(const std::vector<std::string>& row, std::vector<bool>& numeric) {
    static const std::unordered_set<std::string> missing_tokens = {
        "", "NA", "na", "N/A", "n/a", "null", "NULL", "NaN", "nan", "-", ".", "None"
    };
    for (size_t j = 0; j < row.size() && j < numeric.size(); ++j) {
        if (!numeric[j]) continue;
        if (missing_tokens.count(row[j])) continue;

        const std::string& cell = row[j];

        // Detect datetime patterns: "2021-03-24 14:42:03", "2021-03-24T14:42:03", etc.
        // Pattern: digits-digits-digits followed by space/T and digits:digits:digits
        bool has_datetime = false;
        if (cell.size() >= 10) {
            // Check for YYYY-MM-DD or DD-MM-YYYY pattern with colon (time) present
            bool has_colon = false;
            bool has_dash = false;
            for (char c : cell) {
                if (c == ':') has_colon = true;
                if (c == '-') has_dash = true;
            }
            // Datetime: has both dashes and colons (e.g. "2021-03-24 14:42:03")
            if (has_colon && has_dash) has_datetime = true;
        }
        if (has_datetime) {
            numeric[j] = false;
            continue;
        }

        bool has_digit = false, has_alpha = false;
        for (char c : cell) {
            if (std::isdigit(c) || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E') {
                if (std::isdigit(c)) has_digit = true;
            } else if (std::isalpha(c)) {
                has_alpha = true;
            }
        }
        if (has_alpha || (!has_digit && row[j].find('.') == std::string::npos))
            numeric[j] = false;
    }
}

float CSVImporter::parse_cell(const std::string& cell, bool& is_missing) {
    static const std::unordered_set<std::string> missing_tokens = {
        "", "NA", "na", "N/A", "n/a", "null", "NULL", "NaN", "nan", "-", ".", "None"
    };
    if (missing_tokens.count(cell)) {
        is_missing = true;
        return std::numeric_limits<float>::quiet_NaN();
    }
    try {
        return std::stof(cell);
    } catch (...) {
        is_missing = true;
        return std::numeric_limits<float>::quiet_NaN();
    }
}

std::vector<std::string> CSVImporter::supported_missing_tokens() {
    return {"", "NA", "N/A", "null", "NULL", "NaN", "nan", "-", ".", "None"};
}

} // namespace clustering
