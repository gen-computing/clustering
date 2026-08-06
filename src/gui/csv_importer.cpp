#include "csv_importer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace clustering {

// Excel-style column letters: 0=A, 25=Z, 26=AA, 27=AB, ...
static std::string col_letter(size_t n) {
    std::string s;
    while (true) {
        s.insert(s.begin(), char('A' + n % 26));
        if (n < 26) break;
        n = n / 26 - 1;
    }
    return s;
}

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
    std::vector<std::string> col_names;
    std::vector<bool> numeric_cols;
    std::vector<size_t> nan_count;
    size_t d = 0;
    bool first = true;           // no data row parsed yet
    bool header_read = false;    // header line already consumed
    size_t loaded = 0;
    Matrix mat;

    while (std::getline(file, line)) {
        if (cancel_) { result.error = "Cancelled"; loading_ = false; return result; }
        if (max_rows > 0 && (int)loaded >= max_rows) break;

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

        // Skip blank lines (trailing newline, empty rows).
        if (cells.empty() || (cells.size() == 1 && cells[0].empty())) continue;

        if (first && has_header && !header_read) {
            col_names = cells;
            header_read = true;
            continue;
        }

        if (first) {
            // First data row: fix the column count and allocate the matrix.
            // Large matrices automatically spill to disk (RAM cap).
            d = cells.size();
            numeric_cols.assign(d, true);
            nan_count.assign(d, 0);
            col_names.clear();
            for (size_t j = 0; j < d; ++j)
                col_names.push_back(col_letter(j));
            size_t est = total > 0 ? total - (has_header ? 1 : 0) : 0;
            if (est == 0) est = 1;  // count_lines may miss a trailing newline
            if (max_rows > 0) est = std::min(est, (size_t)max_rows);
            mat.resize(est, d);
            first = false;
        }

        // Parse this row directly into the matrix (streaming, no RAM copy).
        detect_types(cells, numeric_cols);
        float* dst = mat[loaded];
        for (size_t j = 0; j < d; ++j) {
            bool miss = false;
            float val = 0.0f;
            if (j < cells.size() && numeric_cols[j]) {
                val = parse_cell(cells[j], miss);
            } else {
                miss = true;
                val = std::numeric_limits<float>::quiet_NaN();
            }
            if (miss) {
                val = std::numeric_limits<float>::quiet_NaN();
                nan_count[j]++;
            }
            dst[j] = val;
        }
        loaded++;

        result.loaded_rows = loaded;
        if (progress) progress(loaded, total);
    }

    loading_ = false;
    progress_ = 1.0f;

    if (loaded == 0 || d == 0) {
        result.error = "No data rows found";
        return result;
    }

    // Trim to the actual row count (estimated size may overshoot).
    if (loaded < mat.rows()) mat.resize(loaded, d);

    // Drop columns where most values are NaN (text columns like labels/timestamps)
    // Threshold: >90% NaN → drop column
    std::vector<size_t> keep_cols;
    std::vector<std::string> keep_names;
    for (size_t j = 0; j < d; ++j) {
        if (nan_count[j] * 10 < loaded * 9) {  // keep if <90% NaN
            keep_cols.push_back(j);
            if (j < col_names.size()) keep_names.push_back(col_names[j]);
        }
    }

    if (keep_cols.size() < d) {
        size_t new_d = keep_cols.size();
        Matrix filtered(loaded, new_d);
        for (size_t i = 0; i < loaded; ++i)
            for (size_t j = 0; j < new_d; ++j)
                filtered[i][j] = mat[i][keep_cols[j]];
        mat = std::move(filtered);
        col_names = std::move(keep_names);
    }

    result.table.set_data(std::move(mat), col_names);
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
