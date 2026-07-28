// ============================================================================
// VersionManager implementation -- Save/load clustering results to disk.
//
// Allows you to save centroids and labels after training, so you can later
// load them back, compare versions, or rollback to a previous model.
//
// STORAGE FORMAT (binary):
//   Each version is stored as a .bin file with this structure:
//     [k: size_t]    -- number of clusters (8 bytes)
//     [d: size_t]    -- number of features (8 bytes)
//     [n: size_t]    -- number of labels (8 bytes)
//     [centroids: k*d*4 bytes]  -- centroid coordinates (float = 4 bytes each)
//     [labels: n*4 bytes]       -- cluster assignment per point
//
// Total file size = 24 + 4*(k*d + n) bytes.
//
// Binary format chosen over text (JSON, CSV) because:
//   - Faster read/write (no parsing overhead)
//   - Smaller file size (float is 4 bytes vs ~8 chars in text)
//   - Exact representation (no precision loss from text conversion)
//   - Simpler code (just dump raw bytes)
// ============================================================================

#include "clustering/versioning.h"
#include <fstream>        // std::ifstream, std::ofstream for file I/O
#include <filesystem>     // std::filesystem for directory operations
#include <algorithm>      // std::sort

namespace clustering {

// Default constructor and destructor use compiler defaults.
// (no special initialization needed -- set_storage_path() does the real setup.)
VersionManager::VersionManager() = default;
VersionManager::~VersionManager() = default;

// ============================================================================
// set_storage_path() -- CONFIGURE WHERE VERSIONS ARE STORED
//
// 1. Saves the path for future save/load calls.
// 2. Creates the directory if it doesn't exist.
//    (std::filesystem::create_directories creates ALL parent directories as needed,
//     like `mkdir -p` on Linux.)
// 3. Scans the directory for existing .bin files and populates the version list.
//    ".bin" extension marks our version files.
//    The filename (without extension) is parsed as the version ID number.
//    Example: "5.bin" -> version 5.
//    Files without numeric names or .bin extension are ignored gracefully.
// ============================================================================

void VersionManager::set_storage_path(const std::string& path) {
    storage_path_ = path;

    // Create directories if they don't exist.
    // create_directories returns true if directories were created.
    std::filesystem::create_directories(path);

    // Clear the existing version list and rebuild from disk.
    versions_.clear();

    // Iterate over all files in the storage directory.
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        // Only process .bin files (our version file format).
        if (entry.path().extension() == ".bin") {
            // Extract the filename WITHOUT extension (stem).
            // entry.path().stem() on "5.bin" returns "5".
            // entry.path().stem() on "versions/3.bin" returns "3".
            try {
                // Parse the filename as an unsigned long long integer.
                size_t id = std::stoull(entry.path().stem().string());
                versions_.push_back(id);
            } catch (...) {
                // Skip files with non-numeric names (e.g., "metadata.bin").
                // Silently ignore: we don't want to crash because of stale files.
            }
        }
    }

    // Sort version IDs in ascending order for list_versions().
    // std::sort uses O(n log n) comparisons.
    std::sort(versions_.begin(), versions_.end());
}

// ============================================================================
// save() -- WRITE CENTROIDS AND LABELS TO DISK
//
// Creates a new .bin file with an auto-incremented version ID.
//
// File format (binary, little-endian on x86/ARM):
//   [8 bytes: k] [8 bytes: d] [8 bytes: n]
//   [k*d*4 bytes: centroids data (row-major)]
//   [n*4 bytes: labels data]
//
// Returns: The new version ID (1, 2, 3, ...).
// Returns 0 if the file couldn't be opened (disk full, permission denied, etc).
// ============================================================================

size_t VersionManager::save(const Matrix& centroids, const Vector& labels) {
    // Generate the next version ID.
    // If no versions exist, start at 1. Otherwise, increment the latest.
    size_t id = versions_.empty() ? 1 : versions_.back() + 1;

    // Build the full file path: e.g., "./versions/3.bin"
    // std::to_string converts the number to a string.
    std::string filename = storage_path_ + "/" + std::to_string(id) + ".bin";

    // Open file in binary write mode.
    // std::ios::binary prevents text-mode newline translation (critical on Windows).
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return 0;  // Couldn't open file (permissions, disk full, path invalid)
    }

    // ---- Write metadata (3 size_t values = 24 bytes total) ----
    size_t k = centroids.rows();  // Number of clusters
    size_t d = centroids.cols();  // Number of features
    size_t n = labels.size();     // Number of data points

    // reinterpret_cast<const char*> treats our size_t variables as a stream of bytes.
    // sizeof(k) tells write() how many bytes to output (8 on 64-bit systems).
    // file.write() copies raw binary data directly to the file.
    file.write(reinterpret_cast<const char*>(&k), sizeof(k));
    file.write(reinterpret_cast<const char*>(&d), sizeof(d));
    file.write(reinterpret_cast<const char*>(&n), sizeof(n));

    // ---- Write centroid data (k * d floats = k * d * 4 bytes) ----
    // centroids.data() gives pointer to start of the flat float array.
    // The centroids are stored row-major: row 0 first, then row 1, etc.
    file.write(reinterpret_cast<const char*>(centroids.data()), k * d * sizeof(float));

    // ---- Write label data (n floats = n * 4 bytes) ----
    file.write(reinterpret_cast<const char*>(labels.data()), n * sizeof(float));

    file.close();

    // Add this version ID to our in-memory list.
    versions_.push_back(id);

    return id;
}

// ============================================================================
// load() -- READ A VERSION FROM DISK
//
// Reads the .bin file and fills the provided Matrix and Vector.
//
// Returns: true if successful, false if file not found or could not be read.
//
// The caller provides EMPTY Matrix and Vector objects -- this method resizes
// them based on the metadata in the file, then fills them with data.
// ============================================================================

bool VersionManager::load(size_t version_id, Matrix& centroids, Vector& labels) const {
    // Build file path: e.g., "./versions/5.bin"
    std::string filename = storage_path_ + "/" + std::to_string(version_id) + ".bin";

    // Open file in binary read mode.
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;  // File doesn't exist or can't be opened
    }

    // ---- Read metadata (3 size_t values = 24 bytes) ----
    size_t k, d, n;

    // file.read() fills the provided buffer with raw binary data.
    file.read(reinterpret_cast<char*>(&k), sizeof(k));
    file.read(reinterpret_cast<char*>(&d), sizeof(d));
    file.read(reinterpret_cast<char*>(&n), sizeof(n));

    // Validate metadata to prevent OOM on corrupted files
    if (k > 100000 || d > 100000 || n > 100000000) {
        return false;  // Suspiciously large values -- likely corrupt file
    }
    if (k == 0 || d == 0) {
        return false;  // Invalid dimensions
    }

    // ---- Allocate space based on metadata ----
    centroids.resize(k, d);  // k rows, d columns
    labels.resize(n);         // n elements

    // ---- Read centroid data (k * d * 4 bytes) ----
    // Read directly into the centroids Matrix's internal buffer.
    // centroids.data() returns a pointer to the start of the data vector.
    file.read(reinterpret_cast<char*>(centroids.data()), k * d * sizeof(float));

    // ---- Read label data (n * 4 bytes) ----
    file.read(reinterpret_cast<char*>(labels.data()), n * sizeof(float));

    file.close();
    return true;
}

// ============================================================================
// list_versions() -- GET ALL SAVED VERSION IDs
//
// Returns a vector of version IDs in ascending order.
// Example: [1, 2, 3, 5] (version 4 might have been deleted).
// ============================================================================

std::vector<size_t> VersionManager::list_versions() const {
    return versions_;
}

// ============================================================================
// latest_version() -- GET THE HIGHEST VERSION ID
//
// Returns 0 if no versions have been saved yet.
// ============================================================================

size_t VersionManager::latest_version() const {
    return versions_.empty() ? 0 : versions_.back();
}

// ============================================================================
// has_versions() -- CHECK IF ANY VERSIONS EXIST
// ============================================================================

bool VersionManager::has_versions() const {
    return !versions_.empty();
}

} // namespace clustering
