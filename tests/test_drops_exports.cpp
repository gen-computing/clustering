// ============================================================================
// test_drops_exports.cpp -- DataTable drop/undo/redo via PreprocessPipeline,
// and the pure-CSV export functions (labels, centroids, preprocessed, report).
// ============================================================================

#include <gtest/gtest.h>
#include "gui/data_table.h"
#include "gui/exports.h"
#include "gui/missing_handler.h"
#include "gui/preprocess_pipeline.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace clustering;

namespace {

Matrix make_data(size_t rows, size_t cols) {
    Matrix m(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            m[i][j] = (float)(i * 10 + j);
    return m;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string temp_path(const char* suffix) {
    static int n = 0;
    return std::string("/tmp/opencode/export_test_") + std::to_string(n++) + suffix;
}

void cleanup(const std::string& p) { std::remove(p.c_str()); }

} // namespace

// ---------------------------------------------------------------------------
// Drop + undo/redo
// ---------------------------------------------------------------------------

TEST(PipelineDrops, DropRowUndoRedo) {
    DataTable t;
    t.set_data(make_data(5, 2), {"a", "b"});
    EXPECT_EQ(t.rows(), 5u);

    t.pipeline().drop_row(2);
    EXPECT_EQ(t.rows(), 4u);
    EXPECT_FLOAT_EQ(t.data()[2][0], 30.0f);
    EXPECT_FLOAT_EQ(t.data()[2][1], 31.0f);

    EXPECT_TRUE(t.pipeline().undo());
    EXPECT_EQ(t.rows(), 5u);
    EXPECT_FLOAT_EQ(t.data()[2][0], 20.0f);
    EXPECT_FLOAT_EQ(t.data()[2][1], 21.0f);
    EXPECT_FLOAT_EQ(t.data()[3][0], 30.0f);

    EXPECT_TRUE(t.pipeline().redo());
    EXPECT_EQ(t.rows(), 4u);
    EXPECT_FLOAT_EQ(t.data()[2][0], 30.0f);
}

TEST(PipelineDrops, DropColumnUndoRedo) {
    DataTable t;
    t.set_data(make_data(4, 3), {"a", "b", "c"});
    EXPECT_EQ(t.cols(), 3u);

    t.pipeline().drop_column(1);
    EXPECT_EQ(t.cols(), 2u);
    EXPECT_EQ(t.column_names()[1], "c");
    EXPECT_FLOAT_EQ(t.data()[2][1], 22.0f);

    EXPECT_TRUE(t.pipeline().undo());
    EXPECT_EQ(t.cols(), 3u);
    EXPECT_EQ(t.column_names()[1], "b");
    EXPECT_FLOAT_EQ(t.data()[2][1], 21.0f);
    EXPECT_FLOAT_EQ(t.data()[3][0], 30.0f);
    EXPECT_FLOAT_EQ(t.data()[3][2], 32.0f);

    EXPECT_TRUE(t.pipeline().redo());
    EXPECT_EQ(t.cols(), 2u);
    EXPECT_EQ(t.column_names()[1], "c");
}

TEST(PipelineDrops, DropColumnKeepsMissingFlags) {
    DataTable t;
    t.set_data(make_data(3, 2), {"a", "b"});
    t.set_missing(1, 1, true);
    t.pipeline().drop_column(1);
    EXPECT_TRUE(t.pipeline().undo());
    EXPECT_EQ(t.cols(), 2u);
    EXPECT_TRUE(t.is_missing(1, 1));
    EXPECT_FLOAT_EQ(t.data()[0][1], 1.0f);
}

TEST(PipelineDrops, DropRowsWithMissingUndoRedo) {
    DataTable t;
    t.set_data(make_data(5, 2), {"a", "b"});
    t.set_missing(1, 0, true);
    t.set_missing(3, 1, true);
    t.pipeline().drop_rows_with_missing();
    EXPECT_EQ(t.rows(), 3u);
    EXPECT_FLOAT_EQ(t.data()[1][0], 20.0f);
    EXPECT_FLOAT_EQ(t.data()[2][0], 40.0f);

    EXPECT_TRUE(t.pipeline().undo());
    EXPECT_EQ(t.rows(), 5u);
    EXPECT_FLOAT_EQ(t.data()[1][1], 11.0f);
    EXPECT_FLOAT_EQ(t.data()[3][0], 30.0f);

    EXPECT_TRUE(t.pipeline().redo());
    EXPECT_EQ(t.rows(), 3u);
}

TEST(PipelineDrops, UndoClearsRedoStack) {
    DataTable t;
    t.set_data(make_data(4, 2));
    t.pipeline().drop_row(0);
    t.pipeline().drop_row(1);
    EXPECT_EQ(t.rows(), 2u);
    t.pipeline().undo();
    EXPECT_EQ(t.rows(), 3u);
    t.pipeline().drop_row(0);
    EXPECT_EQ(t.rows(), 2u);
    EXPECT_FALSE(t.pipeline().can_redo());
}

TEST(PipelineDrops, UndoRestoresPreprocessedCellsThenDrops) {
    DataTable t;
    t.set_data(make_data(4, 2), {"a", "b"});
    t.pipeline().standardize_column(0);
    float v0 = t.data()[0][0];
    t.pipeline().drop_row(2);
    EXPECT_EQ(t.rows(), 3u);
    EXPECT_TRUE(t.pipeline().undo());
    EXPECT_EQ(t.rows(), 4u);
    EXPECT_FLOAT_EQ(t.data()[0][0], v0);
    EXPECT_TRUE(t.pipeline().undo());
    EXPECT_FLOAT_EQ(t.data()[0][0], 0.0f);
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

TEST(Exports, LabelsCsv) {
    DataTable t;
    t.set_data(make_data(3, 2));
    Vector labels(3);
    labels[0] = 0; labels[1] = 1; labels[2] = 0;

    std::string p = temp_path(".csv");
    ASSERT_TRUE(export_labels_csv(t, labels, p));
    std::string out = read_file(p);
    EXPECT_EQ(out, "id,cluster\n0,0\n1,1\n2,0\n");
    cleanup(p);
}

TEST(Exports, CentroidsCsv) {
    Matrix c = make_data(2, 2);
    std::string p = temp_path(".csv");
    ASSERT_TRUE(export_centroids_csv(c, p));
    std::string out = read_file(p);
    EXPECT_EQ(out.substr(0, 6), "f0,f1\n");
    EXPECT_NE(out.find("0.000000,1.000000"), std::string::npos);
    EXPECT_NE(out.find("10.000000,11.000000"), std::string::npos);
    cleanup(p);
}

TEST(Exports, PreprocessedCsvWithMissing) {
    DataTable t;
    t.set_data(make_data(2, 2), {"x", "y"});
    t.set_missing(0, 1, true);
    std::string p = temp_path(".csv");
    ASSERT_TRUE(export_preprocessed_csv(t, p));
    std::string out = read_file(p);
    EXPECT_EQ(out, "x,y\n0.000000,\n10.000000,11.000000\n");
    cleanup(p);
}

TEST(Exports, Report) {
    DataTable t;
    t.set_data(make_data(10, 3), {"a", "b", "c"});
    std::vector<int> sel = {1, 0, 1};
    std::string p = temp_path(".txt");
    ASSERT_TRUE(export_report(t, sel, 0, 3, 78.5f, 7, 0.6f, 0.4f, true, p));
    std::string out = read_file(p);
    EXPECT_NE(out.find("Dataset: 10 rows x 3 cols"), std::string::npos);
    EXPECT_NE(out.find("Selected columns: a c"), std::string::npos);
    EXPECT_NE(out.find("Algorithm: 0  k: 3"), std::string::npos);
    EXPECT_NE(out.find("Inertia: 78.5000"), std::string::npos);
    cleanup(p);
}

TEST(Exports, ReportNoClustering) {
    DataTable t;
    t.set_data(make_data(2, 2));
    std::string p = temp_path(".txt");
    ASSERT_TRUE(export_report(t, {1, 1}, 0, 3, 0, 0, 0, 0, false, p));
    std::string out = read_file(p);
    EXPECT_EQ(out.find("Algorithm"), std::string::npos);
    cleanup(p);
}

TEST(Exports, MissingHandlerDropRowsUndo) {
    DataTable t;
    t.set_data(make_data(4, 2));
    t.set_missing(2, 0, true);
    // MissingHandler::drop_rows records original_rows as post-drop row count;
    // verify pipeline undo restores the removed row with values intact.
    MissingHandler mh;
    MissingHandlerConfig cfg;
    cfg.strategy = MissingStrategy::DropRows;
    mh.set_config(cfg);
    mh.apply(t);
    EXPECT_EQ(t.rows(), 3u);
    EXPECT_TRUE(t.pipeline().undo());
    EXPECT_EQ(t.rows(), 4u);
    EXPECT_FLOAT_EQ(t.data()[2][1], 21.0f);
    EXPECT_TRUE(t.pipeline().redo());
    EXPECT_EQ(t.rows(), 3u);
}
