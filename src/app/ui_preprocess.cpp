// ============================================================================
// ui_preprocess.cpp — Tab 0 panels: Import, Column Stats, Preprocessing.
// ============================================================================

#include "ui.h"
#include <imgui.h>
#include <cstring>

namespace clustering_app {

static char col_names_buf[4096] = {};
static char rename_buf[128] = {};
static int remove_row_idx = 0;

void render_import(AppState& g) {
    if (ImGui::CollapsingHeader("Import", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Header row", &g.has_header);
        if (ImGui::Button("Open CSV File", ImVec2(-1, 30))) open_csv(g);
        if (g.data_loaded) {
            ImGui::Text("Data: %zu rows x %zu cols", g.table.rows(), g.table.cols());
            if (ImGui::Button("Edit Column Names", ImVec2(-1, 0)))
                ImGui::OpenPopup("EditColumns");
            if (ImGui::BeginPopupModal("EditColumns", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                if (ImGui::IsWindowAppearing()) {
                    std::string names;
                    for (size_t j = 0; j < g.table.cols(); ++j) {
                        if (j > 0) names += ", ";
                        names += g.table.column_names()[j];
                    }
                    strncpy(col_names_buf, names.c_str(), sizeof(col_names_buf) - 1);
                }
                ImGui::Text("Comma-separated names:");
                ImGui::InputTextMultiline("##cn", col_names_buf, sizeof(col_names_buf), ImVec2(400, 60));
                if (ImGui::Button("Apply")) {
                    std::string s(col_names_buf);
                    std::vector<std::string> new_names;
                    size_t pos = 0;
                    while (pos < s.size()) {
                        size_t comma = s.find(',', pos);
                        std::string name = s.substr(pos, comma - pos);
                        while (!name.empty() && name.front() == ' ') name.erase(0, 1);
                        while (!name.empty() && name.back() == ' ') name.pop_back();
                        if (!name.empty()) new_names.push_back(name);
                        if (comma == std::string::npos) break;
                        pos = comma + 1;
                    }
                    if (!new_names.empty() && new_names.size() <= g.table.cols()) {
                        auto& cnames = g.table.column_names_mut();
                        for (size_t j = 0; j < g.table.cols(); ++j)
                            cnames[j] = j < new_names.size() ? new_names[j] : "Col_" + std::to_string(j);
                        ImGui::CloseCurrentPopup();
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
            }
        }
    }
}

void render_column_stats(AppState& g) {
    if (ImGui::CollapsingHeader("Column Stats", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (g.data_loaded && g.table.cols() > 0) {
            size_t ncols = g.table.cols();
            if (g.selected_col >= (int)ncols) g.selected_col = 0;
            std::string preview;
            for (size_t j = 0; j < ncols; ++j) { if (j > 0) preview += '\0'; preview += g.table.column_names()[j]; }
            ImGui::Combo("Column", &g.selected_col, preview.c_str());
            size_t col = (size_t)g.selected_col;
            const ColumnStats& s = g.stats.get(col);
            ImGui::Text("Count:   %zu", s.count);
            ImGui::Text("Missing: %zu (%.1f%%)", s.missing_count, s.count > 0 ? 100.0f * s.missing_count / s.count : 0.0f);
            ImGui::Text("Mean:    %.4f", s.mean);
            ImGui::Text("Median:  %.4f", s.median);
            ImGui::Text("Std Dev: %.4f", s.std_dev);
            ImGui::Text("Min:     %.4f", s.min_val);
            ImGui::Text("Max:     %.4f", s.max_val);
            ImGui::Text("Q1:      %.4f", s.q1);
            ImGui::Text("Q3:      %.4f", s.q3);
            auto hd = g.stats.sample_for_histogram(col);
            if (!hd.empty()) ImGui::PlotHistogram("##hist", hd.data(), hd.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(-1, 50));
            ImGui::Separator();
            if (ImGui::Button("Rename Column", ImVec2(-1, 0))) ImGui::OpenPopup("RenameColumn");
            if (ImGui::BeginPopupModal("RenameColumn", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                if (ImGui::IsWindowAppearing()) strncpy(rename_buf, g.table.column_names()[col].c_str(), sizeof(rename_buf) - 1);
                ImGui::Text("New name for '%s':", g.table.column_names()[col].c_str());
                ImGui::InputText("##newname", rename_buf, sizeof(rename_buf));
                if (ImGui::Button("Apply")) {
                    g.table.column_names_mut()[col] = std::string(rename_buf);
                    ImGui::CloseCurrentPopup();
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
            }
            if (ImGui::Button("Remove Column", ImVec2(-1, 0))) {
                g.table.remove_column(col);
                g.stats.invalidate();
                g.clustering_done = false;
                if (g.selected_col >= (int)g.table.cols()) g.selected_col = 0;
            }
        }
    }
}

void render_preprocessing(AppState& g) {
    if (ImGui::CollapsingHeader("Preprocessing")) {
        if (g.data_loaded) {
            static const char* ops[] = {"Normalize", "Standardize", "MinMax Scale", "Log Transform", "Clip Outliers"};
            ImGui::Combo("Operation", &g.preprocess_op, ops, IM_ARRAYSIZE(ops));
            ImGui::Checkbox("All columns", &g.apply_to_all_columns);
            if (ImGui::Button("Apply", ImVec2(-1, 22))) apply_preprocess(g);

            ImGui::Separator();
            ImGui::Text("Missing Data:");
            auto strats = MissingHandler::all_strategies();
            std::string sn;
            for (auto s : strats) { if (!sn.empty()) sn += '\0'; sn += MissingHandler::strategy_name(s); }
            ImGui::Combo("##mstrat", &g.missing_strategy_idx, sn.c_str());
            if (ImGui::Button("Fill Missing", ImVec2(-1, 22))) apply_missing(g);

            ImGui::Separator();
            if (ImGui::Button("Undo", ImVec2(55, 0))) { g.table.pipeline().undo(); g.stats.invalidate(); }
            ImGui::SameLine();
            if (ImGui::Button("Redo", ImVec2(55, 0))) { g.table.pipeline().redo(); g.stats.invalidate(); }
            ImGui::SameLine();
            if (ImGui::Button("Undo All", ImVec2(70, 0))) undo_all(g);
            ImGui::SameLine();
            ImGui::Text("(%zu/%zu)", g.table.pipeline().current(), g.table.pipeline().history().size());

            ImGui::Separator();
            ImGui::Text("Data Operations:");
            if (ImGui::Button("Drop Rows with Missing", ImVec2(-1, 0))) {
                g.table.drop_rows_with_missing();
                g.stats.invalidate();
                g.clustering_done = false;
            }
            ImGui::SetNextItemWidth(80);
            ImGui::InputInt("Row #", &remove_row_idx, 0, 0);
            ImGui::SameLine();
            if (ImGui::Button("Remove Row", ImVec2(-1, 0))) {
                if (remove_row_idx >= 0 && (size_t)remove_row_idx < g.table.rows()) {
                    g.table.remove_row((size_t)remove_row_idx);
                    g.stats.invalidate();
                    g.clustering_done = false;
                }
            }

            const auto& hist = g.table.pipeline().history();
            if (!hist.empty()) {
                ImGui::BeginChild("HistList", ImVec2(0, ImGui::GetTextLineHeight() * 9), ImGuiChildFlags_Borders);
                for (size_t i = 0; i < hist.size(); ++i) {
                    bool cur = (i == g.table.pipeline().current() - 1);
                    ImGui::TextColored(cur ? ImVec4(0.3f, 1, 0.3f, 1) : ImVec4(0.5f, 0.5f, 0.5f, 1),
                        "%s %s", cur ? ">" : " ", hist[i].description.c_str());
                }
                ImGui::EndChild();
            }
        }
    }
}

} // namespace clustering_app
