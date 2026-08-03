// ============================================================================
// ui_table.cpp — Tab 0: Sortable data table.
// ============================================================================

#include "ui.h"
#include <imgui.h>
#include <algorithm>

namespace clustering_app {

void render_data_table(AppState& g) {
    float table_h_frac = g.show_viewport ? 0.5f : 1.0f;
    ImGui::BeginChild("TableChild", ImVec2(0, -ImGui::GetContentRegionAvail().y * (1.0f - table_h_frac)));

    if (g.data_loaded && g.table.rows() > 0) {
        size_t n = g.table.rows(), d = g.table.cols();
        if (ImGui::BeginTable("datatable", (int)d + 1,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX |
            ImGuiTableFlags_ScrollY | ImGuiTableFlags_Sortable | ImGuiTableFlags_Resizable)) {

            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 45);
            for (size_t j = 0; j < d; ++j)
                ImGui::TableSetupColumn(g.table.column_names()[j].c_str());
            ImGui::TableHeadersRow();

            // Sort
            if (ImGuiTableSortSpecs* sorts = ImGui::TableGetSortSpecs()) {
                if (sorts->SpecsDirty) {
                    g.sort_col = sorts->Specs->ColumnUserID;
                    g.sort_ascending = sorts->Specs->SortDirection == ImGuiSortDirection_Ascending;
                    if (g.sort_col >= 0) {
                        g.sorted_indices.resize(n);
                        for (size_t i = 0; i < n; ++i) g.sorted_indices[i] = i;
                        int scol = g.sort_col - 1;
                        if (scol >= 0 && (size_t)scol < d)
                            std::sort(g.sorted_indices.begin(), g.sorted_indices.end(),
                                [&](size_t a, size_t b) {
                                    float va = g.table.data()[a][scol], vb = g.table.data()[b][scol];
                                    return g.sort_ascending ? va < vb : va > vb;
                                });
                    }
                    sorts->SpecsDirty = false;
                }
            }

            // Virtualized rows
            ImGuiListClipper clipper;
            clipper.Begin((int)n);
            while (clipper.Step()) {
                for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                    size_t ri = g.sorted_indices.empty() ? (size_t)i : g.sorted_indices[i];
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", i + 1);
                    for (size_t j = 0; j < d; ++j) {
                        ImGui::TableNextColumn();
                        if (g.table.is_missing(ri, j))
                            ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "NaN");
                        else
                            ImGui::Text("%.4f", g.table.data()[ri][j]);
                    }
                }
            }
            ImGui::EndTable();
        }
    } else {
        ImGui::TextDisabled("Open a CSV file to view data (File > Open CSV)");
    }

    ImGui::EndChild();
}

} // namespace clustering_app
