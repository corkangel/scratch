#include "scratchgui/tools.h"

#include "scratch/stensor.h"

#include "imgui.h"

#include <algorithm>

void DrawTensorLogs()
{
    ImGui::Begin("Tensor Logs");
    std::string buf;
    for (auto& s : get_logs())
    {
        buf += s + "\n\n";
    }
    ImGui::InputTextMultiline("Tensors", (char*)buf.c_str(), buf.size(), ImVec2(1000, 700), ImGuiInputTextFlags_ReadOnly);
}

void DrawTensorTable()
{
    ImGui::BeginTable("Tensors", 7, ImGuiTableFlags_Resizable);

    ImGui::TableSetupColumn("Id", ImGuiTableColumnFlags_WidthFixed, 50);
    ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100);
    ImGui::TableSetupColumn("Operation", ImGuiTableColumnFlags_WidthFixed, 100);
    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 100);
    ImGui::TableSetupColumn("Dims", ImGuiTableColumnFlags_WidthFixed, 150);
    ImGui::TableSetupColumn("Front", ImGuiTableColumnFlags_WidthFixed, 400);
    ImGui::TableSetupColumn("Back", ImGuiTableColumnFlags_WidthFixed, 400);

    ImGui::TableHeadersRow();

    const std::vector<sTensorInfo>& infos = get_tensor_infos();
    for (auto&& info : infos)
    {

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%u", info.id);

        ImGui::TableNextColumn();
        ImGui::Text("%s", info.label ? info.label : "_");

        ImGui::TableNextColumn();
        ImGui::Text("%s", info.operation ? info.operation : "_");

        ImGui::TableNextColumn();
        ImGui::Text("%u", info.time);

        ImGui::TableNextColumn();
        std::stringstream ss;
        uint n = 1;
        for (uint i = 0; i < info.rank; ++i)
        {
            ss << info.dimensions[i];
            if (i != info.rank - 1) ss << ", ";
            n *= info.dimensions[i];
        }
        ImGui::Text("[ %s ]", ss.str().c_str());

        {
            ImGui::TableNextColumn();
            std::stringstream fs;
            for (uint i = 0; i < std::min(sInfoDataSize, n); i++)
            {
                fs << std::fixed << std::setprecision(2);
                fs << info.data_front[i];
                if (i != n - 1) fs << ", ";
            }
            ImGui::Text("[%s]", fs.str().c_str());
        }

        {
            ImGui::TableNextColumn();
            std::stringstream bs;
            for (uint i = 0; i < std::min(sInfoDataSize, n); i++)
            {
                bs << std::fixed << std::setprecision(2);
                bs << info.data_back[sInfoDataSize - i - 1];
                if (i != n - 1) bs << ", ";
            }
            ImGui::Text("[%s]", bs.str().c_str());
        }
    }
    ImGui::EndTable();

    ImGui::End();
}
