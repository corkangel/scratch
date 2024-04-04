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
    ImGui::End();
}

void DrawTensorTable()
{
    ImGui::Begin("Tensor Table");
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

#include "scratch/smodel.h" 

void DrawModel(const sModel& model)
{
    ImGui::Begin("Model");

    if (ImGui::TreeNode("Layers"))
    {
        uint i = 0;
        for (auto&& module : model._layers)
        {
            const sLayer* layer = (sLayer*)module;

            std::string name = std::to_string(i++) + " " + layer->name();
            if (ImGui::TreeNode(name.c_str()))
            {
                const sTensor& a = layer->activations();
                ImGui::Text("Activations: [%d,%d,%d,%d] %u", a.dim_unsafe(0), a.dim_unsafe(1), a.dim_unsafe(2), a.dim_unsafe(3), a.size());
                if (!layer->_activationStats.max.empty())
                {
                    ImGui::Text("mean:%.2f std:%.2f min:%.2f max:%.2f", 
                        layer->_activationStats.mean.back(),
                        layer->_activationStats.std.back(),
                        layer->_activationStats.min.back(),
                        layer->_activationStats.max.back());
                }

                std::map<std::string, const sTensor*>params = module->parameters();
                for (auto&& p : params)
                {
                    const sTensor& t = *p.second;
                    ImGui::Text("%s: %u mean:%.2f std:%.2f min:%.2f max:%.2f", 
                        p.first.c_str(), t.size(),
                        t.mean(), t.std(), t.min(), t.max());

                    std::string vname = p.first + " values";
                    if (ImGui::TreeNode(vname.c_str()))
                    {
                        for (uint i = 0; i < std::min(10u, t.size()); i++)
                        {
                            ImGui::Text("%.2f", t.getAt(i));
                        }
                        ImGui::TreePop();
                    }
                    if (t.grad())
                    {
                        std::string gname = p.first + " gradients";
                        if (ImGui::TreeNode(gname.c_str()))
                        {
                            for (uint i = 0; i < std::min(10u, t.size()); i++)
                            {
                                ImGui::Text("%.2f", t.grad()->getAt(i));
                            }
                            ImGui::TreePop();
                        }
                    }
                }

                ImGui::TreePop();
            }
        }
        ImGui::TreePop();
    }
    ImGui::End();
}