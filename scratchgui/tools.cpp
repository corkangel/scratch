#include "scratchgui/tools.h"

#include "scratch/stensor.h"

#include "imgui.h"
#include "implot.h"

#include <algorithm>
#include <iomanip>

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
#include "scratch/slearner.h"

void DrawModel(const sLearner& learner, const sModel& model)
{
    ImGui::Begin("Model");

    ImGui::Text("Epoch: %u", learner._epoch);
    ImGui::Text("Batch: %u of %u", learner._batch, learner._nImages);
    ImGui::Text("Batch size: %u", learner._batchSize);
    ImGui::Text("Learning rate: %.5f", learner._lr);
    ImGui::Text("Loss: %.5f", model._loss);
    ImGui::Text("Accuracy: %.5f", model._accuracy);

    switch (learner._layerStepState)
    {
        case sLayerStepState::None:
            ImGui::Text("Layer step: None");
            break;

        case sLayerStepState::Forward: 
            ImGui::Text("Layer step: Forward"); 
            ImGui::Text("Layer: %u", learner._layerStepIndex);
            break;

        case sLayerStepState::Middle:
            ImGui::Text("Layer step: Middle"); 
            break;

        case sLayerStepState::Backward: 
            ImGui::Text("Layer step: Backward"); 
            ImGui::Text("Layer: %u", learner._layerStepIndex);
            break;

        case sLayerStepState::End: 
            ImGui::Text("Layer step: End"); 
            break;
    }

    if (ImGui::TreeNodeEx("Layers", ImGuiTreeNodeFlags_DefaultOpen))
    {
        uint i = 0;  
        for (auto&& module : model._layers)
        {
            const sLayer* layer = (sLayer*)module;

            bool color = false;
            if (learner._layerStepState == sLayerStepState::Forward &&  i == learner._layerStepIndex-1)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 0, 1));
                color = true;
            }

            if (learner._layerStepState == sLayerStepState::Backward && i == learner._layerStepIndex)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0, 0, 1));
                color = true;
            }

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 1.0f, 1.0f));
            std::string name = std::to_string(i++) + " " + layer->name();
            if (ImGui::TreeNodeEx(name.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::PopStyleColor();
                const pTensor a = layer->activations();
                ImGui::Text("Activations: [%d,%d,%d,%d] %u", a->dim_unsafe(0), a->dim_unsafe(1), a->dim_unsafe(2), a->dim_unsafe(3), a->size());
                if (!layer->_activationStats.max.empty())
                {
                    ImGui::Text("mean:%.2f std:%.2f min:%.2f max:%.2f", 
                        layer->_activationStats.mean.back(),
                        layer->_activationStats.std.back(),
                        layer->_activationStats.min.back(),
                        layer->_activationStats.max.back());
                }
                std::string activations = "";
                for (uint i = 0; i < std::min(8u, a->size()); i++)
                {
                    char buf[32];
                    sprintf_s(buf, "%.4f,", a->getAt(i));
                    activations += buf;
                }
                ImGui::Text("[%s]", activations.c_str());

                std::string aname = name + " activations";                 
                if (ImGui::TreeNode(aname.c_str()))
                {
                    for (uint i = 0; i < std::min(10u, a->size()); i++)
                    {
                        ImGui::Text("%.5f", a->getAt(i));
                    }
                    ImGui::TreePop();
                }

                std::string gname = name + " gradients";
                if (a->grad().isnull())
                {
                    ImGui::Text("No gradients");
                }
                else
                {
                    if (ImGui::TreeNode(gname.c_str()))
                    {
                        for (uint i = 0; i < std::min(10u, a->grad()->size()); i++)
                        {
                            ImGui::Text("%.5f", a->grad()->getAt(i));
                        }
                        ImGui::TreePop();
                    }
                }

                const std::map<std::string, pTensor>params = module->parameters();
                for (auto&& p : params)
                {
                    const sTensor& t = *p.second;
                    ImGui::Text("%s: %u mean:%.2f std:%.2f min:%.2f max:%.2f", 
                        p.first.c_str(), t.size(),
                        t.mean(), t.std(), t.min(), t.max());

                    ImGui::Text("%s: [%d,%d,%d,%d]", p.first.c_str(), t.dim_unsafe(0), t.dim_unsafe(1), t.dim_unsafe(2), t.dim_unsafe(3));
                    std::string vname = p.first + " values";
                    if (ImGui::TreeNode(vname.c_str()))
                    {
                        for (uint i = 0; i < std::min(10u, t.size()); i++)
                        {
                            ImGui::Text("%.5f", t.getAt(i));
                        }
                        ImGui::TreePop();
                    }
                    if (!t.grad().isnull())
                    {
                        std::string gname = p.first + " gradients";
                        if (ImGui::TreeNode(gname.c_str()))
                        {
                            for (uint i = 0; i < std::min(10u, t.size()); i++)
                            {
                                ImGui::Text("%.5f", t.grad()->getAt(i));
                            }
                            ImGui::TreePop();
                        }
                    }
                }

                ImGui::TreePop();
            }
            else
            {
                ImGui::PopStyleColor();
            }

            if (color)
            {
                ImGui::PopStyleColor();
            }
        }
        ImGui::TreePop();
    }
    ImGui::End();
}

void DrawActivationStats(const sLearner& learner, const sModel& model)
{
    
    ImGui::Begin("LinearStats");
    if (ImPlot::BeginPlot("Activations Mean")) {

        for (uint i = 0; i < 4; i++)
        {
            const std::vector<float>& means = ((sLayer*)model._layers[i])->_activationStats.mean;
            if (means.size() > 0)
            {
                std::vector<float> x(means.size());
                for (uint j = 0; j < uint(means.size()); j++)
                    x[j] = float(j);

                ImPlot::PlotLine(("Layer " + std::to_string(i)).c_str(), x.data(), means.data(), uint(means.size()));
            }
        }
        ImPlot::EndPlot();
    }
    ImGui::End();
}

bool DrawMenu(sLearner& learner, initFunc init)
{
    bool alive = true;

    ImGui::Begin("SGD");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    if (ImGui::Button("Restart"))
        init();

    ImGui::SameLine();
    if (ImGui::Button("Step Layer"))
        learner.step_layer();

    ImGui::SameLine();
    if (ImGui::Button("Step Batch"))
        learner.step();

    ImGui::SameLine();
    if (ImGui::Button("Step Epoch"))
        learner.step_epoch();

    ImGui::SameLine();
    if (ImGui::Button("Fit"))
        learner.fit(5);

    ImGui::SameLine();
    if (ImGui::Button("Quit"))
        alive = false;

    ImGui::End();

    return alive;
}



struct TreeData
{
    pTensor selected_tensor = nullptr;
};
TreeData tree_data;


void TensorName(const pTensor& t, char buf[256], const char* label)
{
    if (t.isnull())
    {
        snprintf(buf, 256, "null %s", label);
        return;
    }
    snprintf(buf, 256, "[%2u,%2u,%2u,%2u] %s", t->dim_unsafe(0), t->dim_unsafe(1), t->dim_unsafe(2), t->dim_unsafe(3), label);
}

void DrawTensor(const pTensor& t)
{
    if (t->rank() == 0)
    {
        ImGui::Text("null");
        return;
    }

    uint rowWidth = 1;
    for (uint i = 1; i < t->rank(); ++i)
    {
        rowWidth *= t->dim(i);
    }

    const float mean = t->mean();
    const float stddev = t->std();
    const float maxv = t->max();
    const float minv = t->min();

    ImGui::BeginChild("Tensor");
    ImGui::Text("Dims:[%u,%u,%u,%u] Size:[%u] RowWidth:[%u] Mean:[%3f] StdDev:[%3f] Range:[%3f,%3f]",
        t->dim_unsafe(0), t->dim_unsafe(1), t->dim_unsafe(2), t->dim_unsafe(3),
        t->size(),
        rowWidth, mean, stddev, minv, maxv);

    ImGui::Separator();

    ImGui::Text("Values:");
    const uint finalDim = t->dim(t->rank() - 1);
    ImGui::Text("    ");
    ImGui::SameLine();
    for (uint i = 0; i < rowWidth; ++i)
    {
        ImGui::Text("%+6u", i);
        ImGui::SameLine();
        if (i % finalDim == finalDim - 1)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(.5f, .5f, .5f, 1));
            ImGui::Text("|");
            ImGui::PopStyleColor();
            ImGui::SameLine();
        }
    }
    ImGui::Text("");

    const uint nrows = t->dim(0);
    for (uint row = 0; row < nrows; row++)
    {
        ImGui::Text("[%2d]", row);
        ImGui::SameLine();

        for (uint i = 0; i < rowWidth; ++i)
        {
            ImGui::Text("%+.3f", t->getAt(row * rowWidth + i));
            ImGui::SameLine();

            if (i % finalDim == finalDim - 1)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(.5f, .5f, .5f, 1));
                ImGui::Text("|");
                ImGui::PopStyleColor();
                ImGui::SameLine();
            }
        }
        ImGui::Text("");
    }
    ImGui::EndChild();
}

void DrawTreeNodeActivations(const sModel& model, const sLayer* layer)
{
    const pTensor& a = layer->activations();
    if (!a.isnull())
    {
        if (ImGui::Selectable("Activations"))
        {
            tree_data.selected_tensor = layer->activations();
        }
    }

    pTensor& grads = layer->activations()->grad();
    if (!grads.isnull())
    {
        if (ImGui::Selectable("Activations Gradients"))
        {
            tree_data.selected_tensor = layer->activations()->grad();
        }
    }
}

void DrawTreeParameters(const sModel& model, const sLayer* layer)
{
    const std::map<std::string, pTensor> params = layer->parameters();
    if (!params.empty())
    {
        if (ImGui::TreeNodeEx("Parameters"))
        {

            for (auto&& p : params)
            {
                const std::string& name = p.first;
                const sTensor& t = *p.second;
                if (ImGui::TreeNode(name.c_str()))
                {
                    char buf[256];

                    TensorName(p.second, buf, "Values");
                    if (ImGui::Selectable(buf))
                    {
                        tree_data.selected_tensor = p.second;
                    }

                    if (!p.second->grad().isnull())
                    {
                        TensorName(p.second, buf, "Gradients");
                        if (ImGui::Selectable(buf))
                        {
                            tree_data.selected_tensor = p.second->grad();
                        }
                    }

                    ImGui::TreePop();
                }
            }
            ImGui::TreePop();
        }
    }
}

void DrawSummary(const sModel& model)
{
    char buf[256];

    TensorName(model._cachedInput, buf, "Input");
    if (ImGui::Selectable(buf))
    {
        tree_data.selected_tensor = model._cachedInput;
    }

    TensorName(model._cachedOutput, buf, "Output");
    if (ImGui::Selectable(buf))
    {
        tree_data.selected_tensor = model._cachedOutput;
    }

    TensorName(model._cachedTarget, buf, "Target");
    if (ImGui::Selectable(buf))
    {
        tree_data.selected_tensor = model._cachedTarget;
    }
    ImGui::Text("Loss: %.5f", model._loss);
    ImGui::Text("Accuracy: %.5f", model._accuracy);
}

void DrawTreeLayers(const sModel& model)
{
    if (ImGui::TreeNodeEx("Layers", ImGuiTreeNodeFlags_DefaultOpen))
    {
        uint i = 0;
        for (auto&& module : model._layers)
        {
            const sLayer* layer = (sLayer*)module;
            const pTensor& t = layer->activations();

            char buf[256];
            snprintf(buf, sizeof(buf), "%u [%2u,%2u,%2u,%2u] %s", i, t->dim_unsafe(0), t->dim_unsafe(1), t->dim_unsafe(2), t->dim_unsafe(3), layer->name());
            if (ImGui::TreeNodeEx(buf))
            {
                DrawTreeNodeActivations(model, layer);
                DrawTreeParameters(model, layer);
                ImGui::TreePop();
            }
            i++;
        }
        ImGui::TreePop();
    }
}

void DrawTreeNodes(const sModel& model)
{
    DrawSummary(model);
    DrawTreeLayers(model);
}

void DrawTreeDetail(const sModel& model)
{
    if (!tree_data.selected_tensor.isnull())
    {
        DrawTensor(tree_data.selected_tensor);
    }
}

void DrawTree(const sModel& model)
{
    static bool node_open = false;
    static int selected_node = -1;
    static float property_value = 0.0f;

    ImGui::Begin("ModelTree");

    // Split the window into two columns
    ImGui::Columns(2, "_cols"); 

    DrawTreeNodes(model);

    // Move to the next column
    ImGui::NextColumn();

    DrawTreeDetail(model);

    // End of columns
    ImGui::Columns(1);

    ImGui::End();
}