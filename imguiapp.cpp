

#include "stensor.h"

#include "imguidemo.h"
#include "implot.h"

sTensor& meanshift_centroids();
sTensor& meanshift_samples();

void meanshift_step();
void meanshift_iterate_rows();
void meanshift_init();
void test_tensors();

class MyApp : public App
{
public:
    bool OnDraw() override
    {
        bool alive = true;

        ImGui::Begin("Scratch");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        if (ImGui::Button("Quit"))
             alive = false;

        if (ImGui::Button("Meanshift step"))
        {
            meanshift_step();

            //meanshift_iterate_rows();

        }

        ImGui::End();

        ImGui::Begin("Tensor Logs");
        //std::string buf;
        //for (auto& s : get_logs())
        //{
        //    buf += s + "\n\n";
        //}
        //ImGui::InputTextMultiline("Tensors", (char*)buf.c_str(), buf.size(), ImVec2(1000, 700), ImGuiInputTextFlags_ReadOnly);

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
                for (uint i = 0; i < min(sInfoDataSize, n); i++)
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
                for (uint i = 0; i < min(sInfoDataSize, n); i++)
                {
                    bs << std::fixed << std::setprecision(2);
                    bs << info.data_back[sInfoDataSize - i - 1];
                    if (i != n - 1) bs << ", ";
                }
                ImGui::Text("[%s]", bs.str().c_str());
            }

            //ImGui::TableNextColumn();
            //std::stringstream bs;
            //ImGui::Text("Back:[ %s ]", bs.str().c_str());


        }
        ImGui::EndTable();

        ImGui::End();

        sTensor centroids_x = meanshift_centroids().column(0);
        sTensor centroids_y = meanshift_centroids().column(1);

        sTensor samples_x = meanshift_samples().column(0);
        sTensor samples_y = meanshift_samples().column(1);

        ImGui::Begin("MeanShift");
        if (ImPlot::BeginPlot("MeanShift")) {
            ImPlot::PlotScatter("Samples", samples_x.data(), samples_y.data(), 1500);
            ImPlot::PlotScatter("Centroids", centroids_x.data(), centroids_y.data(), 6);
            ImPlot::EndPlot();
        }
        ImGui::End();

        return alive;
    }

    bool OnCreate() override
    {
        meanshift_init();

        //meanshift_iterate_rows();

        //test_tensors();

        return true;
    }

    int count = 0;
};

// Main code
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{

    MyApp app;
    app.Init(hInstance, hPrevInstance, lpCmdLine, nCmdShow);
    app.Run();
    app.Cleanup();

    return 0;
}
