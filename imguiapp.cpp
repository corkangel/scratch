

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
        std::string buf;
        for (auto& s : get_logs())
        {
            buf += s + "\n\n";
        }
        ImGui::InputTextMultiline("Tensors", (char*)buf.c_str(), buf.size(), ImVec2(1000, 700), ImGuiInputTextFlags_ReadOnly);
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
