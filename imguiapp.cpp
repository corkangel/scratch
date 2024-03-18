

#include "stensor.h"

#include "imguidemo.h"
#include "implot.h"

class MyApp : public App
{
public:
    bool OnDraw() override
    {

        if (count++ < 5)
        {
            void meanshift_step();
            meanshift_step();
        }
        bool alive = true;

        ImGui::Begin("Scratch");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        if (ImGui::Button("Quit"))
             alive = false;

        ImGui::End();

        ImGui::Begin("Tensor Logs");
        std::string buf;
        for (auto& s : get_logs())
        {
            buf += s + "\n";
        }
        ImGui::Text(buf.c_str());
        ImGui::End();



        int   bar_data[11] = { 1,2,3,4,5,6,7,8,8,5 };
        float x_data[4] = { .1f, .2f, .3f, .4f };
        float y_data[4] = { .8f, .7f, .6f, .3f };

        ImGui::Begin("My Window");
        if (ImPlot::BeginPlot("My Plot")) {
            ImPlot::PlotBars("My Bar Plot", bar_data, 11);
            ImPlot::PlotLine("My Line Plot", x_data, y_data, 4);
            ImPlot::EndPlot();
        }
        ImGui::End();

        return alive;
    }

    bool OnCreate() override
    {
        void meanshift_init();
        meanshift_init();

        //void test_tensors();
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
