

#include "scratch/stensor.h"
#include "scratchgui/tools.h"

#include "imguiapp.h"


void sgd_init();
void sgd_step();
void sgd_step_epoch();
void sgd_fit(uint epochs);

class SgdApp : public App
{
public:
    bool OnDraw() override
    {
        bool alive = true;

        ImGui::Begin("SGD");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        if (ImGui::Button("Step Batch"))
            sgd_step(); 

        ImGui::SameLine();
        if (ImGui::Button("Step Epoch"))
            sgd_step_epoch();

        ImGui::SameLine();
        if (ImGui::Button("Fit"))
            sgd_fit(5);

        ImGui::SameLine();
        if (ImGui::Button("Quit"))
             alive = false;

        ImGui::End();

        DrawTensorLogs();

        return alive;
    }

    bool OnCreate() override
    {

        sgd_init();
        return true;
    }
};

// Main code
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{

    SgdApp app;
    app.Init(hInstance, hPrevInstance, lpCmdLine, nCmdShow);
    app.Run();
    app.Cleanup();

    return 0;
}
