

#include "scratch/stensor.h"
#include "scratchgui/tools.h"

#include "imguiapp.h"


void sgd_init();

class SgdApp : public App
{
public:
    bool OnDraw() override
    {
        bool alive = true;

        ImGui::Begin("SGD");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

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
