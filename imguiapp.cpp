

#include "stensor.h"

#include "imguidemo.h"

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

        ImGui::End();

        ImGui::Begin("Tensor Logs");
        std::string buf;
        for (auto& s : get_logs())
        {
            buf += s + "\n";
        }
        ImGui::Text(buf.c_str());
        ImGui::End();

        return alive;
    }

    bool OnCreate() override
    {
        void meanshift();
        meanshift();

        //void test_tensors();
        //test_tensors();

        return true;
    }
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
