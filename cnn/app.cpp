

#include "scratch/stensor.h"
#include "scratch/smodel.h"
#include "scratch/slearner.h"
#include "scratchgui/tools.h"

#include "imguiapp.h"

void cnn_init();
const std::vector<float> cnn_activation_means(const uint layer);
const sModel& cnn_model();
sLearner& cnn_learner();
const float* cnn_images_train();

class CnnApp : public App
{
public:
    bool OnDraw() override
    {
        bool alive = DrawMenu(cnn_learner(), cnn_init);

        ImGui::Begin("images");
        ImVec2 imageSize(64, 64); // Size of the image
        for (uint i = 0; i < 10; i++)
        {
            ImGui::Image((void*)(intptr_t)textures[i], imageSize);
        }
        ImGui::Image((void*)(intptr_t)texture, imageSize);
        ImGui::End();

        DrawTensorTable();

        DrawModel(cnn_learner(), cnn_model());

        return alive;
    }

    bool OnCreate() override
    {
        texture  = CreateTexture2DFromImageFile(GetDevice(), "test.png");

        cnn_init();

        ImVec2 imageSize(28, 28); // Size of the image
        for (uint i = 0; i < 10; i++)
        {
            textures[i] = CreateTexture2DFromMinst(GetDevice(), cnn_images_train() + i * 28 * 28);
        }

        return true;
    }

    ID3D11ShaderResourceView* texture = nullptr;
    ID3D11ShaderResourceView* textures[10];
};

// Main code
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    CnnApp app;
    app.Init(hInstance, hPrevInstance, lpCmdLine, nCmdShow);
    app.Run();
    app.Cleanup();

    return 0;
}
