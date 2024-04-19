

#include "scratch/stensor.h"
#include "scratch/smodel.h"
#include "scratch/slearner.h"
#include "scratchgui/tools.h"

#include "imguiapp.h"

void cnn_init();
const sModel& cnn_model();
sLearner& cnn_learner();
const float* cnn_images_train(const uint index);
const float* cnn_edge1();
const float* cnn_edge2();

class CnnApp : public App
{
public:
    bool OnDraw() override
    {
        bool alive = DrawMenu(cnn_learner(), cnn_init);

        //ImGui::Begin("images");
        //ImVec2 imageSize(128, 128); // Size of the image
        //ImGui::Image((void*)(intptr_t)textures[0], imageSize);
        //ImGui::Image((void*)(intptr_t)textures[1], imageSize);
        //ImGui::Image((void*)(intptr_t)textures[2], imageSize);
        //ImGui::Image((void*)(intptr_t)textures[3], imageSize);
        //ImGui::Image((void*)(intptr_t)texture, imageSize);
        //ImGui::End();

        //DrawTensorTable();

        //DrawModel(cnn_learner(), cnn_model());

        DrawTree(cnn_model());


        return alive;
    }

    bool OnCreate() override
    {
        texture  = CreateTexture2DFromImageFile(GetDevice(), "test.png");

        cnn_init();

        ImVec2 imageSize(28, 28); // Size of the image
        textures[0] = CreateTexture2DFromMinst(GetDevice(), cnn_images_train(0));
        textures[1] = CreateTexture2DFromMinst(GetDevice(), cnn_images_train(7));
        textures[2] = CreateTexture2DFromMinst(GetDevice(), cnn_edge1());
        textures[3] = CreateTexture2DFromMinst(GetDevice(), cnn_edge2());

        return true;
    }

    ID3D11ShaderResourceView* texture = nullptr;
    ID3D11ShaderResourceView* textures[4];
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
