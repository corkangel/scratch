

#include "scratch/stensor.h"
#include "scratch/smodel.h"
#include "scratch/slearner.h"
#include "scratchgui/tools.h"

#include "imguiapp.h"
#include "implot.h"

void cnn_init();
void cnn_step();
void cnn_step_layer();
void cnn_step_epoch();
void cnn_fit(uint epochs);
const std::vector<float> cnn_activation_means(const uint layer);
const sModel& cnn_model();
const sLearner& cnn_learner();
const float* cnn_images_train();

class CnnApp : public App
{
public:
    bool OnDraw() override
    {
        bool alive = true;

        ImGui::Begin("CNN");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        if (ImGui::Button("Restart"))
            cnn_init();

        ImGui::SameLine();
        if (ImGui::Button("Step Layer"))
            cnn_step_layer();

        ImGui::SameLine();
        if (ImGui::Button("Step Batch"))
            cnn_step(); 

        ImGui::SameLine();
        if (ImGui::Button("Step Epoch"))
            cnn_step_epoch();

        ImGui::SameLine();
        if (ImGui::Button("Fit"))
            cnn_fit(5);

        ImGui::SameLine();
        if (ImGui::Button("Quit"))
             alive = false;

        

        ImGui::End();

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


        ImGui::Begin("LinearStats");
        if (ImPlot::BeginPlot("Activations Mean")) {

            for (uint i = 0; i < 4; i++)
            {
                const std::vector<float> means = cnn_activation_means(i);
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
