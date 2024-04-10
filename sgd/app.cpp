

#include "scratch/stensor.h"
#include "scratch/smodel.h"
#include "scratch/slearner.h"
#include "scratchgui/tools.h"

#include "imguiapp.h"
#include "implot.h"

void sgd_init();
void sgd_step();
void sgd_step_layer();
void sgd_step_epoch();
void sgd_fit(uint epochs);
const std::vector<float> sgd_activation_means(const uint layer);
const sModel& sgd_model();
const sLearner& sgd_learner();


class SgdApp : public App
{
public:
    bool OnDraw() override
    {
        bool alive = true;

        ImGui::Begin("SGD");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        if (ImGui::Button("Restart"))
            sgd_init();

        ImGui::SameLine();
        if (ImGui::Button("Step Layer"))
            sgd_step_layer();

        ImGui::SameLine();
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

        ImVec2 imageSize(256, 256); // Size of the image
        ImGui::Image((void*)(intptr_t)texture, imageSize);

        ImGui::End();

        DrawTensorTable();

        DrawModel(sgd_learner(), sgd_model());


        ImGui::Begin("LinearStats");
        if (ImPlot::BeginPlot("Activations Mean")) {

            for (uint i = 0; i < 4; i++)
            {
                const std::vector<float> means = sgd_activation_means(i);
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

        sgd_init();
        return true;
    }

    ID3D11ShaderResourceView* texture = nullptr;
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
