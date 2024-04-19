

#include "scratch/stensor.h"
#include "scratch/smodel.h"
#include "scratch/slearner.h"
#include "scratchgui/tools.h"

#include "imguiapp.h"

void sgd_init();
const std::vector<float> sgd_activation_means(const uint layer);
const sModel& sgd_model();
sLearner& sgd_learner();
const float* sgd_images_train();

class SgdApp : public App
{
public:
    bool OnDraw() override
    {
        bool alive = DrawMenu(sgd_learner(), sgd_init);

        //DrawTensorTable();

        DrawModel(sgd_learner(), sgd_model());

        //DrawActivationStats(sgd_learner(), sgd_model());

        DrawTree(sgd_model());

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
