
#include "scratch/stensor.h"
#include "scratch/smodel.h"
#include "scratch/slearner.h"
#include "scratch/minst.h"


const uint g_imageArraySize = 28 * 28;
const uint g_numImagesTrain = 60000; // should be 60000
const uint g_numImagesValid = 10000;
const uint g_numCategories = 10;
const uint g_numHidden = 50;

const uint batchSize = 100;
const uint epochs = 4;
const float lr = 0.5f;

struct CnnData
{
    sLearner* learner = nullptr;
    sModel* model = nullptr;

    pTensor images_train = sTensor::Empty();
    pTensor categories_train = sTensor::Empty();

    //sTensor images_valid = sTensor::Empty();
    //sTensor categories_valid = sTensor::Empty();

    pTensor edge1 = sTensor::Zeros(28, 28);
    pTensor edge2 = sTensor::Zeros(28, 28);

    ~CnnData()
    {
        delete model;
        delete learner;
    }
};

CnnData data;

const float* cnn_images_train()
{
    return data.images_train->data() + 7*784;
}

const float* cnn_edge1() { return data.edge1->data(); }
const float* cnn_edge2() { return data.edge2->data(); }


void cnn_init()
{
    data.images_train = minstLoadImages("Resources/Data/minst/train-images.idx3-ubyte", g_numImagesTrain, g_imageArraySize);
    data.categories_train = minstLoadLabels("Resources/Data/minst/train-labels.idx1-ubyte", g_numImagesTrain);

    // not used yet
    //data.images_valid = minstLoadImages("Resources/Data/minst/t10k-images.idx3-ubyte", g_numImagesValid, g_imageArraySize);
    //data.categories_valid = minstLoadLabels("Resources/Data/minst/t10k-labels.idx1-ubyte", g_numImagesValid);

    data.model = new sModel(g_imageArraySize, g_numHidden, 10);

    data.learner = new sLearner(*data.model, data.images_train, data.categories_train, batchSize, lr);

    sTensor::enableAutoLog = true;

    pTensor top_edge = sTensor::Zeros(3, 3);
    top_edge->data()[0] = -1.f;
    top_edge->data()[1] = -1.f;
    top_edge->data()[2] = -1.f;
    top_edge->data()[6] = 1.f;
    top_edge->data()[7] = 1.f;
    top_edge->data()[8] = 1.f;

    pTensor left_edge = sTensor::Zeros(3, 3);
    left_edge->data()[0] = -1.f;
    left_edge->data()[3] = -1.f;
    left_edge->data()[6] = -1.f;
    left_edge->data()[2] = 1.f;
    left_edge->data()[5] = 1.f;
    left_edge->data()[8] = 1.f;

    if (1)
    {
        // need to pad in two dimensions
        pTensor ready = data.images_train->unsqueeze(2)->view_(60000, 28, 28)->pad3d(1);
        pTensor unfolded = unfold_multiple(ready, 3)->reshape_(60000 * (28 * 28), 9);

        pTensor flattened_top = top_edge->view_(9, 1);
        pTensor imgs = unfolded->MatMult(flattened_top)->reshape_(60000, 784);
        data.edge1 = imgs->row(0)->view_(28, 28);
        data.edge2 = imgs->row(7)->view_(28, 28);
    }

    if (0)
    {
        // unfold test
        pTensor image = data.images_train->slice_rows(7, 8)->view_(28,28)->pad2d(1);
        pTensor unfolded_image = unfold_single(image, 3);
        pTensor flattened_top = top_edge->view_(9, 1);
        data.edge1 = unfolded_image->MatMult(flattened_top)->view_(28,28);

        pTensor flattened_left = left_edge->view_(9, 1);
        data.edge2 = unfolded_image->MatMult(flattened_left)->view_(28,28);
    }

    // manually compute the convolution
    //for (uint i = 0; i < 28-2; i++)
    //{
    //    for (uint j = 0; j < 28-2; j++)
    //    {
    //        if (i == 3 && j == 14)
    //        {
    //            int debug = 1;
    //        }
    //        pTensor slice = image->slice2d(i, i + 3, j, j + 3);

    //        data.edge1->set2d(i, j, (slice * top_edge)->sum());
    //        data.edge2->set2d(i, j, (slice * left_edge)->sum());
    //    }
    //}

    sTensor::enableAutoLog = false;
}

const sModel& cnn_model()
{
    return *data.model;
}

sLearner& cnn_learner()
{
    return *data.learner;
}
