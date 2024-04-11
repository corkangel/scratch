
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


    ~CnnData()
    {
        delete model;
        delete learner;
    }
};

CnnData data;

const float* cnn_images_train()
{
    return data.images_train->data();
}

void cnn_init()
{
    data.images_train = minstLoadImages("Resources/Data/fashion/train-images.idx3-ubyte", g_numImagesTrain, g_imageArraySize);
    data.categories_train = minstLoadLabels("Resources/Data/fashion/train-labels.idx1-ubyte", g_numImagesTrain);

    // not used yet
    //data.images_valid = minstLoadImages("Resources/Data/minst/t10k-images.idx3-ubyte", g_numImagesValid, g_imageArraySize);
    //data.categories_valid = minstLoadLabels("Resources/Data/minst/t10k-labels.idx1-ubyte", g_numImagesValid);

    data.model = new sModel(g_imageArraySize, g_numHidden, 10);

    data.learner = new sLearner(*data.model, data.images_train, data.categories_train, batchSize, lr);
}

const std::vector<float> cnn_activation_means(const uint layer)
{
    return ((sLayer*)data.model->_layers[layer])->_activationStats.mean;
}

const sModel& cnn_model()
{
    return *data.model;
}

sLearner& cnn_learner()
{
    return *data.learner;
}
