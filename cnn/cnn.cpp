
#include "scratch/stensor.h"
#include "scratch/smodel.h"
#include "scratch/slearner.h"
#include "scratch/minst.h"


const uint g_imageSize = 28;
const uint g_imageArraySize = g_imageSize * g_imageSize;
const uint g_numImagesTrain = 500; // should be 60000
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

const float* cnn_images_train(const uint index)
{
    return data.images_train->data() + index * 784;
}

const float* cnn_edge1() { return data.edge1->data(); }
const float* cnn_edge2() { return data.edge2->data(); }

void cnn_init()
{
    data.images_train = minstLoadImages("Resources/Data/fashion/train-images.idx3-ubyte", g_numImagesTrain, g_imageArraySize)->reshape_(g_numImagesTrain, uint(1), g_imageSize, g_imageSize);
    data.categories_train = minstLoadLabels("Resources/Data/fashion/train-labels.idx1-ubyte", g_numImagesTrain);

    data.model = new sModel(g_imageArraySize, g_numHidden, 10);

    data.model->add_layer(new sManualConv2d(1, 4)); // 14x14
    data.model->add_layer(new sRelu());
    data.model->add_layer(new sManualConv2d(4, 8)); // 7x7
    data.model->add_layer(new sRelu());
    data.model->add_layer(new sManualConv2d(8, 16)); // 4x4
    data.model->add_layer(new sRelu());
    data.model->add_layer(new sManualConv2d(16, 16)); // 2x2
    data.model->add_layer(new sRelu());
    data.model->add_layer(new sManualConv2d(16, 10)); // 1x1
    data.model->add_layer(new sSoftMax());

    // format for CNN is (batch, channels, height x width)
    pTensor images = data.images_train;
    data.learner = new sLearner(*data.model, images, data.categories_train, batchSize, lr);
}

const sModel& cnn_model()
{
    return *data.model;
}

sLearner& cnn_learner()
{
    return *data.learner;
}
