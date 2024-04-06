
#include "scratch/stensor.h"
#include "scratch/smodel.h"
#include "scratch/slearner.h"
#include "scratch/minst.h"


const uint g_imageArraySize = 28 * 28;
const uint g_numImagesTrain = 1000; // should be 60000
const uint g_numImagesValid = 10000;
const uint g_numCategories = 10;
const uint g_numHidden = 50;

const uint batchSize = 100;
const uint epochs = 4;
const float lr = 0.5f;

struct SgdData
{
    sLearner* learner = nullptr;
    sModel* model = nullptr;

    pTensor images_train = sTensor::Empty();
    pTensor categories_train = sTensor::Empty();

    //sTensor images_valid = sTensor::Empty();
    //sTensor categories_valid = sTensor::Empty();

    ~SgdData()
    {
        delete model;
        delete learner;
    }
};

SgdData data;

void sgd_init()
{
    data.images_train = minstLoadImages("Resources/Data/minst/train-images.idx3-ubyte", g_numImagesTrain, g_imageArraySize);
    data.categories_train = minstLoadLabels("Resources/Data/minst/train-labels.idx1-ubyte", g_numImagesTrain);

    // not used yet
    //data.images_valid = minstLoadImages("Resources/Data/minst/t10k-images.idx3-ubyte", g_numImagesValid, g_imageArraySize);
    //data.categories_valid = minstLoadLabels("Resources/Data/minst/t10k-labels.idx1-ubyte", g_numImagesValid);

    data.model = new sModel(g_imageArraySize, g_numHidden, 10);

    data.learner = new sLearner(*data.model, data.images_train, data.categories_train, batchSize, lr);
}

void sgd_step()
{
    data.learner->step();
}

void sgd_step_layer()
{
    data.learner->step_layer();
}

void sgd_step_epoch()
{
    data.learner->step_epoch();
}

void sgd_fit(uint epochs)
{
    data.learner->fit(epochs);
}

const std::vector<float> sgd_activation_means(const uint layer)
{
    return ((sLayer*)data.model->_layers[layer])->_activationStats.mean;
}

const sModel& sgd_model()
{
    return *data.model;
}
const sLearner& sgd_learner()
{
    return *data.learner;
}

//sTensor w1 = sTensor::NormalDistribution(0.0f, 0.5f, g_imageArraySize, g_numHidden);
//sTensor b1 = sTensor::Zeros(uint(1), g_numHidden);
//sTensor w2 = sTensor::Randoms(g_numHidden, uint(1));
//sTensor b2 = sTensor::Zeros(uint(1),uint(1));

//forward_backward(g_images_train, w1, b1, w2, b2, g_categories_train);
// 
//sTensor preds = model(g_images_train, w1, b1, w2, b2).set_label("preds");
//float L = loss(preds, g_categories_train);

//sModel mmm(g_imageArraySize, g_numHidden, 10);
//sTensor preds = mmm.forward(g_images_train);
//float L = mmm.loss(g_categories_train);

//sTensor s = softmax(preds).log_();
//sTensor ss = log_softmax(preds);
//sTensor sss = log_softmax2(preds);

//float loss = nll_loss(log_softmax2(preds), g_categories_train);
//float 

//sTensor preds = model.forward(xb);
//float L = nll_loss(preds, yb);
//sTensor argmax = preds.argmax();
//float acc = accuracy(argmax, yb);

//
//float accuracy(const sTensor& preds, const sTensor& target)
//{
//    sTensor am = preds.argmax();
//    sTensor correct = am.equal(target);
//    return correct.mean();
//}
//
//sTensor model(const sTensor& x, const sTensor& w1, const sTensor& b1, const sTensor& w2, const sTensor& b2)
//{
//    sTensor l1 = lin(x, w1, b1);
//    sTensor l2 = relu(l1);
//    sTensor res = lin(l2, w2, b2);
//    return res;
//}
//
//void lin_grad(const sTensor& out, sTensor& inp, sTensor& weights, sTensor& bias)
//{
//    inp.set_grad(out.grad()->MatMult(weights.Transpose()));
//
//    weights.set_grad(inp.Transpose().MatMult(*out.grad()));
//
//    bias.set_grad(out.grad()->sum_rows());
//
//}
//
//void forward_backward(sTensor& inp, sTensor& w1, sTensor& b1, sTensor& w2, sTensor& b2, const sTensor& target)
//{
//    // forward pass
//    sTensor l1 = lin(inp, w1, b1).set_label("l1");
//    sTensor l2 = relu(l1).set_label("l2");
//    sTensor out = lin(l2, w2, b2).set_label("out");
//    sTensor diff = (out.squeeze() - target).set_label("diff");
//    float loss = diff.mse();
//
//    // backward pass
//    out.set_grad(diff.unsqueeze(1) * 2.0f / float(inp.dim(0))); // 2x is the derivative of the loss function x^2
//    lin_grad(out, l2, w2, b2);
//
//    l1.set_grad(l1.greater_than(0.0f) * (*l2.grad()));
//    lin_grad(l1, inp, w1, b1);
//}
