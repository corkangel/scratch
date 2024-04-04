
#include "scratch/stensor.h"
#include "scratch/smodel.h"
#include "scratch/minst.h"

const uint g_imageArraySize = 28 * 28;
const uint g_numImagesTrain = 1000;
const uint g_numImagesValid = 10000;
const uint g_numCategories = 10;
const uint g_numHidden = 50;

float accuracy(const sTensor& preds, const sTensor& target)
{
    sTensor am = preds.argmax();
    sTensor correct = am.equal(target);
    return correct.mean();
}

void sgd_init()
{
    sTensor g_images_train = minstLoadImages("Resources/Data/minst/train-images.idx3-ubyte", g_numImagesTrain, g_imageArraySize);
    sTensor g_categories_train = minstLoadLabels("Resources/Data/minst/train-labels.idx1-ubyte", g_numImagesTrain);

    sTensor g_images_valid = minstLoadImages("Resources/Data/minst/t10k-images.idx3-ubyte", g_numImagesValid, g_imageArraySize);
    sTensor g_categories_valid = minstLoadLabels("Resources/Data/minst/t10k-labels.idx1-ubyte", g_numImagesValid);

    auto start = std::chrono::high_resolution_clock::now();
    
    sModel model(g_imageArraySize, g_numHidden, 10);

    const uint batchSize = 100;
    const uint epochs = 4;
    const float lr = 0.5f;

    float L = 0.0f;

    for (uint epoch = 0; epoch < epochs; epoch++)
    {
        for (uint i = 0; i < g_numImagesTrain; i += batchSize)
        {
            sTensor xb = g_images_train.slice_rows(i, i + batchSize);
            sTensor yb = g_categories_train.slice_rows(i, i + batchSize);

            sTensor preds = model.forward(xb);
            L = model.loss(xb, yb);
            slog("loss: %f", L);

            for (auto& layer : model._layers)
            {
                if (layer->activations().grad())
                {
                    layer->update_weights(lr);
                }
            }

            for (auto& layer : model._layers)
                layer->zero_grad();
        }
        slog("loss: %f", L);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    slog("duration: %u ms", duration);
    //sTensor::enableAutoLog = false;
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
