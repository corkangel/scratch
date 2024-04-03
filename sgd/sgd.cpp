
#include "scratch/stensor.h"

#include <fstream>

const uint g_imageArraySize = 28 * 28;
const uint g_numImagesTrain = 6000;
const uint g_numImagesValid = 10000;
const uint g_numCategories = 10;
const uint g_numHidden = 50;

// constant to covert from 255 to float in 0-to-1 range
const float convert255 = float(1) / float(255);

sTensor loadImages(const char* filename, const uint numImages)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});

    const uint headerSize = 16;
    //assert(buffer.size() == numImages * g_imageArraySize + headerSize);

    sTensor images = sTensor::Dims(numImages, g_imageArraySize);

    // read image bytes into the images tensor
    for (uint i = 0; i < numImages; i++)
    {
        unsigned char* imagePtr = &buffer[headerSize + i * g_imageArraySize];
        for (uint j = 0; j < g_imageArraySize; j++)
        {
            images.set2d(i, j, float(imagePtr[j + 1]) * convert255);
        }
    }
    return images;
}

sTensor loadLabels(const char* filename, const uint numImages)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});

    const uint headerSize = 8;
    //assert(buffer.size() == numImages + headerSize);

    sTensor categories = sTensor::Zeros(numImages);

    // read label bytes into the categories tensor
    for (uint i = 0; i < numImages; i++)
    {
        unsigned char* labelPtr = &buffer[headerSize + i];

        // hot encoded, so use value as index
        //categories.set2d(i, labelPtr[0], 1.0f);

        // raw
        categories.set1d(i, labelPtr[0]);
    }
    return categories;
}


sTensor lin(const sTensor& x, const sTensor& w, const sTensor& b)
{
    return x.MatMult(w) + b;
}

sTensor relu(sTensor& x)
{
    return x.clamp_min_(0.0f);
}

sTensor model(const sTensor& x, const sTensor& w1, const sTensor& b1, const sTensor& w2, const sTensor& b2)
{
    sTensor l1 = lin(x, w1, b1);
    sTensor l2 = relu(l1);
    sTensor res = lin(l2, w2, b2);
    return res;
}

float loss(sTensor& res, const sTensor& categories)
{
    sTensor diffs = (res.squeeze() - categories);
    return diffs.pow_(2).mean();
}


void lin_grad(const sTensor& out, sTensor& inp, sTensor& weights, sTensor& bias)
{
    inp.set_grad(out.grad()->MatMult(weights.Transpose()));

    weights.set_grad(inp.Transpose().MatMult(*out.grad()));

    bias.set_grad(out.grad()->sum_rows());

}

void forward_backward(sTensor& inp, sTensor& w1, sTensor& b1, sTensor& w2, sTensor& b2, const sTensor& target)
{
    // forward pass
    sTensor l1 = lin(inp, w1, b1).set_label("l1");
    sTensor l2 = relu(l1).set_label("l2");
    sTensor out = lin(l2, w2, b2).set_label("out");
    sTensor diff = (out.squeeze() - target).set_label("diff");
    float loss = diff.mse();

    // backward pass
    out.set_grad(diff.unsqueeze(1) * 2.0f / float(inp.dim(0))); // 2x is the derivative of the loss function x^2
    lin_grad(out, l2, w2, b2);

    l1.set_grad(l1.greater_than(0.0f) * (*l2.grad()));
    lin_grad(l1, inp, w1, b1);
}

class sModule
{
public:
    virtual sTensor& forward(const sTensor& input) = 0;
    virtual void backward(sTensor& input) = 0;
    virtual float loss(const sTensor& target) { assert(0);  return 0.0f; }

    virtual sTensor& activations() { assert(0); return sTensor::null; }

    virtual ~sModule() {}
};


class sRelu : public sModule
{
public:

    sTensor _activations;

    sRelu() : sModule(), _activations(sTensor::Empty())
    {
    }

    sTensor& forward(const sTensor& input) override
    {
        _activations = input.clone().clamp_min_(0.0f);
        return _activations;
    }

    void backward(sTensor& input) override
    {
        input.set_grad(input.greater_than(0.0f) * (*_activations.grad()));
    }

    sTensor& activations() override
    {
        return _activations;
    }
};

class sLinear : public sModule
{
public:
    sTensor _activations;
    sTensor _weights;
    sTensor _bias;


    sLinear(uint in_features, uint out_features) : 
        sModule(),
        _activations(sTensor::Empty()),
        _weights(sTensor::Empty()),
        _bias(sTensor::Empty())
    {
        _weights = sTensor::NormalDistribution(0.0f, 0.5f, in_features, out_features);
        _bias = sTensor::Zeros(uint(1), out_features);
    }

    sTensor& forward(const sTensor& input) override
    {
        _activations = lin(input, _weights, _bias);
        return _activations;
    }

    void backward(sTensor& input) override
    {
        input.set_grad(_activations.grad()->MatMult(_weights.Transpose()));

        _weights.set_grad(input.Transpose().MatMult(*_activations.grad()));

        _bias.set_grad(_activations.grad()->sum_rows());
    }

    sTensor& activations() override
    {
        return _activations;
    }
};

class sMSE : public sModule
{
public:
    sTensor _activations;
    sTensor _diff;

    sMSE() : sModule(), _activations(sTensor::Empty()), _diff(sTensor::Empty())
    {
    }

    sTensor& forward(const sTensor& input) override
    {
        _activations = input;
        return _activations;
    }

    void backward(sTensor& input) override
    {
        // loss must have been called first to populate _diff!
        input.set_grad(_diff.unsqueeze(1) * 2.0f / float(input.dim(0))); // 2x is the derivative of the loss function x^2
    }

    sTensor& activations() override
    {
        return _activations;
    }

    float loss(const sTensor& target) override
    {
        _diff = (_activations.squeeze() - target);
        return _diff.mse();
    }
};

class sModel : public sModule
{
public:
    std::vector<sModule*> _layers;
    sMSE *_smeLayer;
    const int _nInputs;
    const int _nHidden;

    sModel(const int nInputs, const int nHidden) :
        sModule(), _nInputs(nInputs), _nHidden(nHidden)
    {
        _layers.emplace_back(new sLinear(g_imageArraySize, g_numHidden));
        _layers.emplace_back(new sRelu());
        _layers.emplace_back(new sLinear(g_numHidden, 1));

        _smeLayer = new sMSE();
        _layers.emplace_back(_smeLayer);
    }

    ~sModel()
    {
        for (auto& layer : _layers)
        {
            delete layer;
        }
    }

    sTensor& forward(const sTensor& input) override
    {
        sTensor& x = input.clone_shallow();
        for (auto& layer : _layers)
        {
            x.ref_shallow_(layer->forward(x));
        }
        return _layers.back()->activations();
    }

    void backward(sTensor& input) override
    {
        assert(0); // use loss to backprop
    }

    float loss(const sTensor& target) override
    {
        const float L = _smeLayer->loss(target);

        const uint n = uint(_layers.size());
        for (uint i = n - 1; i > 0; i--)
        {
            sTensor& x = _layers[i-1]->activations();
            _layers[i]->backward(x);
        }

        return L;
    }
};

void sgd_init()
{
    sTensor g_images_train = loadImages("Resources/Data/minst/train-images.idx3-ubyte", g_numImagesTrain);
    sTensor g_categories_train = loadLabels("Resources/Data/minst/train-labels.idx1-ubyte", g_numImagesTrain);

    sTensor g_images_valid = loadImages("Resources/Data/minst/t10k-images.idx3-ubyte", g_numImagesValid);
    sTensor g_categories_valid = loadLabels("Resources/Data/minst/t10k-labels.idx1-ubyte", g_numImagesValid);

    //sTensor w1 = sTensor::NormalDistribution(0.0f, 0.5f, g_imageArraySize, g_numHidden);
    //sTensor b1 = sTensor::Zeros(uint(1), g_numHidden);
    //sTensor w2 = sTensor::Randoms(g_numHidden, uint(1));
    //sTensor b2 = sTensor::Zeros(uint(1),uint(1));

    //forward_backward(g_images_train, w1, b1, w2, b2, g_categories_train);
    // 
    //sTensor preds = model(g_images_train, w1, b1, w2, b2).set_label("preds");
    //float L = loss(preds, g_categories_train);

    sTensor::enableAutoLog = true;
    auto start = std::chrono::high_resolution_clock::now();
    
    sModel mmm(g_imageArraySize, g_numHidden);
    mmm.forward(g_images_train);
    float L = mmm.loss(g_categories_train);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    slog("duration: %u ms", duration);
    sTensor::enableAutoLog = false;

}

