
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
    sTensor diffs = (res.squeeze_() - categories);
    return diffs.pow_(2).mean();
}


void lin_grad(sTensor& inp, const sTensor& out, sTensor& weights, sTensor& bias)
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
    sTensor diff = (out.squeeze_() - target).set_label("diff");
    float loss = diff.pow_(2).mean();

    // backward pass
    out.set_grad(diff.unsqueeze_(1) * 2.0f / float(inp.dim(0))); // 2x is the derivative of the loss function x^2
    lin_grad(l2, out, w2, b2);

    l1.set_grad(l1.greater_than(0.0f) * (*l2.grad()));
    lin_grad(inp, l1, w1, b1);
}

class sModule
{
public:
    virtual sTensor& forward(const sTensor& input, const sTensor& target) = 0;
    virtual void backward() = 0;

    sTensor operator()(const sTensor& input, const sTensor& target)
    {
        return forward(input, target);
    }
    virtual ~sModule() {}
};


class sRelu : public sModule
{
public:

    sTensor _input;
    sTensor _output;

    sRelu() : sModule(), _input(sTensor::Empty()), _output(sTensor::Empty())
    {
    }

    sTensor& forward(const sTensor& input, const sTensor& _) override
    {
        _input.ref_shallow_(input);
        _output = input.clone().clamp_min_(0.0f);
        return _output;
    }

    void backward() override
    {
        _input.set_grad(_output.greater_than(0.0f) * (*_output.grad()));
    }
};

class sLinear : public sModule
{
public:
    sTensor _weights;
    sTensor _bias;
    sTensor _input;
    sTensor _output;

    sLinear(uint in_features, uint out_features) : 
        sModule(),
        _weights(sTensor::NormalDistribution(0.0f, 0.5f, in_features, out_features)),
        _bias(sTensor::Zeros(uint(1), out_features)),
        _input(sTensor::Empty()),
        _output(sTensor::Empty())
    {
    }

    sTensor& forward(const sTensor& input, const sTensor& _) override
    {
        _input.ref_shallow_(input);
        _output = lin(_input, _weights, _bias);
        return _output;
    }

    void backward() override
    {
        _input.set_grad(_output.grad()->MatMult(_weights.Transpose()));

        _weights.set_grad(_input.Transpose().MatMult(*_output.grad()));

        _bias.set_grad(_output.grad()->sum_rows());
    }
};

class seMSE : public sModule
{
public:
    sTensor _input;
    sTensor _output;
    sTensor _diff;

    seMSE() : sModule(), _input(sTensor::Empty()), _output(sTensor::Empty()), _diff(sTensor::Empty())
    {
    }

    sTensor& forward(const sTensor& input, const sTensor& target) override
    {
        _input.ref_shallow_(input);

        sTensor inputCopy = _input.clone_shallow();
        _diff = (inputCopy.squeeze_() - target);
        float loss = _diff.mse();
        
        _output = sTensor::Zeros(1, 1);
        _output(0,0) = loss;
        return _output;
    }

    void backward() override
    {
        _input.set_grad(_diff.unsqueeze_(1) * 2.0f / float(_input.dim(0))); // 2x is the derivative of the loss function x^2
    }
};

class sModel : public sModule
{
public:
    sTensor _input;
    std::vector<sModule*> layers;
    seMSE loss;

    sModel() : sModule(), _input(sTensor::Empty()), layers(), loss()
    {
        layers.emplace_back(new sLinear(g_imageArraySize, g_numHidden));
        layers.emplace_back(new sRelu());
        layers.emplace_back(new sLinear(g_numHidden, 1));
    }

    ~sModel()
    {
        for (auto& layer : layers)
        {
            delete layer;
        }
    }

    sTensor& forward(const sTensor& input, const sTensor& target) override
    {
        _input.ref_shallow_(input);

        sTensor& x = input.clone_shallow();
        for (auto& layer : layers)
        {
            x.ref_shallow_(layer->forward(x, target));
        }
        return loss.forward(x, target);
    }

    void backward() override
    {
        loss.backward();
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            (*it)->backward();
        }
    }
};

void sgd_init()
{
    sTensor g_images_train = loadImages("Resources/Data/minst/train-images.idx3-ubyte", g_numImagesTrain);
    sTensor g_categories_train = loadLabels("Resources/Data/minst/train-labels.idx1-ubyte", g_numImagesTrain);

    sTensor g_images_valid = loadImages("Resources/Data/minst/t10k-images.idx3-ubyte", g_numImagesValid);
    sTensor g_categories_valid = loadLabels("Resources/Data/minst/t10k-labels.idx1-ubyte", g_numImagesValid);

    sTensor w1 = sTensor::NormalDistribution(0.0f, 0.5f, g_imageArraySize, g_numHidden);
    sTensor b1 = sTensor::Zeros(uint(1), g_numHidden);
    sTensor w2 = sTensor::Randoms(g_numHidden, uint(1));
    sTensor b2 = sTensor::Zeros(uint(1),uint(1));

    sTensor::enableAutoLog = true;
    auto start = std::chrono::high_resolution_clock::now();

    //sTensor preds = model(g_images_train, w1, b1, w2, b2).set_label("preds");
    //float L = loss(preds, g_categories_train);

    //sModel mmm;
    //mmm.forward(g_images_train, g_categories_train);
    //mmm.backward();

    forward_backward(g_images_train, w1, b1, w2, b2, g_categories_train);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    slog("duration: %u ms", duration);
    sTensor::enableAutoLog = false;

}

