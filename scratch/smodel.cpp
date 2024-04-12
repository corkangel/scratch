#include "smodel.h"


pTensor lin(const pTensor& x, const pTensor& w, const pTensor& b)
{
    return x->MatMult(w) + b;
}

pTensor relu(pTensor& x)
{
    return x->clamp_min_(0.0f);
}

pTensor softmax(const pTensor& x)
{
    pTensor exps = x->exp();
    pTensor sums = exps->sum_columns()->unsqueeze_(1);
    return (exps / sums);
}

pTensor log_softmax(const pTensor& x)
{
    pTensor exps = x->exp();
    return x - exps->sum_columns()->log_()->unsqueeze_(1);
}

pTensor logsumexp(const pTensor& x)
{
    float m = x->max();
    return (x - m)->exp()->sum(1)->log_() + m;
}

pTensor log_softmax2(const pTensor& x)
{
    return x - logsumexp(x)->unsqueeze_(1);
}

float nll_loss(pTensor& input, const pTensor& target)
{
    return -input->index_select(target->squeeze())->mean();
}

float cross_entropy_loss(const pTensor& input, const pTensor& target)
{
    return nll_loss(log_softmax2(input), target);
}

// unfold a tensor for the specified kernel, assuming it contains a single image
pTensor unfold_single(pTensor& image, const uint ksize, const uint stride)
{
    const uint h = image->dim(0);
    const uint w = image->dim(1);

    const uint oh = (h - ksize) / stride + 1;
    const uint ow = (w - ksize) / stride + 1;

    pTensor out = sTensor::Dims(oh * ow, ksize * ksize);

    for (uint i = 0; i < oh; i++)
    {
        for (uint j = 0; j < ow; j++)
        {
            uint row = i * ow + j;
            for (uint k1 = 0; k1 < ksize; k1++)
            {
                for (uint k2 = 0; k2 < ksize; k2++)
                {
                    out->set2d(row, k1 * ksize + k2, image->get2d(i + k1, j * stride + k2));
                }
            }
        }
    }
    return out;
}

// unfold a tensor for the specified kernel, assuming it contains multiple images
pTensor unfold_multiple(pTensor& images, const uint ksize, const uint stride)
{
    const uint nImages = images->dim(0);
    const uint h = images->dim(1);
    const uint w = images->dim(2);

    const uint oh = (h - ksize) / stride + 1;
    const uint ow = (w - ksize) / stride + 1;

    pTensor out = sTensor::Dims(nImages, oh * ow, ksize * ksize);

    const float* data = images->data();
    float* out_data = out->data();
    for (uint nImage = 0; nImage < nImages; nImage++)
    {
        for (uint i = 0; i < oh; i++)
        {
            for (uint j = 0; j < ow; j++)
            {
                uint row = i * ow + j;
                for (uint k1 = 0; k1 < ksize; k1++)
                {
                    for (uint k2 = 0; k2 < ksize; k2++)
                    {
                        const float value = data[nImage * h * w + (i + k1) * w + j * stride + k2];
                        out_data[nImage * oh * ow * ksize * ksize + row * ksize * ksize + k1 * ksize + k2] = value;

                        //const float value = images->get3d(nImage, i * stride + k1, j * stride + k2);
                        //out->set3d(nImage, row, k1 * ksize + k2, value);
                    }
                }
            }
        }
    }
    return out;
}


// ---------------- sLayer ----------------

static uint g_layers = 0;

sLayer::sLayer() : _id(g_layers++), _activations(sTensor::Empty())
{
}

void sLayer::collect_stats()
{
    _activationStats.max.push_back(_activations->max());
    _activationStats.min.push_back(_activations->min());
    _activationStats.mean.push_back(_activations->mean());
    _activationStats.std.push_back(_activations->std());
}

void sLayer::zero_grad()
{
    _activations->zero_grad();
    
    std::map<std::string, pTensor> parms = parameters();
    for (auto& parm : parms)
    {
        parm.second->zero_grad();
    }
}


// ---------------- sRelu ----------------

sRelu::sRelu() : sLayer()
{
}

const pTensor sRelu::forward(pTensor& input)
{
    _activations = input->clone()->clamp_min_(0.0f);
    return _activations;
}

void sRelu::backward(pTensor& input)
{
    input->set_grad(input->greater_than(0.0f) * _activations->grad());
}

// ---------------- sLinear ----------------

sLinear::sLinear(uint in_features, uint out_features) :
    sLayer(),
    _weights(sTensor::Empty()),
    _bias(sTensor::Empty())
{
    _weights = sTensor::NormalDistribution(0.0f, 0.1f, in_features, out_features);
    _bias = sTensor::Zeros(uint(1), out_features);
    //collect_stats();
}

const pTensor sLinear::forward(pTensor& input)
{
    _activations = input->MatMult(_weights) + _bias;
    collect_stats();
    return _activations;
}

void sLinear::backward(pTensor& input)
{
    input->set_grad(_activations->grad()->MatMult(_weights->Transpose()));

    _weights->set_grad(input->Transpose()->MatMult(_activations->grad()));

    _bias->set_grad(_activations->grad()->sum_rows());
}

void sLinear::update_weights(const float lr)
{
    _weights = _weights - (_weights->grad() * lr);
    _bias = _bias - (_bias->grad()->unsqueeze(0) * lr);
}

std::map<std::string,pTensor> sLinear::parameters() const
{
    std::map<std::string, pTensor> p;
    p["weights"] = _weights;
    p["bias"] = _bias;
    return p;
}


// ---------------- sConv2d ----------------

sConv2d::sConv2d(const uint num_channels, const uint num_features, const uint kernel_size, const uint stride, const uint padding) :
    sLayer(),
    _num_channels(num_channels),
    _num_features(num_features),
    _kernel_size(kernel_size),
    _stride(stride),
    _padding(kernel_size/2),
    _kernels(sTensor::Empty()),
    _bias(sTensor::Empty())
{
    _kernels = sTensor::NormalDistribution(0.0f, 0.1f, num_features * num_channels, kernel_size * kernel_size);
    _bias = sTensor::Zeros(uint(1), num_features);
}

const pTensor sConv2d::forward(pTensor& input)
{
    // format is (batch, channels, rows, cols)
    const uint batchSize = input->dim(0);
    const uint width = uint(std::sqrt(input->dim(2)));
    const uint height = width;

    pTensor padded_images = input->clone_shallow()->view_(batchSize * _num_channels, width, height);
    if (_padding != 0)
    {
        padded_images = padded_images->pad3d(1);
    }
    
    pTensor unfolded_images = unfold_multiple(padded_images, _kernel_size, _stride);
    const uint sz = unfolded_images->size_dims(2);
    unfolded_images->reshape_(sz, _kernel_size * _kernel_size);

    // mismatch dimensions!!!!!


    _activations = (unfolded_images->MatMult(_kernels->Transpose()) + _bias);
    _activations->reshape_(batchSize, _num_features, (width / _stride) * (height / _stride));
    return _activations;
}

void sConv2d::backward(pTensor& input)
{
    input->set_grad(_activations->grad()->MatMult(_kernels->Transpose()));

    _kernels->set_grad(input->Transpose()->MatMult(_activations->grad()));

    _bias->set_grad(_activations->grad()->sum_rows());
}

void sConv2d::update_weights(const float lr)
{
    _kernels = _kernels - (_kernels->grad() * lr);
    _bias = _bias - (_bias->grad()->unsqueeze(0) * lr);
}

std::map<std::string, pTensor> sConv2d::parameters() const
{
    std::map<std::string, pTensor> p;
    p["kernels"] = _kernels;
    p["bias"] = _bias;
    return p;
}


// ---------------- sMSE ----------------


sMSE::sMSE() : sLayer(), _diff(sTensor::Empty())
{
}

const pTensor sMSE::forward(pTensor& input)
{
    _activations = input;
    return _activations;
}

void sMSE::backward(pTensor& input)
{
    // loss must have been called first to populate _diff!
    input->set_grad(_diff->unsqueeze(1) * 2.0f / float(input->dim(0))); // 2x is the derivative of the loss function x^2
}

float sMSE::loss(pTensor& input, const pTensor& target)
{
    _diff = (_activations->squeeze() - target);
    return _diff->mse();
}


// ---------------- sSoftMax ----------------

sSoftMax::sSoftMax() : sLayer(), _diff(sTensor::Empty())
{
}

const pTensor sSoftMax::forward(pTensor& input)
{
    _activations = input;
    return _activations;
}

void sSoftMax::backward(pTensor& input)
{
    // loss must have been called first to populate _diff!
    input->set_grad(_diff / float(input->dim(0)));
}

float sSoftMax::loss(pTensor& input, const pTensor& target)
{
    // need gradients for each of the activations, not just the target
    const uint nrows = _activations->dim(0);
    const uint ncols = _activations->dim(1);
    pTensor grads = _activations->clone();

    for (uint r = 0; r < nrows; r++)
    {
        uint t = uint(target->get2d(r, 0));
        for (uint c = 0; c < ncols; c++)
        {
            grads->set2d(r, c, _activations->get2d(r, c) - ((c == t) ? 1 : 0));
        }
    }
    _diff = grads;

    // cross_entropy_loss
    pTensor sf2 = log_softmax2(_activations);
    pTensor sf3 = sf2->index_select(target->squeeze(1));
    float nll = -sf3->mean();
    return nll;
}

// ---------------- sModel ----------------

sModel::sModel(const uint nInputs, const uint nHidden, const uint nOutputs) :
    sModule(), _nInputs(nInputs), _nHidden(nHidden), _nOutputs(nOutputs), _loss(0), _accuracy(0)
{
}

sModel::~sModel()
{
    for (auto& layer : _layers)
    {
        delete layer;
    }
}

const pTensor sModel::forward(pTensor& input)
{
    pTensor x = input;
    for (auto& layer : _layers)
    {
        x = layer->forward(x);
    }
    return _layers.back()->activations();
}

void sModel::backward(pTensor& input)
{
    assert(0); // use loss to backprop
}

float sModel::loss(pTensor& input, const pTensor& target)
{
    sLayer* lastLayer = (sLayer*)_layers.back();
    const float L = lastLayer->loss(input, target);

    const uint n = uint(_layers.size());
    for (int i = n - 1; i >= 0; i--)
    {
        pTensor& x = (i == 0) ? input : _layers[i - 1]->activations();
        _layers[i]->backward(x);
    }
    _loss = L;

    // calc accuracy
    const pTensor preds = _layers.back()->activations();
    pTensor am = preds->argmax();
    pTensor correct = am->equal(target);
    _accuracy = correct->mean();

    return L;
}

void sModel::add_layer(sLayer* layer)
{
    _layers.emplace_back(layer);
}

