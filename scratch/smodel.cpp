#include "smodel.h"


sTensor lin(const sTensor& x, const sTensor& w, const sTensor& b)
{
    return x.MatMult(w) + b;
}

sTensor relu(sTensor& x)
{
    return x.clamp_min_(0.0f);
}

sTensor softmax(const sTensor& x)
{
    sTensor exps = x.exp();
    sTensor sums = exps.sum_columns().unsqueeze_(1);
    return (exps / sums);
}

sTensor log_softmax(const sTensor& x)
{
    sTensor exps = x.exp();
    return x - exps.sum_columns().log_().unsqueeze_(1);
}

sTensor logsumexp(const sTensor& x)
{
    float m = x.max();
    return (x - m).exp().sum(1).log_() + m;
}

sTensor log_softmax2(const sTensor& x)
{
    return x - logsumexp(x).unsqueeze_(1);
}

float nll_loss(sTensor& input, const sTensor& target)
{
    return -input.index_select(target.squeeze()).mean();
}

float cross_entropy_loss(const sTensor& input, const sTensor& target)
{
    return nll_loss(log_softmax2(input), target);
}



// ---------------- sRelu ----------------

sRelu::sRelu() : sModule(), _activations(sTensor::Empty())
{
}

sTensor& sRelu::forward(const sTensor& input)
{
    _activations = input.clone().clamp_min_(0.0f);
    return _activations;
}

void sRelu::backward(sTensor& input)
{
    input.set_grad(input.greater_than(0.0f) * (*_activations.grad()));
}

sTensor& sRelu::activations()
{
    return _activations;
}

// ---------------- sLinear ----------------

sLinear::sLinear(uint in_features, uint out_features) :
    sModule(),
    _activations(sTensor::Empty()),
    _weights(sTensor::Empty()),
    _bias(sTensor::Empty())
{
    _weights = sTensor::NormalDistribution(0.0f, 0.1f, in_features, out_features);
    _bias = sTensor::Zeros(uint(1), out_features);
}

sTensor& sLinear::forward(const sTensor& input)
{
    _activations = input.MatMult(_weights) + _bias;
    return _activations;
}

void sLinear::backward(sTensor& input)
{
    input.set_grad(_activations.grad()->MatMult(_weights.Transpose()));

    _weights.set_grad(input.Transpose().MatMult(*_activations.grad()));

    _bias.set_grad(_activations.grad()->sum_rows());
}

sTensor& sLinear::activations()
{
    return _activations;
}

void sLinear::update_weights(const float lr)
{
    _weights = _weights - (*_weights.grad() * lr);
    _bias = _bias - (_bias.grad()->unsqueeze(0) * lr);
}
void sLinear::zero_grad()
{
    _weights.zero_grad();
    _bias.zero_grad();
}

// ---------------- sMSE ----------------


sMSE::sMSE() : sModule(), _activations(sTensor::Empty()), _diff(sTensor::Empty())
{
}

sTensor& sMSE::forward(const sTensor& input)
{
    _activations = input;
    return _activations;
}

void sMSE::backward(sTensor& input)
{
    // loss must have been called first to populate _diff!
    input.set_grad(_diff.unsqueeze(1) * 2.0f / float(input.dim(0))); // 2x is the derivative of the loss function x^2
}

sTensor& sMSE::activations()
{
    return _activations;
}

float sMSE::loss(sTensor& input, const sTensor& target)
{
    _diff = (_activations.squeeze() - target);
    return _diff.mse();
}


// ---------------- sSoftMax ----------------

sSoftMax::sSoftMax() : sModule(), _activations(sTensor::Empty()), _diff(sTensor::Empty())
{
}

sTensor& sSoftMax::forward(const sTensor& input)
{
    _activations = input;
    return _activations;
}

void sSoftMax::backward(sTensor& input)
{
    // loss must have been called first to populate _diff!
    input.set_grad(_diff);
}

float sSoftMax::loss(sTensor& input, const sTensor& target)
{
    _diff = (_activations.squeeze() - target);
    return cross_entropy_loss(_activations, target);
}

sTensor& sSoftMax::activations()
{
    return _activations;
}

// ---------------- sModel ----------------

sModel::sModel(const uint nInputs, const uint nHidden, const uint nOutputs) :
    sModule(), _nInputs(nInputs), _nHidden(nHidden), _nOutputs(nOutputs)
{
    _layers.emplace_back(new sLinear(_nInputs, _nHidden));
    _layers.emplace_back(new sRelu());
    _layers.emplace_back(new sLinear(_nHidden, _nOutputs));

    //_smeLayer = new sMSE();
    //_layers.emplace_back(_smeLayer);

    _smLayer = new sSoftMax();
    _layers.emplace_back(_smLayer);
}

sModel::~sModel()
{
    for (auto& layer : _layers)
    {
        delete layer;
    }
}

sTensor& sModel::forward(const sTensor& input)
{
    sTensor& x = input.clone_shallow();
    for (auto& layer : _layers)
    {
        x.ref_shallow_(layer->forward(x));
    }
    return _layers.back()->activations();
}

void sModel::backward(sTensor& input)
{
    assert(0); // use loss to backprop
}

float sModel::loss(sTensor& input, const sTensor& target)
{
    //const float L = _smeLayer->loss(target);
    const float L = _smLayer->loss(input, target);

    const uint n = uint(_layers.size());
    for (int i = n - 1; i >= 0; i--)
    {
        sTensor& x = (i == 0) ? input : _layers[i - 1]->activations();
        _layers[i]->backward(x);
    }
    return L;
}

