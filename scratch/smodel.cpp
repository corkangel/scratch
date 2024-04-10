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



// ---------------- sLayer ----------------

void sLayer::collect_stats()
{
    _activationStats.max.push_back(_activations->max());
    _activationStats.min.push_back(_activations->min());
    _activationStats.mean.push_back(_activations->mean());
    _activationStats.std.push_back(_activations->std());
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
void sLinear::zero_grad()
{
    _weights->zero_grad();
    _bias->zero_grad();
}

const std::map<std::string,pTensor> sLinear::parameters() const
{
    std::map<std::string, pTensor> p;
    p["weights"] = _weights;
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
    //const float L = _smeLayer->loss(target);
    const float L = _smLayer->loss(input, target);

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

