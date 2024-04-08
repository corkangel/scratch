#pragma once

#include "stensor.h"

#include <map>

struct sStats
{
    std::vector<float> mean;
    std::vector<float> std;
    std::vector<float> min;
    std::vector<float> max;
};

class sModule
{
public:
    virtual const pTensor forward(const pTensor& input) = 0;
    virtual void backward(pTensor& input) = 0;
    virtual float loss(pTensor& input, const pTensor& target) { assert(0);  return 0.0f; }

    virtual const pTensor activations() const { assert(0); return sTensor::nullPtr; }

    virtual void update_weights(const float lr) {}
    virtual void zero_grad() {}

    virtual const std::map<std::string, pTensor> parameters() const { return std::map<std::string, pTensor>(); }
    virtual const char* name() const { return "sModule"; }

    virtual ~sModule() {}
};

class sLayer : public sModule
{
public:
    sLayer() : _activations(sTensor::Empty()) {}
    const pTensor activations() const override { return _activations;}
    pTensor _activations;

    void collect_stats();
    sStats _activationStats;
};

class sRelu : public sLayer
{
public:
    sRelu();
    const char* name() const override { return "sRelu"; }

    const pTensor forward(const pTensor& input) override;
    void backward(pTensor& input) override;
};


class sLinear : public sLayer
{
public:
    sLinear(uint in_features, uint out_features);
    const char* name() const override { return "sLinear"; }

    const pTensor forward(const pTensor& input) override;
    void backward(pTensor& input) override;
    void update_weights(const float lr) override;
    void zero_grad() override;

    const std::map<std::string, pTensor> parameters() const override;

    pTensor _weights;
    pTensor _bias;
};

class sMSE : public sLayer
{
public:
    sMSE();

    const pTensor forward(const pTensor& input) override;
    void backward(pTensor& input) override;
    float loss(pTensor& input, const pTensor& target) override;

    pTensor _diff;
};

class sSoftMax : public sLayer
{
public:
    sSoftMax();
    const char* name() const override { return "sSoftMax"; }

    const pTensor forward(const pTensor& input) override;
    void backward(pTensor& input) override;
    float loss(pTensor& input, const pTensor& target) override;

    pTensor _diff;
};

class sModel : public sModule
{
public:
  
    sModel(const uint nInputs, const uint nHidden, const uint nOutputs);
    ~sModel();

    const pTensor forward(const pTensor& input) override;
    void backward(pTensor& input) override;
    float loss(pTensor& input, const pTensor& target) override;

    std::vector<sModule*> _layers;
    sMSE* _smeLayer;
    sSoftMax* _smLayer;
    const uint _nInputs;
    const uint _nHidden;
    const uint _nOutputs;
    float _loss;
    float _accuracy;
};

// model utils

pTensor lin(const pTensor& x, const pTensor& w, const pTensor& b);
pTensor relu(pTensor& x);
pTensor softmax(const pTensor& x);
pTensor log_softmax(const pTensor& x);
pTensor logsumexp(const pTensor& x);
pTensor log_softmax2(const pTensor& x);
float nll_loss(pTensor& input, const pTensor& target);
float cross_entropy_loss(const pTensor& input, const pTensor& target);

