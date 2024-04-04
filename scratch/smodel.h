#pragma once

#include "stensor.h"

class sModule
{
public:
    virtual sTensor& forward(const sTensor& input) = 0;
    virtual void backward(sTensor& input) = 0;
    virtual float loss(sTensor& input, const sTensor& target) { assert(0);  return 0.0f; }

    virtual sTensor& activations() { assert(0); return sTensor::null; }

    virtual void update_weights(const float lr) {}
    virtual void zero_grad() {}

    virtual ~sModule() {}
};


class sRelu : public sModule
{
public:
    sRelu();

    sTensor& forward(const sTensor& input) override;
    void backward(sTensor& input) override;
    sTensor& activations() override;

    sTensor _activations;
};


class sLinear : public sModule
{
public:
    sLinear(uint in_features, uint out_features);

    sTensor& forward(const sTensor& input) override;
    void backward(sTensor& input) override;
    sTensor& activations() override;
    void update_weights(const float lr) override;
    void zero_grad() override;

    sTensor _activations;
    sTensor _weights;
    sTensor _bias;

};

class sMSE : public sModule
{
public:
    sMSE();

    sTensor& forward(const sTensor& input) override;
    void backward(sTensor& input) override;
    sTensor& activations() override;
    float loss(sTensor& input, const sTensor& target) override;

    sTensor _activations;
    sTensor _diff;
};

class sSoftMax : public sModule
{
public:
    sSoftMax();

    sTensor& forward(const sTensor& input) override;
    void backward(sTensor& input) override;
    float loss(sTensor& input, const sTensor& target) override;
    sTensor& activations() override;

    sTensor _activations;
    sTensor _diff;
};

class sModel : public sModule
{
public:
  
    sModel(const uint nInputs, const uint nHidden, const uint nOutputs);
    ~sModel();

    sTensor& forward(const sTensor& input) override;
    void backward(sTensor& input) override;
    float loss(sTensor& input, const sTensor& target) override;

    std::vector<sModule*> _layers;
    sMSE* _smeLayer;
    sSoftMax* _smLayer;
    const uint _nInputs;
    const uint _nHidden;
    const uint _nOutputs;
};

// model utils

sTensor lin(const sTensor& x, const sTensor& w, const sTensor& b);
sTensor relu(sTensor& x);
sTensor softmax(const sTensor& x);
sTensor log_softmax(const sTensor& x);
sTensor logsumexp(const sTensor& x);
sTensor log_softmax2(const sTensor& x);
float nll_loss(sTensor& input, const sTensor& target);
float cross_entropy_loss(const sTensor& input, const sTensor& target);

