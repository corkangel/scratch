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
    virtual const sTensor& forward(const sTensor& input) = 0;
    virtual void backward(sTensor& input) = 0;
    virtual float loss(sTensor& input, const sTensor& target) { assert(0);  return 0.0f; }

    virtual const sTensor& activations() const { assert(0); return sTensor::null; }

    virtual void update_weights(const float lr) {}
    virtual void zero_grad() {}

    virtual std::map<std::string, const sTensor*> parameters() const { return std::map<std::string, const sTensor*>(); }
    virtual const char* name() const { return "sModule"; }

    virtual ~sModule() {}
};

class sLayer : public sModule
{
public:
    sLayer() : _activations(sTensor::Empty()) {}
    const sTensor& activations() const override { return _activations;}
    sTensor _activations;

    void collect_stats();
    sStats _activationStats;
};

class sRelu : public sLayer
{
public:
    sRelu();
    const char* name() const override { return "sRelu"; }

    const sTensor& forward(const sTensor& input) override;
    void backward(sTensor& input) override;
};


class sLinear : public sLayer
{
public:
    sLinear(uint in_features, uint out_features);
    const char* name() const override { return "sLinear"; }

    const sTensor& forward(const sTensor& input) override;
    void backward(sTensor& input) override;
    void update_weights(const float lr) override;
    void zero_grad() override;

    std::map<std::string, const sTensor*> parameters() const override;

    sTensor _weights;
    sTensor _bias;
};

class sMSE : public sLayer
{
public:
    sMSE();

    const sTensor& forward(const sTensor& input) override;
    void backward(sTensor& input) override;
    float loss(sTensor& input, const sTensor& target) override;

    sTensor _diff;
};

class sSoftMax : public sLayer
{
public:
    sSoftMax();
    const char* name() const override { return "sSoftMax"; }

    const sTensor& forward(const sTensor& input) override;
    void backward(sTensor& input) override;
    float loss(sTensor& input, const sTensor& target) override;

    sTensor _diff;
};

class sModel : public sModule
{
public:
  
    sModel(const uint nInputs, const uint nHidden, const uint nOutputs);
    ~sModel();

    const sTensor& forward(const sTensor& input) override;
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

