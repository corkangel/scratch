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
    virtual const pTensor forward(pTensor& input) = 0;
    virtual void backward(pTensor& input) = 0;
    virtual float loss(pTensor& input, const pTensor& target);

    virtual const pTensor activations() const;

    virtual void update_weights(const float lr) {}
    virtual void zero_grad() {}

    virtual std::map<std::string, pTensor> parameters() const { return std::map<std::string, pTensor>(); }
    virtual const char* name() const { return "sModule"; }

    virtual ~sModule() {}
};

class sLayer : public sModule
{
public:
    sLayer();
    const pTensor activations() const override { return _activations;}
    void zero_grad() override;

    uint _id;
    pTensor _activations;

    void collect_stats();
    sStats _activationStats;
};

class sRelu : public sLayer
{
public:
    sRelu();
    const char* name() const override { return "sRelu"; }

    const pTensor forward(pTensor& input) override;
    void backward(pTensor& input) override;
};


class sLinear : public sLayer
{
public:
    sLinear(uint in_features, uint out_features);
    const char* name() const override { return "sLinear"; }

    const pTensor forward(pTensor& input) override;
    void backward(pTensor& input) override;
    void update_weights(const float lr) override;

    std::map<std::string, pTensor> parameters() const override;

    pTensor _weights;
    pTensor _bias;
};

class sManualConv2d : public sLayer
{
public:
    sManualConv2d(const uint num_channels, const uint num_features, const uint kernel_size = 3, const uint stride = 2, const uint padding = 1);
    const char* name() const override { return "sManualConv2d"; }

    const pTensor forward(pTensor& input) override;
    void backward(pTensor& input) override;
    void update_weights(const float lr) override;

    std::map<std::string, pTensor> parameters() const override;

    const uint _num_channels;
    const uint _num_features;
    const uint _kernel_size;
    const uint _stride;
    const uint _padding;
    pTensor _weights;
    pTensor _bias;
};

class sConv2d : public sLayer
{
public:
    sConv2d(const uint num_channels, const uint num_features, const uint kernel_size = 3, const uint stride = 2, const uint padding = 1);
    const char* name() const override { return "sConv2d"; }

    const pTensor forward(pTensor& input) override;
    void backward(pTensor& input) override;
    void update_weights(const float lr) override;

    std::map<std::string, pTensor> parameters() const override;

    const uint _num_channels;
    const uint _num_features;
    const uint _kernel_size;
    const uint _stride;
    const uint _padding;
    pTensor _weights;
    pTensor _bias;
};

class sMSE : public sLayer
{
public:
    sMSE();

    const pTensor forward(pTensor& input) override;
    void backward(pTensor& input) override;
    float loss(pTensor& input, const pTensor& target) override;

    pTensor _diff;
};

class sSoftMax : public sLayer
{
public:
    sSoftMax();
    const char* name() const override { return "sSoftMax"; }

    const pTensor forward(pTensor& input) override;
    void backward(pTensor& input) override;
    float loss(pTensor& input, const pTensor& target) override;

    pTensor _diff;
    pTensor _sm;
};


class sCrossEntropy : public sLayer
{
public:
    sCrossEntropy();
    const char* name() const override { return "sCrossEntropy"; }

    const pTensor forward(pTensor& input) override;
    void backward(pTensor& input) override;
    float loss(pTensor& input, const pTensor& target) override;

    pTensor _diff;
    pTensor _sm;
};

class sModel : public sModule
{
public:
  
    sModel(const uint nInputs, const uint nHidden, const uint nOutputs);
    ~sModel();

    const pTensor forward(pTensor& input) override;
    void backward(pTensor& input) override;
    float loss(pTensor& input, const pTensor& target) override;

    void add_layer(sLayer* layer);

    std::vector<sModule*> _layers;
    const uint _nInputs;
    const uint _nHidden;
    const uint _nOutputs;
    float _loss;
    float _accuracy;

    pTensor _cachedInput;
    pTensor _cachedOutput;
    pTensor _cachedTarget;
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

pTensor unfold_single(pTensor& image, const uint ksize, const uint stride);
pTensor unfold_multiple(pTensor& images, const uint ksize, const uint stride);

pTensor fold_multiple(pTensor& images, const uint ksize, const uint stride);

pTensor conv_manual_simple(pTensor& input, const pTensor& kernel, const uint stride, const uint padding);
pTensor conv_manual_batch(pTensor& input, const pTensor& kernel, const uint stride, const uint padding);

// output of the matmul is interleaved, this undoes that
pTensor reorder_data(const pTensor& input);
