#pragma once

#include "stensor.h"
#include "smodel.h"

enum class sLayerStepState : uint
{
    None,
    Forward,
    Middle,
    Backward,
    End
};

inline const char* layer_step_str(sLayerStepState state)
{
    switch (state)
    {
    case sLayerStepState::None: return "None";
    case sLayerStepState::Forward: return "Forward";
    case sLayerStepState::Middle: return "Middle";
    case sLayerStepState::Backward: return "Backward";
    case sLayerStepState::End: return "End";
    default: return "Unknown";
    }
}

class sLearner
{
public:
    sLearner(sModel& model, const pTensor& images, const pTensor& categories, uint batchSize, float lr)
        : _model(model), _images(images), _categories(categories), _batchSize(batchSize), _lr(lr), _nImages(images->dim(0))
    {
    }

    void fit(uint epochs)
    {
        if (_layerStepState != sLayerStepState::None)
            finish_layer_steps();

        for (uint epoch = 0; epoch < epochs; epoch++)
        {
            step_epoch();
        }
    }

    void step_epoch()
    {
        for (uint i = 0; i < _images->dim(0); i += _batchSize)
        {
            pTensor xb = _images->slice_rows(i, i + _batchSize);
            pTensor yb = _categories->slice_rows(i, i + _batchSize);

            step_batch(xb, yb);
        }
        _epoch++;
    }

    void step_batch(pTensor& xb, const pTensor& yb)
    {
        pTensor preds = _model.forward(xb);
        float L = _model.loss(xb, yb);
        slog("loss: %f", L);

        // do this before instead of after so they can be inspected
        for (auto& layer : _model._layers)
            layer->zero_grad();

        for (auto& layer : _model._layers)
        {
            if (!layer->activations()->grad().isnull())
            {
                layer->update_weights(_lr);
            }
        }
    }

    void step()
    {
        slog("batch: %d epoch: %d", _batch, _epoch);
        pTensor xb = _images->slice_rows(_batch, _batch + _batchSize);
        pTensor yb = _categories->slice_rows(_batch, _batch + _batchSize);

        step_batch(xb, yb);

        _batch += _batchSize;
        if (_batch >= _nImages)
        {
            _batch = 0;
            _epoch++;
        }
    }

    void step_layer_forward()
    {
        _layerStepInput = _model._layers[_layerStepIndex]->forward(_layerStepInput);
        _layerStepIndex++;

        if (_layerStepIndex == _model._layers.size())
        {
            _layerStepState = sLayerStepState::Middle;
            _model.loss(_layerStepInput, _categories);
        }
    }

    void step_layer_backwards()
    {
        pTensor input = (_layerStepIndex == 0) ? _images : _model._layers[_layerStepIndex - 1]->activations();
        _model._layers[_layerStepIndex]->backward(input);

        _layerStepIndex--;
        if (_layerStepIndex == 0)
        {
            _layerStepState = sLayerStepState::End;
        }
    }

    void step_layer()
    {
        switch (_layerStepState)
        {
        case sLayerStepState::None:
            assert(_layerStepIndex == 0);
            _layerStepState = sLayerStepState::Forward;
            _layerStepInput = _images;
            step_layer_forward();
            break;

        case sLayerStepState::Forward:
            step_layer_forward();
            break;

        case sLayerStepState::Middle:
            _layerStepState = sLayerStepState::Backward;
            _layerStepIndex--;
            step_layer_backwards();
            break;

        case sLayerStepState::Backward:
            step_layer_backwards();
            break;

        case sLayerStepState::End:
            _layerStepState = sLayerStepState::None;
            break;

        default:
            // Handle other cases or throw an error
            break;
        }
    }

    void finish_layer_steps()
    {

    }

    sModel& _model;
    const pTensor _images;
    const pTensor _categories;
    const uint _batchSize;
    float _lr;

    const uint _nImages;
    uint _epoch = 0;
    uint _batch = 0;

    sLayerStepState _layerStepState = sLayerStepState::None;
    uint _layerStepIndex = 0;
    pTensor _layerStepInput;
    pTensor _layerStepOutput;

};
