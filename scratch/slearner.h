#pragma once

#include "stensor.h"
#include "smodel.h"

class sLearner
{
public:
    sLearner(sModel& model, const sTensor& images, const sTensor& categories, uint batchSize, float lr)
        : _model(model), _images(images), _categories(categories), _batchSize(batchSize), _lr(lr), _nImages(images.dim(0))
    {
    }

    void fit(uint epochs)
    {
        for (uint epoch = 0; epoch < epochs; epoch++)
        {
            step_epoch();
        }
    }

    void step_epoch()
    {
        for (uint i = 0; i < _images.dim(0); i += _batchSize)
        {
            sTensor xb = _images.slice_rows(i, i + _batchSize);
            sTensor yb = _categories.slice_rows(i, i + _batchSize);

            step_batch(xb, yb);
        }
    }

    void step_batch(sTensor& xb, const sTensor& yb)
    {
        sTensor preds = _model.forward(xb);
        float L = _model.loss(xb, yb);
        slog("loss: %f", L);

        for (auto& layer : _model._layers)
        {
            if (layer->activations().grad())
            {
                layer->update_weights(_lr);
            }
        }

        for (auto& layer : _model._layers)
            layer->zero_grad();
    }

    void step()
    {
        slog("batch: %d epoch: %d", _batch, _epoch);
        sTensor xb = _images.slice_rows(_batch, _batch + _batchSize);
        sTensor yb = _categories.slice_rows(_batch, _batch + _batchSize);

        step_batch(xb, yb);

        _batch += _batchSize;
        if (_batch >= _nImages)
        {
            _batch = 0;
            _epoch++;
        }
    }

    sModel& _model;
    const sTensor& _images;
    const sTensor& _categories;
    const uint _batchSize;
    float _lr;

    const uint _nImages;
    uint _epoch = 0;
    uint _batch = 0;
};
