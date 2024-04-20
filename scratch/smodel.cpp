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
    pTensor t = target->rank() > 1 ? target->squeeze() : target;
    return -input->index_select(t)->mean();
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
    const uint nChannels = images->dim(1);
    const uint nRows = images->dim(2);
    const uint nCols = images->dim(3);

    const uint outputHeight = (nRows - ksize) / stride + 1;
    const uint outputWidth = (nCols - ksize) / stride + 1;

    pTensor out = sTensor::Dims(nImages, nChannels * ksize * ksize, outputHeight * outputWidth);

    const float* data = images->data();
    float* out_data = out->data();
    for (uint nImage = 0; nImage < nImages; nImage++)
    {
        for (uint nChannel = 0; nChannel < nChannels; nChannel++)
        {
            for (uint i = 0; i < outputHeight; i++)
            {
                for (uint j = 0; j < outputWidth; j++)
                {
                    for (uint k1 = 0; k1 < ksize; k1++)
                    {
                        for (uint k2 = 0; k2 < ksize; k2++)
                        {
                            const uint source_index = (nImage * nChannels * nRows * nCols) + (nChannel * nRows * nCols) + (i + k1) * nCols + j * stride + k2;
                            const uint row = nImage * nChannels * outputHeight * outputWidth + nChannel * outputHeight * outputWidth + i * outputWidth + j;
                            const uint col = k1 * ksize + k2;
                            out_data[row * ksize * ksize + col] = data[source_index];
                        }
                    }
                }
            }
        }
    }
    return out;
}


 // https://github.com/pjreddie/darknet/blob/master/src/col2im.c

void col2im_add_pixel(float* im, const int height, const int width, const int channels,
    int row, int col, const int channel, const int pad, const float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width * (row + height * channel)] += val;
}

pTensor fold_multiple(pTensor& images, const uint ksize, const uint stride)
{
    const uint nImages = images->dim(0);
    const uint nPatches = images->dim(1);
    const uint nBlocks = images->dim(2);

    const uint nChannels = nPatches / (ksize * ksize);

    const uint h_col = uint(std::sqrt(nBlocks));
    const uint w_col = uint(std::sqrt(nBlocks));

    const uint height = (h_col - 1) * stride + ksize;
    const uint width = (w_col - 1) * stride + ksize;

    pTensor out = sTensor::Dims(nImages, nChannels, height, width);

    const float* data_col = images->data();
    float* data_im = out->data();

    for (uint nImage = 0; nImage < nImages; nImage++)
    {
        const uint channels_col = nChannels * ksize * ksize;
        for (uint c = 0; c < channels_col; c++)
        {
            const uint w_offset = c % ksize;
            const uint h_offset = (c / ksize) % ksize;
            const uint c_im = c / ksize / ksize;
            for (uint h = 0; h < h_col; ++h) {
                for (uint w = 0; w < w_col; ++w) {
                    const int im_row = h_offset + h * stride;
                    const int im_col = w_offset + w * stride;
                    const int col_index = (c * h_col + h) * w_col + w;
                    const float val = data_col[col_index];
                    col2im_add_pixel(data_im, height, width, nChannels,
                        im_row, im_col, c_im, 0, val);
                }
            }
        }
    }
    return out;
}

// single image and single kernel
pTensor conv_manual_simple(pTensor& input, const pTensor& kernel, const uint stride, const uint padding)
{
    const uint inRows = input->dim(0);
    const uint inCols = input->dim(1);
    const uint kRows = kernel->dim(0);
    const uint kCols = kernel->dim(1);

    pTensor padded = padding == 0 ? input : input->pad2d(padding);

    const uint outRows = (inRows - kRows + 2 * padding) / stride + 1;
    const uint outCols = (inCols - kCols + 2 * padding) / stride + 1;
    pTensor result = sTensor::Zeros(outRows, outCols);

    for (uint i = 0; i < outRows; i++)
    {
        for (uint j = 0; j < outCols; j++)
        {
            pTensor slice = padded->slice2d(i*stride, i*stride + kCols, j*stride, j*stride + kRows);
            result->set2d(i, j, (slice * kernel)->sum());
        }
    }
    return result;
}

// multiple images and multiple kernels 
// input: (batches, channels, rows, cols)
// kernels: (features, channels, kRows, kCols)
// output: (batches, features, outRows, outCols)
pTensor conv_manual_batch(pTensor& input, const pTensor& kernels, const uint stride, const uint padding)
{
    const uint batchSizeN = input->dim(0);
    const uint inChannels = input->dim(1);
    const uint inRows = input->dim(2);
    const uint inCols = input->dim(3);

    const uint nKernels = kernels->dim(0);
    const uint kChannels = kernels->dim(1);
    const uint kRows = kernels->dim(2);
    const uint kCols = kernels->dim(3);

    assert(inChannels == kChannels);

    pTensor padded = padding == 0 ? input : input->pad_images(padding);

    const uint outRows = (inRows - kRows + 2 * padding) / stride + 1;
    const uint outCols = (inCols - kCols + 2 * padding) / stride + 1;
    pTensor result = sTensor::Zeros(batchSizeN, nKernels, outRows, outCols);

    const float* s = padded->data();
    float* r = result->data();
    for (uint n = 0; n < batchSizeN; n++)
    {
        const uint batchBytes= inChannels * (inRows + 2 * padding) * (inCols + 2 * padding);
        const uint batchBegin = n * batchBytes;
        pTensor slice = sTensor::Dims(kRows, kCols);
        float* ss = slice->data();

        for (uint k = 0; k < nKernels; k++)
        {
            for (uint c = 0; c < inChannels; c++)
            {
                const uint channelSize = (inRows + 2 * padding) * (inCols + 2 * padding);
                const uint channelBegin = batchBegin + c * channelSize;
                for (uint i = 0; i < outRows; i++)
                {
                    for (uint j = 0; j < outCols; j++)
                    {
                        // populate slice from the image data for this channel
                        for (uint k1 = 0; k1 < kRows; k1++)
                        {
                            for (uint k2 = 0; k2 < kCols; k2++)
                            {
                                const uint index = channelBegin + k2 + k1 * (inCols + 2 * padding) + (j* stride) + i * stride * (inCols + 2 * padding);
                                ss[k2 + k1 * kCols] = s[index];
                            }
                        }
                        pTensor kernel = kernels->select2d(k, c)->squeeze_();
                        const float* kk = kernel->data();
                        result->add4d(n, k, i, j, (slice * kernel)->sum());
                    }
                }
            }

        }
    }
    return result;
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
    _weights = sTensor::NormalDistribution(0.0f, 0.05f, in_features, out_features);
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


// ---------------- sManualConv2d ----------------

sManualConv2d::sManualConv2d(const uint num_channels, const uint num_features, const uint kernel_size, const uint stride, const uint padding) :
    sLayer(),
    _num_channels(num_channels),
    _num_features(num_features),
    _kernel_size(kernel_size),
    _stride(stride),
    _padding(padding == 0 ? 0 : kernel_size / 2),
    _weights(sTensor::Empty()),
    _bias(sTensor::Empty())
{
    _weights = sTensor::NormalDistribution(0.0f, 0.1f, num_features, num_channels, kernel_size, kernel_size);
    _bias = sTensor::Zeros(uint(1), num_features, uint(1), uint(1));
}

const pTensor sManualConv2d::forward(pTensor& input)
{
    // format is (batch, channels, rows, cols)
    const uint batchSize = input->dim(0);
    const uint channels = input->dim(1);
    assert(channels == _num_channels);

    _activations = conv_manual_batch(input, _weights, _stride, _padding);
    _activations += _bias;
    return _activations;
}

void sManualConv2d::backward(pTensor& input)
{
    pTensor ag = _activations->grad();

    {
        // update gradients for input
        pTensor w = _weights->clone_shallow()->reshape_(_num_channels * _kernel_size * _kernel_size, _num_features)->Transpose();
        input->set_grad(ag->MatMult(w));
    }

    pTensor ag_reshaped = ag->clone_shallow()->reshape_(_activations->dim(0), _activations->dim(1), _activations->dim(2), _activations->dim(3));

    //for each weight:
    //    set to zero
    //    for each convolution position in the input :
    //        get the slice of input values
    //        multiply by gradient of corresponding activation / output
    //        add to weight


    const uint batchSizeN = input->dim(0);
    const uint inChannels = input->dim(1);
    const uint inRows = input->dim(2);
    const uint inCols = input->dim(3);

    const uint nKernels = _weights->dim(0);
    const uint kChannels = _weights->dim(1);
    const uint kRows = _weights->dim(2);
    const uint kCols = _weights->dim(3);

    pTensor padded = _padding == 0 ? input : input->pad_images(_padding);

    const uint outRows = (inRows - kRows + 2 * _padding) / _stride + 1;
    const uint outCols = (inCols - kCols + 2 * _padding) / _stride + 1;

    //pTensor slice = sTensor::Dims(_kernel_size, _kernel_size);
    //float* ss = slice->data();
    float* w = _weights->data();
    float* p = padded->data();

    //for (uint i = 0; i < _weights->size(); i++)
    //{
    //     w[i] = 0.0f;
    //     for (uint i = 0; i < outRows; i++)
    //     {
    //         for (uint j = 0; j < outCols; j++)
    //         {
    //             pTensor slice = padded->slice2d(i * _stride, i * _stride + _kernel_size, j * _stride, j * _stride + _kernel_size);
    //             float activation_grad = ag->data()[i * outCols + j];
    //             w[i] += (slice * activation_grad)->sum();
    //         }
    //     }
    //}

    pTensor grads = _weights->grad().isnull() ? _weights->clone() : _weights->grad();
    grads->zero_();

    for (uint n = 0; n < batchSizeN; n++)
    {
        const uint batchBytes = inChannels * (inRows + 2 * _padding) * (inCols + 2 * _padding);
        const uint batchBegin = n * batchBytes;

        for (uint k = 0; k < nKernels; k++)
        {
            for (uint c = 0; c < inChannels; c++)
            {
                for (uint k1 = 0; k1 < kRows; k1++)
                {
                    for (uint k2 = 0; k2 < kCols; k2++)
                    {
                        for (uint i = 0; i < outRows; i++)
                        {
                            for (uint j = 0; j < outCols; j++)
                            {
                                // populate slice from padded input
                                pTensor slice = padded->slice4d(n, n + 1, c, c + 1, i, i + 3, j, j + 3);

                                float activation_grad = ag_reshaped->get4d(n, k, i, j);
                                grads += (slice * activation_grad);
                            }
                        }
                    }
                }
            }
        }
    }

    _weights->set_grad(grads);
    _bias->set_grad(ag->sum_rows());
    _bias->grad()->reshape_(uint(1), _num_features, uint(1), uint(1));
}

void sManualConv2d::update_weights(const float lr)
{
    _weights = _weights - (_weights->grad() * lr);
    _bias = _bias - (_bias->grad() * lr);
}

std::map<std::string, pTensor> sManualConv2d::parameters() const
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
    _weights(sTensor::Empty()),
    _bias(sTensor::Empty())
{
    _weights = sTensor::NormalDistribution(0.0f, 0.1f, num_channels * num_features, kernel_size * kernel_size);
    _bias = sTensor::Zeros(uint(1), num_features, uint(1), uint(1));
}

const pTensor sConv2d::forward(pTensor& input)
{
    // format is (batch, channels, rows, cols)
    const uint batchSize = input->dim(0);
    const uint channels = input->dim(1);
    assert(channels == _num_channels);

    const uint width = input->dim(2);
    const uint height = input->dim(3);

    pTensor padded_images = input->clone_shallow();
    if (_padding != 0)
    {
        padded_images = padded_images->pad_images(1);
    }
    
    pTensor unfolded_images = unfold_multiple(padded_images, _kernel_size, _stride);
    const uint nBlocks = unfolded_images->dim(2);
    const uint sz = unfolded_images->size() / (_kernel_size * _kernel_size);
    unfolded_images->reshape_(sz, _kernel_size * _kernel_size);


    pTensor col = unfolded_images->MatMult(_weights->Transpose());
    col->reshape_(batchSize, _num_channels * _kernel_size * _kernel_size, nBlocks);
    _activations = fold_multiple(col, _kernel_size, _stride);
    _activations += _bias;
    return _activations;
}

void sConv2d::backward(pTensor& input)
{
    input->set_grad(_activations->grad()->MatMult(_weights->Transpose()));

    _weights->set_grad(input->Transpose()->MatMult(_activations->grad()));

    _bias->set_grad(_activations->grad()->sum_rows());
}

void sConv2d::update_weights(const float lr)
{
    _weights = _weights - (_weights->grad() * lr);
    _bias = _bias - (_bias->grad()->unsqueeze(0) * lr);
}

std::map<std::string, pTensor> sConv2d::parameters() const
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
    _diff = (target - _activations->squeeze());
    return _diff->mse();
}

// ---------------- sCrossEntropy ----------------
// ---------------- sSoftMax ----------------

sSoftMax::sSoftMax() : sLayer(), _diff(sTensor::Empty()), _sm(sTensor::Empty())
{
}

const pTensor sSoftMax::forward(pTensor& input)
{
    _activations = input;
    pTensor tmp = _activations->clone_shallow()->reshape_(input->dim(0) * input->dim(1), input->dim(2) * input->dim(3));
    _sm = softmax(tmp);
    return _activations;
}

void sSoftMax::backward(pTensor& input)
{
    // loss must have been called first to populate _diff!
    input->set_grad(_diff / float(input->dim(0)));
}

float sSoftMax::loss(pTensor& _, const pTensor& target)
{
    // _activations IS the input
    
    // need gradients for each of the activations, not just the target
    const uint nrows = _activations->dim(0);
    const uint ncols = _activations->dim(1);

    // remove trailing dimensions from activations
    while (_activations->rank() > 2)
    {
        _activations = _activations->squeeze(_activations->rank()-1);
    }

    // remove trailing dimensions from target
    pTensor tar = target->clone_shallow();
    while (tar->rank() > 1)
    {
        tar = tar->squeeze(tar->rank()-1);
    }

    pTensor grads = _activations->clone();
    for (uint r = 0; r < nrows; r++)
    {
        uint t = uint(tar->get1d(r));
        for (uint c = 0; c < ncols; c++)
        {
            const float ytrue = (c == t) ? 1.f : 0.f;
            const float ypred = _sm->get2d(r, c);
            grads->set2d(r, c, ypred - ytrue);
        }
    }
    _diff = grads;

    float l = cross_entropy_loss(_activations, target);
    return l;
}

// ---------------- sCrossEntropy ----------------

sCrossEntropy::sCrossEntropy() : sLayer(), _diff(sTensor::Empty()), _sm(sTensor::Empty())
{
}

const pTensor sCrossEntropy::forward(pTensor& input)
{
    _activations = input;
    pTensor tmp = _activations->clone_shallow()->reshape_(input->dim(0) * input->dim(1), input->dim(2) * input->dim(3));
    _sm = softmax(tmp);
    return _activations;
}

void sCrossEntropy::backward(pTensor& input)
{
    // loss must have been called first to populate _diff!
    input->set_grad(_diff / float(input->dim(0)));
}

float sCrossEntropy::loss(pTensor& _, const pTensor& target)
{
    // _activations IS the input

    // need gradients for each of the activations, not just the target
    const uint nrows = _activations->dim(0);
    const uint ncols = _activations->dim(1);

    // remove trailing dimensions from activations
    while (_activations->rank() > 2)
    {
        _activations = _activations->squeeze(_activations->rank() - 1);
    }

    // remove trailing dimensions from target
    pTensor tar = target->clone_shallow();
    while (tar->rank() > 1)
    {
        tar = tar->squeeze(tar->rank() - 1);
    }

    pTensor grads = _activations->clone();
    for (uint r = 0; r < nrows; r++)
    {
        uint t = uint(tar->get1d(r));
        for (uint c = 0; c < ncols; c++)
        {
            const float ytrue = (c == t) ? 1.f : 0.f;
            const float ypred = _sm->get2d(r, c);
            grads->set2d(r, c, ypred - ytrue);
        }
    }
    _diff = grads;

    float l = cross_entropy_loss(_activations, target);
    return l;
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
    _cachedInput = input;
    pTensor x = input;
    for (auto& layer : _layers)
    {
        x = layer->forward(x);
    }
    _cachedOutput = _layers.back()->activations();
    return _cachedOutput;
}

void sModel::backward(pTensor& input)
{
    assert(0); // use loss to backprop
}

float sModel::loss(pTensor& input, const pTensor& target)
{
    _cachedTarget = target;

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


// output of the matmul is interleaved, this undoes that
pTensor reorder_data(const pTensor& input)
{
    // Get the dimensions of the input tensor
    const uint n = input->dim(0); // 2
    const uint sz = input->dim(1); // 784

    // Create a new tensor to hold the reordered data
    pTensor output = sTensor::Dims(sz, n);

    // Copy the data from the input tensor to the output tensor in the desired order
    for (uint i = 0; i < n; i++)
    {
        for (uint j = 0; j < sz; j++)
        {
            float value = input->get2d(i, j);
            output->set2d(j, i, value);
        }
    }
    return output;
}