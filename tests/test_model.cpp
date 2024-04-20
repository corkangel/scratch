//tests

#include "scratch/minst.h"
#include "scratch/stensor.h"
#include "scratch/smodel.h"
#include "tests.h"


const uint numInputs = 8;
const uint numHidden = 2;
const uint numOutputs = 10;

const uint g_imageSize = 28;
const uint g_imageArraySize = g_imageSize * g_imageSize;
const uint g_numImagesTrain = 500; // should be 60000. "few-images" has 500-ish images in it for testing.
const uint g_numCategories = 10;
const uint g_numHidden = 50;

pTensor images;
pTensor categories;

// convolution kernels
pTensor top_edge = sTensor::Zeros(3, 3);
pTensor bottom_edge = sTensor::Zeros(3, 3);
pTensor left_edge = sTensor::Zeros(3, 3);
pTensor right_edge = sTensor::Zeros(3, 3);

void init_kernels()
{
    top_edge->data()[0] = -1.f;
    top_edge->data()[1] = -1.f;
    top_edge->data()[2] = -1.f;
    top_edge->data()[6] = 1.f;
    top_edge->data()[7] = 1.f;
    top_edge->data()[8] = 1.f;
}

void t_model_one()
{
    sModel model(10, 50, 10);
    model.add_layer(new sLinear(numInputs, numOutputs));

    pTensor input = sTensor::Ones(uint(1), numInputs);
    model.forward(input);
}

void t_model_conv_manual_simple()
{
    pTensor image = sTensor::Zeros(8, 8);

    // no padding, lose a row and column
    pTensor result = conv_manual_simple(image, top_edge, 2, 0);
    expect_eq_int(3, result->dim(0));
 
    // padding, no loss
    pTensor result1 = conv_manual_simple(image, top_edge, 2, 1);
    expect_eq_int(4, result1->dim(0));

    // stride 1, no padding
    pTensor result2 = conv_manual_simple(image, top_edge, 1, 0);
    expect_eq_int(6, result2->dim(0));

    // stride 1, padding
    pTensor result3 = conv_manual_simple(image, top_edge, 1, 1);
    expect_eq_int(8, result3->dim(0));
}

void t_model_conv_manual_batch1()
{
    pTensor input = sTensor::Ones(4, 1, 8, 8);
    pTensor kernels = sTensor::Ones(1, 1, 3, 3);

    // no padding, 1 kernel
    pTensor result1 = conv_manual_batch(input, kernels, 2, 0);
    expect_eq_int(result1->dim(0), 4);
    expect_eq_int(result1->dim(1), 1);
    expect_eq_int(result1->dim(2), 3);
    expect_eq_int(result1->dim(3), 3);

    // padding, 1 kernel
    pTensor result2 = conv_manual_batch(input, kernels, 2, 1);
    expect_eq_int(result2->dim(0), 4);
    expect_eq_int(result2->dim(1), 1);
    expect_eq_int(result2->dim(2), 4);
    expect_eq_int(result2->dim(3), 4);
}

void t_model_conv_manual_batch2()
{
    pTensor input = sTensor::Ones(12, 5, 8, 8);
    pTensor kernels = sTensor::Ones(4, 5, 3, 3);

    // no padding, 
    pTensor result1 = conv_manual_batch(input, kernels, 2, 0);
    expect_eq_int(result1->dim(0), 12);
    expect_eq_int(result1->dim(1), 4);
    expect_eq_int(result1->dim(2), 3);
    expect_eq_int(result1->dim(3), 3);

    // padding, 
    pTensor result2 = conv_manual_batch(input, kernels, 2, 1);
    expect_eq_int(result2->dim(0), 12);
    expect_eq_int(result2->dim(1), 4);
    expect_eq_int(result2->dim(2), 4);
    expect_eq_int(result2->dim(3), 4);
}

/*
# t_model_conv_layer_shape
import torch
from torch import nn

batch_size = 6
input_channels = 5
output_features = 2
stride = 2
padding = 1

input_data = torch.randn(64, input_channels, 4, 4)
c1 = nn.Conv2d(input_channels, output_features, kernel_size=3, stride=2, padding=1)
mid = c1(input_data)

c2 = nn.Conv2d(2, 10, kernel_size=2, stride=2, padding=0)
output_data = c2(mid)

# Print the sizes of all the tensors
print(f"Input data size: {input_data.size()}")

print(f"Convolutional layer1 weight size: {c1.weight.size()}")
print(f"Convolutional layer1 bias size: {c1.bias.size()}")
print(f"Convolutional layer2 weight size: {c2.weight.size()}")
print(f"Convolutional layer2 bias size: {c2.bias.size()}")

print(f"Output data size: {output_data.size()}")

Input data size: torch.Size([64, 5, 4, 4])
Convolutional layer1 weight size: torch.Size([2, 5, 3, 3])
Convolutional layer1 bias size: torch.Size([2])
Convolutional layer2 weight size: torch.Size([10, 2, 2, 2])
Convolutional layer2 bias size: torch.Size([10])
Output data size: torch.Size([64, 10, 1, 1])
*/
void t_model_conv_layer_shape()
{
    constexpr uint batchSize = 6;
    constexpr uint kSize = 3;
    constexpr uint nInputChannels = 5;
    constexpr uint nOutputChannels = 2;
    constexpr uint nPixels = 4;

    pTensor input = sTensor::Ones(batchSize, nInputChannels, nPixels, nPixels);
    sManualConv2d conv(nInputChannels, nOutputChannels, kSize);

    expect_eq_int(batchSize, input->dim(0));
    expect_eq_int(nInputChannels, input->dim(1));
    expect_eq_int(nPixels, input->dim(2));
    expect_eq_int(nPixels, input->dim(3));

    expect_eq_int(nOutputChannels, conv._weights->dim(0));
    expect_eq_int(nInputChannels, conv._weights->dim(1));
    expect_eq_int(3, conv._weights->dim(2));
    expect_eq_int(3, conv._weights->dim(3));

    expect_eq_int(nOutputChannels, conv._bias->dim(1));
    
    pTensor result = conv.forward(input);
    expect_eq_int(batchSize, result->dim(0));
    expect_eq_int(nOutputChannels, result->dim(1));
    expect_eq_int(2, result->dim(2));
    expect_eq_int(2, result->dim(3));

    sManualConv2d conv2(2, 10, 2, 2, 0);
    pTensor result2 = conv2.forward(result);

    expect_eq_int(10, conv2._weights->dim(0));
    expect_eq_int(2, conv2._weights->dim(1));
    expect_eq_int(2, conv2._weights->dim(2));
    expect_eq_int(2, conv2._weights->dim(3));
}

void t_model_conv_layer_forward()
{
    constexpr uint batchSize = 3;
    constexpr uint kSize = 3;

    pTensor result;
    {
        constexpr uint nPixels = 28;
        constexpr uint nInputChannels = 1;
        constexpr uint nOutputChannels = 4;

        // format for CNN is (batch, input_channels, rows, columns)
        pTensor input = sTensor::Ones(batchSize, nInputChannels, nPixels, nPixels);
        sManualConv2d conv(nInputChannels, nOutputChannels, kSize);
        result = conv.forward(input);
        expect_eq_int(batchSize, result->dim(0));
        expect_eq_int(nOutputChannels, result->dim(1));
        expect_eq_int(14, result->dim(2));
        expect_eq_int(14, result->dim(3));
    }

    {
        constexpr uint nPixels = 14;
        constexpr uint nInputChannels = 4;
        constexpr uint nKernels = 8;

        // format for CNN is (batch, input_channels, rows, columns)
        sManualConv2d conv(nInputChannels, nKernels, kSize);
        result = conv.forward(result);
        expect_eq_int(batchSize, result->dim(0));
        expect_eq_int(nKernels, result->dim(1));
        expect_eq_int(7, result->dim(2));
        expect_eq_int(7, result->dim(3));
    }
}

/*
# t_model_conv_layer_values_single
batch_size = 1
input_channels = 1
output_features = 4
stride = 2
padding = 1

input_data = torch.ones(1, input_channels, 8, 8)
conv_layer = nn.Conv2d(input_channels, 4, kernel_size=3, stride=2, padding=1)
conv_layer.weight.data.fill_(.1)
conv_layer.bias.data.fill_(.1)

# Pass the input data through the convolutional layer
output_data = conv_layer(input_data)

print(f"Output shape: {output_data.shape}")
print(f"Output data: {output_data}")

Output shape: torch.Size([1, 4, 4, 4])
Output data: tensor([[
       [[0.5000, 0.7000, 0.7000, 0.7000],
        [0.7000, 1.0000, 1.0000, 1.0000],
        [0.7000, 1.0000, 1.0000, 1.0000],
        [0.7000, 1.0000, 1.0000, 1.0000]],
        ...
       [[0.5000, 0.7000, 0.7000, 0.7000],
        [0.7000, 1.0000, 1.0000, 1.0000],
        [0.7000, 1.0000, 1.0000, 1.0000],
        [0.7000, 1.0000, 1.0000, 1.0000]]]], grad_fn=<ConvolutionBackward0>)
*/
void t_model_conv_layer_values_single()
{
    constexpr uint batchSize = 1;
    constexpr uint kSize = 3;
    constexpr uint nInputChannels = 1;
    constexpr uint nOutputChannels = 4;
    constexpr uint nPixels = 8;

    // format for CNN is (batch, input_channels, rows, columns)
    pTensor input = sTensor::Ones(batchSize, nInputChannels, nPixels, nPixels);
    sManualConv2d conv(nInputChannels, nOutputChannels, kSize, 2, 1);
    conv._weights->fill_(.1f);
    conv._bias->fill_(.1f);

    pTensor result = conv.forward(input);
    expect_eq_int(batchSize, result->dim(0));
    expect_eq_int(nOutputChannels, result->dim(1));
    expect_eq_int(4, result->dim(2));
    expect_eq_int(4, result->dim(3));

    // first item
    expect_eq_float(0.5f, result->data()[0]);
    expect_eq_float(0.7f, result->data()[1]);
    expect_eq_float(0.7f, result->data()[2]);
    expect_eq_float(0.7f, result->data()[3]);

    // last item
    expect_eq_float(0.5f, result->data()[3 * 4 * 4]);
    expect_eq_float(0.7f, result->data()[3 * 4 * 4 + 1]);
    expect_eq_float(0.7f, result->data()[3 * 4 * 4 + 2]);
    expect_eq_float(1.0f, result->data()[3 * 4 * 4 + 15]);
}

/*
# t_model_conv_layer_values_batch
batch_size = 6
input_channels = 1
output_features = 4
stride = 2
padding = 1

input_data = torch.ones(1, input_channels, 8, 8)
conv_layer = nn.Conv2d(input_channels, 4, kernel_size=3, stride=2, padding=1)
conv_layer.weight.data.fill_(.1)
conv_layer.bias.data.fill_(.1)

# Pass the input data through the convolutional layer
output_data = conv_layer(input_data)

print(f"Output shape: {output_data.shape}")
print(f"Output data: {output_data}")

Output shape: torch.Size([6, 4, 4, 4])
Output data: tensor([[
     [[0.5000, 0.7000, 0.7000, 0.7000],
      [0.7000, 1.0000, 1.0000, 1.0000],
      [0.7000, 1.0000, 1.0000, 1.0000],
      [0.7000, 1.0000, 1.0000, 1.0000]],
      ...
     [[0.5000, 0.7000, 0.7000, 0.7000],
      [0.7000, 1.0000, 1.0000, 1.0000],
      [0.7000, 1.0000, 1.0000, 1.0000],
      [0.7000, 1.0000, 1.0000, 1.0000]]]], grad_fn=<ConvolutionBackward0>)
*/
void t_model_conv_layer_values_batch()
{
    constexpr uint batchSize = 6;
    constexpr uint kSize = 3;
    constexpr uint nInputChannels = 1;
    constexpr uint nOutputChannels = 4;
    constexpr uint nPixels = 8;

    // format for CNN is (batch, input_channels, rows, columns)
    pTensor input = sTensor::Ones(batchSize, nInputChannels, nPixels, nPixels);
    sManualConv2d conv(nInputChannels, nOutputChannels, kSize, 2, 1);
    conv._weights->fill_(.1f);
    conv._bias->fill_(.1f);

    pTensor result = conv.forward(input);
    expect_eq_int(batchSize, result->dim(0));
    expect_eq_int(nOutputChannels, result->dim(1));
    expect_eq_int(4, result->dim(2));
    expect_eq_int(4, result->dim(3));

    // first item
    expect_eq_float(0.5f, result->data()[0]);
    expect_eq_float(0.7f, result->data()[1]);
    expect_eq_float(0.7f, result->data()[2]);
    expect_eq_float(1.0f, result->data()[15]);

    // last item
    uint result_size = result->size();
    expect_eq_float(0.5f, result->at(5 * 4 * 4 * 4));
    expect_eq_float(0.7f, result->at(5 * 4 * 4 * 4 + 1));
    expect_eq_float(0.7f, result->at(5 * 4 * 4 * 4 + 2));
    expect_eq_float(1.0f, result->at(5 * 4 * 4 * 4 + 15));
}

/*
# t_model_conv_layer_values_batch_channels
batch_size = 6
input_channels = 5
output_features = 4
stride = 2
padding = 1

input_data = torch.ones(batch_size, input_channels, 8, 8)
conv_layer = nn.Conv2d(input_channels, output_features, kernel_size=3, stride=2, padding=1)
conv_layer.weight.data.fill_(.1)
conv_layer.bias.data.fill_(.1)

# Pass the input data through the convolutional layer
output_data = conv_layer(input_data)

print(f"Output shape: {output_data.shape}")
print(f"Output data: {output_data}")

Output shape: torch.Size([6, 4, 4, 4])
Output data: tensor([[
     [[2.1000, 3.1000, 3.1000, 3.1000],
      [3.1000, 4.6000, 4.6000, 4.6000],
      [3.1000, 4.6000, 4.6000, 4.6000],
      [3.1000, 4.6000, 4.6000, 4.6000]],
      ...
     [[2.1000, 3.1000, 3.1000, 3.1000],
      [3.1000, 4.6000, 4.6000, 4.6000],
      [3.1000, 4.6000, 4.6000, 4.6000],
      [3.1000, 4.6000, 4.6000, 4.6000]]]], grad_fn=<ConvolutionBackward0>)
*/
void t_model_conv_layer_values_batch_channels()
{
    constexpr uint batchSize = 2;
    constexpr uint kSize = 3;
    constexpr uint nInputChannels = 2;
    constexpr uint nOutputChannels = 4;
    constexpr uint nPixels = 4;

    // format for CNN is (batch, input_channels, rows, columns)
    pTensor input = sTensor::Ones(batchSize, nInputChannels, nPixels, nPixels);
    sManualConv2d conv(nInputChannels, nOutputChannels, kSize, 2, 1);
    conv._weights->fill_(.1f);
    conv._bias->fill_(.1f);

    pTensor result = conv.forward(input);
    expect_eq_int(batchSize, result->dim(0));
    expect_eq_int(nOutputChannels, result->dim(1));
    expect_eq_int(2, result->dim(2));
    expect_eq_int(2, result->dim(3));

    // first item
    expect_eq_float(0.9f, result->data()[0]);
    expect_eq_float(1.3f, result->data()[1]);
    expect_eq_float(1.3f, result->data()[2]);
    expect_eq_float(1.9f, result->data()[15]);

    // last item
    uint result_size = result->size();
    expect_eq_float(0.9f, result->at(1 * 4 * 2 * 2));
    expect_eq_float(1.3f, result->at(1 * 4 * 2 * 2 + 1));
    expect_eq_float(1.3f, result->at(1 * 4 * 2 * 2 + 2));
    expect_eq_float(1.9f, result->at(1 * 4 * 2 * 2 + 15));
}

/*
# t_model_conv_layer_mse_loss
batch_size = 2
input_channels = 5
output_features = 2

input_data = torch.ones(batch_size, input_channels, 4, 4)

conv_layer1 = nn.Conv2d(input_channels, output_features, kernel_size=3, stride=2, padding=1)
conv_layer1.weight.data.fill_(.1)
conv_layer1.bias.data.fill_(.1)
mid = conv_layer1(input_data)
print(f"mid shape: {mid.shape} 0:{mid[0][0][0][0]} 1:{mid[0][0][0][1]} ")

conv_layer2 = nn.Conv2d(2, 1, kernel_size=2, stride=2, padding=0)
conv_layer2.weight.data.fill_(.1)
conv_layer2.bias.data.fill_(.1)
output = conv_layer2(mid)
print(f"output shape: {output.shape} 0:{output[0][0][0][0]}")

mse = nn.MSELoss()

target = torch.ones(2, 1, 1, 1)
loss = mse(output, target);
loss.backward()

print(f"loss: {loss}")

print(f"Weights gradient shape: {conv_layer1.weight.grad.shape}")
print(f"Bias gradient shape: {conv_layer1.bias.grad.shape}")

mid shape: torch.Size([2, 2, 2, 2])
mid tensor([[
         [[  1.6046,   2.4740],
          [  2.8507,   4.4220]],
          ...
         [[ 7.9638, 12.1238],
          [12.8189, 19.5296]]]], grad_fn=<ConvolutionBackward0>)

output shape: torch.Size([2, 1, 1, 1])
output tensor(
        [[[[ 2.3702]]],
        [[[10.5872]]]], grad_fn=<ConvolutionBackward0>)

loss: 46.896270751953125

*/
void t_model_conv_layer_mse_loss()
{
    constexpr uint batchSize = 2;
    constexpr uint kSize = 3;
    constexpr uint nInputChannels = 5;
    constexpr uint nOutputChannels = 2;
    constexpr uint nPixels = 4;

    // format for CNN is (batch, input_channels, rows, columns)
    //pTensor input = sTensor::Ones(batchSize, nInputChannels, nPixels, nPixels);
    pTensor input = sTensor::Linear(0.0f, 0.1f, batchSize, nInputChannels, nPixels, nPixels)->pow_(1.5f)->multiply_(0.1f);

    sManualConv2d conv1(nInputChannels, nOutputChannels, kSize, 2, 1);
    conv1._weights->fill_(.1f);
    conv1._bias->fill_(.1f);
    pTensor mid = conv1.forward(input);
    expect_eq_int(mid->dim(0), batchSize);
    expect_eq_int(mid->dim(1), nOutputChannels);
    expect_eq_int(mid->dim(2), 2);
    expect_eq_int(mid->dim(3), 2);

    expect_eq_float(mid->at(0), 1.6046f);
    expect_eq_float(mid->at(1), 2.4740f);
    expect_eq_float(mid->at(2), 2.8507f);
    expect_eq_float(mid->at(3), 4.4220f);

    expect_eq_float(mid->at(12), 7.9638f);
    expect_eq_float(mid->at(13), 12.1238f);
    expect_eq_float(mid->at(14), 12.8189f);
    expect_eq_float(mid->at(15), 19.5296f);

    sManualConv2d conv2(2, 1, 2, 2, 0);
    conv2._weights->fill_(.1f);
    conv2._bias->fill_(.1f);
    pTensor output = conv2.forward(mid);
    expect_eq_int(output->dim(0), batchSize);
    expect_eq_int(output->dim(1), 1);
    expect_eq_int(output->dim(2), 1);
    expect_eq_int(output->dim(3), 1);
    expect_eq_float(output->at(0), 2.3702f);
    expect_eq_float(output->at(1), 10.5872f);

    sMSE mse;
    mse.forward(output);

    pTensor target = sTensor::Ones(batchSize);
    float l = mse.loss(output->squeeze(), target);
    expect_eq_float(l, 46.8964f);

    expect_eq_int(conv1._weights->dim(0), nOutputChannels);
    expect_eq_int(conv1._weights->dim(1), nInputChannels);
    expect_eq_int(conv1._weights->dim(2), 3);
    expect_eq_int(conv1._weights->dim(3), 3);
    expect_eq_int(conv1._bias->dim(1), nOutputChannels); 

    expect_eq_int(conv2._weights->dim(0), 1);
    expect_eq_int(conv2._weights->dim(1), 2);
    expect_eq_int(conv2._weights->dim(2), 2);
    expect_eq_int(conv2._weights->dim(3), 2);
    expect_eq_int(conv2._bias->dim(1), 1);
}

/*
# t_model_mse_layer
loss_fn = nn.MSELoss()

predictions = torch.ones(3, 5)
true_values = torch.arange(1, 16).reshape(3,5)
loss = loss_fn(predictions, true_values)

print(loss)
tensor(67.6667)
*/
void t_model_mse_layer()
{
    pTensor predictions = sTensor::Ones(3, 5);
    pTensor true_values = sTensor::Linear(1,1,15)->reshape_(3, 5);

    sMSE mse;
    mse.forward(predictions);
    float l = mse.loss(predictions, true_values);
    expect_eq_float(67.6667f, l);
}

/*
# t_model_softmax_func
softmax2 = nn.Softmax(dim=1)
data2d = torch.arange(15).reshape(3,5).exp() * 0.1
probs2d = softmax2(data2d)
print (f"Probs2d: {probs2d}")

Probs2d: tensor([[0.0045, 0.0053, 0.0085, 0.0302, 0.9516],
        [0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]])
*/
void t_model_softmax_func()
{
    pTensor data2d = sTensor::Linear(0, 0.1f, 15)->reshape_(3, 5)->exp();
    const pTensor probabilities = softmax(data2d);
    expect_eq_float(probabilities->data()[0], 0.1559f);
    expect_eq_float(probabilities->data()[14], 0.3608f);
}

/*
# t_model_cross_entropy_loss
def logsumexp(x):
    m = x.max(-1)[0]
    return m + (x-m[:,None]).exp().sum(-1).log()

def log_softmax(x): return x - x.logsumexp(-1,keepdim=True)

def nll(input, target): return -input[range(target.shape[0]), target].mean()

targets = torch.tensor([1,2,0,1,2])
preds = torch.tensor([[0,1,2],[2,3,2],[5,3,2],[2,1,2],[2,5,2]]).float()

loss_fn = nn.CrossEntropyLoss()
l2 = loss_fn(preds, targets)
print("l2",  l2)

sm_pred = log_softmax(preds)
print ("preds", sm_pred)
l1 = nll(sm_pred, targets)
print("l1", l1)

l2 tensor(1.6172)
preds tensor([[-2.4076, -1.4076, -0.4076],
        [-1.5514, -0.5514, -1.5514],
        [-0.1698, -2.1698, -3.1698],
        [-0.8620, -1.8620, -0.8620],
        [-3.0949, -0.0949, -3.0949]])
l1 tensor(1.6172)

*/
void t_model_cross_entropy_loss()
{
    pTensor targets = sTensor::Dims(5);
    targets->data()[0] = 1;
    targets->data()[1] = 2;
    targets->data()[2] = 0;
    targets->data()[3] = 1;
    targets->data()[4] = 2;

    pTensor preds = sTensor::Dims(5,3);
    preds->data()[0] = 0; preds->data()[1] = 1; preds->data()[2] = 2;
    preds->data()[3] = 2; preds->data()[4] = 3; preds->data()[5] = 2;
    preds->data()[6] = 5; preds->data()[7] = 3; preds->data()[8] = 2;
    preds->data()[9] = 2; preds->data()[10] = 1; preds->data()[11] = 2;
    preds->data()[12] = 2; preds->data()[13] = 5; preds->data()[14] = 2;

    pTensor sm_pred = log_softmax(preds);
    float l1 = nll_loss(sm_pred, targets);
    expect_eq_float(l1, 1.6172f);

    float l2 = cross_entropy_loss(preds, targets);
    expect_eq_float(l2, 1.6172f);
}

/*
# t_model_softmax_layer
softmax = nn.Softmax(dim=1)

loss_fn = nn.CrossEntropyLoss()

# Assume we have 5 samples and 3 classes
outputs =  torch.arange(15).reshape(5,3).exp() * 0.1
targets = torch.tensor([0, 1, 2, 0, 1])  # ground truth

outputs_softmax = softmax(outputs)
loss = loss_fn(outputs, targets)

print(loss)
tensor(1.1553)
*/
void t_model_softmax_layer()
{
    pTensor data2d = sTensor::Linear(0, 0.1f, 15)->reshape_(5, 3)->exp();
    sSoftMax sm;
    pTensor data_softmax = sm.forward(data2d);

    pTensor targets = sTensor::Ones(5);
    targets->data()[0] = 0;
    targets->data()[1] = 1;
    targets->data()[2] = 2;
    targets->data()[3] = 0;
    targets->data()[4] = 1;
    float L = sm.loss(data2d, targets);
    expect_eq_float(L, 1.1553f);
}

/*
# t_model_unfold1
unfold = nn.Unfold(kernel_size=(3, 3), stride=1)
input = torch.randn(2, 5, 14, 14) # 2 samples, 5 channels, 14x14 input
output = unfold(input)
print (output.size())
torch.Size([2, 45, 144])

batches: 2
patches: 45 = 3 * 3 * 5 (ks * ks * channels)
blocks: 144 = (14 - 3 + 1) * (14 - 3 + 1)
*/
void t_model_unfold1()
{
    int bs = 2;
    int ch = 5;

    pTensor input = sTensor::Ones(bs, ch, 14, 14);
    pTensor unfolded = unfold_multiple(input, 3, 1);
    expect_eq_int(bs, unfolded->dim(0));
    expect_eq_int(45, unfolded->dim(1));
    expect_eq_int(144, unfolded->dim(2));
}

/*
# t_model_unfold2
unfold = nn.Unfold(kernel_size=(3, 3), stride=2)
input = torch.randn(4, 6, 14, 14) # 4 samples, 6 channels, 14x14 input
output = unfold(input)
print (output.size())
torch.Size([4, 54, 36])

batches: 4
patches: 54 = 3 * 3 * 6 (ks * ks * channels)
blocks: 36 = ((14 - 3) // stride + 1) * ((14 - 3) // stride + 1)
*/
void t_model_unfold2()
{
    int bs = 4;
    int ch = 6;

    pTensor input = sTensor::Ones(bs, ch, 14, 14);
    pTensor unfolded = unfold_multiple(input, 3, 2);
    expect_eq_int(bs, unfolded->dim(0));
    expect_eq_int(54, unfolded->dim(1));
    expect_eq_int(36, unfolded->dim(2));
}

/*
# t_model_fold1
fold = nn.Fold(output_size=(14, 14), kernel_size=(3, 3), stride=1)
input = torch.randn(2, 5 * 3 * 3, 144)
output = fold(input)
print (output.size())
torch.Size([1, 5, 14, 14])

in_blocks: 144 = (14 - ks + 1) * (14 - ks + 1)
width: 14 =  (sqrt(144) - 1) * stride + ks;
*/
void t_model_fold1()
{
    int bs = 2;
    int ch = 5;
    int blocks = 144;

    pTensor input = sTensor::Ones(bs, ch * 3 * 3, blocks);
    pTensor folded = fold_multiple(input, 3, 1);
    expect_eq_int(bs, folded->dim(0));
    expect_eq_int(ch, folded->dim(1));
    expect_eq_int(14, folded->dim(2));
    expect_eq_int(14, folded->dim(3));
}

/*
# t_model_fold2
fold = nn.Fold(output_size=(14, 14), kernel_size=(3, 3), stride=2)
input = torch.randn(2, 5 * 3 * 3, 36)
output = fold(input)
print (output.size())
torch.Size([2, 5, 14, 14])

in_blocks: 36 = ((14 - ks) // 2 + 1) * ((14 - ks) // 2 + 1)
width: 14 =  (sqrt(144) - 1) * stride + ks;
*/
void t_model_fold2()
{
    int bs = 2;
    int ch = 5;
    int blocks = 144;

    pTensor input = sTensor::Ones(bs, ch * 3 * 3, blocks);
    pTensor folded = fold_multiple(input, 3, 1);
    expect_eq_int(bs, folded->dim(0));
    expect_eq_int(ch, folded->dim(1));
    expect_eq_int(14, folded->dim(2));
    expect_eq_int(14, folded->dim(3));
}


/*
# t_model_conv_layer_backwards
import torch
from torch import nn

batch_size = 6
input_channels = 5
output_features = 2

input_data = ((torch.arange(0, batch_size * 80) * 0.1).pow(1.5) * 0.1).reshape(batch_size,input_channels, 4, 4)

conv_layer1 = nn.Conv2d(input_channels, output_features, kernel_size=3, stride=2, padding=1)
conv_layer1.weight.data.fill_(.1)
conv_layer1.bias.data.fill_(.1)
mid = conv_layer1(input_data)
print(f"mid shape: {mid.shape} 0:{mid[0][0][0][0]} 1:{mid[0][0][0][1]} ")

conv_layer2 = nn.Conv2d(2, 10, kernel_size=2, stride=2, padding=0)
conv_layer2.weight.data.fill_(.1)
conv_layer2.bias.data.fill_(.1)
output = conv_layer2(mid)
print(f"output shape: {output.shape} 0:{output[0][0][0][0]}")

softmax = nn.Softmax(dim=1)
softmax(output)

target = torch.tensor([2,7,2,0,1,2])
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output.squeeze(), target)
loss.backward()

print(f"loss: {loss}")

print(f"Weights gradient shape: {conv_layer1.weight.grad.shape}")
print(f"Bias gradient shape: {conv_layer1.bias.grad.shape}")
mid shape: torch.Size([6, 2, 2, 2]) 0:1.6045881509780884 1:2.4739937782287598
output shape: torch.Size([6, 10, 1, 1]) 0:2.3702378273010254
loss: 2.3025853633880615
Weights gradient shape: torch.Size([2, 5, 3, 3])
Bias gradient shape: torch.Size([2])
*/
void t_model_conv_layer_backwards()
{
    constexpr uint batchSize = 1;
    constexpr uint kSize = 3;
    constexpr uint nInputChannels = 5;
    constexpr uint nOutputChannels = 2;
    constexpr uint nPixels = 4;

    // format for CNN is (batch, input_channels, rows, columns)
    pTensor input = sTensor::Linear(0.0f, 0.1f, batchSize, nInputChannels, nPixels, nPixels)->pow_(1.5f)->multiply_(0.1f);

    sManualConv2d conv1(nInputChannels, nOutputChannels, kSize, 2, 1);
    conv1._weights->fill_(.1f);
    conv1._bias->fill_(.1f);
    pTensor mid = conv1.forward(input);

    sManualConv2d conv2(2, 10, 2, 2, 0);
    conv2._weights->fill_(.1f);
    conv2._bias->fill_(.1f);
    pTensor output = conv2.forward(mid);

    sSoftMax softmax;
    softmax.forward(output);

    pTensor target = sTensor::Dims(batchSize);
    target->set1d(0, 2);
    //target->set1d(1, 1);
    //target->set1d(2, 2);
    //target->set1d(3, 0);
    //target->set1d(4, 1);
    //target->set1d(5, 2);

    float l = softmax.loss(output->squeeze(), target);
    expect_eq_float(l, 2.3025f);

    softmax.backward(conv2._activations);
    conv2.backward(conv1._activations);
    conv1.backward(input);

}

/*
# t_model_conv_layer_backwards1

import torch
import torch.nn as nn
import torch.optim as optim

# Create a single 1x8x8 image
input = (torch.arange(0., 25.)*0.1).reshape(1, 1, 5, 5)

# Create a Conv2d layer
conv = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=0)
conv.weight.data.fill_(0.1)
conv.bias.data.fill_(0.1)

# Pass the image through the Conv2d layer
output = conv(input)

# Register a hook on the output to print the gradients
output.register_hook(lambda grad: print("Gradients on the activations:", grad.shape, grad[0][0]))

# Create a target tensor
target = torch.tensor([1])

# Create a CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

print ("Output shape", output.view(1, -1).shape)
print("Output ", output[0][0])

# Compute the loss
loss = criterion(output.view(1, -1), target)
print ("loss:", loss)

# Zero the gradients
conv.zero_grad()

# Backward pass
loss.backward()

# Print the gradients
for name, param in conv.named_parameters():
    print("Params Gradients", name, param.grad[0])

print ("Weights Before:", conv.weight.data[0])

# Create an optimizer
optimizer = optim.SGD(conv.parameters(), lr=0.01)

# Update the weights
optimizer.step()

print ("Weights After:", conv.weight.data[0])

Output shape torch.Size([1, 4])
Output  tensor([[0.6400, 0.8200],
        [1.5400, 1.7200]], grad_fn=<SelectBackward0>)
loss: tensor(1.8483, grad_fn=<NllLossBackward0>)
Gradients on the activations: torch.Size([1, 1, 2, 2]) tensor([[ 0.1316, -0.8425],
        [ 0.3236,  0.3874]])
Params Gradients weight tensor([[[0.6199, 0.6199, 0.6199],
         [0.6199, 0.6199, 0.6199],
         [0.6199, 0.6199, 0.6199]]])
Params Gradients bias tensor(5.9605e-08)
Weights Before: tensor([[[0.1000, 0.1000, 0.1000],
         [0.1000, 0.1000, 0.1000],
         [0.1000, 0.1000, 0.1000]]])
Weights After: tensor([[[0.0938, 0.0938, 0.0938],
         [0.0938, 0.0938, 0.0938],
         [0.0938, 0.0938, 0.0938]]])
*/

void t_model_conv_layer_backwards1()
{
    constexpr uint batchSize = 1;
    constexpr uint kSize = 3;
    constexpr uint nInputChannels = 1;
    constexpr uint nOutputChannels = 1;
    constexpr uint nPixels = 5;
    constexpr float lr = 0.01f;

    // format for CNN is (batch, input_channels, rows, columns)
    pTensor input = sTensor::Linear(0.0f, 0.1f, nPixels* nPixels)->reshape_(nInputChannels, nOutputChannels, nPixels, nPixels);

    sManualConv2d conv1(nInputChannels, nOutputChannels, kSize, 2, 0);
    conv1._weights->fill_(.1f);
    conv1._bias->fill_(.1f);
    pTensor output = conv1.forward(input);

    pTensor target = sTensor::Dims(1);
    target->set1d(0, 1);

    sCrossEntropy celoss;
    celoss.forward(output);

    output->reshape_(1, 4);
    float l = celoss.loss(output, target);
    expect_eq_float(l, 1.8483f);

    output->reshape_(1, 1, 2, 2);

    expect_eq_float(output->at(0), 0.6400f);
    expect_eq_float(output->at(1), 0.8200f);
    expect_eq_float(output->at(2), 1.5400f);
    expect_eq_float(output->at(3), 1.7200f);

    expect_eq_float(celoss._diff->at(0), 0.1316f);
    expect_eq_float(celoss._diff->at(1), -0.8425f);
    expect_eq_float(celoss._diff->at(2), 0.3236f);
    expect_eq_float(celoss._diff->at(3), 0.3874f);

    celoss.backward(conv1._activations);
    output->grad()->reshape_(4, 1 );

    conv1.backward(input);

    celoss.update_weights(lr);
    conv1.update_weights(lr);

    expect_eq_float(conv1._weights->at(0), 0.0938f);
    expect_eq_float(conv1._weights->at(1), 0.0938f);
    expect_eq_float(conv1._weights->at(2), 0.0938f);
}


void test_model()
{
    init_kernels();

    images = minstLoadImages("Resources/Data/minst/few-images.idx3-ubyte", g_numImagesTrain, g_imageArraySize);
    categories = minstLoadLabels("Resources/Data/minst/train-labels.idx1-ubyte", g_numImagesTrain);

    sTEST(model_one);
    sTEST(model_unfold1);
    sTEST(model_unfold2);
    sTEST(model_fold1);
    sTEST(model_fold2);
    sTEST(model_mse_layer);
    sTEST(model_softmax_func)
    sTEST(model_softmax_layer)
    sTEST(model_cross_entropy_loss);
    sTEST(model_conv_manual_simple);
    sTEST(model_conv_manual_batch1);
    sTEST(model_conv_manual_batch2);
    sTEST(model_conv_layer_shape);
    sTEST(model_conv_layer_forward);
    sTEST(model_conv_layer_values_single);
    sTEST(model_conv_layer_values_batch);
    sTEST(model_conv_layer_values_batch_channels);
    sTEST(model_conv_layer_mse_loss);
    //sTEST(model_conv_layer_backwards);
    sTEST(model_conv_layer_backwards1);

}