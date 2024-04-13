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

    bottom_edge->data()[0] = 1.f;
    bottom_edge->data()[1] = 1.f;
    bottom_edge->data()[2] = 1.f;
    bottom_edge->data()[6] = -1.f;
    bottom_edge->data()[7] = -1.f;
    bottom_edge->data()[8] = -1.f;

    left_edge->data()[0] = -1.f;
    left_edge->data()[3] = -1.f;
    left_edge->data()[6] = -1.f;
    left_edge->data()[2] = 1.f;
    left_edge->data()[5] = 1.f;
    left_edge->data()[8] = 1.f;

    right_edge->data()[0] = 1.f;
    right_edge->data()[3] = 1.f;
    right_edge->data()[6] = 1.f;
    right_edge->data()[2] = -1.f;
    right_edge->data()[5] = -1.f;
    right_edge->data()[8] = -1.f;
}

void t_model_one()
{
    sModel model(10, 50, 10);
    model.add_layer(new sLinear(numInputs, numOutputs));

    pTensor input = sTensor::Ones(uint(1), numInputs);
    model.forward(input);
}

void t_model_conv_manual()
{
    pTensor edge1 = sTensor::Zeros(28, 28);
    pTensor edge2 = sTensor::Zeros(28, 28);

    //manually compute the convolution
    for (uint i = 0; i < 28-2; i++)
    {
        for (uint j = 0; j < 28-2; j++)
        {
            if (i == 3 && j == 14)
            {
                int debug = 1;
            }
            pTensor slice = images->slice2d(i, i + 3, j, j + 3);

            edge1->set2d(i, j, (slice * top_edge)->sum());
            edge2->set2d(i, j, (slice * left_edge)->sum());
        }
    }
}

void t_model_conv_single_image()
{
    pTensor edge1 = sTensor::Zeros(28, 28);
    pTensor edge2 = sTensor::Zeros(28, 28);

    pTensor image = images->slice_rows(7, 8)->view_(28, 28)->pad2d(1);
    pTensor unfolded_image = unfold_single(image, 3, 1);
    pTensor flattened_top = top_edge->view_(9, 1);
    edge1 = unfolded_image->MatMult(flattened_top)->view_(28, 28);

    pTensor flattened_left = left_edge->view_(9, 1);
    edge2 = unfolded_image->MatMult(flattened_left)->view_(28, 28);
}

void t_model_conv_batch_image()
{
    pTensor edge1 = sTensor::Zeros(28, 28);
    pTensor edge2 = sTensor::Zeros(28, 28);

    // need to pad in two dimensions
    pTensor padded_images = images->clone_shallow()->reshape_(g_numImagesTrain, uint(1), g_imageSize, g_imageSize)->pad_images(1);
    pTensor unfolded = unfold_multiple(padded_images, 3, 1)->reshape_(g_numImagesTrain * (g_imageSize * g_imageSize), uint(9));

    pTensor flattened_top = top_edge->view_(9, 1);

    pTensor imgs = unfolded->MatMult(flattened_top)->reshape_(g_numImagesTrain, uint(784));
    edge1 = imgs->row2d(0)->view_(28, 28);
    edge2 = imgs->row2d(7)->view_(28, 28);
}

void t_model_conv_batch_image_batch_kernels()
{
    pTensor edge1 = sTensor::Zeros(28, 28);
    pTensor edge2 = sTensor::Zeros(28, 28);

    // need to pad in two dimensions
    pTensor ready = images->unsqueeze(2)->view_(g_numImagesTrain, g_imageSize, g_imageSize)->pad3d(1);
    pTensor unfolded = unfold_multiple(ready, 3, 1)->reshape_(g_numImagesTrain * (g_imageSize * g_imageSize), uint(9));

    pTensor flattened_top = top_edge->view_(1, 9);
    pTensor flattened_bottom = bottom_edge->view_(1, 9);
    pTensor flattened_left = left_edge->view_(1, 9);
    pTensor flattened_right = right_edge->view_(1, 9);

    pTensor kernel_stack_transposed = sTensor::Dims(0, 9);
    kernel_stack_transposed->cat0_(flattened_top);
    kernel_stack_transposed->cat0_(flattened_bottom);
    kernel_stack_transposed->cat0_(flattened_left);
    kernel_stack_transposed->cat0_(flattened_right);
    kernel_stack_transposed->transpose_();

    pTensor imgs = unfolded->MatMult(kernel_stack_transposed)->reshape_(g_numImagesTrain, uint(784), uint(4));
    edge1 = reorder_data(imgs->select(0, 0)->squeeze_());
    edge2 = reorder_data(imgs->select(0, 7)->squeeze_());
}

void t_model_conv_layer()
{
    constexpr uint batchSize = 64;
    constexpr uint nInputChannels = 1;
    constexpr uint nKernels = 4;
    constexpr uint nPixels = 28;
    constexpr uint kSize = 3;

    // format for CNN is (batch, input_channels, rows, columns)
    pTensor input = sTensor::Ones(batchSize, nInputChannels, nPixels, nPixels);

    sConv2d conv(nInputChannels, nKernels, kSize);

    const pTensor result = conv.forward(input);
    expect_eq_int(batchSize, result->dim(0));
    expect_eq_int(nKernels, result->dim(1));
    expect_eq_int(14, result->dim(2));
    expect_eq_int(14, result->dim(3));

}

void test_model()
{
    init_kernels();

    images = minstLoadImages("Resources/Data/minst/few-images.idx3-ubyte", g_numImagesTrain, g_imageArraySize);
    categories = minstLoadLabels("Resources/Data/minst/train-labels.idx1-ubyte", g_numImagesTrain);

    sTEST(model_one);
    sTEST(model_conv_manual);
    sTEST(model_conv_single_image);
    sTEST(model_conv_batch_image);
    sTEST(model_conv_batch_image_batch_kernels);
    sTEST(model_conv_layer);
}