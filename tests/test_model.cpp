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
    pTensor kernels = sTensor::Ones(1, 3, 3);

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
    pTensor input = sTensor::Ones(4, 3, 8, 8);
    pTensor kernels = sTensor::Ones(7, 3, 3);

    // no padding, 1 kernel
    pTensor result1 = conv_manual_batch(input, kernels, 2, 0);
    expect_eq_int(result1->dim(0), 4);
    expect_eq_int(result1->dim(1), 7);
    expect_eq_int(result1->dim(2), 3);
    expect_eq_int(result1->dim(3), 3);

    // padding, 1 kernel
    pTensor result2 = conv_manual_batch(input, kernels, 2, 1);
    expect_eq_int(result2->dim(0), 4);
    expect_eq_int(result2->dim(1), 7);
    expect_eq_int(result2->dim(2), 4);
    expect_eq_int(result2->dim(3), 4);
}

void t_model_conv_layer()
{
    constexpr uint batchSize = 1;
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
        //pTensor input = sTensor::Ones(batchSize, nInputChannels, nPixels, nPixels);
        sManualConv2d conv(nInputChannels, nKernels, kSize);
        result = conv.forward(result);
        expect_eq_int(batchSize, result->dim(0));
        expect_eq_int(nKernels, result->dim(1));
        expect_eq_int(7, result->dim(2));
        expect_eq_int(7, result->dim(3));
    }
}


void t_model_unfold1()
{
    /*
    unfold = nn.Unfold(kernel_size=(3, 3), stride=1)
    input = torch.randn(2, 5, 14, 14) # 2 samples, 5 channels, 14x14 input
    output = unfold(input)
    print (output.size())
    torch.Size([2, 45, 144])

    batches: 2
    patches: 45 = 3 * 3 * 5 (ks * ks * channels)
    blocks: 144 = (14 - 3 + 1) * (14 - 3 + 1)
    */

    int bs = 2;
    int ch = 5;

    pTensor input = sTensor::Ones(bs, ch, 14, 14);
    pTensor unfolded = unfold_multiple(input, 3, 1);
    expect_eq_int(bs, unfolded->dim(0));
    expect_eq_int(45, unfolded->dim(1));
    expect_eq_int(144, unfolded->dim(2));
}

void t_model_unfold2()
{
    /*
    unfold = nn.Unfold(kernel_size=(3, 3), stride=2)
    input = torch.randn(4, 6, 14, 14) # 4 samples, 6 channels, 14x14 input
    output = unfold(input)
    print (output.size())
    torch.Size([4, 54, 36])

    batches: 4
    patches: 54 = 3 * 3 * 6 (ks * ks * channels)
    blocks: 36 = ((14 - 3) // stride + 1) * ((14 - 3) // stride + 1)
    */

    int bs = 4;
    int ch = 6;

    pTensor input = sTensor::Ones(bs, ch, 14, 14);
    pTensor unfolded = unfold_multiple(input, 3, 2);
    expect_eq_int(bs, unfolded->dim(0));
    expect_eq_int(54, unfolded->dim(1));
    expect_eq_int(36, unfolded->dim(2));
}


void t_model_fold1()
{
    /*
    fold = nn.Fold(output_size=(14, 14), kernel_size=(3, 3), stride=1)
    input = torch.randn(2, 5 * 3 * 3, 144)
    output = fold(input)
    print (output.size())
    torch.Size([1, 5, 14, 14])

    in_blocks: 144 = (14 - ks + 1) * (14 - ks + 1)
    width: 14 =  (sqrt(144) - 1) * stride + ks;
    */

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

void t_model_fold2()
{
    /*
    fold = nn.Fold(output_size=(14, 14), kernel_size=(3, 3), stride=2)
    input = torch.randn(2, 5 * 3 * 3, 36)
    output = fold(input)
    print (output.size())
    torch.Size([2, 5, 14, 14])

    in_blocks: 36 = ((14 - ks) // 2 + 1) * ((14 - ks) // 2 + 1)
    width: 14 =  (sqrt(144) - 1) * stride + ks;
    */

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
    sTEST(model_conv_manual_simple);
    sTEST(model_conv_manual_batch1);
    sTEST(model_conv_manual_batch2);
    //sTEST(model_conv_single_image);
    //sTEST(model_conv_batch_image);
    //sTEST(model_conv_batch_image_batch_kernels);
    sTEST(model_conv_layer);
}