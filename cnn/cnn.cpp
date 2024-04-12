
#include "scratch/stensor.h"
#include "scratch/smodel.h"
#include "scratch/slearner.h"
#include "scratch/minst.h"


const uint g_imageSize = 28;
const uint g_imageArraySize = g_imageSize * g_imageSize;
const uint g_numImagesTrain = 60000; // should be 60000
const uint g_numImagesValid = 10000;
const uint g_numCategories = 10;
const uint g_numHidden = 50;

const uint batchSize = 100;
const uint epochs = 4;
const float lr = 0.5f;

struct CnnData
{
    sLearner* learner = nullptr;
    sModel* model = nullptr;

    pTensor images_train = sTensor::Empty();
    pTensor categories_train = sTensor::Empty();

    //sTensor images_valid = sTensor::Empty();
    //sTensor categories_valid = sTensor::Empty();

    pTensor edge1 = sTensor::Zeros(28, 28);
    pTensor edge2 = sTensor::Zeros(28, 28);

    ~CnnData()
    {
        delete model;
        delete learner;
    }
};

CnnData data;

const float* cnn_images_train(const uint index)
{
    return data.images_train->data() + index * 784;
}

const float* cnn_edge1() { return data.edge1->data(); }
const float* cnn_edge2() { return data.edge2->data(); }

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

void cnn_init()
{
    data.images_train = minstLoadImages("Resources/Data/fashion/train-images.idx3-ubyte", g_numImagesTrain, g_imageArraySize);
    data.categories_train = minstLoadLabels("Resources/Data/fashion/train-labels.idx1-ubyte", g_numImagesTrain);

    // not used yet
    //data.images_valid = minstLoadImages("Resources/Data/minst/t10k-images.idx3-ubyte", g_numImagesValid, g_imageArraySize);
    //data.categories_valid = minstLoadLabels("Resources/Data/minst/t10k-labels.idx1-ubyte", g_numImagesValid);

    data.model = new sModel(g_imageArraySize, g_numHidden, 10);

    // format for CNN is (batch, channels, height, width)
    //pTensor images = data.images_train->clone_shallow()->reshape_(g_numImagesTrain, 1, g_imageSize, g_imageSize);
    data.learner = new sLearner(*data.model, data.images_train, data.categories_train, batchSize, lr);

    sTensor::enableAutoLog = true;

    pTensor top_edge = sTensor::Zeros(3, 3);
    top_edge->data()[0] = -1.f;
    top_edge->data()[1] = -1.f;
    top_edge->data()[2] = -1.f;
    top_edge->data()[6] = 1.f;
    top_edge->data()[7] = 1.f;
    top_edge->data()[8] = 1.f;

    pTensor bottom_edge = sTensor::Zeros(3, 3);
    bottom_edge->data()[0] = 1.f;
    bottom_edge->data()[1] = 1.f;
    bottom_edge->data()[2] = 1.f;
    bottom_edge->data()[6] = -1.f;
    bottom_edge->data()[7] = -1.f;
    bottom_edge->data()[8] = -1.f;

    pTensor left_edge = sTensor::Zeros(3, 3);
    left_edge->data()[0] = -1.f;
    left_edge->data()[3] = -1.f;
    left_edge->data()[6] = -1.f;
    left_edge->data()[2] = 1.f;
    left_edge->data()[5] = 1.f;
    left_edge->data()[8] = 1.f;

    pTensor right_edge = sTensor::Zeros(3, 3);
    right_edge->data()[0] = 1.f;
    right_edge->data()[3] = 1.f;
    right_edge->data()[6] = 1.f;
    right_edge->data()[2] = -1.f;
    right_edge->data()[5] = -1.f;
    right_edge->data()[8] = -1.f;

    if (0) // batch image, batch kernel test
    {
        // need to pad in two dimensions
        pTensor ready = data.images_train->unsqueeze(2)->view_(60000, 28, 28)->pad3d(1);
        pTensor unfolded = unfold_multiple(ready, 3, 1)->reshape_(60000 * (28 * 28), 9);

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

        pTensor imgs = unfolded->MatMult(kernel_stack_transposed)->reshape_(60000, 784, 4);
        data.edge1 = reorder_data(imgs->select(0, 0)->squeeze_());
        data.edge2 = reorder_data(imgs->select(0, 7)->squeeze_());
    }

    if (1) // batch image, single kernel unfold test
    {
        // need to pad in two dimensions
        pTensor padded_images = data.images_train->unsqueeze(2)->view_(60000, 28, 28)->pad3d(1);
        pTensor unfolded = unfold_multiple(padded_images, 3, 1)->reshape_(60000 * (28 * 28), 9);

        pTensor flattened_top = top_edge->view_(9, 1);

        pTensor imgs = unfolded->MatMult(flattened_top)->reshape_(60000, 784);
        data.edge1 = imgs->row2d(0)->view_(28, 28);
        data.edge2 = imgs->row2d(7)->view_(28, 28);
    }

    if (0)  // single image, single kernel unfold test
    {
        pTensor image = data.images_train->slice_rows(7, 8)->view_(28,28)->pad2d(1);
        pTensor unfolded_image = unfold_single(image, 3, 1);
        pTensor flattened_top = top_edge->view_(9, 1);
        data.edge1 = unfolded_image->MatMult(flattened_top)->view_(28,28);

        pTensor flattened_left = left_edge->view_(9, 1);
        data.edge2 = unfolded_image->MatMult(flattened_left)->view_(28,28);
    }

    // manually compute the convolution
    //for (uint i = 0; i < 28-2; i++)
    //{
    //    for (uint j = 0; j < 28-2; j++)
    //    {
    //        if (i == 3 && j == 14)
    //        {
    //            int debug = 1;
    //        }
    //        pTensor slice = image->slice2d(i, i + 3, j, j + 3);

    //        data.edge1->set2d(i, j, (slice * top_edge)->sum());
    //        data.edge2->set2d(i, j, (slice * left_edge)->sum());
    //    }
    //}

    sTensor::enableAutoLog = false;
}

const sModel& cnn_model()
{
    return *data.model;
}

sLearner& cnn_learner()
{
    return *data.learner;
}
