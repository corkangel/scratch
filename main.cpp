//main

#include "stensor.h"

void test_tensors();

void meanshift()
{
    const uint numCentroids = 5;
    const uint numSample = 200;
    const float spread = 5.f;

    sTensor centroids = sTensor::Randoms(numCentroids, uint(2)).multiply_(70.f).add_(-35.f);
    sTensor samples = sTensor::Dims(0, 2);

    std::cout << "centroids: " << centroids << std::endl;

    // generate samples
    for (sTensorRowIterator riter = centroids.begin_rows(); riter != centroids.end_rows(); ++riter)
    {
        sTensor centroid = *riter; // 2 matrix
        centroid.unsqueeze_(0);    // 1x2 matrix

        sTensor batch = sTensor::NormalDistribution(0.f, spread, numSample, uint(2)); // 250x2 matrix
        batch = batch + centroid; // 250x2 matrix broadcasted with 1x2 matrix
        samples.cat0_(batch);
    }
    std::cout << "samples: " << samples << std::endl;
   
    sTensor new_samples = sTensor::Dims(0, 2);

    // process a batch of samples once
    for (sTensorRowIterator riter = samples.begin_rows(); riter != samples.end_rows(); ++riter)
    {
        sTensor sample = *riter; // 2 matrix
        sample.unsqueeze_(0);    // 1x2 matrix
        //std::cout << "---------\r\n sample: " << sample << std::endl;
        sTensor diffs = (samples - sample);
        //std::cout << "diffs: " << diffs << std::endl;
        diffs.pow_(2);
        //std::cout << "diffs_pow2: " << diffs << std::endl;
        sTensor sum_columns = diffs.sum_columns();
        //std::cout << "sum_columns: " << sum_columns << std::endl;
        sTensor sum_columns_sqrt = sum_columns.sqrt_();
        //std::cout << "sum_columns_sqrt: " << sum_columns_sqrt << std::endl;
        sTensor weights = sum_columns_sqrt.gaussian_(2.5f);
        //std::cout << "weights: " << weights << std::endl;
        sTensor sum_weights = (samples * weights).sum_rows();
        //std::cout << "sum_weights: " << sum_weights << std::endl;
        const float sum = weights.sum();
        sTensor sample_new = sum_weights / sum; // 1x2 matrix
        //std::cout << "sample_new: " << sample_new << std::endl;

        // all the above steps in one line
        sTensor all = (weights * (samples-sample).pow_(2).sum_columns().sqrt_().gaussian_(2.5f)).sum_rows() / weights.sum();
        assert(all(0,0) == sample_new(0,0));

        new_samples.cat0_(sample_new);
    }
    std::cout << "new_samples: " << new_samples << std::endl;
}


int main()
{
    test_tensors();
    //meanshift();
    return 0;
}

