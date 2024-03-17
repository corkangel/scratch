//main

#include "t2.h"

void test_t2()
{
    {
        // raw array init
        uint dims[] = { 2, 2 };
        sTensor t1(2U, dims);

        // initializer list
        sTensor t2({ 2, 2 });
    }

    {
        sTensor ones = sTensor::Ones(2, 3);
        sTensor randoms = sTensor::Randoms(2, 3);
        sTensor nd = sTensor::NormalDistribution(0.f, 1.f, 3, 8);
        sTensor dd = sTensor::Dims(3, 2);

        sTensor wide = sTensor::Linear(0.f, 0.1f, 2, 4, 10);
        //std::cout << wide << std::endl;

        wide.view_(4, 2, 10);
        //std::cout << wide << std::endl;

        assert(wide == wide);
        assert(wide != ones);

        ones = randoms;
        assert(ones == randoms);

        ones.add_(99.f);
        std::cout << "ones: " << ones << std::endl;

        {
            sTensor o1 = sTensor::Ones(3, 3);
            sTensor o2 = sTensor::Integers(1, 3, 3);

            sTensor add = o1 + o2;
            assert(add(1, 1) == 6.f);

            sTensor mult = o1 * o2;
            assert(mult(1, 1) == 5.f);

            sTensor sub = o1 - o2;
            assert(sub(1, 1) == -4.f);
        }
    }

    {
        sTensor a = sTensor::Randoms(3, 1);
        sTensor b = sTensor::Integers(10, 3, 1);
        sTensor c = a * b;
        c = c + 22.f;

        //std::cout << "A " << a << std::endl;
        //std::cout << "B " << b << std::endl;
        //std::cout << "C " << c << std::endl;
    }

    {
        sTensor mm = sTensor::Randoms(3, 3);
        mm(1,1) = 99.f;
        assert(mm(1,1) == 99.f);

        sTensor rows = mm.sum_rows();
        std::cout << "rows: " << rows << std::endl;

        sTensor cols = mm.sum_columns();
        std::cout << "cols: " << cols << std::endl;
    }

    {
        sTensor mm = sTensor::Integers(1, 3, 3);
        sTensor mm2 = sTensor::Integers(2, 3, 3);
        sTensor mm3 = mm.MatMult(mm2);
        assert(mm3(1,1) == 96.f);
        std::cout << "mm: " << mm3 << std::endl;
    }

    {
        sTensor m1 = sTensor::Fill(3.3f, 6, 3);
        sTensor m2 = sTensor::Integers(1, 1, 3);

        sTensor result_mult = m1 * m2;
        assert(result_mult(1, 1) == 6.6f);
        std::cout << "broadcast mult : " << result_mult << std::endl;

        sTensor result_add = m1 + m2;
        assert(result_add(1, 1) == 5.3f);
        std::cout << "broadcast add : " << result_add << std::endl;
    }

    {
        sTensor mm = sTensor::Randoms(3);
        sTensor mm2 = sTensor::Ones(3);
        float dp = mm.DotProduct(mm2);
        std::cout << "dp: " << dp << std::endl;
    }
}


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
    //test_t2();
    meanshift();
    return 0;
}

