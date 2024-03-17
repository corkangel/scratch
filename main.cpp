//main

#include <iostream>
#include <chrono>

//#include "matrix.h"
#include "smatrix.h"
#include "tensor.h"
#include "t2.h"

void test_smatrix()
{
    sMatrix<float, 2> m;
    std::cout << m.size() << std::endl;
    std::cout << m.bytes() << std::endl;

    sVec2 v;
    std::cout << v.size() << std::endl;
    std::cout << v.bytes() << std::endl;

    v(0) = 23.0f;
    assert(v(0) == 23.0f);

    sMat3x3 zz = sMat3x3::Zeros();
    assert(zz(1,1) == 0.f);

    sMat3x3 oo = sMat3x3::Ones();
    assert(oo(1,1) == 1.f);

}

void test_tensor()
{
    Tensor m = Tensor::Dims(4, 4);
    std::cout << "sz:" << m.size()  << " bytes: " << m.bytes() << std::endl;

    for (uint r = 0; r < m.rank; ++r)
    {
        std::cout << m.dimensions[r] << std::endl;
    }

    m(1, 1) = 11.f;
    assert(m(1,1) == 11.f);

    const Tensor& cm = m;
    std::cout << cm(1,1) << std::endl;

    Tensor m2 = Tensor::Zeros(3, 3);
    assert(m2(1,1) == 0.f);

    // 2D vector
    std::vector<float> v = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
    Tensor m3(v);
    assert(m3(2) == 3.f);

    Tensor mm({ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f });
    float dot = DotProduct(mm, mm);
    assert(dot == 285.f);

    // 3x3 Matrix

    std::vector<std::vector<float>> vv = { { 1.f, 2.f, 3.f }, { 4.f, 5.f, 6.f }, { 7.f, 8.f, 9.f } };
    Tensor m3x3(vv);
    assert(m3x3(1,1) == 5.f);
   
    Tensor mmm = m3x3 * m3x3;
    assert(mmm(1,1) == 25.f);

    Tensor mscalar = m3x3 * 2.f; 
    assert(mscalar(1,1) == 10.f);

    Tensor mult = MatrixMultiply(m3x3, m3x3);
    assert(mult(1,1) == 81.f);

    Tensor m10x150 = Tensor::Random(10, 150);

    Tensor nd150x10 = Tensor::NormalDistribution(0.f, 1.f, 150, 10);

    Tensor multRand = MatrixMultiply(m10x150, nd150x10);

    // view changes

    Tensor wide = Tensor::Ones(2, 8);
    assert(wide(1,7) == 1.f);

    Tensor w1 = wide.Row(0);
    assert(w1.size() == 8);

    wide.view_(8, 2);
    assert(wide(7,1) == 1.f);

    // row

    Tensor row_dest = m3x3.Row(2);
    assert(row_dest(1) == 8.f);

    // wide is now 8x2
    Tensor w2 = wide.Row(0);
    assert(w2.size() == 2);

}

void perf()
{
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i)
    {
        Tensor m1 = Tensor::Random(784, 10);
        Tensor m2 = Tensor::Random(5, 784);
        MatrixMultiply(m2, m1);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

}

void meanshift()
{
    const uint numCentroids = 5;
    const uint numSample = 200;
    const float spread = 5.f;

    Tensor centroids = Tensor::Random(numCentroids, uint(2)) * 75.f;
    Tensor samples = Tensor::Dims(0, 2);

    for (int i=0; i < numCentroids; ++i)
    {
        Tensor centroid = centroids.Row(i); // 2 matrix
        centroid.unsqueeze_(0); // 1x2 matrix

        Tensor batch = Tensor::NormalDistribution(0.f, spread, numSample, uint(2)); // 250x2 matrix
        Tensor broad = Tensor::Broadcast0(centroid, numSample); // 250x2 matrix)
        batch += broad;
        samples.cat0_(batch);
    }

    assert(samples.dim(0) == numSample * numCentroids); // 5 x 200 = 1000
    assert(samples.dim(1) == 2); // x,y positions

    for (uint n = 0; n < numSample * numCentroids; ++n)
    {
        Tensor one = samples.Row(n); // 2 item vector
        one.unsqueeze_(0); // 1x2 matrix
        Tensor oneBroad = Tensor::Broadcast0(one, numSample * numCentroids); // 1000x2 matrix
        Tensor diff = samples - oneBroad; //  1000x2 matrix
        diff.pow_(2);
        Tensor weights = diff.sum1(); // 1000x2 -> 1000x1 matrix
        weights.sqrt_();
        weights.gaussian_(2.5f); // still 1000x1

        Tensor weightsBroad = Tensor::Broadcast1(weights, 2); // 1000x2 matrix
        
        Tensor res = MatrixMultiply(weightsBroad, samples); // 1000x2 matrix

        // todo - divide by sum of weights
        // todo set the row 

        if (n % 100 == 0)
            std::cout << res << std::endl;
    }

    //std::cout << res << std::endl;

    //for (uint n = 0; n < numSample * numCentroids; ++n)
    //{
    //    Tensor sample = samples.Row(n);
    //    Tensor weight = 

}

void test_t2()
{
    // raw array init
    uint dims[] = {2, 2};
    sTensor t1(2U, dims);

    // initializer list
    sTensor t2({ 2, 2 });

    sTensor ones = sTensor::Ones(2, 3);
    sTensor randoms = sTensor::Randoms(2, 3);
    sTensor nd = sTensor::NormalDistribution(0.f, 1.f, 3, 8);
    sTensor dd = sTensor::Dims(3, 2);

    sTensor wide = sTensor::Linear(0.f, 0.1f, 2, 4, 10);
    std::cout << wide << std::endl;

    wide.view_(4, 2, 10);
    std::cout << wide << std::endl;
}

int main()
{
    //test_smatrix();
    //test_tensor();
    //perf();
    //meanshift();
    test_t2();
    return 0;
}

