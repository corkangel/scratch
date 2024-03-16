//main

#include <iostream>
#include <chrono>

#include "matrix.h"
#include "smatrix.h"

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

void test_dmatrix()
{
    dMat m = dMat::Dims(4, 4);
    std::cout << "sz:" << m.size()  << " bytes: " << m.bytes() << std::endl;

    for (uint r = 0; r < m.rank; ++r)
    {
        std::cout << m.dimensions[r] << std::endl;
    }

    m(1, 1) = 11.f;
    assert(m(1,1) == 11.f);

    const dMat& cm = m;
    std::cout << cm(1,1) << std::endl;

    dMat m2 = dMat::Zeros(3, 3);
    assert(m2(1,1) == 0.f);

    // 2D vector
    std::vector<float> v = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
    dMat m3(v);
    assert(m3(2) == 3.f);

    dMat mm({ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f });
    float dot = DotProduct(mm, mm);
    assert(dot == 285.f);

    // 3x3 Matrix

    std::vector<std::vector<float>> vv = { { 1.f, 2.f, 3.f }, { 4.f, 5.f, 6.f }, { 7.f, 8.f, 9.f } };
    dMat m3x3(vv);
    assert(m3x3(1,1) == 5.f);
   
    dMat mmm = m3x3 * m3x3;
    assert(mmm(1,1) == 25.f);

    dMat mscalar = m3x3 * 2.f; 
    assert(mscalar(1,1) == 10.f);

    dMat mult = MatrixMultiply(m3x3, m3x3);
    assert(mult(1,1) == 81.f);

    dMat m10x150 = dMat::Random(10, 150);

    dMat nd150x10 = dMat::NormalDistribution(0.f, 1.f, 150, 10);

    dMat multRand = MatrixMultiply(m10x150, nd150x10);

    // view changes

    dMat wide = dMat::Ones(2, 8);
    assert(wide(1,7) == 1.f);
    wide.view_(8, 2);
    assert(wide(7,1) == 1.f);

    // row

    dMat row_dest = m3x3.Row(2);
    assert(row_dest(1) == 8.f);

    dMat w = wide.Row(0);
    assert(w.size() == 8);

}

void perf()
{
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i)
    {
        dMat m1 = dMat::Random(784, 10);
        dMat m2 = dMat::Random(5, 784);
        MatrixMultiply(m2, m1);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

}

void meanshift()
{
    const uint numCentroids = 10;
    const uint numSample = 250;
    const float spread = 10.f;

    dMat centroids = dMat::Random(numCentroids, uint(2)) * 100.f;
    dMat samples = dMat::Dims(0, 2);

    for (int i=0; i < numCentroids; ++i)
    {
        dMat centroid = centroids.Row(i); // 2 matrix
        centroid.unsqueeze_(0); // 1x2 matrix
        //std::cout << centroid << std::endl;

        dMat batch = dMat::NormalDistribution(0.f, spread, numSample, uint(2)); // 250x2 matrix
        dMat broad = dMat::Broadcast0(centroid, numSample); // 250x2 matrix)
        batch += broad;
        samples.cat0_(batch);
    }

    std::cout << samples << std::endl;
    //std::cout << samples.mean() << std::endl;
}

int main()
{
    //test_smatrix();
    //test_dmatrix();
    //perf();
    meanshift();
    return 0;
}

