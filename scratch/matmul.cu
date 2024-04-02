#include "cuda_runtime.h"

#include "matmul.ch"

#include <stdio.h>
#include <thread>
#include <chrono>

void ErrorCheck(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

__device__ float get2d(const float* storage, const uint stride, const uint row, const uint col)
{
    return storage[row * stride + col];
}

__device__ void add2d(float* storage, const uint stride, const uint row, const uint col, const float value)
{
    storage[row * stride + col] += value;
}

__global__ void cuMmatmul(
    const uint nrows, 
    const uint ncols, 
    const uint other_ncols,
    const uint result_ncols,
    const float* left, 
    const float* right, 
    float* result)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nrows || y >= other_ncols)
    {
        return;
    }

    int debug = 1;
    debug++;
    printf("debug: %d\n", debug);


    //for (uint k = 0; k < ncols; k++)
    //{
    //    add2d(result, result_ncols, x, y, get2d(left, ncols, x, k) * get2d(right, other_ncols, k, y));
    //}

    float tmp[10];
    for (uint i=0; i < 10; i++)
    {
        tmp[i] = result[i];
    }
}


__global__ void testit()
{
    int debug = 1;
    debug++;
    printf("debug: %d\n", debug);
}

void FreeOnDevice(CudaTensor& t)
{
    if (t._storage && t._storageOwned)
    {
        ErrorCheck(cudaFree(t._storage));
        t._storage = nullptr;
        t._storageOwned = false;
    }
}

CudaTensor ToDevice(const CudaTensor& t)
{
    CudaTensor deviceT = t;
    ErrorCheck(cudaMalloc((void**)&deviceT._storage, deviceT._storageSize * sizeof(float)));
    ErrorCheck(cudaMemcpy(deviceT._storage, t._storage, t._storageSize, cudaMemcpyHostToDevice));
    deviceT._storageOwned = true;
    return deviceT;
}

void ToHost(const CudaTensor& deviceT, CudaTensor& hostT)
{
    ErrorCheck(cudaMemcpy(hostT._storage, deviceT._storage, hostT._storageSize, cudaMemcpyDeviceToHost));
}

void cpuMmatmul(CudaTensor cpuLeft, CudaTensor cpuRight, CudaTensor cpuResult)
{
    CudaTensor deviceLeft = ToDevice(cpuLeft);
    CudaTensor deviceRight = ToDevice(cpuRight);
    CudaTensor deviceResult = ToDevice(cpuResult);

    const uint nrows = cpuLeft._dimensions[0];
    const uint ncols = cpuLeft._dimensions[1];
    const uint other_ncols = cpuRight._dimensions[1];
    const uint result_ncols = cpuResult._dimensions[1];
    const uint nOps = nrows * other_ncols;

    dim3 blockSize(4, 4);
    dim3 gridSize(nOps / 4 + 1, nOps / 4 + 1);

    //cuMmatmul<<<20,20>>>(
    //    nrows,
    //    ncols,
    //    other_ncols,
    //    result_ncols,
    //    deviceLeft._storage,
    //    deviceRight._storage,
    //    deviceResult._storage);

    testit<<<3,3>>>();

    std::this_thread::sleep_for(std::chrono::seconds(30));


    ErrorCheck(cudaDeviceSynchronize());

    ToHost(deviceResult, cpuResult);
    FreeOnDevice(deviceLeft);
    FreeOnDevice(deviceRight);
    FreeOnDevice(deviceResult);
}

