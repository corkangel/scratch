#include "cuda_runtime.h"

#include "matmul.ch"

__device__ float get2d(CudaTensor& t, const uint row, const uint col)
{
    return t._storage[row * t._dimensions[1] + col];
}

__device__ void add2d(CudaTensor& t, const uint row, const uint col, const float value)
{
    t._storage[row * t._dimensions[1] + col] += value;
}

__global__ void cuMmatmul(CudaTensor left, CudaTensor right, CudaTensor result)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const uint nrows = left._dimensions[0];
    const uint ncols = left._dimensions[1];
    const uint other_ncols = right._dimensions[1];
    if (x >= nrows || y >= other_ncols)
    {
        return;
    }

    for (uint k = 0; k < ncols; k++)
    {
        add2d(result, x, y, get2d(left, x, k) * get2d(right, k, y));
    }
}

void FreeOnDevice(CudaTensor& t)
{
    if (t._storage && t._storageOwned)
    {
        cudaFree(t._storage);
        t._storage = nullptr;
        t._storageOwned = false;
    }
}

CudaTensor ToDevice(const CudaTensor& t)
{
    CudaTensor deviceT = t;
    cudaMalloc((void**)&deviceT._storage, deviceT._storageSize * sizeof(float));
    cudaMemcpy(deviceT._storage, t._storage, t._storageSize, cudaMemcpyHostToDevice);
    deviceT._storageOwned = true;
    return deviceT;
}

void ToHost(const CudaTensor& deviceT, CudaTensor& hostT)
{
    cudaMemcpy(hostT._storage, deviceT._storage, hostT._storageSize, cudaMemcpyDeviceToHost);
}

void cpuMmatmul(CudaTensor cpuLeft, CudaTensor cpuRight, CudaTensor cpuResult)
{
    CudaTensor deviceLeft = ToDevice(cpuLeft);
    CudaTensor deviceRight = ToDevice(cpuRight);
    CudaTensor deviceResult = ToDevice(cpuResult);

    const uint nrows = cpuLeft._dimensions[0];
    const uint other_ncols = cpuRight._dimensions[1];
    const uint nOps = nrows * other_ncols;

    dim3 blockSize(32, 32);
    dim3 gridSize(nOps / 32 + 1, nOps / 32 + 1);

    cuMmatmul<<<gridSize, blockSize >>>(deviceLeft, deviceRight, deviceResult);
    cudaDeviceSynchronize();

    ToHost(deviceResult, cpuResult);
    FreeOnDevice(deviceLeft);
    FreeOnDevice(deviceRight);
    FreeOnDevice(deviceResult);
}

