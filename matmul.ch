
using uint = unsigned int;

constexpr uint TENSOR_MAX_DIMENSIONS = 4;

struct CudaTensor
{
    uint _dimensions[TENSOR_MAX_DIMENSIONS];
    float* _storage;
    uint _storageSize;
    bool _storageOwned = false;
};

void cpuMmatmul(CudaTensor left, CudaTensor right, CudaTensor result);
