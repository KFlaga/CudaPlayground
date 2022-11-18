#include "cuda_all.h"
#include <device_launch_parameters.h>

#include "binarize.h"
#include "matrix_device.h"
#include <memory>

using namespace OmniSense;

static __global__ void BinarizeKernel(
    mat_fr In, mat_fr Out,
    float threshold, float lowValue, float highValue,
    dim3 blockSize)
{
    int row = threadIdx.y + blockIdx.y * blockSize.y;
    for (int c = 0; c < In.cols; ++c)
    {
        Out(row, c) = In(row, c) > threshold ? highValue : lowValue;
    }
}

static CUDA_HOST_API int findBlockSize(int elements)
{
    int s = 4;
    while (s < (elements / 16) && s < 1024)
    {
        s *= 2;
    }
    return s;
}

namespace OmniSense
{
namespace CUDA
{
namespace General
{
    void Binarize(const mat_fr In, mat_fr Out, float threshold, float lowValue, float highValue)
    {
        int blockSize = findBlockSize(Out.rows);

        auto d_In = toDeviceMemoryPad(In, blockSize, 1, true);
        auto d_Out = toDeviceMemoryPad(Out, blockSize, 1, false);

        dim3 dimBlock(1, blockSize);
        dim3 dimGrid(1, d_Out.mat.rows / dimBlock.y);
        BinarizeKernel KERNEL_ARGS(dimGrid, dimBlock) (d_In, d_Out, threshold, lowValue, highValue, dimBlock);
        checkCudaErrors(cudaPeekAtLastError());

        copyFromDeviceMemory(d_Out.mat, Out);
    }
}
}
}
