#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "convolution.h"
#include "matrix_helpers.cuh"
#include <memory>

using namespace OmniSense;

static __global__ void ConvolutionKernel(mat_fr A, mat_fr B, mat_fr C)
{
}

namespace OmniSense
{
namespace CUDA
{
namespace General
{
    void Convolve(const mat_fr A, const mat_fr B, mat_fr C, ConvolveBoundary boundary)
    {
        int blockSize = findBlockSize(C.cols * C.rows);

        auto d_A = toDeviceMemoryWithPadding(A, blockSize, blockSize, true);
        auto d_B = toDeviceMemoryWithPadding(B, blockSize, blockSize, true);
        auto d_C = toDeviceMemoryWithPadding(C, blockSize, blockSize, false);

        int sharedMemoryNeeded = blockSize * blockSize * 2 * sizeof(float);

        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(d_C.mat.cols / dimBlock.x, d_C.mat.rows / dimBlock.y);
        ConvolutionKernel KERNEL_ARGS(dimGrid, dimBlock, sharedMemoryNeeded) (d_A, d_B, d_C);

        copyFromDeviceMemory(d_C.mat, C);
    }
}
}
}
