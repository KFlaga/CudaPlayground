#include "cuda_all.h"
#include <device_launch_parameters.h>

#include "convolution.h"
#include "matrix_device.h"
#include <memory>

using namespace CudaPlayground;

// extern __shared__ float shared_cache[];

static __global__ void ConvolutionKernel_InsideBoundary(mat_fr source, mat_fr block, mat_fr dest, int blockSize)
{
    // Assumes dest is inside boundary of source (so dest.rows + block.rows = source.rows)

    // To handle unpadded matrices
    int row = threadIdx.y + blockIdx.y * blockSize;
    int col = threadIdx.x + blockIdx.x * blockSize;
    if (row >= dest.rows || col >= dest.cols) {
        return;
    }

    float val = 0;
    block.forEach([&](int r, int c, float b)
    {
        float a = source(row + r, col + c);
        val += a * b;
    });

    dest(row, col) = val;
}

static CUDA_HOST_API int findBlockSize(const mat_fr& elements)
{
    if (elements.rows * elements.cols >= 600) {
        return 12;
    }
    if (elements.rows * elements.cols >= 120) {
        return 8;
    }
    return 4;
}

namespace CudaPlayground
{
namespace CUDA
{
namespace General
{
    void Convolve(const mat_fr source, const mat_fr block, mat_fr dest, ConvolveBoundary boundary)
    {
        int blockSize = findBlockSize(dest);

        auto d_source = [&]() {
            if (boundary == ConvolveBoundary::ExtendZero)
            {
                return toDeviceMemoryExtendedBlock(source, block, true);
            }
            else
            {
                return toDeviceMemory(source, true);
            }
        }();

        auto d_block = toDeviceMemory(block, true);
        auto d_dest = toDeviceMemory(dest, false);

        auto d_dest_mat = [&]()
        {
            if (boundary == ConvolveBoundary::ExtendZero)
            {
                return d_dest.mat;
            }
            else
            {
                return d_dest.mat.sub(block.rows/2, block.cols/2, dest.rows - (block.rows + 1) / 2, dest.cols - (block.cols + 1) / 2);
            }
        }();

        if (boundary == ConvolveBoundary::Zero)
        {
            auto m = d_dest.mat;
            checkCudaErrors(cudaMemset2D(m.elements, m.stride * sizeof(float), 0, m.cols * sizeof(float), m.rows));
        }
        else if (boundary == ConvolveBoundary::Copy)
        {
            copyFromHostMemory(source, d_dest.mat);
        }

        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(pad(d_dest_mat.cols, blockSize) / dimBlock.x, pad(d_dest_mat.rows, blockSize) / dimBlock.y);
        ConvolutionKernel_InsideBoundary KERNEL_ARGS(dimGrid, dimBlock) (d_source.mat, d_block.mat, d_dest_mat, blockSize);
        checkCudaErrors(cudaPeekAtLastError());

        copyFromDeviceMemory(d_dest.mat, dest);
    }
}
}
}
