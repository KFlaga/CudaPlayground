#include "cuda_all.h"
#include <device_launch_parameters.h>

#include "median.h"
#include "matrix_device.h"
#include <malloc.h>

using namespace OmniSense;

template<typename F>
static CUDA_DEVICE_API bool forEachX(int r, int c, int radius, int rows, int cols, F&& f)
{
    for (int dr = -radius; dr <= radius; ++dr)
    {
        for (int dc = -radius; dc <= radius; ++dc)
        {
            int rr = r + dr;
            int cc = c + dc;
            if (rr >= 0 && rr < rows && cc >= 0 && cc < cols)
            {
                if (!f(rr, cc)) {
                    return false;
                }
            }
        }
    }
    return true;
}

struct X { unsigned char lower; unsigned char equal; };

static __global__ void MedianKernelSimple(
    mat_fr In, mat_fr Out,
    int radius,
    dim3 blockSize)
{
    int row = threadIdx.y + blockIdx.y * blockSize.y;
    int col = threadIdx.x + blockIdx.x * blockSize.x;
    if (row >= In.rows || col >= In.cols)
    {
        return;
    }

    int size = 2 * radius + 1;
    int mSize = size * size;
    X* m = (X*)alloca(mSize);
    memset(m, 0, mSize);

    forEachX(row, col, radius, In.rows, In.cols, [&](int rr, int cc) {
        X x = { 0, 0 };
        int count = 0;

        forEachX(row, col, radius, In.rows, In.cols, [&](int rr2, int cc2) {
            x.lower += (int)(In(rr2, cc2) < In(rr, cc));
            x.equal += (int)(In(rr2, cc2) == In(rr, cc));
            count++;
            return true;
        });

        int medianPos = (count + 1) / 2;
        if (x.lower == medianPos ||
            (x.lower < medianPos && (x.lower + x.equal) >= medianPos))
        {
            Out(row, col) = In(rr, cc);
            return false;
        }
        return true;
    });
}

static __global__ void MedianKernelOptimized(
    mat_fr In, mat_fr Out,
    int radius,
    dim3 blockSize)
{
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

namespace OmniSense
{
namespace CUDA
{
namespace General
{
    void Median(const mat_fr In, mat_fr Out, int radius)
    {
        int blockSize = findBlockSize(Out);

        auto d_In = toDeviceMemory(In, true);
        auto d_Out = toDeviceMemory(Out, false);

        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(pad(Out.cols, blockSize) / dimBlock.x, pad(Out.rows, blockSize) / dimBlock.y);
        MedianKernelSimple KERNEL_ARGS(dimGrid, dimBlock) (d_In, d_Out, radius, dimBlock);
        //MedianKernelOptimized KERNEL_ARGS(dimGrid, dimBlock) (d_In, d_Out, radius, dimBlock);
        checkCudaErrors(cudaPeekAtLastError());

        copyFromDeviceMemory(d_Out.mat, Out);
    }
}
}
}
