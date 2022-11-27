#include "cuda_all.h"

#include <cuda/std/cmath>
#include "bilateral_filter.h"
#include "matrix_device.h"

using namespace CudaPlayground;

extern __shared__ float shared_cache[];

template<typename F>
static __global__ void  __launch_bounds__(256)
    BilateralFilterKernel(mat_fr source, mat_fr dest, int radius, F filter, int blockSize)
{
    // To handle unpadded matrices
    const int r = threadIdx.y + blockIdx.y * blockSize;
    const int c = threadIdx.x + blockIdx.x * blockSize;
    if (r >= source.rows || c >= source.cols) {
        return;
    }

    //int filterSize = 2 * radius + 1;

    const int minR = max(0, r - radius);
    const int minC = max(0, c - radius);
    const int maxR = min(source.rows - 1, r + radius);
    const int maxC = min(source.cols - 1, c + radius);

    auto neighbourhood = source.sub(minR, minC, (maxR - minR) + 1, (maxC - minC) + 1);

    float val = 0;
    float weight = 0;
    const float I_rc = source(r, c);

    neighbourhood.forEach([&](int nr, int nc, float I_n) {
        float dI2 = (I_n - I_rc) * (I_n - I_rc);
        float dx2 = ((r - nr - minR) * (r - nr - minR) + (c - nc - minC) * (c - nc - minC));

        float f = filter(dI2, dx2);

        weight += f;
        val += I_n * f;
    });

    dest(r, c) = val / weight;
}

template<typename F>
static __global__ void  __launch_bounds__(256)
    BilateralFilterKernel_2(mat_fr source, mat_fr dest, int radius, F filter, int blockSize)
{
    // Only padded matrices are allowed
    const int r = threadIdx.y + blockIdx.y * blockSize;
    const int c = threadIdx.x + blockIdx.x * blockSize;

    const int minR = max(0, r - radius);
    const int minC = max(0, c - radius);
    const int maxR = min(source.rows - 1, r + radius);
    const int maxC = min(source.cols - 1, c + radius);

    // Block is always one column
    // Use this to copy a block of In to shared memory
    // We need:
    //      minR
    // minC      maxC
    //      maxR
    //
    // 

    // copy memory -> split column by threads
    int firstColumn = blockIdx.x * blockSize;

    int threadsInBlock = blockSize;
    int columnsInBlock = min(source.cols - firstColumn, threadsInBlock);
    int cellsInBlock = ((maxR - minR) + 1) * source.cols;

   //  auto neighbourhood = source.sub(minR, minC, (maxR - minR) + 1, (maxC - minC) + 1);
    mat_fr block = { source.cols, (maxR - minR) + 1, source.cols, (float*)shared_cache };


    auto neighbourhood = block.sub(0, minC, (maxR - minR) + 1, (maxC - minC) + 1);

    float val = 0;
    float weight = 0;
    const float I_rc = neighbourhood(radius, radius);

    neighbourhood.forEach([&](int nr, int nc, float I_n) {
        float dI2 = (I_n - I_rc) * (I_n - I_rc);
        float dx2 = ((r - nr - minR) * (r - nr - minR) + (c - nc - minC) * (c - nc - minC));

        float f = filter(dI2, dx2);

        weight += f;
        val += I_n * f;
    });

    dest(r, c) = val / weight;
}

static CUDA_HOST_API int findBlockSize(const mat_fr& elements)
{
    if (elements.cols >= 512) {
        return 256;
    }
    return (elements.cols + 1) / 2;
}

namespace CudaPlayground
{
namespace CUDA
{
namespace General
{
    void BilateralFilter(const mat_fr source,  mat_fr dest, int radius, SmoothingKernel filter)
    {
        int blockSize = findBlockSize(dest);

        auto d_source = toDeviceMemory(source, true);
        auto d_dest = toDeviceMemory(dest, false);

        dim3 dimBlock(min(source.cols, blockSize), 1);
        dim3 dimGrid(pad(d_dest.mat.cols, blockSize) / dimBlock.x, d_dest.mat.rows);

        std::visit([&](auto& f)
        {
            BilateralFilterKernel KERNEL_ARGS(dimGrid, dimBlock) (d_source.mat, d_dest.mat, radius, f, blockSize);
        }
        , filter);

        checkCudaErrors(cudaGetLastError());

        copyFromDeviceMemory(d_dest.mat, dest);
    }

    void BilateralFilter_2(const mat_fr source,  mat_fr dest, int radius, SmoothingKernel filter)
    {
        int blockSize = findBlockSize(dest);

        auto d_source = toDeviceMemory(source, true);
        auto d_dest = toDeviceMemory(dest, false);

        dim3 dimBlock(min(source.cols, blockSize), 1);
        dim3 dimGrid(pad(d_dest.mat.cols, blockSize) / dimBlock.x, d_dest.mat.rows);

        std::visit([&](auto& f)
        {
            BilateralFilterKernel KERNEL_ARGS(dimGrid, dimBlock) (d_source.mat, d_dest.mat, radius, f, blockSize);
        }
        , filter);

        checkCudaErrors(cudaGetLastError());

        copyFromDeviceMemory(d_dest.mat, dest);
    }
}
}
}
