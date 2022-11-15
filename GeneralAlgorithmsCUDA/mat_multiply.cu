#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "mat_multiply.h"
#include "matrix_helpers.cuh"
#include <memory>

using namespace OmniSense;

extern __shared__ float shared_cache[];

static __global__ void MatMulKernel(mat_fr A, mat_fr B, mat_fr C, int blockSize)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread computes one element of C by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within submatrix of C
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are required to compute Csub
    // Multiply each pair of sub-matrices together and accumulate the results
    for (int m = 0; m < (A.cols / blockSize); ++m)
    {
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        mat_fr Asub_shared{ blockSize, blockSize, blockSize, ((float*)shared_cache) + 0 };
        mat_fc Bsub_shared{ blockSize, blockSize, blockSize, ((float*)shared_cache) + blockSize * blockSize }; // B has column-wise layout as we access it that way
        Asub_shared(row, col) = A(row + blockRow * blockSize, col + m * blockSize);
        Bsub_shared(row, col) = B(row + m * blockSize, col + blockCol * blockSize);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int e = 0; e < blockSize; ++e)
        {
            Cvalue += Asub_shared(row, e) * Bsub_shared(e, col);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    C(row + blockRow * blockSize, col + blockCol * blockSize) = Cvalue;
}

namespace OmniSense
{
namespace CUDA
{
namespace General
{
    void MatMul(const mat_fr A, const mat_fr B, mat_fr C)
    {
        int blockSize = findBlockSize(C.cols * C.rows);

        auto d_A = toDeviceMemoryWithPadding(A, blockSize, blockSize, true);
        auto d_B = toDeviceMemoryWithPadding(B, blockSize, blockSize, true);
        auto d_C = toDeviceMemoryWithPadding(C, blockSize, blockSize, false);

        int sharedMemoryNeeded = blockSize * blockSize * 2 * sizeof(float);

        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(d_C.mat.cols / dimBlock.x, d_C.mat.rows / dimBlock.y);
        MatMulKernel KERNEL_ARGS(dimGrid, dimBlock, sharedMemoryNeeded) (d_A, d_B, d_C, blockSize);

        copyFromDeviceMemory(d_C.mat, C);
    }
}
}
}
