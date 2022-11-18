#include "cuda_all.h"
#include <device_launch_parameters.h>

#include "mat_multiply.h"
#include "matrix_device.h"
#include <memory>

#include <cublas.h>
#include <cublas_v2.h>

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

    int Bsub_offset = blockSize * blockSize;

    // Loop over all the sub-matrices of A and B that are required to compute Csub
    // Multiply each pair of sub-matrices together and accumulate the results
    for (int m = 0; m < (A.cols / blockSize); ++m)
    {
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        mat_fr Asub_shared{ blockSize, blockSize, blockSize, ((float*)shared_cache) + 0 };
        mat_fc Bsub_shared{ blockSize, blockSize, blockSize, ((float*)shared_cache) + Bsub_offset }; // B has column-wise layout as we access it that way
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
    void MatMul(const mat_fr A, const mat_fr B, mat_fr C)
    {
        int blockSize = findBlockSize(C);

        auto d_A = toDeviceMemoryPad(A, blockSize, blockSize, true);
        auto d_B = toDeviceMemoryPad(B, blockSize, blockSize, true);
        auto d_C = toDeviceMemoryPad(C, blockSize, blockSize, false);

        int sharedMemoryNeeded = blockSize * blockSize * 2 * sizeof(float);

        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(d_C.mat.cols / dimBlock.x, d_C.mat.rows / dimBlock.y);
        MatMulKernel KERNEL_ARGS(dimGrid, dimBlock, sharedMemoryNeeded) (d_A, d_B, d_C, blockSize);

        copyFromDeviceMemory(d_C.mat, C);
    }

    void MatMul(const mat_fc A, const mat_fc B, mat_fc C)
    {
        static cublasHandle_t ctx = nullptr;
        if (ctx == nullptr) {
            cublasCreate_v2(&ctx);
        }

        auto d_A = toDeviceMemory(A, true);
        auto d_B = toDeviceMemory(B, true);
        auto d_C = toDeviceMemory(C, false);

        float a = 1.0f;
        float b = 0.0f;

        cublasSgemm(ctx,
                    cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N,
                    C.rows, C.cols, A.cols,
                    &a,
                    d_A.mat.elements, d_A.mat.stride,
                    d_B.mat.elements, d_B.mat.stride,
                    &b,
                    d_C.mat.elements, d_C.mat.stride);
        
        copyFromDeviceMemory(d_C.mat, C);
    }
}
}
}
