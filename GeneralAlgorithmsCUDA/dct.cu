#include "cuda_all.h"
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "dct.h"
#include "matrix_device.h"
#include <cuda/std/cmath>
#include "cuda_stream.h"

using namespace CudaPlayground;

extern __shared__ float shared_cache[];

struct DCT_II
{
    static __device__ float first(float in0, float F)
    {
        return in0 * coeff(F, 0);
    }

    static __device__ float F(int i, int size)
    {
        return ((float)M_PI * i) / size;
    }

    static __device__ float coeff(float F, int k)
    {
        return __cosf((k + 0.5f) * F);
    }
};

struct DCT_III
{
    static __device__ float first(float in0, float F)
    {
        return in0 * 0.5f; 
    }

    static __device__ float F(int i, int size)
    {
        return ((float)M_PI * (i + 0.5f)) / size;
    }

    static __device__ float coeff(float F, int k)
    {
        return __cosf(k  * F);
    }
};

#pragma hd_warning_disable
const mat_fr DCT_LUT8 = []() {
    static float lut[64];
    mat_fr X{ 8, 8, 8, lut };
    X.forEach([](int i, int k, float& x)
    {
        x = 4.0f * std::cosf((k + 0.5f) * i * (float)M_PI / 8.0f);
    });
    return X;
}();

#pragma hd_warning_disable
const mat_fr IDCT_LUT8 = []() {
    static float lut[64];
    mat_fr X{ 8, 8, 8, lut };
    X.forEach([](int i, int k, float& x)
    {
        if (i == 0)
        {
            x = 2.0f;
        }
        else
        {
            x = 4.0f * std::cosf(k * (i * 0.5f) * (float)M_PI / 8.0f);
        }
    });
    return X;
}();


// Each thread computes single cell, one block per column/row, block have size of (1,N)
template<typename DCT_X, typename V1, typename V2>
static __device__ void CUDA_DCT_simple_1d_v1(V1 In, V2 Out, int blockSize)
{
    int tIdx = threadIdx.x + blockIdx.x * blockSize;

    // Now each thread iterates over inShared once 
    float scaling = (float)M_SQRT2 * __frsqrt_rn((float)In.size);
    float F = DCT_X::F(tIdx, In.size);
    float x = DCT_X::first(In(0), F);
    for (int k = 1; k < In.size; k++)
    {
        x += In(k) * DCT_X::coeff(F, k);
    }
    Out(tIdx) = x * scaling;
}

template<typename DCT_X>
static __global__ void CUDA_DCT_simple_v1_columns(mat_fr In, mat_fc Out, int blockSize)
{
    int col = threadIdx.y + blockIdx.y * blockSize;
    auto columnIn = In.column(col);
    auto columnOut = Out.column(col);
    CUDA_DCT_simple_1d_v1<DCT_X>(columnIn, columnOut, blockSize);
}

template<typename DCT_X>
static __global__ void CUDA_DCT_simple_v1_rows(mat_fc In, mat_fr Out, int blockSize)
{
    int row = threadIdx.y + blockIdx.y * blockSize;
    auto rowIn = In.row(row);
    auto rowOut = Out.row(row);
    CUDA_DCT_simple_1d_v1<DCT_X>(rowIn, rowOut, blockSize);
}

template<typename DCT_X, typename V1, typename V2>
static __device__ void CUDA_DCT_simple_1d_v2(V1 In, V2 Out, int blockSize)
{
    int tIdx = threadIdx.x;
    int bIdx = blockIdx.y;

    // Load whole In to shared memory first
    int loadPerThread = div_ceil(In.size, blockSize);
    float* inShared = (float*)shared_cache;
    for (int i = 0; i < loadPerThread; ++i)
    {
        int j = tIdx * loadPerThread + i;
        inShared[j] = In(j);
    }

    __syncthreads();

    // Now each thread iterates over inShared once 
    float scaling = (float)M_SQRT2 * __frsqrt_rn((float)In.size);
    float F = DCT_X::F(tIdx, In.size);
    float x = DCT_X::first(inShared[0], F);
    for (int k = 1; k < In.size; k++)
    {
        x += inShared[k] * DCT_X::coeff(F, k);
    }
    Out(tIdx + bIdx + blockSize) = x * scaling;
}

template<typename DCT_X>
static __global__ void CUDA_DCT_simple_v2_columns(mat_fr In, mat_fc Out, int blockSize)
{
    int col = blockIdx.x;
    auto columnIn = In.column(col);
    auto columnOut = Out.column(col);
    CUDA_DCT_simple_1d_v2<DCT_X>(columnIn, columnOut, blockSize);
}

template<typename DCT_X>
static __global__ void CUDA_DCT_simple_v2_rows(mat_fc In, mat_fr Out, int blockSize)
{
    int row = blockIdx.x;
    auto rowIn = In.row(row);
    auto rowOut = Out.row(row);
    CUDA_DCT_simple_1d_v2<DCT_X>(rowIn, rowOut, blockSize);
}

template<typename DCT_X>
void DCT_2d_simple_split_v1_impl(const mat_fr In, mat_fr Out)
{
    int blockSize = min(In.cols, min(In.rows, 16)); // Works only if cols and rows are multiple on one another, or of 16

    auto d_In = toDeviceMemoryPad(In, blockSize, blockSize, true);
    auto d_Out = toDeviceMemoryPad(Out, blockSize, blockSize, false);

    size_t tempStride;
    float* tempElements;
    checkCudaErrors(cudaMallocPitch(&tempElements, &tempStride, d_Out.mat.cols * sizeof(float), d_Out.mat.rows));
    checkCudaErrors(cudaMemset2D(tempElements, tempStride, 0, d_Out.mat.cols * sizeof(float), d_Out.mat.rows));
    DeviceMatrixGuard<mat_fc> d_Temp{ mat_fc{ d_Out.mat.rows, d_Out.mat.cols, (int)(tempStride / sizeof(float)), tempElements } };

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(pad(d_In.mat.cols, blockSize) / dimBlock.x, pad(d_In.mat.rows, blockSize) / dimBlock.y);

    CUDA_DCT_simple_v1_columns<DCT_X> KERNEL_ARGS(dimGrid, dimBlock) (d_In, d_Temp, blockSize);
    CUDA_DCT_simple_v1_rows<DCT_X> KERNEL_ARGS(dimGrid, dimBlock) (d_Temp, d_Out, blockSize);

    copyFromDeviceMemory(d_Out, Out);
}

template<typename DCT_X>
void DCT_2d_simple_split_v2_impl(const mat_fr In, mat_fr Out)
{
    int blockSize = min(In.cols, min(In.rows, 256));

    auto d_In = toDeviceMemoryPad(In, blockSize, blockSize, true);
    auto d_Out = toDeviceMemoryPad(Out, blockSize, blockSize, false);

    size_t tempStride;
    float* tempElements;
    checkCudaErrors(cudaMallocPitch(&tempElements, &tempStride, d_Out.mat.cols * sizeof(float), d_Out.mat.rows));
    checkCudaErrors(cudaMemset2D(tempElements, tempStride, 0, d_Out.mat.cols * sizeof(float), d_Out.mat.rows));
    DeviceMatrixGuard<mat_fc> d_Temp{ mat_fc{ d_Out.mat.rows, d_Out.mat.cols, (int)(tempStride / sizeof(float)), tempElements } };

    {
        dim3 dimBlock(blockSize, 1);
        dim3 dimGrid(d_In.mat.cols, d_In.mat.rows / blockSize);
        int sharedMemorySize = In.rows * sizeof(float);
        CUDA_DCT_simple_v2_columns<DCT_X> KERNEL_ARGS(dimGrid, dimBlock, sharedMemorySize) (d_In, d_Temp, blockSize);
    }
    {
        dim3 dimBlock(blockSize, d_In.mat.cols / blockSize);
        dim3 dimGrid(d_In.mat.rows, d_In.mat.cols / blockSize);
        int sharedMemorySize = In.cols * sizeof(float);
        CUDA_DCT_simple_v2_rows<DCT_X> KERNEL_ARGS(dimGrid, dimBlock, sharedMemorySize) (d_Temp, d_Out, blockSize);
    }

    copyFromDeviceMemory(d_Out, Out);
}

namespace CudaPlayground
{
namespace CUDA
{
namespace General
{
    void DCT_2d_simple_v1(const mat_fr In, mat_fr Out)
    {
        DCT_2d_simple_split_v1_impl<DCT_II>(In, Out);
    }

    void IDCT_2d_simple_v1(const mat_fr In, mat_fr Out)
    {
        DCT_2d_simple_split_v1_impl<DCT_III>(In, Out);
    }

    void DCT_2d_simple_v2(const mat_fr In, mat_fr Out)
    {
        DCT_2d_simple_split_v2_impl<DCT_II>(In, Out);
    }

    void IDCT_2d_simple_v2(const mat_fr In, mat_fr Out)
    {
        DCT_2d_simple_split_v2_impl<DCT_III>(In, Out);
    }
}
}
}
