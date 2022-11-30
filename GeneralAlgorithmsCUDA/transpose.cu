#include "cuda_all.h"

#include "transpose.h"
#include "matrix_device.h"

using namespace CudaPlayground;


template<int N>
__global__ void TransposeKernel(mat_fr In, mat_fr Out)
{
	__shared__ float block[N][N + 1];

	int col = blockIdx.x * N + threadIdx.x;
	int row = blockIdx.y * N + threadIdx.y;
	if ((col < In.cols) && (row < In.rows))
	{
		block[threadIdx.y][threadIdx.x] = In(row, col);
	}

	__syncthreads();

	col = blockIdx.y * N + threadIdx.x;
	row = blockIdx.x * N + threadIdx.y;
	if ((col < Out.cols) && (row < Out.rows))
	{
		Out(row, col) = block[threadIdx.x][threadIdx.y];
	}
}

static CUDA_HOST_API int findBlockSize(mat_fr m)
{
	int size = m.rows * m.cols;
	if (size >= 64 * 64)
	{
		return 16;
	}
	if (size >= 8 * 8)
	{
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
    void Transpose(const mat_fr In, mat_fr Out)
    {
        int blockSize = findBlockSize(Out);

        auto d_In = toDeviceMemory(In, true);
        auto d_Out = toDeviceMemory(Out, false);

        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(d_In.mat.cols / dimBlock.x, d_In.mat.rows / dimBlock.y);
		switch (blockSize)
		{
		case 4:
			TransposeKernel<4> KERNEL_ARGS(dimGrid, dimBlock) (d_In, d_Out);
			break;
		case 8:
			TransposeKernel<8> KERNEL_ARGS(dimGrid, dimBlock) (d_In, d_Out);
			break;
		default:
			TransposeKernel<16> KERNEL_ARGS(dimGrid, dimBlock) (d_In, d_Out);
			break;
		}
        checkCudaErrors(cudaPeekAtLastError());

        copyFromDeviceMemory(d_Out.mat, Out);
    }

	void Transpose(mat_fr InOut)
	{
		Transpose(InOut, InOut);
	}
}
}
}
