#pragma once

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/cuda_interop.h>
#include <future>

namespace CudaPlayground
{
	namespace CUDA
	{
		namespace General
		{
			CUDA_HOST_API void MatMul(const mat_fr A, const mat_fr B, mat_fr C);
			CUDA_HOST_API void MatMul(const mat_fc A, const mat_fc B, mat_fc C);

			// Result of async call (matrix C) will be valid when stream is synchronized
			CUDA_HOST_API void MatMulAsync(const mat_fr A, const mat_fr B, mat_fr C, cudaStream_t stream);
		}
	}

	namespace General
	{
		template<typename MatrixT>
		CUDA_HOST_API void MatMul(const MatrixT A, const MatrixT B, MatrixT C);
	}
}
