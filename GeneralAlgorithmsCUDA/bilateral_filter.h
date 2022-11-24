#pragma once

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/smoothing_kernel.h>

namespace CudaPlayground
{
	namespace CUDA
	{
		namespace General
		{
			CUDA_HOST_API void BilateralFilter(const mat_fr In, mat_fr Out, int radius, SmoothingKernel filter);
		}
	}

	namespace General
	{
		template<typename MatrixT>
		CUDA_HOST_API void BilateralFilter(const MatrixT In, MatrixT Out, int radius, SmoothingKernel filter);
	}
}
