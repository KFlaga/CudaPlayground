#pragma once

#include "matrix.h"
#include "convolution.h"
#include <vector> // TODO: maybe use arbitrary dimension (or maybe only 3d) matrix

namespace CudaPlayground
{
	namespace CUDA
	{
		namespace General
		{
			CUDA_HOST_API void MultiBlockFilter(const mat_fr In, mat_fr Out, const std::vector<mat_fr>& filters, ConvolveBoundary);
		}
	}

	namespace General
	{
		template<typename MatrixT>
		CUDA_HOST_API void MultiBlockFilter(const MatrixT In, MatrixT Out, const std::vector<mat_fr>& filters, ConvolveBoundary);
	}
}
