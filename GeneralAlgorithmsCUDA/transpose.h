#pragma once

#include "matrix.h"

namespace CudaPlayground
{
	namespace CUDA
	{
		namespace General
		{
			CUDA_HOST_API void Transpose(const mat_fr In, mat_fr Out);
			CUDA_HOST_API void Transpose(mat_fr InOut); // inline only for square
		}
	}

	namespace General
	{
		template<typename MatrixT>
		CUDA_HOST_API void Transpose(const MatrixT In, MatrixT Out);
		template<typename MatrixT>
		CUDA_HOST_API void Transpose(MatrixT InOut);
	}
}
