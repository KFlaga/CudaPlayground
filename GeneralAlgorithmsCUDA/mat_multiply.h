#pragma once

#include "matrix.h"

namespace CudaPlayground
{
	namespace CUDA
	{
		namespace General
		{
			CUDA_HOST_API void MatMul(const mat_fr A, const mat_fr B, mat_fr C);
			CUDA_HOST_API void MatMul(const mat_fc A, const mat_fc B, mat_fc C);
		}
	}

	namespace General
	{
		template<typename MatrixT>
		CUDA_HOST_API void MatMul(const MatrixT A, const MatrixT B, MatrixT C);
	}
}
