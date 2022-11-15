#pragma once

#include "matrix.h"

namespace OmniSense
{
	namespace CUDA
	{
		namespace General
		{
			CUDA_HOST_API void MatMul(const mat_fr A, const mat_fr B, mat_fr C);
		}
	}

	namespace General
	{
		template<typename MatrixT>
		CUDA_HOST_API void MatMul(const MatrixT A, const MatrixT B, MatrixT C);
	}
}
