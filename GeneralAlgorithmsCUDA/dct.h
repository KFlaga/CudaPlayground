#pragma once

#include "matrix.h"

namespace CudaPlayground
{
	namespace CUDA
	{
		namespace General
		{
			CUDA_HOST_API void DCT_2d_simple_v1(const mat_fr In, mat_fr Out);
			CUDA_HOST_API void IDCT_2d_simple_v1(const mat_fr In, mat_fr Out);

			CUDA_HOST_API void DCT_2d_simple_v2(const mat_fr In, mat_fr Out);
			CUDA_HOST_API void IDCT_2d_simple_v2(const mat_fr In, mat_fr Out);
		}
	}

	namespace General
	{
		template<typename MatrixT>
		CUDA_HOST_API void DCT_2d_simple(const MatrixT In, MatrixT Out);

		template<typename MatrixT>
		CUDA_HOST_API void IDCT_2d_simple(const MatrixT In, MatrixT Out);
	}
}
