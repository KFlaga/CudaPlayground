#pragma once

#include "matrix.h"

namespace OmniSense
{
	namespace CUDA
	{
		namespace General
		{
			CUDA_HOST_API void Median(const mat_fr In, mat_fr Out, int radius);
		}
	}

	namespace General
	{
		template<typename MatrixT>
		CUDA_HOST_API void Median(const MatrixT In, MatrixT Out, int radius);
	}
}
