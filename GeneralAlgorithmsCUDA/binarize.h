#pragma once

#include "matrix.h"

namespace OmniSense
{
	namespace CUDA
	{
		namespace General
		{
			CUDA_HOST_API void Binarize(const mat_fr In, mat_fr Out, float threshold, float lowValue, float highValue);
		}
	}

	namespace General
	{
		template<typename MatrixT>
		CUDA_HOST_API void Binarize(const MatrixT In, MatrixT Out, float threshold, float lowValue, float highValue);
	}
}
