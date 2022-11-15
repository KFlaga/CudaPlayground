#pragma once

#include "matrix.h"

namespace OmniSense
{
	enum class ConvolveBoundary
	{
		Zero,
		Copy,
		Keep
	};

	namespace CUDA
	{
		namespace General
		{
			CUDA_HOST_API void Convolve(const mat_fr A, const mat_fr B, mat_fr C, ConvolveBoundary);

			CUDA_HOST_API void ConvolveHorizontalMask(const mat_fr A, const mat_fr mask, mat_fr C, ConvolveBoundary);

			CUDA_HOST_API void ConvolveVerticalMask(const mat_fr A, const mat_fr mask, mat_fr C, ConvolveBoundary);
		}
	}

	namespace General
	{
		template<typename MatrixT>
		CUDA_HOST_API void Convolve(const MatrixT A, const MatrixT B, MatrixT C, ConvolveBoundary);

		template<typename MatrixT>
		CUDA_HOST_API void ConvolveHorizontalMask(const MatrixT A, const MatrixT mask, MatrixT C, ConvolveBoundary);

		template<typename MatrixT>
		CUDA_HOST_API void ConvolveVerticalMask(const MatrixT A, const MatrixT mask, MatrixT C, ConvolveBoundary);
	}
}
