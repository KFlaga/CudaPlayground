#pragma once

#include "matrix.h"

namespace CudaPlayground
{
	enum class ConvolveBoundary
	{
		Zero,
		Copy,
		Keep, // dont care
		ExtendZero
	};

	namespace CUDA
	{
		namespace General
		{
			// Optimized for small 'block' matrices compared to 'source'

			CUDA_HOST_API void Convolve(const mat_fr source, const mat_fr block, mat_fr dest, ConvolveBoundary);

			CUDA_HOST_API void ConvolveHorizontalMask(const mat_fr source, const mat_fr mask, mat_fr dest, ConvolveBoundary);

			CUDA_HOST_API void ConvolveVerticalMask(const mat_fr source, const mat_fr mask, mat_fr dest, ConvolveBoundary);
		}
	}

	namespace General
	{
		template<typename MatrixT>
		CUDA_HOST_API void Convolve(const MatrixT source, const MatrixT block, MatrixT dest, ConvolveBoundary);

		template<typename MatrixT>
		CUDA_HOST_API void ConvolveHorizontalMask(const MatrixT source, const MatrixT mask, MatrixT dest, ConvolveBoundary);

		template<typename MatrixT>
		CUDA_HOST_API void ConvolveVerticalMask(const MatrixT source, const MatrixT mask, MatrixT dest, ConvolveBoundary);
	}
}
