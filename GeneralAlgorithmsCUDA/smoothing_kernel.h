#pragma once

#include <GeneralAlgorithmsCUDA/cuda_interop.h>
#include <variant>

#ifndef  __NVCC__
#include <cmath>
#endif

namespace CudaPlayground
{
	struct GaussianGaussianSmoothing
	{
		GaussianGaussianSmoothing(float sigma_i, float sigma_x);
		CUDA_DEVICE_API float operator()(float dI2, float dx2);

		float inv_sigma_i;
		float inv_sigma_x;
	};

	struct GaussianSmoothing
	{
		GaussianSmoothing(float sigma);
		CUDA_DEVICE_API float operator()(float x2);

		float inv_sigma;
	};

	struct TriangleSmoothing
	{
		TriangleSmoothing(float radius);
		CUDA_DEVICE_API float operator()(float x2);

		float radius;
		float inv_radius;
	};

	template<typename S1, typename S2>
	struct MultiSmoothing
	{
		CUDA_DEVICE_API float operator()(float dI2, float dx2)
		{
			return filterIntensity(dI2) * filterDistance(dx2);
		}

		S1 filterIntensity;
		S2 filterDistance;
	};

	using SmoothingKernel = std::variant<
		GaussianGaussianSmoothing,
		MultiSmoothing<GaussianSmoothing, TriangleSmoothing>,
		MultiSmoothing<TriangleSmoothing, GaussianSmoothing>,
		MultiSmoothing<TriangleSmoothing, TriangleSmoothing>
	>;

	inline GaussianGaussianSmoothing::GaussianGaussianSmoothing(float si, float sx)
		: inv_sigma_i{ 1.0f / si }
		, inv_sigma_x{ 1.0f / sx }
	{}

	__forceinline CUDA_DEVICE_API float GaussianGaussianSmoothing::operator()(float dI2, float dx2)
	{
		float i_p = dI2 * 0.5f * inv_sigma_i * inv_sigma_i;
		float x_p = dx2 * 0.5f * inv_sigma_x * inv_sigma_x;

#ifdef  __NVCC__
		return __expf(-i_p - x_p);
#else
		return std::expf(-i_p - x_p);
#endif
	}

	inline GaussianSmoothing::GaussianSmoothing(float s)
		: inv_sigma{ 1.0f / s }
	{}

	__forceinline CUDA_DEVICE_API float GaussianSmoothing::operator()(float x2)
	{
		float p = x2 * 0.5f * inv_sigma * inv_sigma;

#ifdef  __NVCC__
		return __expf(-p);
#else
		return std::expf(-p);
#endif
	}

	inline TriangleSmoothing::TriangleSmoothing(float r)
		: radius{ r }
		, inv_radius{ 1.0f / r }
	{}

	__forceinline CUDA_DEVICE_API float TriangleSmoothing::operator()(float x2)
	{
#ifdef  __NVCC__
		float x = __fsqrt_rn(x2);
		return max(0.0f, (radius - x) * inv_radius);
#else
		float x = std::sqrtf(x2);
		return std::max(0.0f, (radius - x) * inv_radius);
#endif
	}
}
