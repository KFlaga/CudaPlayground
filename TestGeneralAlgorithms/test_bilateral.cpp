#include "gtest/gtest.h"
#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/bilateral_filter.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>
#include "test_helpers.h"

using namespace CudaPlayground;

TEST(BilateralFilter, fixed_triangle_triangle)
{
	TriangleSmoothing ti{ 3.0f };
	TriangleSmoothing tx{ 2.0f };

	MultiSmoothing<TriangleSmoothing, TriangleSmoothing> tts{ ti, tx };
	SmoothingKernel smoothing{ tts };

	auto A = MatrixDynamic<mat_fr>::fromRows({
		{ 1, 2, 1 } ,
		{ 1, 4, 4 } ,
		{ 1, 4, 1 } ,
	});

	float r = std::sqrtf(2.0f);
	auto w = [&](float di, float dx)
	{
		return ti(di * di) * tx(dx * dx);
	};
	auto f = [&](float i, float di, float dx)
	{
		return i * w(di, dx);
	};

	auto C_exp = MatrixDynamic<mat_fr>::fromRows({
		{
			(f(1, 0, 0) + f(2, 1, 1) + f(1, 0, 1) + f(4, 3, r)) / (w(0, 0) + w(1, 1) + w(0, 1) + w(3, r)),
			(f(1, 1, 1) + f(2, 0, 0) + f(1, 1, 1) + f(1, 1, r) + f(4, 2, 1) + f(4, 2, r)) / (w(1, 1) + w(0, 0) + w(1, 1) + w(1, r) + w(2, 1) + w(2, r)),
			(f(2, 1, 1) + f(1, 0, 0) + f(4, 3, r) + f(4, 3, 1)) / (w(1, 1) + w(0, 0) + w(3, r) + w(3, 1)),
		},
		{
			(f(1, 0, 1) + f(2, 1, r) + f(1, 0, 0) + f(4, 3, 1) + f(1, 1, 0) + f(4, 3, r)) / (w(0, 1) + w(1, r) + w(0, 0) + w(3, 1) + w(1, 0) + w(3, r)),
			(f(1, 3, r) + f(2, 2, 1) + f(1, 3, r) + f(1, 3, 1) + f(4, 0, 0) + f(4, 0, 1) + f(1, 3, r) + f(4, 0, 1) + f(1, 3, r)) / (w(3, r) + w(2, 1) + w(3, r) + w(3, 1) + w(0, 0) + w(0, 1) + w(3, r) + w(0, 1) + w(3, r)),
			(f(2, 2, r) + f(1, 3, 1) + f(4, 0, 1) + f(4, 0, 0) + f(4, 0, r) + f(1, 3, 1)) / (w(2, r) + w(3, 1) + w(0, 1) + w(0, 0) + w(0, r) + w(3, 1)),
		},
		{
			(f(1, 0, 1) + f(4, 3, r) + f(1, 0, 0) + f(4, 3, 1)) / (w(0, 1) + w(3, r) + w(0, 0) + w(3, 1)),
			(f(1, 3, r) + f(4, 0, 1) + f(4, 0, r) + f(1, 3, 1) + f(4, 0, 0) + f(1, 3, 1)) / (w(3, r) + w(0, 1) + w(0, r) + w(3, 1) + w(0, 0) + w(3, 1)),
			(f(4, 3, r) + f(4, 3, 1) + f(4, 3, 1) + f(1, 0, 0)) / (w(3, r) + w(3, 1) + w(3, 1) + w(0, 0)),
		},
	});

	auto C_cuda = MatrixDynamic<mat_fr>(3, 3);
	auto C_cpu = MatrixDynamic<mat_fr>(3, 3);

	General::BilateralFilter<mat_fr>(A, C_cpu, 1, smoothing);
	CUDA::General::BilateralFilter(A, C_cuda, 1, smoothing);

	assertEqual(C_cpu, C_exp, 1e-3f);
	assertEqual(C_cuda, C_exp, 1e-3f);
}

TEST(BilateralFilter, gauss_gauss)
{
	auto A = MatrixDynamic<mat_fr>(100, 100);
	auto C_cuda = MatrixDynamic<mat_fr>(100, 100);
	auto C_cpu = MatrixDynamic<mat_fr>(100, 100);

	fillRand(A, 0, 1);

	GaussianGaussianSmoothing ggs{ 0.5, 2.0 };
	SmoothingKernel smoothing{ ggs };

	CUDA::General::BilateralFilter(A, C_cuda, 7, smoothing);
	General::BilateralFilter<mat_fr>(A, C_cpu, 7, smoothing);
	assertEqual(C_cpu, C_cuda, 1e-3f);
}
