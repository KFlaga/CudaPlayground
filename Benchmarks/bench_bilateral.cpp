#pragma once

#include "nanobench.h"
#include "bench_helpers.h"

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/bilateral_filter.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>

using namespace CudaPlayground;

ADD_BENCHMARK(BilateralFilter)
{
	auto test = [&](int size, int radius, SmoothingKernel smoothing)
	{
		auto A = MatrixDynamic<mat_fr>(size, size);
		auto C_cuda = MatrixDynamic<mat_fr>(size, size);
		auto C_cpu = MatrixDynamic<mat_fr>(size, size);

		fillRand(A, 0, 1);

		std::string smoothingName = std::holds_alternative<GaussianGaussianSmoothing>(smoothing) ? "GG" : "TT";

		bench.run(format("BilateralFilter CPU %i x %i | %i | %s", size, size, radius, smoothingName.c_str()), [&]()
		{
			General::BilateralFilter<mat_fr>(A, C_cpu, radius, smoothing);
		});

		bench.run(format("BilateralFilter CUDA %i x %i | %i | %s", size, size, radius, smoothingName.c_str()), [&]()
		{
			CUDA::General::BilateralFilter(A, C_cuda, radius, smoothing);
		});

		bench.run(format("BilateralFilter 2 CUDA %i x %i | %i | %s", size, size, radius, smoothingName.c_str()), [&]()
		{
			CUDA::General::BilateralFilter_2(A, C_cuda, radius, smoothing);
		});
	};

	test(128, 2, GaussianGaussianSmoothing{ 2.0f, 2.0f });
	test(128, 2, MultiSmoothing<TriangleSmoothing, TriangleSmoothing>{ 2.0f, 2.0f });
	test(128, 4, GaussianGaussianSmoothing{ 2.0f, 2.0f });
	test(128, 4, MultiSmoothing<TriangleSmoothing, TriangleSmoothing>{ 2.0f, 2.0f });
	test(128, 8, GaussianGaussianSmoothing{ 2.0f, 2.0f });
	test(128, 8, MultiSmoothing<TriangleSmoothing, TriangleSmoothing>{ 2.0f, 2.0f });

	test(512, 2, GaussianGaussianSmoothing{ 2.0f, 2.0f });
	test(512, 2, MultiSmoothing<TriangleSmoothing, TriangleSmoothing>{ 2.0f, 2.0f });
	test(512, 4, GaussianGaussianSmoothing{ 2.0f, 2.0f });
	test(512, 4, MultiSmoothing<TriangleSmoothing, TriangleSmoothing>{ 2.0f, 2.0f });
	test(512, 8, GaussianGaussianSmoothing{ 2.0f, 2.0f });
	test(512, 8, MultiSmoothing<TriangleSmoothing, TriangleSmoothing>{ 2.0f, 2.0f });
};
