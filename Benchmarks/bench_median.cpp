#pragma once

#include "nanobench.h"
#include "bench_helpers.h"

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/median.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>

using namespace OmniSense;

ADD_BENCHMARK(Median)
{
	auto test = [&](int rowsA, int colsA, int radius)
	{
		auto A = MatrixDynamic<mat_fr>(rowsA, colsA);
		auto C_cuda = MatrixDynamic<mat_fr>(rowsA, colsA);
		auto C_cpu = MatrixDynamic<mat_fr>(rowsA, colsA);

		fillRand(A, 0, 1);

		bench.run(format("Median CPU %i x %i R %i", rowsA, colsA, radius), [&]()
		{
			General::Median<mat_fr>(A, C_cpu, radius);
		});

		bench.run(format("Median CUDA %i x %i R %i", rowsA, colsA, radius), [&]()
		{
			CUDA::General::Median(A, C_cuda, radius);
		});
	};

	test(10, 10, 1);
	test(100, 100, 1);
	test(1000, 1000, 1);

	test(10, 10, 2);
	test(100, 100, 2);
	test(1000, 1000, 2);

	test(10, 10, 4);
	test(100, 100, 4);
	test(1000, 1000, 4);
};
