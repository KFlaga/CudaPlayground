#pragma once

#include "nanobench.h"
#include "bench_helpers.h"

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/binarize.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>

using namespace OmniSense;

ADD_BENCHMARK(Binarize)
{
	auto test = [&](int size)
	{
		auto A = MatrixDynamic<mat_fr>(size, size);
		auto C_cuda = MatrixDynamic<mat_fr>(size, size);
		auto C_cpu = MatrixDynamic<mat_fr>(size, size);
		fillRand(A, 0, 1);

		bench.run(format("Binarize CPU %i x %i", size, size), [&]()
		{
			General::Binarize<mat_fr>(A, C_cpu, 0.5f, 0.0f, 1.0f);
		});

		bench.run(format("Binarize CUDA %i x %i", size, size), [&]()
		{
			CUDA::General::Binarize(A, C_cuda, 0.5f, 0.0f, 1.0f);
		});
	};

	test(10);
	test(100);
	test(1000);
};
