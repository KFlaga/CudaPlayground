#pragma once

#include "nanobench.h"
#include "bench_helpers.h"

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/mat_multiply.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>

using namespace CudaPlayground;

ADD_BENCHMARK(Multiply)
{
	auto test = [&](int rowsC, int colsC, int sizeAB)
	{
		auto A = MatrixDynamic<mat_fr>(rowsC, sizeAB);
		auto B = MatrixDynamic<mat_fr>(sizeAB, colsC);
		auto C_cuda = MatrixDynamic<mat_fr>(rowsC, colsC);
		auto C_cpu = MatrixDynamic<mat_fr>(rowsC, colsC);

		fillRand(A, 0, 1);
		fillRand(B, 0, 1);

		bench.run(format("MatMul CPU %i x %i * %i x %i", rowsC, sizeAB, sizeAB, colsC), [&]()
		{
			General::MatMul<mat_fr>(A, B, C_cpu);
		});

		bench.run(format("MatMul CUDA %i x %i * %i x %i", rowsC, sizeAB, sizeAB, colsC), [&]()
		{
			CUDA::General::MatMul(A, B, C_cuda);
		});

		auto Ac = MatrixDynamic<mat_fc>(rowsC, sizeAB);
		auto Bc = MatrixDynamic<mat_fc>(sizeAB, colsC);
		auto Cc_cuda = MatrixDynamic<mat_fc>(rowsC, colsC);

		fillRand(Ac, 0, 1);
		fillRand(Bc, 0, 1);

		bench.run(format("MatMul CUDA CM %i x %i * %i x %i", rowsC, sizeAB, sizeAB, colsC), [&]()
		{
			CUDA::General::MatMul(Ac, Bc, Cc_cuda);
		});
	};

	test(10, 10, 10);
	test(100, 100, 100);
	test(128, 128, 128);
	test(512, 128, 32);
	test(32, 128, 512);
	test(512, 512, 512);
};
