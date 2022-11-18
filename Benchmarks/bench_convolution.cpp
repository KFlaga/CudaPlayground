#pragma once

#include "nanobench.h"
#include "bench_helpers.h"

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/convolution.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>

using namespace OmniSense;

ADD_BENCHMARK(Convolution)
{
	auto test = [&](int rowsA, int colsA, int blockA, int blockB, ConvolveBoundary boundary)
	{
		auto A = MatrixDynamic<mat_fr>(rowsA, colsA);
		auto B = MatrixDynamic<mat_fr>(blockA, blockB);
		auto C_cuda = MatrixDynamic<mat_fr>(rowsA, colsA);
		auto C_cpu = MatrixDynamic<mat_fr>(rowsA, colsA);

		std::string boundaryName = [&]() {
			switch (boundary)
			{
			case ConvolveBoundary::Copy: return "Copy";
			case ConvolveBoundary::ExtendZero: return "ExtendZero";
			case ConvolveBoundary::Keep: return "Keep";
			case ConvolveBoundary::Zero: default: return "Zero";
			}
		}();

		fillRand(A, 0, 1);
		fillRand(B, 0, 1);

		bench.run(format("Convolution CPU %i x %i * %i x %i %s", rowsA, colsA, blockA, blockB, boundaryName.c_str()), [&]()
		{
			General::Convolve<mat_fr>(A, B, C_cpu, boundary);
		});

		bench.run(format("Convolution CUDA %i x %i * %i x %i %s", rowsA, colsA, blockA, blockB, boundaryName.c_str()), [&]()
		{
			CUDA::General::Convolve(A, B, C_cuda, boundary);
		});
	};

	test(10, 10, 3, 3, ConvolveBoundary::Copy);
	test(100, 100, 3, 3, ConvolveBoundary::Copy);
	test(100, 100, 7, 7, ConvolveBoundary::Copy);
	test(100, 100, 13, 13, ConvolveBoundary::Copy);
	test(1000, 1000, 3, 3, ConvolveBoundary::Copy);
	test(1000, 1000, 7, 7, ConvolveBoundary::Copy);
	test(1000, 1000, 13, 13, ConvolveBoundary::Copy);
	test(100, 100, 7, 7, ConvolveBoundary::Copy);
	test(100, 100, 7, 7, ConvolveBoundary::Zero);
	test(100, 100, 7, 7, ConvolveBoundary::ExtendZero);
	test(100, 100, 100, 100, ConvolveBoundary::ExtendZero);
};
