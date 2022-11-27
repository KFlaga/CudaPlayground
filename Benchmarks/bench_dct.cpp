#pragma once

#include "nanobench.h"
#include "bench_helpers.h"

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/dct.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>

using namespace CudaPlayground;

ADD_BENCHMARK(DCT_Simple)
{
	auto test = [&](int size)
	{
		auto A = MatrixDynamic<mat_fr>(size, size);
		fillRand(A, 0, 1);

		auto DCT_cpu = MatrixDynamic<mat_fr>(size, size);
		auto DCT_cuda_merged = MatrixDynamic<mat_fr>(size, size);
		auto DCT_cuda_split_v1 = MatrixDynamic<mat_fr>(size, size);
		auto DCT_cuda_split_v2 = MatrixDynamic<mat_fr>(size, size);
		auto A_cpu = MatrixDynamic<mat_fr>(size, size);
		auto A_cuda_merged = MatrixDynamic<mat_fr>(size, size);
		auto A_cuda_split_v1 = MatrixDynamic<mat_fr>(size, size);
		auto A_cuda_split_v2 = MatrixDynamic<mat_fr>(size, size);

		if (size <= 64)
		{
			bench.run(format("DCT CPU %i x %i", size, size), [&]()
			{
				General::DCT_2d_simple<mat_fr>(A, DCT_cpu);
				General::IDCT_2d_simple<mat_fr>(DCT_cpu, A_cpu);
			});
		}

		bench.run(format("DCT CUDA v1 %i x %i", size, size), [&]()
		{
			CUDA::General::DCT_2d_simple_v1(A, DCT_cuda_split_v1);
			CUDA::General::IDCT_2d_simple_v1(DCT_cuda_split_v1, A_cuda_split_v1);
		});

		bench.run(format("DCT CUDA v2 %i x %i", size, size), [&]()
		{
			CUDA::General::DCT_2d_simple_v2(A, DCT_cuda_split_v2);
			CUDA::General::IDCT_2d_simple_v2(DCT_cuda_split_v2, A_cuda_split_v2);
		});
	};

	test(16);
	test(64);
	test(256);
	test(1024);
	test(4096);
};
