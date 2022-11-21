#pragma once

#include "nanobench.h"
#include "bench_helpers.h"

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/mat_multiply.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>
#include <GeneralAlgorithmsCUDA/cuda_stream.h>
#include <GeneralAlgorithmsCUDA/pinned_allocator.h>

#include <iostream>

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
	//test(512, 512, 512);
};

template<typename MatrixAllocatorT>
static void testAsync(ankerl::nanobench::Bench& bench, int size, int ops, int streamCount)
{
	bench.batch(ops);

	std::vector<MatrixAllocatorT> As;
	std::vector<MatrixAllocatorT> Bs;
	std::vector<MatrixAllocatorT> Cs;
	for (int s = 0; s < streamCount; ++s)
	{
		As.emplace_back(size, size);
		Bs.emplace_back(size, size);
		Cs.emplace_back(size, size);
		fillRand(As[s], 0, 1);
		fillRand(Bs[s], 0, 1);
	}

	std::vector<CudaStream> streams(streamCount);
	int operationsPerStream = ops / streamCount;

	bench.run(format("%i x %i | x%i on %i streams | host sync", size, size, ops, streamCount), [&]()
	{
		for (int i = 0; i < operationsPerStream; ++i)
		{
			for (int s = 0; s < streamCount; ++s)
			{
				CUDA::General::MatMulAsync(As[s], Bs[s], Cs[s], streams[s]);
			}
		}

		for (auto& stream : streams)
		{
			stream.sync();
		}
	});

	bench.run(format("%i x %i | x%i on %i streams | host async", size, size, ops, streamCount), [&]()
	{
		std::vector<std::future<void>> fs{};
		for (int s = 0; s < streamCount; ++s)
		{
			fs.push_back(std::async(std::launch::async, [&, n=s]()
			{
				for (int i = 0; i < operationsPerStream; ++i)
				{
					CUDA::General::MatMulAsync(As[n], Bs[n], Cs[n], streams[n]);
				}
				streams[n].sync();
			}));
		}

		for (auto& f : fs)
		{
			f.wait_for(std::chrono::seconds(5));
		}
	});
}

template<typename MatrixAllocatorT>
static void testsuiteAsync(ankerl::nanobench::Bench& bench)
{
	// MatrixDynamic<mat_fr, PinnedAllocator>
	testAsync<MatrixAllocatorT>(bench, 20, 4, 1);
	testAsync<MatrixAllocatorT>(bench, 20, 4, 2);
	testAsync<MatrixAllocatorT>(bench, 20, 4, 4);

	testAsync<MatrixAllocatorT>(bench, 20, 12, 1);
	testAsync<MatrixAllocatorT>(bench, 20, 12, 2);
	testAsync<MatrixAllocatorT>(bench, 20, 12, 4);

	testAsync<MatrixAllocatorT>(bench, 100, 4, 1);
	testAsync<MatrixAllocatorT>(bench, 100, 4, 2);
	testAsync<MatrixAllocatorT>(bench, 100, 4, 4);

	testAsync<MatrixAllocatorT>(bench, 100, 12, 1);
	testAsync<MatrixAllocatorT>(bench, 100, 12, 2);
	testAsync<MatrixAllocatorT>(bench, 100, 12, 4);

	testAsync<MatrixAllocatorT>(bench, 500, 4, 1);
	testAsync<MatrixAllocatorT>(bench, 500, 4, 2);
	testAsync<MatrixAllocatorT>(bench, 500, 4, 4);

	testAsync<MatrixAllocatorT>(bench, 500, 12, 1);
	testAsync<MatrixAllocatorT>(bench, 500, 12, 2);
	testAsync<MatrixAllocatorT>(bench, 500, 12, 4);
}

ADD_BENCHMARK(MultiplyAsync)
{
	std::cout << "PAGED HOST MEMORY" << std::endl;
	testsuiteAsync<MatrixDynamic<mat_fr>>(bench);

	std::cout << "PAGE-LOCKED HOST MEMORY" << std::endl;
	testsuiteAsync<MatrixDynamic<mat_fr, PinnedAllocator>>(bench);
};
