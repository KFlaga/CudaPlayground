#include "gtest/gtest.h"
#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/dct.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>
#include "test_helpers.h"

using namespace CudaPlayground;


TEST(DCT, simple)
{
	auto A = MatrixDynamic<mat_fr>(8, 8);
	auto C_cuda = MatrixDynamic<mat_fr>(8, 8);
	auto C_cpu = MatrixDynamic<mat_fr>(8, 8);

	fillRand(A, 0, 1);

	auto B = MatrixDynamic<mat_fc>(8, 8);
	B.forEach([&](int r, int c, float)
	{
		B(r, c) = A(r, c);
	});
	auto B_cpu = MatrixDynamic<mat_fc>(8, 8);

	General::DCT_2d_simple<mat_fr>(A, C_cpu);
	General::DCT_2d_simple<mat_fc>(B, B_cpu);
	assertEqual(B_cpu, C_cpu, 1e-4f, 1e-4f);

	CUDA::General::DCT_2d_simple_v1(A, C_cuda);
	
	assertEqual(C_cuda, C_cpu, 1e-2f, 1e-3f); // may have quite an error due to fast cosine

	auto A_cuda = MatrixDynamic<mat_fr>(8, 8);
	auto A_cpu = MatrixDynamic<mat_fr>(8, 8);

	CUDA::General::IDCT_2d_simple_v1(C_cuda, A_cuda);
	General::IDCT_2d_simple<mat_fr>(C_cpu, A_cpu);

	assertEqual(A_cuda, A_cpu, 1e-2f, 2e-3f);
	assertEqual(A_cuda, A, 1e-2f, 2e-3f);
}

TEST(DCT, simple_gpu_versions)
{
	auto A = MatrixDynamic<mat_fr>(64, 64);
	auto DCT_cuda_v1 = MatrixDynamic<mat_fr>(64, 64);
	auto DCT_cuda_v2 = MatrixDynamic<mat_fr>(64, 64);

	auto A_cuda_v1 = MatrixDynamic<mat_fr>(64, 64);
	auto A_cuda_v2 = MatrixDynamic<mat_fr>(64, 64);

	fillRand(A, 0, 1);

	CUDA::General::DCT_2d_simple_v1(A, DCT_cuda_v1);
	CUDA::General::IDCT_2d_simple_v1(DCT_cuda_v1, A_cuda_v1);

	CUDA::General::DCT_2d_simple_v2(A, DCT_cuda_v2);
	CUDA::General::IDCT_2d_simple_v2(DCT_cuda_v2, A_cuda_v2);

	assertEqual(DCT_cuda_v1, DCT_cuda_v2, 1e-4f, 1e-4f);

	assertEqual(A_cuda_v1, A, 1e-2f, 2e-3f);
	assertEqual(A_cuda_v2, A, 1e-2f, 2e-3f);
}
