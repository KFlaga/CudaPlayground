#include "gtest/gtest.h"
#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/mat_multiply.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>
#include "test_helpers.h"

using namespace OmniSense;

TEST(MatrixFunctions, multiply_small)
{
	auto A = MatrixDynamic<mat_fr>(24, 16);
	auto B = MatrixDynamic<mat_fr>(16, 32);
	auto C_cuda = MatrixDynamic<mat_fr>(24, 32);
	auto C_cpu = MatrixDynamic<mat_fr>(24, 32);

	fill(A, 1);
	fill(B, 1);

	CUDA::General::MatMul(A, B, C_cuda);
	General::MatMul<mat_fr>(A, B, C_cpu);
	assertEqual(C_cuda, C_cpu);
}

TEST(MatrixFunctions, multiply_big)
{
	auto A = MatrixDynamic<mat_fr>(480, 160);
	auto B = MatrixDynamic<mat_fr>(160, 320);
	auto C_cuda = MatrixDynamic<mat_fr>(480, 320);
	auto C_cpu = MatrixDynamic<mat_fr>(480, 320);

	fill(A, 1);
	fill(B, 1);

	CUDA::General::MatMul(A, B, C_cuda);
	General::MatMul<mat_fr>(A, B, C_cpu);
	assertEqual(C_cuda, C_cpu);
}

TEST(MatrixFunctions, multiply_uneven)
{
	auto A = MatrixDynamic<mat_fr>(37, 75);
	auto B = MatrixDynamic<mat_fr>(75, 44);
	auto C_cuda = MatrixDynamic<mat_fr>(37, 44);
	auto C_cpu = MatrixDynamic<mat_fr>(37, 44);

	fill(A, 1);
	fill(B, 1);

	CUDA::General::MatMul(A, B, C_cuda);
	General::MatMul<mat_fr>(A, B, C_cpu);
	assertEqual(C_cuda, C_cpu);
}
