#include "gtest/gtest.h"
#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/transpose.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>
#include "test_helpers.h"

using namespace CudaPlayground;


TEST(Transpose, toOther)
{
	auto A = MatrixDynamic<mat_fr>(64, 64);
	auto C_cpu = MatrixDynamic<mat_fr>(64, 64);
	auto C_cuda = MatrixDynamic<mat_fr>(64, 64);

	fillRand(A, 0, 1);

	General::Transpose<mat_fr>(A, C_cpu);
	CUDA::General::Transpose(A, C_cuda);

	for (int r = 0; r < 64; ++r)
	{
		for (int c = 0; c < 64; ++c)
		{
			EXPECT_FLOAT_EQ(A(r, c), C_cpu(c, r)) << " at r = " << r << " c = " << c;
			ASSERT_FLOAT_EQ(A(r, c), C_cuda(c, r)) << " at r = " << r << " c = " << c;
		}
	}
}

TEST(Transpose, toSame)
{
	auto A = MatrixDynamic<mat_fr>(64, 64);
	auto C_cpu = MatrixDynamic<mat_fr>(64, 64);
	auto C_cuda = MatrixDynamic<mat_fr>(64, 64);

	fillRand(A, 0, 1);
	for (int r = 0; r < 64; ++r)
	{
		for (int c = 0; c < 64; ++c)
		{
			C_cpu(r, c) = A(r, c);
			C_cuda(r, c) = A(r, c);
		}
	}

	General::Transpose<mat_fr>(C_cpu);
	CUDA::General::Transpose(C_cuda);

	for (int r = 0; r < 64; ++r)
	{
		for (int c = 0; c < 64; ++c)
		{
			EXPECT_FLOAT_EQ(A(r, c), C_cpu(c, r)) << " at r = " << r << " c = " << c;
			ASSERT_FLOAT_EQ(A(r, c), C_cuda(c, r)) << " at r = " << r << " c = " << c;
		}
	}
}
