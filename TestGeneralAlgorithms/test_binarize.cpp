#include "gtest/gtest.h"
#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/binarize.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>
#include "test_helpers.h"

using namespace CudaPlayground;

TEST(MatrixBinarize, binarize)
{
	auto A = MatrixDynamic<mat_fr>(100, 100);
	auto C_cuda = MatrixDynamic<mat_fr>(100, 100);
	auto C_cpu = MatrixDynamic<mat_fr>(100, 100);

	fillRand(A, 0, 1);

	CUDA::General::Binarize(A, C_cuda, 0.5f, 0.0f, 1.0f);
	General::Binarize<mat_fr>(A, C_cpu, 0.5f, 0.0f, 1.0f);
	
	for (int r = 0; r < C_cuda.rows; ++r)
	{
		for (int c = 0; c < C_cuda.cols; ++c)
		{
			if (A(r, c) > 0.5001f)
			{
				ASSERT_FLOAT_EQ(C_cpu(r, c), 1.f) << " at r = " << r << " c = " << c << ", A = " << A(r, c);
				ASSERT_FLOAT_EQ(C_cuda(r, c), 1.f) << " at r = " << r << " c = " << c << ", A = " << A(r,c);
			}
			else if (A(r, c) < 0.4999f)
			{
				ASSERT_FLOAT_EQ(C_cpu(r, c), 0.f) << " at r = " << r << " c = " << c << ", A = " << A(r, c);
				ASSERT_FLOAT_EQ(C_cuda(r, c), 0.f) << " at r = " << r << " c = " << c << ", A = " << A(r, c);
			}
		}
	}
}
