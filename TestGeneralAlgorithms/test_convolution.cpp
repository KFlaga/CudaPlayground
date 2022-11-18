#include "gtest/gtest.h"
#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/convolution.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>
#include "test_helpers.h"

using namespace OmniSense;

TEST(MatrixConvolution, convolve_copyBoundary)
{
	auto A = MatrixDynamic<mat_fr>::fromRows({
		{ 1, 1, 1, 1, 1, 1 } ,
		{ 1, 2, 2, 2, 2, 2 } ,
		{ 1, 2, 3, 3, 3, 3 } ,
		{ 1, 2, 3, 4, 4, 4 } ,
		{ 1, 2, 3, 4, 5, 5 } ,
	});

	auto B = MatrixDynamic<mat_fr>::fromRows({
		{ 0, 1, 0 } ,
		{ 1, 1, 1 } ,
		{ 0, 1, 0 } ,
	});

	auto C_exp = MatrixDynamic<mat_fr>::fromRows({
		{ 1, 1,  1,  1,  1,  1 } ,
		{ 1, 8,  10, 10, 10, 2 } ,
		{ 1, 10, 13, 15, 15, 3 } ,
		{ 1, 10, 15, 18, 20, 4 } ,
		{ 1, 2,   3,  4,  5, 5 } ,
	});

	auto C_cuda = MatrixDynamic<mat_fr>(A.rows, A.cols);
	auto C_cpu = MatrixDynamic<mat_fr>(A.rows, A.cols);

	General::Convolve<mat_fr>(A, B, C_cpu, ConvolveBoundary::Copy);
	CUDA::General::Convolve(A, B, C_cuda, ConvolveBoundary::Copy);
	assertEqual(C_cpu, C_exp);
	assertEqual(C_cuda, C_exp);
}

TEST(MatrixConvolution, convolve_zeroBoundary)
{
	auto A = MatrixDynamic<mat_fr>::fromRows({
		{ 1, 1, 1, 1, 1, 1 } ,
		{ 1, 2, 2, 2, 2, 2 } ,
		{ 1, 2, 3, 3, 3, 3 } ,
		{ 1, 2, 3, 4, 4, 4 } ,
		{ 1, 2, 3, 4, 5, 5 } ,
	});

	auto B = MatrixDynamic<mat_fr>::fromRows({
		{ 0, 1, 0 } ,
		{ 1, 1, 1 } ,
		{ 0, 1, 0 } ,
	});

	auto C_exp = MatrixDynamic<mat_fr>::fromRows({
		{ 0, 0,  0,  0,  0,  0 } ,
		{ 0, 8,  10, 10, 10, 0 } ,
		{ 0, 10, 13, 15, 15, 0 } ,
		{ 0, 10, 15, 18, 20, 0 } ,
		{ 0, 0,  0,  0,  0,  0 } ,
	});

	auto C_cuda = MatrixDynamic<mat_fr>(A.rows, A.cols);
	auto C_cpu = MatrixDynamic<mat_fr>(A.rows, A.cols);

	General::Convolve<mat_fr>(A, B, C_cpu, ConvolveBoundary::Zero);
	CUDA::General::Convolve(A, B, C_cuda, ConvolveBoundary::Zero);
	assertEqual(C_cpu, C_exp);
	assertEqual(C_cuda, C_exp);
}

TEST(MatrixConvolution, convolve_extendZeroBoundary)
{
	auto A = MatrixDynamic<mat_fr>::fromRows({
		{ 1, 1, 1, 1, 1, 1 } ,
		{ 1, 2, 2, 2, 2, 2 } ,
		{ 1, 2, 3, 3, 3, 3 } ,
		{ 1, 2, 3, 4, 4, 4 } ,
		{ 1, 2, 3, 4, 5, 5 } ,
	});

	auto B = MatrixDynamic<mat_fr>::fromRows({
		{ 0, 1, 0 } ,
		{ 1, 1, 1 } ,
		{ 0, 1, 0 } ,
	});

	auto C_exp = MatrixDynamic<mat_fr>::fromRows({
		{ 3, 5,  5,  5,  5,  4 } ,
		{ 5, 8,  10, 10, 10, 8 } ,
		{ 5, 10, 13, 15, 15, 12 } ,
		{ 5, 10, 15, 18, 20, 16 } ,
		{ 4, 8,  12, 16, 18, 14 } ,
	});

	auto C_cuda = MatrixDynamic<mat_fr>(A.rows, A.cols);
	auto C_cpu = MatrixDynamic<mat_fr>(A.rows, A.cols);

	General::Convolve<mat_fr>(A, B, C_cpu, ConvolveBoundary::ExtendZero);
	CUDA::General::Convolve(A, B, C_cuda, ConvolveBoundary::ExtendZero);
	assertEqual(C_cpu, C_exp);
	assertEqual(C_cuda, C_exp);
}
