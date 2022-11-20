#include "gtest/gtest.h"
#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/median.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>
#include "test_helpers.h"

using namespace OmniSense;

TEST(MatrixBinarize, median3x3_fixed)
{
	auto A = MatrixDynamic<mat_fr>::fromRows({
		{ 1, 1, 1, 1, 1, 1 } ,
		{ 1, 2, 2, 2, 2, 2 } ,
		{ 1, 2, 3, 3, 3, 3 } ,
		{ 1, 2, 3, 4, 4, 4 } ,
		{ 1, 2, 3, 4, 5, 5 } ,
	});

	auto C_exp = MatrixDynamic<mat_fr>::fromRows({
		{ 1, 1, 1, 1, 1, 1 } ,
		{ 1, 1, 2, 2, 2, 2 } ,
		{ 1, 2, 2, 3, 3, 3 } ,
		{ 1, 2, 3, 3, 4, 4 } ,
		{ 1, 2, 3, 4, 4, 4 } ,
	});

	auto C_cuda = MatrixDynamic<mat_fr>(A.rows, A.cols);
	auto C_cpu = MatrixDynamic<mat_fr>(A.rows, A.cols);

	General::Median<mat_fr>(A, C_cpu, 1);
	assertEqual(C_cpu, C_exp);

	CUDA::General::Median(A, C_cuda, 1);
	assertEqual(C_cuda, C_exp);
}

//TEST(MatrixBinarize, median5x5)
//{
//	auto A = MatrixDynamic<mat_fr>(100, 100);
//	auto C_cuda = MatrixDynamic<mat_fr>(100, 100);
//	auto C_cpu = MatrixDynamic<mat_fr>(100, 100);
//
//	fillRand(A, 0, 1);
//
//	CUDA::General::Median(A, C_cuda, 2);
//	General::Median<mat_fr>(A, C_cpu, 2);
//	assertEqual(C_cpu, C_cuda);
//}

//TEST(MatrixBinarize, median15x15)
//{
//	auto A = MatrixDynamic<mat_fr>(100, 100);
//	auto C_cuda = MatrixDynamic<mat_fr>(100, 100);
//	auto C_cpu = MatrixDynamic<mat_fr>(100, 100);
//
//	fillRand(A, 0, 1);
//
//	CUDA::General::Median(A, C_cuda, 7);
//	General::Median<mat_fr>(A, C_cpu, 7);
//	assertEqual(C_cpu, C_cuda);
//}
