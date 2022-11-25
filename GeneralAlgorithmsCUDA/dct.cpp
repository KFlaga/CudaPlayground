#include "mat_multiply.h"
#include <malloc.h>
#include <cmath>

namespace CudaPlayground
{
namespace General
{

// DCT II, from wikipedia
template<typename VectorT>
void DCT_simple_1d(const VectorT In, VectorT Out)
{
	float F = M_PI / In.size;
	for (int i = 0; i < In.size; ++i)
	{
		float x = 0.0f;
		for (int k = 0; k < In.size; k++)
		{
			x += In(k) * std::cosf((k + 0.5f) * i * F);
		}
		Out(i) = x;
	}
}

// DCT III
template<typename VectorT>
void IDCT_simple_1d(const VectorT In, VectorT Out)
{
	float F = M_PI / In.size;
	for (int i = 0; i < In.size; ++i)
	{
		float x = In(0) * 0.5f;
		for (int k = 0; k < In.size; k++)
		{
			x += In(k) * std::cosf(k * (i + 0.5f) * F);
		}
		Out(i) = x;
	}
}

template<typename DCTFunc>
void DCT_2d_col_row(const mat_fr In, mat_fr Out, DCTFunc dct_1d)
{
	// Vertical pass
	for (int c = 0; c < In.cols; ++c)
	{
		auto columnIn = In.column(c);
		auto columnOut = Out.column(c);
		dct_1d(columnIn, columnOut);
	}

	// Horizontal pass
	float* tempMem = (float*)alloca(Out.cols * sizeof(float));
	Row<float, typename mat_fr::storage_type> temp{ Out.cols, 1, tempMem };

	for (int r = 0; r < In.rows; ++r)
	{
		auto rowIn = Out.row(r);
		dct_1d(rowIn, temp);

		auto rowOut = Out.row(r);
		std::memcpy(rowOut.elements, temp.elements, temp.size * sizeof(float));
	}
}

template<typename DCTFunc>
void DCT_2d_col_row(const mat_fc In, mat_fc Out, DCTFunc dct_1d)
{
	// Horizontal pass
	for (int r = 0; r < In.rows; ++r)
	{
		auto rowIn = In.row(r);
		auto rowOut = Out.row(r);
		dct_1d(rowIn, rowOut);
	}

	// Vertical pass
	float* tempMem = (float*)alloca(Out.rows * sizeof(float));
	Column<float, typename mat_fc::storage_type> temp{ Out.rows, 1, tempMem };

	for (int c = 0; c < In.cols; ++c)
	{
		auto columnIn = Out.column(c);
		dct_1d(columnIn, temp);

		auto columnOut = Out.column(c);
		std::memcpy(columnOut.elements, temp.elements, temp.size * sizeof(float));
	}
}


template<typename MatrixT>
void DCT_2d_simple(const MatrixT In, MatrixT Out)
{
	DCT_2d_col_row(In, Out, DCT_simple_1d);
}

template<typename MatrixT>
void IDCT_2d_simple(const MatrixT In, MatrixT Out)
{
	DCT_2d_col_row(In, Out, IDCT_simple_1d);
}

template void DCT_2d_simple(const mat_fr In, mat_fr Out);
template void DCT_2d_simple(const mat_fc In, mat_fc Out);
template void IDCT_2d_simple(const mat_fr In, mat_fr Out);
template void IDCT_2d_simple(const mat_fc In, mat_fc Out);
}
}
