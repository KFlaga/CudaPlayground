#include "dct.h"
#include <malloc.h>
#include <cmath>

namespace CudaPlayground
{
namespace General
{

// DCT II, from wikipedia
template<typename VectorT>
static void DCT_simple_1d(const VectorT In, VectorT Out)
{
	float scaling = std::sqrtf(2.0f / (float)In.size);

	float F = (float)M_PI / In.size;
	for (int i = 0; i < In.size; ++i)
	{
		float x = 0;
		for (int k = 0; k < In.size; k++)
		{
			x += In(k) * std::cosf((k + 0.5f) * i * F);
		}
		Out(i) = x * scaling;
	}
}

// DCT III
template<typename VectorT>
static void IDCT_simple_1d(const VectorT In, VectorT Out)
{
	float scaling = std::sqrtf(2.0f / (float)In.size);

	float F = (float)M_PI / In.size;
	for (int i = 0; i < In.size; ++i)
	{
		float x = In(0) * 0.5f;
		for (int k = 1; k < In.size; k++)
		{
			x += In(k) * std::cosf(k * (i + 0.5f) * F);
		}
		Out(i) = x * scaling;
	}
}

template<typename MatrixT, typename DCTFunc>
static void DCT_2d_col_row(const MatrixT In, MatrixT Out, DCTFunc dct_1d)
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
	decltype(Out.column(0)) temp{ Out.rows, 1, tempMem };

	for (int c = 0; c < In.cols; ++c)
	{
		auto columnIn = Out.column(c);
		dct_1d(columnIn, temp);

		auto columnOut = Out.column(c);
		for (int k = 0; k < temp.size; ++k)
		{
			columnOut(k) = temp.elements[k];
		}
	}
}

template<typename MatrixT>
void DCT_2d_simple(const MatrixT In, MatrixT Out)
{
	DCT_2d_col_row(In, Out, [](auto vecIn, auto vecOut) { DCT_simple_1d(vecIn, vecOut); });
}

template<typename MatrixT>
void IDCT_2d_simple(const MatrixT In, MatrixT Out)
{
	DCT_2d_col_row(In, Out, [](auto vecIn, auto vecOut) { IDCT_simple_1d(vecIn, vecOut); });
}

template void DCT_2d_simple(const mat_fr In, mat_fr Out);
template void DCT_2d_simple(const mat_fc In, mat_fc Out);
template void IDCT_2d_simple(const mat_fr In, mat_fr Out);
template void IDCT_2d_simple(const mat_fc In, mat_fc Out);
}
}
