#pragma once

#include "gtest/gtest.h"

template<typename M>
void fill(M& m, float v)
{
	for (int r = 0; r < m.rows; ++r)
	{
		for (int c = 0; c < m.cols; ++c)
		{
			m(r, c) = v;
		}
	}
}

template<typename M>
void assertEqual(M& m1, M& m2)
{
	for (int r = 0; r < m1.rows; ++r)
	{
		for (int c = 0; c < m2.cols; ++c)
		{
			ASSERT_FLOAT_EQ(m1(r, c), m2(r, c)) << " at r = " << r << " c = " << c;
		}
	}
}
