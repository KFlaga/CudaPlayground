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
void fillRand(M& m, float min = 0.f, float max = 1.f)
{
	for (int r = 0; r < m.rows; ++r)
	{
		for (int c = 0; c < m.cols; ++c)
		{
			m(r, c) = min + (max - min) * (rand() % 10000) / 10000.f;
		}
	}
}

template<typename M>
void assertEqual(M& m1, M& m2, float error = 1e-9f)
{
	ASSERT_EQ(m1.rows, m2.rows);
	ASSERT_EQ(m1.cols, m2.cols);
	for (int r = 0; r < m1.rows; ++r)
	{
		for (int c = 0; c < m2.cols; ++c)
		{
			ASSERT_NEAR(m1(r, c), m2(r, c), error) << " at r = " << r << " c = " << c;
		}
	}
}
