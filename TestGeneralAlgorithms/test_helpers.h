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

template<typename M1, typename M2>
void assertEqual(M1& m1, M2& m2, float absErr = 1e-6f, float relErr = 1e-6f)
{
	ASSERT_EQ(m1.rows, m2.rows);
	ASSERT_EQ(m1.cols, m2.cols);
	for (int r = 0; r < m1.rows; ++r)
	{
		for (int c = 0; c < m2.cols; ++c)
		{
			float rel = (std::abs(m1(r, c)) + std::abs(m2(r, c))) * relErr / 2.0f;
			float maxError = std::max(rel, absErr);

			ASSERT_NEAR(m1(r, c), m2(r, c), maxError) << " at r = " << r << " c = " << c;
		}
	}
}
