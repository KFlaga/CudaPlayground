#pragma once

#include "nanobench.h"
#include <functional>
#include <cstdlib>


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

void registerBench(std::string name, std::function<void()>);


#define ADD_BENCHMARK(name) \
	struct _benchmark_##name { \
		_benchmark_##name() { \
			registerBench(#name, [&]() { _run(); }); \
			bench.title(#name); \
			bench.warmup(1); \
		} \
		void _run();\
		ankerl::nanobench::Bench bench; \
	}; \
	_benchmark_##name _instance_benchmark_##name{}; \
	void _benchmark_##name::_run()


template<typename... Ts>
std::string format(std::string fmt, Ts&&... args)
{
	char x[1024];
	sprintf_s(x, 1024, fmt.c_str(), args...);
	return std::string(x);
}
