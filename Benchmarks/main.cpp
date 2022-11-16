#include "nanobench.h"

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/binarize.h>
#include <GeneralAlgorithmsCUDA/mat_multiply.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>


using namespace OmniSense;

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


int main()
{
	ankerl::nanobench::Bench b;

	b.title("Binarize")
		//.relative(true)
	;

	{
		auto A = MatrixDynamic<mat_fr>(100, 100);
		auto C_cuda = MatrixDynamic<mat_fr>(100, 100);
		auto C_cpu = MatrixDynamic<mat_fr>(100, 100);
		fillRand(A, 0, 1);

		b.run("Binarize CPU 100x100", [&]()
		{
			General::Binarize<mat_fr>(A, C_cpu, 0.5f, 0.0f, 1.0f);
		});

		b.run("Binarize CUDA 100x100", [&]()
		{
			CUDA::General::Binarize(A, C_cuda, 0.5f, 0.0f, 1.0f);
		});
	}

	{
		auto A = MatrixDynamic<mat_fr>(1000, 1000);
		auto C_cuda = MatrixDynamic<mat_fr>(1000, 1000);
		auto C_cpu = MatrixDynamic<mat_fr>(1000, 1000);
		fillRand(A, 0, 1);

		b.run("Binarize CPU 1000x1000", [&]()
		{
			General::Binarize<mat_fr>(A, C_cpu, 0.5f, 0.0f, 1.0f);
		});

		b.run("Binarize CUDA 1000x1000", [&]()
		{
			CUDA::General::Binarize(A, C_cuda, 0.5f, 0.0f, 1.0f);
		});
	}

	{
		auto A = MatrixDynamic<mat_fr>(480, 160);
		auto B = MatrixDynamic<mat_fr>(160, 320);
		auto C_cuda = MatrixDynamic<mat_fr>(480, 320);
		auto C_cpu = MatrixDynamic<mat_fr>(480, 320);

		fillRand(A, 0, 1);
		fillRand(B, 0, 1);

		b.run("MatMul CPU", [&]()
		{
			General::MatMul<mat_fr>(A, B, C_cpu);
		});

		b.run("MatMul CUDA", [&]()
		{
			CUDA::General::MatMul(A, B, C_cuda);
		});
	}
}
