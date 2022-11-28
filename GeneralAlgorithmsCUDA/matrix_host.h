#pragma once

#include <GeneralAlgorithmsCUDA/cpp_stuff.h>
#include <GeneralAlgorithmsCUDA/matrix.h>
#include <vector>

namespace CudaPlayground
{
	template<typename MatrixT, template<typename...> class Allocator = std::allocator>
	struct MatrixDynamic : public MatrixT
	{
		MOVE_ONLY_CLASS(MatrixDynamic);

		MatrixDynamic(int rows, int cols)
			: MatrixT{ rows, cols, MatrixT::storage_type::stride(rows, cols), nullptr }
			, data(rows* cols, 0)
		{
			MatrixT::elements = data.data();
		}

		MatrixDynamic(MatrixDynamic&& other)
			: MatrixT((MatrixT)other)
			, data(std::move(other.data))
		{
			MatrixT::elements = data.data();
		}

		MatrixDynamic& operator=(MatrixDynamic&& other)
		{
			MatrixT::rows = other.rows;
			MatrixT::cols = other.cols;
			MatrixT::stride = other.stride;
			data = std::move(other.data);
			MatrixT::elements = data.data();
			return *this;
		}

		operator MatrixT () const
		{
			return MatrixT{ MatrixT::rows, MatrixT::cols, MatrixT::stride, data.get() };
		}

		static MatrixDynamic fromRows(const std::vector<std::vector<typename MatrixT::value_type>>& rowsIn)
		{
			MatrixDynamic out((int)rowsIn.size(), (int)rowsIn[0].size());
			out.forEach([&](int r, int c, auto& val)
			{
				val = rowsIn[r][c];
			});
			return out;
		}

	private:
		std::vector<typename MatrixT::value_type, Allocator<typename MatrixT::value_type>> data;
	};

	template<typename MatrixT, template<typename...> class Allocator = std::allocator>
	struct MatrixExtMem : public MatrixT
	{
		MOVE_ONLY_CLASS(MatrixExtMem);

		MatrixExtMem(int rows, int cols, int stride, std::unique_ptr<uint8_t, void(*)(void*)> mem)
			: MatrixT{ rows, cols, stride, (typename MatrixT::value_type*)mem.get() }
			, data(std::move(mem))
		{
		}

		MatrixExtMem(MatrixExtMem&& other)
			: MatrixT((MatrixT)other)
			, data(std::move(other.data))
		{
		}

		MatrixExtMem& operator=(MatrixExtMem&& other)
		{
			MatrixT::rows = other.rows; other.rows = 0;
			MatrixT::cols = other.cols; other.cols = 0;
			MatrixT::stride = other.stride; other.stride = 0;
			data = std::move(other.data);
			MatrixT::elements = (typename MatrixT::value_type*)data.get();
			return *this;
		}

		operator MatrixT () const
		{
			return MatrixT{ MatrixT::rows, MatrixT::cols, MatrixT::stride, (typename MatrixT::value_type*)data.get() };
		}

	private:
		std::unique_ptr<uint8_t, void(*)(void*)> data;
	};
}
