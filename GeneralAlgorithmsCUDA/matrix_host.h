#pragma once

#include "cpp_stuff.h"
#include "matrix.h"
#include <vector>

namespace CudaPlayground
{
	template<typename MatrixT>
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
		std::vector<typename MatrixT::value_type> data;
	};
}
