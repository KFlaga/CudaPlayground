#pragma once

#include <GeneralAlgorithmsCUDA/cuda_interop.h>
#include <cstdint>
#include <utility>

namespace CudaPlayground
{
    namespace MatrixStorages
    {
        CONCEPT_DECLARE(
            template<typename T>
            concept MatrixStorage = requires(T* t, int row, int col, int stride)
            {
                { T::get(t, row, col, stride) } -> std::convertible_to<T*>;
                { T::stride(row, col) } -> std::convertible_to<int>;
            };
        );

        struct RowMajor
        {
            template<typename T>
            CUDA_COMMON_API static T* get(T* ptr, int row, int col, int stride)
            {
                return &ptr[row * stride + col];
            }

            CUDA_COMMON_API static int stride(int /*rows*/, int cols)
            {
                return cols;
            }

            template<typename F, typename M>
            CUDA_COMMON_API static void forEach(F&& f, M& m, int r1, int r2, int c1, int c2)
            {
                for (int c = c1; c < c2; ++c)
                {
                    for (int r = r1; r < r2; ++r)
                    {
                        f(r, c, m(r, c));
                    }
                }
            }

            CUDA_COMMON_API static int size(int rows, int /*cols*/, int stride)
            {
                return rows * stride;
            }
        };

        struct ColumnMajor
        {
            template<typename T>
            CUDA_COMMON_API static T* get(T* ptr, int row, int col, int stride)
            {
                return &ptr[col * stride + row];
            }

            CUDA_COMMON_API static int stride(int rows, int /*cols*/)
            {
                return rows;
            }

            template<typename F, typename M>
            CUDA_COMMON_API static void forEach(F&& f, M& m, int r1, int r2, int c1, int c2)
            {
                for (int r = r1; r < r2; ++r)
                {
                    for (int c = c1; c < c2; ++c)
                    {
                        f(r, c, m(r, c));
                    }
                }
            }

            CUDA_COMMON_API static int size(int /*rows*/, int cols, int stride)
            {
                return cols * stride;
            }
        };
    }

    template<typename T, CONCEPT(MatrixStorages::MatrixStorage) StorageType = MatrixStorages::RowMajor>
    struct Column
    {
        using value_type = T;
        using storage_type = StorageType;

        int size;
        int stride;
        T* elements;

        CUDA_COMMON_API T& operator()(int i)
        {
            OMS_ASSERT(i < size, "Vector::operator()");
            return *StorageType::get(elements, i, 0, stride);
        }

        CUDA_COMMON_API T operator()(int i) const
        {
            OMS_ASSERT(i < size, "Vector::operator()");
            return *StorageType::get(elements, i, 0, stride);
        }
    };

    template<typename T, CONCEPT(MatrixStorages::MatrixStorage) StorageType = MatrixStorages::RowMajor>
    struct Row
    {
        using value_type = T;
        using storage_type = StorageType;

        int size;
        int stride;
        T* elements;

        CUDA_COMMON_API T& operator()(int i)
        {
            OMS_ASSERT(i < size, "Vector::operator()");
            return *StorageType::get(elements, 0, i, stride);
        }

        CUDA_COMMON_API T operator()(int i) const
        {
            OMS_ASSERT(i < size, "Vector::operator()");
            return *StorageType::get(elements, 0, i, stride);
        }
    };

    template<typename T, CONCEPT(MatrixStorages::MatrixStorage) StorageType = MatrixStorages::RowMajor>
    struct Matrix
    {
        using value_type = T;
        using storage_type = StorageType;

        int rows;
        int cols;
        int stride;

        T* elements;

        CUDA_COMMON_API T& operator()(int row, int col)
        {
            OMS_ASSERT(row < rows && col < cols, "Matrix::operator()");
            return *StorageType::get(elements, row, col, stride);
        }

        CUDA_COMMON_API T operator()(int row, int col) const
        {
            OMS_ASSERT(row < rows && col < cols, "Matrix::operator()");
            return *StorageType::get(elements, row, col, stride);
        }

        CUDA_COMMON_API int memSize() const
        {
            return StorageType::size(rows, cols, stride);
        }

        CUDA_COMMON_API Matrix sub(int row, int col, int size) const
        {
            OMS_ASSERT(row < rows && col < cols && row+size <= rows && col+size <= cols, "Matrix::sub()");
            return Matrix{ size, size, stride, StorageType::get(elements, row, col, stride) };
        }

        CUDA_COMMON_API Matrix sub(int row, int col, int rowSize, int colSize) const
        {
            OMS_ASSERT(row < rows && col < cols && row+rowSize <= rows && col+colSize <= cols, "Matrix::sub()");
            return Matrix{ rowSize, colSize, stride, StorageType::get(elements, row, col, stride) };
        }

        CUDA_COMMON_API Row<T, StorageType> row(int r) const
        {
            OMS_ASSERT(r < rows, "Matrix::row()");
            return Row<T, StorageType>{ cols, stride, StorageType::get(elements, r, 0, stride) };
        }

        CUDA_COMMON_API Column<T, StorageType> column(int c) const
        {
            OMS_ASSERT(c < cols, "Matrix::column()");
            return Column<T, StorageType>{ rows, stride, StorageType::get(elements, 0, c, stride) };
        }

        CUDA_COMMON_API Matrix sameSize() const
        {
            return Matrix{ rows, cols, stride, nullptr };
        }

        template<typename F>
        CUDA_COMMON_API void forEach(F&& f)
        {
            StorageType::forEach(std::forward<F>(f), *this, 0, rows, 0, cols);
        }

        template<typename F>
        CUDA_COMMON_API void forEach(F&& f) const
        {
            StorageType::forEach(std::forward<F>(f), *this, 0, rows, 0, cols);
        }
        
        template<typename F>
        CUDA_COMMON_API void forEachOutsideBoundary(int radiusRows, int radiusCols, F&& f)
        {
            callOnSub(0, 0, radiusRows, cols, f); // Top
            callOnSub(rows - radiusRows, 0, radiusRows, cols, f); // Bottom
            callOnSub(radiusRows, 0, rows - 2 * radiusRows, radiusCols, f); // Left
            callOnSub(radiusRows, cols - radiusCols, rows - 2 * radiusRows, radiusCols, f); // Right
        }

        template<typename F>
        CUDA_COMMON_API void forEachInsideBoundary(int radiusRows, int radiusCols, F&& f)
        {
            callOnSub(radiusRows, radiusCols, rows - radiusRows * 2, cols - radiusCols * 2, f);
        }

    private:
        template<typename F>
        CUDA_COMMON_API void callOnSub(int offsetRows, int offsetCols, int rowSize, int colSize, F& f)
        {
            sub(offsetRows, offsetCols, rowSize, colSize)
                .forEach([&](int r, int c, auto& Cval) { f(r + offsetRows, c + offsetCols, Cval); });
        }
    };

    using mat_fr = Matrix<float, MatrixStorages::RowMajor>;
    using mat_fc = Matrix<float, MatrixStorages::ColumnMajor>;
}
