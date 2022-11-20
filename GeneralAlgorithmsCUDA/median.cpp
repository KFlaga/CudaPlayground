#include "median.h"
#include <malloc.h>

namespace OmniSense
{
namespace General
{
template<typename F>
void forEachX(int r, int c, int radius, int rows, int cols, F&& f)
{
    for (int dr = -radius; dr <= radius; ++dr)
    {
        for (int dc = -radius; dc <= radius; ++dc)
        {
            int rr = r + dr;
            int cc = c + dc;
            if (rr >= 0 && rr < rows && cc >= 0 && cc < cols)
            {
                f(rr, cc);
            }
        }
    }
}

template<typename F>
void forEachC(int size, F&& f)
{
    for (int dr = 0; dr <= size; ++dr)
    {
        for (int dc = 0; dc <= size; ++dc)
        {
            if (f(dr, dc))
            {
                return;
            }
        }
    }
}

struct alignas(4) X 
{ 
    unsigned char lower = 0;
    unsigned char equal = 0;
    unsigned char count = 0;
};

template<typename MatrixT>
void Median(const MatrixT In, MatrixT Out, int radius)
{
    if (radius > 7)
    {
        throw std::runtime_error("Median: exceeded max radius of 7");
    }

    int size = 2 * radius + 1;
    int mSize = size * size;
    X* m = (X*)alloca(mSize); // should be at least 16byte aligned
    Matrix<X, MatrixStorages::ColumnMajor> Xs{ size, size, size, m };

    // Optimized for mat_fr only
    for (int r = 0; r < In.rows; ++r)
    {
        // Explicitly compute first block
        forEachX(r, 0, radius, In.rows, In.cols, [&](int rr, int cc) {
            X x = { 0, 0, 0 };
            if (cc < radius)
            {
                forEachX(r, 0, radius, In.rows, In.cols, [&](int rr2, int cc2) {
                    x.lower += (int)(In(rr2, cc2) < In(rr, cc));
                    x.equal += (int)(In(rr2, cc2) == In(rr, cc));
                    x.count++;
                });
            }
            Xs(rr, cc) = x;
        });

        int firstColumn = 0;
        int lastColumn = size - 1;

        for (int c = 0; c < In.cols; ++c)
        {
            // Compute only single column of block, at In(r, c + radius) | Xs(firstColumn-1)
            // 

            // Add new column values to  block
            if (c + radius < In.cols)
            {
                for (int xrow = 0; xrow < size; ++xrow)
                {
                    X x{};

                    int in_row = r + radius;
                    int in_col = c + radius;
                    float xval = In(in_row, in_col);

                    auto compute_column = [&](int bl_col)
                    {
                        int in_col2 = c + bl_col - radius;
                        if (in_col2 < 0 || in_col2 >= In.cols)
                        {
                            break;
                        }

                        for (int bl_row = 0; bl_row < size; ++bl_row)
                        {
                            int in_row2 = r + bl_row - radius;
                            if (in_row2 < 0 || in_row2 >= In.rows)
                            {
                                break;
                            }

                            float in2 = In(in_row2, in_col2);

                            // We can compute value for new column at the same time
                            x.lower += (int)(in2 > xval);
                            x.equal += (int)(in2 == xval);
                            x.count++;

                            if (bl_col != lastColumn)
                            {
                                X(bl_row, bl_col).lower += (int)(in2 < xval);
                                X(bl_row, bl_col).equal += (int)(in2 == xval);
                                X(bl_row, bl_col).count++;
                            }
                        }
                    };

                    for (int bl_col = firstColumn; bl_col < size; ++bl_col)
                    {
                        compute_column(bl_col);
                    }
                    for (int bl_col = 0; bl_col < firstColumn; ++bl_col)
                    {
                        compute_column(bl_col);
                    }

                    Xs(xrow, lastColumn) = x;
                }
            }

            // Find median value
            forEachC(size, [&](int rr, int cc)
            {
                X x = Xs(rr, cc);
                int medianPos = (x.count + 1) / 2;
                if (x.lower == medianPos ||
                    (x.lower < medianPos && (x.lower + x.equal) >= medianPos))
                {
                    int in_row = r + rr - radius;
                    int in_col = c + cc - radius;

                    Out(r, c) = In(in_row, in_col);

                    return false;
                }
            });

            // Subtract first column values from rest of block
            
            int in_col2 = c - radius;
            if (in_col2 < 0)
            {
                break;
            }

            int bl_col = firstColumn;
            for (int bl_row = 0; bl_row < size; ++bl_row)
            {
                int in_row2 = r + bl_row - radius;
                if (in_row2 < 0 || in_row2 >= In.rows)
                {
                    break;
                }

                float in2 = In(in_row2, in_col2);
                X(bl_row, bl_col).lower -= (int)(in2 < xval);
                X(bl_row, bl_col).equal -= (int)(in2 == xval);
                X(bl_row, bl_col).count--;
            }

            firstColumn = (firstColumn + 1) % size;
            lastColumn = (lastColumn + 1) % size;
        }
    }

    // Simple implmenetation
    //Out.forEach([&](int r, int c, auto& Cval) {
    //    forEachX(r, c, radius, In.rows, In.cols, [&](int rr, int cc) {
    //        int count = 0;

    //        forEachX(r, c, radius, In.rows, In.cols, [&](int rr2, int cc2) {
    //            x.lower += (int)(In(rr2, cc2) < In(rr, cc));
    //            x.equal += (int)(In(rr2, cc2) == In(rr, cc));
    //            count++;
    //            return true;
    //        });

    //        int medianPos = (count + 1) / 2;
    //        if (x.lower == medianPos ||
    //            (x.lower < medianPos && (x.lower + x.equal) >= medianPos))
    //        {
    //            Cval = In(rr, cc);
    //            return false;
    //        }
    //        return true;
    //    });
    //});
}

template void Median(const mat_fr, mat_fr, int);
template void Median(const mat_fc, mat_fc, int);
}
}
