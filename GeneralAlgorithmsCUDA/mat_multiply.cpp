#include "mat_multiply.h"

namespace CudaPlayground
{
namespace General
{
template<typename MatrixT>
void MatMul(const MatrixT A, const MatrixT B, MatrixT C)
{
    C.forEach([&](int r, int c, auto& Cval) {
        float x = 0;
        for (int i = 0; i < A.cols; ++i)
        {
            x += A(r, i) * B(i, c);
        }
        Cval = x;
    });
}

template void MatMul(const mat_fr, const mat_fr, mat_fr);
template void MatMul(const mat_fc, const mat_fc, mat_fc);
}
}
