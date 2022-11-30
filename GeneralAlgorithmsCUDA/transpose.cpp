#include "transpose.h"

namespace CudaPlayground
{
namespace General
{
template<typename MatrixT>
void Transpose(const MatrixT In, MatrixT Out)
{
    Out.forEach([&](int r, int c, auto& Cval) {
        Cval = In(c, r);
    });
}

template<typename MatrixT>
void Transpose(MatrixT InOut)
{
    for (int r = 0; r < InOut.rows; ++r)
    {
        for (int c = r; c < InOut.cols; ++c)
        {
            std::swap(InOut(r, c), InOut(c, r));
        }
    }
}

template void Transpose(const mat_fr In, mat_fr Out);
template void Transpose(const mat_fc In, mat_fc Out);
template void Transpose(mat_fr InOut);
template void Transpose(mat_fc InOut);
}
}
