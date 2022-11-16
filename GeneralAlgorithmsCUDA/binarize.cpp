#include "mat_multiply.h"

namespace OmniSense
{
namespace General
{
template<typename MatrixT>
void Binarize(const MatrixT In, MatrixT Out, float threshold, float lowValue, float highValue)
{
    Out.forEach([&](int r, int c, auto& Cval) {
        Cval = In(r, c) > threshold ? highValue : lowValue;
    });
}

template void Binarize(const mat_fr In, mat_fr Out, float threshold, float lowValue, float highValue);
template void Binarize(const mat_fc In, mat_fc Out, float threshold, float lowValue, float highValue);
}
}
