#include "bilateral_filter.h"

namespace CudaPlayground
{
namespace General
{
template<typename MatrixT, typename F>
void BilateralFilter(const MatrixT In, MatrixT Out, int radius, F&& filter)
{
    Out.forEach([&](int r, int c, float& outVal) {
        int minR = std::max(0, r - radius);
        int minC = std::max(0, c - radius);
        int maxR = std::min(In.rows - 1, r + radius);
        int maxC = std::min(In.cols - 1, c + radius);

        auto neighbourhood = In.sub(minR, minC, (maxR - minR) + 1, (maxC - minC) + 1);

        float val = 0;
        float weight = 0;
        float I_rc = In(r, c);

        neighbourhood.forEach([&](int nr, int nc, float I_n) {
            float dI2 = (I_n - I_rc) * (I_n - I_rc);
            float dx2 = (float)((r - nr - minR) * (r - nr - minR) + (c - nc - minC) * (c - nc - minC));

            float f = filter(dI2, dx2);

            weight += f;
            val += I_n * f;
        });

        outVal = val / weight;
    });
}

template<typename MatrixT>
void BilateralFilter(const MatrixT In, MatrixT Out, int radius, SmoothingKernel filter)
{
    std::visit([&](auto& f)
    {
        BilateralFilter(In, Out, radius, f);
    }
    , filter);
}

template void BilateralFilter(const mat_fr In, mat_fr Out, int radius, SmoothingKernel);
template void BilateralFilter(const mat_fc In, mat_fc Out, int radius, SmoothingKernel);
}
}
