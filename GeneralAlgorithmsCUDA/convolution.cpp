#include "convolution.h"

namespace OmniSense
{
    namespace General
    {
        // Return convolution A * B
        // C must have size of A
        // Elements on boundaries ( of length rows/cols of B/2 )
        // are not computed and set to 0 or to A(r,c)
        // Size of B must be odd
        template<typename MatrixT>
        void Convolve(const MatrixT A, const MatrixT B, MatrixT C, ConvolveBoundary boundary)
        {
            int radiusRows = B.rows / 2;
            int radiusCols = B.cols / 2;

            C.forEachInsideBoundary(radiusRows, radiusCols, [&](int r, int c, auto& Cval) {
                typename MatrixT::value_type val = 0;
                B.forEach([&](int dr, int dc, auto Bval)
                {
                    val += A(r + dr - radiusRows, c + dc - radiusCols) * Bval;
                });
                Cval = val;
            });

            if (boundary == ConvolveBoundary::Zero)
            {
                C.forEachOutsideBoundary(radiusRows, radiusCols, [](int, int, auto& Cval) { Cval = 0; });
            }
            else if (boundary == ConvolveBoundary::Copy)
            {
                C.forEachOutsideBoundary(radiusRows, radiusCols, [&](int r, int c, auto& Cval) { Cval = A(r, c); });
            }
        }

        template void Convolve(const mat_fr, const mat_fr, mat_fr, ConvolveBoundary);
        template void Convolve(const mat_fc, const mat_fc, mat_fc, ConvolveBoundary);
    }
}
