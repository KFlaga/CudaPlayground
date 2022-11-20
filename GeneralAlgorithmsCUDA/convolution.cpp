#include "convolution.h"

namespace OmniSense
{
    namespace General
    {
        // Return convolution A * B; C must have size of A
        template<typename MatrixT>
        void ConvolveShrink(const MatrixT A, const MatrixT B, MatrixT C, ConvolveBoundary boundary)
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

        template<typename MatrixT>
        void ConvolveExtendedIgnoreOutside(const MatrixT A, const MatrixT B, MatrixT C)
        {
            int radiusRows = B.rows / 2;
            int radiusCols = B.cols / 2;

            C.forEach([&](int r, int c, auto& Cval) {
                typename MatrixT::value_type val = 0;
                B.forEach([&](int dr, int dc, auto Bval) {
                    int rr = r + dr - radiusRows;
                    int cc = c + dc - radiusCols;
                    if (rr >= 0 && rr < A.rows && cc >= 0 && cc < A.cols) {
                        val += A(rr, cc) * Bval;
                    }
                });
                Cval = val;
            });
        }

        template<typename MatrixT>
        void Convolve(const MatrixT A, const MatrixT B, MatrixT C, ConvolveBoundary boundary)
        {
            if (boundary == ConvolveBoundary::ExtendZero)
            {
                ConvolveExtendedIgnoreOutside(A, B, C);
            }
            else
            {
                ConvolveShrink(A, B, C, boundary);
            }
        }

        template void Convolve(const mat_fr, const mat_fr, mat_fr, ConvolveBoundary);
        template void Convolve(const mat_fc, const mat_fc, mat_fc, ConvolveBoundary);
    }
}
