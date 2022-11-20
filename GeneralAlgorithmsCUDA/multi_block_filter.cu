//#include "cuda_all.h"
//
//#include "multi_block_filter.h"
//#include "matrix_device.h"
//#include "convolution.h"
//#include <memory>
//#include <algorithm>
//
//using namespace CudaPlayground;
//
//
//static CUDA_HOST_API int findBlockSize(const mat_fr& elements)
//{
//    if (elements.rows * elements.cols >= 600) {
//        return 12;
//    }
//    if (elements.rows * elements.cols >= 120) {
//        return 8;
//    }
//    return 4;
//}
//
//template<typename Container>
//auto findBlockExtentionSize(const Container& ms)
//{
//    int biggestBlockRows = 0;
//    int biggestBlockCols = 0;
//    for (auto& m : ms)
//    {
//        biggestBlockRows = std::max(m.rows, biggestBlockRows);
//        biggestBlockCols = std::max(m.cols, biggestBlockCols);
//    }
//    return std::make_pair(biggestBlockRows, biggestBlockRows);
//}
//
//
//namespace CudaPlayground
//{
//namespace CUDA
//{
//namespace General
//{
//    void MultiBlockFilter(const mat_fr source, mat_fr dest, const std::vector<mat_fr>& filters, ConvolveBoundary boundary)
//    {
//        if (filters.size() == 0)
//        {
//            // TODO: assert on equal layout
//            std::memcpy(dest.elements, source.elements, source.memSize());
//        }
//
//        int blockSize = findBlockSize(dest);
//        auto [biggestFilterRows, biggestFilterCols] = findBlockExtentionSize(filters);
//
//        // Allocate memory once for all operations
//        auto d_source = [&]() {
//            if (boundary == ConvolveBoundary::ExtendZero)
//            {
//                return toDeviceMemoryExtendedBlock(source, biggestFilterRows, biggestFilterCols, true);
//            }
//            else
//            {
//                return toDeviceMemory(source, true);
//            }
//        }();
//        auto d_filter = toDeviceMemory(mat_fr{ biggestFilterRows, biggestFilterCols }, false);
//
//
//        // TODO
//    }
//}
//}
//}
