#pragma once

#include "matrix.h"
#include "cuda_interop.h"
#include "cpp_stuff.h"

namespace CudaPlayground
{
    template<typename MatrixT>
    struct DeviceMatrixGuard
    {
        MatrixT mat;
        cudaStream_t stream = nullptr;

        MOVE_ONLY_CLASS(DeviceMatrixGuard);

        DeviceMatrixGuard(MatrixT m, cudaStream_t s = nullptr)
            : mat{ m }
            , stream{ s }
        {}

        DeviceMatrixGuard(DeviceMatrixGuard&& other) noexcept
            : mat{ other.mat }
            , stream{ other.stream }
        {
            other.mat.elements = nullptr;
            other.stream = nullptr;
        }

        DeviceMatrixGuard& operator=(DeviceMatrixGuard&& other) noexcept
        {
            mat = other.mat;
            stream = other.stream;
            other.mat.elements = nullptr;
            other.mat.stream = nullptr;
            return *this;
        }

        ~DeviceMatrixGuard()
        {
            if (stream != nullptr)
            {
                cudaFreeAsync(mat.elements, stream);
            }
            else if (mat.elements != nullptr)
            {
                cudaFree(mat.elements);
            }
        }

        operator MatrixT () const
        {
            return mat;
        }
    };

    inline CUDA_COMMON_API int pad(int x, int alignment)
    {
        return ((x + (alignment - 1)) / alignment) * alignment;
    }

    inline CUDA_COMMON_API int div_ceil(int x, int alignment)
    {
        return ((x + (alignment - 1)) / alignment);
    }

    CUDA_HOST_API DeviceMatrixGuard<mat_fr> toDeviceMemory(const mat_fr host, bool copy);
    CUDA_HOST_API DeviceMatrixGuard<mat_fc> toDeviceMemory(const mat_fc host, bool copy);

    CUDA_HOST_API DeviceMatrixGuard<mat_fr> toDeviceMemoryAsync(const mat_fr host, bool copy, cudaStream_t);

    CUDA_HOST_API DeviceMatrixGuard<mat_fr> toDeviceMemoryPad(const mat_fr host,
                                                              int blockSizeRows,
                                                              int blockSizeCols,
                                                              bool copy);

    CUDA_HOST_API DeviceMatrixGuard<mat_fr> toDeviceMemoryPadAsync(const mat_fr host,
                                                                   int blockSizeRows,
                                                                   int blockSizeCols,
                                                                   bool copy,
                                                                   cudaStream_t);

    CUDA_HOST_API DeviceMatrixGuard<mat_fr> toDeviceMemoryExtendedBlock(const mat_fr host,
                                                                        const mat_fr block,
                                                                        bool copy);

    CUDA_HOST_API DeviceMatrixGuard<mat_fr> toDeviceMemoryExtendedBlock(const mat_fr host,
                                                                        int blockRows,
                                                                        int blockCols,
                                                                        bool copy);

    CUDA_HOST_API void copyFromDeviceMemory(const mat_fr dev, mat_fr host);
    CUDA_HOST_API void copyFromHostMemory(const mat_fr host, mat_fr dev);

    CUDA_HOST_API void copyFromDeviceMemoryAsync(const mat_fr dev, mat_fr host, cudaStream_t);

    CUDA_HOST_API void copyFromDeviceMemoryExtendedBlockPad(const mat_fr dev, const mat_fr block, mat_fr host);

    CUDA_HOST_API void copyFromDeviceMemory(const mat_fc dev, mat_fc host);
}
