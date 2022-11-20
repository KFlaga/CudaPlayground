#include <cuda_runtime.h>
#include "matrix_device.h"
#include <utility>

namespace CudaPlayground
{
    CUDA_HOST_API DeviceMatrixGuard<mat_fr> toDeviceMemory(const mat_fr host, bool copy)
    {
        size_t stride;
        float* data;
        checkCudaErrors(cudaMallocPitch(&data, &stride, host.cols * sizeof(float), host.rows));
        if (copy)
        {
            checkCudaErrors(cudaMemcpy2D(
                data, stride,
                host.elements, host.stride * sizeof(float),
                host.cols * sizeof(float), host.rows,
                cudaMemcpyKind::cudaMemcpyHostToDevice
            ));
        }

        return mat_fr{ host.rows, host.cols, (int)(stride/sizeof(float)), data };
    }

    CUDA_HOST_API DeviceMatrixGuard<mat_fc> toDeviceMemory(const mat_fc host, bool copy)
    {
        size_t stride;
        float* data;
        checkCudaErrors(cudaMallocPitch(&data, &stride, host.rows * sizeof(float), host.cols));
        if (copy)
        {
            checkCudaErrors(cudaMemcpy2D(
                data, stride,
                host.elements, host.stride * sizeof(float),
                host.rows * sizeof(float), host.cols,
                cudaMemcpyKind::cudaMemcpyHostToDevice
            ));
        }

        return mat_fc{ host.rows, host.cols, (int)(stride / sizeof(float)), data };
    }

    CUDA_HOST_API DeviceMatrixGuard<mat_fr> toDeviceMemoryPad(
        const mat_fr host,
        int blockSizeRows,
        int blockSizeCols,
        bool copy)
    {
        int rows = pad(host.rows, blockSizeRows);
        int cols = pad(host.cols, blockSizeCols);

        size_t stride;
        float* data;
        checkCudaErrors(cudaMallocPitch(&data, &stride, cols * sizeof(float), rows));
        checkCudaErrors(cudaMemset2D(data, stride, 0, cols * sizeof(float), rows));

        if (copy)
        {
            checkCudaErrors(cudaMemcpy2D(
                data, stride,
                host.elements, host.stride * sizeof(float),
                host.cols * sizeof(float), host.rows,
                cudaMemcpyKind::cudaMemcpyHostToDevice
            ));
        }
        return mat_fr{ rows, cols, (int)(stride / sizeof(float)), data };
    }

    CUDA_HOST_API DeviceMatrixGuard<mat_fr> toDeviceMemoryExtendedBlock(
        const mat_fr host,
        const mat_fr block,
        int blockSizeRows,
        int blockSizeCols,
        bool copy)
    {
        int extRows = host.rows + (block.rows-1)/2;
        int extCols = host.cols + (block.cols-1)/2;

        size_t stride;
        float* data;
        checkCudaErrors(cudaMallocPitch(&data, &stride, extCols * sizeof(float), extRows));
        checkCudaErrors(cudaMemset2D(data, stride, 0, extCols * sizeof(float), extRows));

        if (copy)
        {
            int dataStart = (block.rows / 2) * (stride / sizeof(float)) + block.cols / 2;
            checkCudaErrors(cudaMemcpy2D(
                data + dataStart, (stride + (block.cols / 2)) * sizeof(float), // extra stride is for extended block
                host.elements, host.stride * sizeof(float),
                host.cols * sizeof(float), host.rows,
                cudaMemcpyKind::cudaMemcpyHostToDevice
            ));
        }
        return mat_fr{ extRows, extCols, (int)(stride / sizeof(float)), data };
    }

    inline CUDA_HOST_API void copyMemory(const mat_fr from, mat_fr& to, cudaMemcpyKind kind)
    {
        checkCudaErrors(cudaMemcpy2D(
            to.elements, to.stride * sizeof(float),
            from.elements, from.stride * sizeof(float),
            to.cols * sizeof(float), to.rows,
            kind
        ));
    }

    CUDA_HOST_API void copyFromDeviceMemory(const mat_fr dev, mat_fr& host)
    {
        copyMemory(dev, host, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    }

    CUDA_HOST_API void copyFromHostMemory(const mat_fr host, mat_fr& dev)
    {
        copyMemory(host, dev, cudaMemcpyKind::cudaMemcpyHostToDevice);
    }

    CUDA_HOST_API void copyFromDeviceMemoryExtendedBlockPad(const mat_fr dev, const mat_fr block, mat_fr& host)
    {
        int dataStart = (block.rows / 2) * dev.stride + block.cols / 2;
        checkCudaErrors(cudaMemcpy2D(
            host.elements, host.stride * sizeof(float),
            dev.elements + dataStart, (dev.stride + (block.cols / 2)) * sizeof(float),
            host.cols * sizeof(float), host.rows,
            cudaMemcpyKind::cudaMemcpyDeviceToHost
        ));
    }

    inline CUDA_HOST_API void copyMemory(const mat_fc from, mat_fc& to, cudaMemcpyKind kind)
    {
        checkCudaErrors(cudaMemcpy2D(
            to.elements, to.stride * sizeof(float),
            from.elements, from.stride * sizeof(float),
            to.rows * sizeof(float), to.cols,
            kind
        ));
    }

    CUDA_HOST_API void copyFromDeviceMemory(const mat_fc dev, mat_fc& host)
    {
        copyMemory(dev, host, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    }
}
