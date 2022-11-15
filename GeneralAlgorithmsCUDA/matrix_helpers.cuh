#include "matrix.h"
#include "cuda_interop.h"
#include "cpp_stuff.h"
#include <utility>

namespace OmniSense
{
    namespace detail
    {
        template<typename MatrixT, typename Allocator, typename cudaMemcpyKind_t>
        CUDA_HOST_API MatrixT copyMatrix(const MatrixT A, bool copy, cudaMemcpyKind_t memcpyKind, Allocator&& allocator)
        {
            MatrixT B = A.sameSize();
            size_t size = A.cols * A.rows * sizeof(typename MatrixT::value_type);
            allocator(&B.elements, &B.stride, A.cols, A.rows);
            if (copy)
            {
                cudaMemcpy(B.elements, A.elements, size, memcpyKind);
            }
            return B;
        }
    }

    template<typename MatrixT>
    struct DeviceMatrixGuard
    {
        MatrixT mat;

        MOVE_ONLY_CLASS(DeviceMatrixGuard);

        DeviceMatrixGuard(MatrixT m) : mat{ m } {}

        DeviceMatrixGuard(DeviceMatrixGuard&& other)
            : mat{other.mat}
        {
            other.mat.elements = nullptr;
        }

        DeviceMatrixGuard& operator=(DeviceMatrixGuard&& other)
        {
            mat = other.mat;
            other.mat.elements = nullptr;
            return *this;
        }

        ~DeviceMatrixGuard()
        {
            if (mat.elements != nullptr)
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

    inline CUDA_HOST_API DeviceMatrixGuard<mat_fr> toDeviceMemory(const mat_fr host, bool copy)
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

    inline CUDA_HOST_API DeviceMatrixGuard<mat_fr> toDeviceMemoryWithPadding(const mat_fr host, int blockSizeRows, int blockSizeCols, bool copy)
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

    inline CUDA_HOST_API void copyFromDeviceMemory(const mat_fr dev, mat_fr& host)
    {
        checkCudaErrors(cudaMemcpy2D(
            host.elements, host.stride * sizeof(float),
            dev.elements, dev.stride * sizeof(float),
            host.cols * sizeof(float), host.rows,
            cudaMemcpyKind::cudaMemcpyDeviceToHost
        ));
    }

    inline CUDA_HOST_API int findBlockSize(int elements)
    {
        // TODO: its arbitrary now
        int blocks16 = (elements / 16);
        if (blocks16 >= 400) {
            return 16;
        }
        int blocks8 = (elements / 8);
        if (blocks8 >= 100) {
            return 8;
        }
        return 4;
    }
}
