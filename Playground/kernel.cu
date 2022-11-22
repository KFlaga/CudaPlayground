
#include <GeneralAlgorithmsCUDA/cuda_all.h>
#include <GeneralAlgorithmsCUDA/cuda_interop.h>
#include <GeneralAlgorithmsCUDA/matrix_device.h>
#include <GeneralAlgorithmsCUDA/cuda_stream.h>

#include <cstdio>
#include <cmath>
#include <iostream>

#include <chrono>

using namespace CudaPlayground;

template<int iterations>
__global__ void DummyKernel(mat_fr out)
{
    int row = threadIdx.y + blockIdx.y;
    int col = threadIdx.x + blockIdx.x;

    float x = ((row * 1285) ^ (col * 7748201)) * 456.78f;
    for (int i = 0; i < iterations; ++i)
    {
        float s = std::sin(x);
        float c = std::cos(x);
        x *= (s + c);
    }
    out(row, col) = x;
}

template<int streamCount, int extend, int iterations, bool stackOps>
void test_streams()
{
    checkCudaErrors(cudaSetDevice(0));

    constexpr int size = extend * extend * sizeof(float);

    CudaStream streams[streamCount];

    dim3 dimBlock(8, 8);
    dim3 dimGrid(extend / dimBlock.x, extend / dimBlock.y);

    float* mhost[4];
    mat_fr host[4];

    for (int i = 0; i < 4; ++i)
    {
        checkCudaErrors(cudaMallocHost(&mhost[i], size));
        host[i] = { extend, extend, extend, mhost[i] };
    }

    auto p1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 20; ++i)
    {
        if constexpr (stackOps)
        {
            float* mdev[4];

            for (int s = 0; s < 4; ++s)
            {
                checkCudaErrors(cudaMallocAsync(&mdev[s], size, streams[s % streamCount]));
            }
            for (int s = 0; s < 4; ++s)
            {
                checkCudaErrors(cudaMemcpyAsync(mdev[s], mhost[s], size, cudaMemcpyKind::cudaMemcpyHostToDevice, streams[s % streamCount]));
            }
            for (int s = 0; s < 4; ++s)
            {
                DummyKernel<iterations> KERNEL_ARGS(dimGrid, dimBlock, 0, streams[s % streamCount]) ({ extend, extend, extend, mdev[s] });
            }
            for (int s = 0; s < 4; ++s)
            {
                checkCudaErrors(cudaMemcpyAsync(mhost[s], mdev[s], extend * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost, streams[s % streamCount]));
            }
            for (int s = 0; s < 4; ++s)
            {
                checkCudaErrors(cudaFreeAsync(mdev[s], streams[s % streamCount]));
            }
        }
        else
        {
            for (int s = 0; s < 4; ++s)
            {
                auto dev = toDeviceMemoryAsync(host[s], true, streams[s % streamCount]);
                DummyKernel<iterations> KERNEL_ARGS(dimGrid, dimBlock, 0, streams[s % streamCount]) (dev.mat);
                copyFromDeviceMemoryAsync(dev.mat, host[s], streams[s % streamCount]);
            }
        }
    }

    for (auto& s : streams)
    {
        s.sync();
    }

    auto p2 = std::chrono::high_resolution_clock::now();

    std::cout << "S " << streamCount << " E " << extend << " I " << iterations << "\n";
    std::cout << "TOOK " << (p2 - p1).count()/1000 << " us\n\n";
}

int main()
{
    test_streams<1, 16, 30000, true>();
    test_streams<4, 16, 30000, true>();
    test_streams<1, 16, 30000, false>();
    test_streams<4, 16, 30000, false>();

    return 0;
}