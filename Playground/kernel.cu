
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

template<int iterations>
__global__ void DummyKernel2(mat_fr out)
{
    int row = threadIdx.y + blockIdx.y;
    int col = threadIdx.x + blockIdx.x;

    float x = ((row * 1123787) + (col * 23)) * 1.78f;
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

void CUDART_CB callback(void*)
{
    // Dummy
}

template<int extend, int iterations>
void assignCudaWork(float* mhost, int size, cudaStream_t stream, dim3 dimGrid, dim3 dimBlock)
{
    float* mdev;

    cudaMallocAsync(&mdev, size, stream);
    cudaMemcpyAsync(mdev, mhost, size, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

    DummyKernel<iterations> KERNEL_ARGS(dimGrid, dimBlock, 0, stream) ({ extend, extend, extend, mdev });
    DummyKernel2<iterations> KERNEL_ARGS(dimGrid, dimBlock, 0, stream) ({ extend, extend, extend, mdev });
    DummyKernel<iterations> KERNEL_ARGS(dimGrid, dimBlock, 0, stream) ({ extend, extend, extend, mdev });

    cudaMemcpyAsync(mhost, mdev, extend * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);

    cudaLaunchHostFunc(stream, &callback, nullptr);

    cudaMemcpyAsync(mdev, mhost, size, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

    DummyKernel<iterations> KERNEL_ARGS(dimGrid, dimBlock, 0, stream) ({ extend, extend, extend, mdev });
    DummyKernel2<iterations> KERNEL_ARGS(dimGrid, dimBlock, 0, stream) ({ extend, extend, extend, mdev });
    DummyKernel<iterations> KERNEL_ARGS(dimGrid, dimBlock, 0, stream) ({ extend, extend, extend, mdev });

    cudaMemcpyAsync(mhost, mdev, extend * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(mdev, stream);
}

template<int extend, int iterations>
void test_graphs()
{
    constexpr int size = extend * extend * sizeof(float);

    dim3 dimBlock(8, 8);
    dim3 dimGrid(extend / dimBlock.x, extend / dimBlock.y);

    checkCudaErrors(cudaSetDevice(0));

    float* mhost;
    checkCudaErrors(cudaMallocHost(&mhost, size + 160*4));

    CudaStream stream;

    cudaGraphExec_t graphExec = NULL;


    auto p2 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i)
    {
        mhost = mhost + 16;

        cudaGraph_t graph;
        checkCudaErrors(cudaGraphCreate(&graph, 0));
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        assignCudaWork<extend, iterations>(mhost, size, stream, dimGrid, dimBlock);
        cudaStreamEndCapture(stream, &graph);

        cudaGraphExecUpdateResult updateResult;
        cudaGraphNode_t errorNode;
        if (graphExec != NULL)
        {
            cudaGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
        }

        if (graphExec == NULL || updateResult != cudaGraphExecUpdateSuccess)
        {
            if (graphExec != NULL)
            {
                cudaGraphExecDestroy(graphExec);
            }
            cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
        }

        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
        cudaGraphDestroy(graph);

        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
    }

    auto p3 = std::chrono::high_resolution_clock::now();

    std::cout << " E " << extend << " I " << iterations << "\n";
    std::cout << "TOOK " << (p3 - p2).count() / 1000 << " us\n\n";
}

template<int extend, int iterations>
void test_graphs_ref()
{
    constexpr int size = extend * extend * sizeof(float);

    dim3 dimBlock(8, 8);
    dim3 dimGrid(extend / dimBlock.x, extend / dimBlock.y);

    checkCudaErrors(cudaSetDevice(0));

    float* mhost;
    checkCudaErrors(cudaMallocHost(&mhost, size));

    CudaStream stream;

    auto p2 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i)
    {
        mhost = mhost + 16;

        assignCudaWork<extend, iterations>(mhost, size, stream, dimGrid, dimBlock);
        cudaStreamSynchronize(stream);
    }

    auto p3 = std::chrono::high_resolution_clock::now();

    std::cout << "REF TOOK " << (p3 - p2).count() / 1000 << " us\n\n";
}

int main()
{
    //test_streams<1, 16, 30000, true>();
    //test_streams<4, 16, 30000, true>();
    //test_streams<1, 16, 30000, false>();
    //test_streams<4, 16, 30000, false>();

    test_graphs<16, 30000>();
    test_graphs_ref<16, 30000>();

    return 0;
}