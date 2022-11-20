#include "cuda_stream.h"
#include "cuda_all.h"

namespace CudaPlayground
{
	CUDA_HOST_API CudaStream::CudaStream()
	{
		cudaStreamCreate(&stream);
	}

	CUDA_HOST_API CudaStream::~CudaStream()
	{
		if (stream != nullptr)
		{
			cudaStreamDestroy(stream);
		}
	}

	CUDA_HOST_API CudaStream::CudaStream(CudaStream&& other) noexcept
	{
		std::swap(stream, other.stream);
	}

	CUDA_HOST_API CudaStream& CudaStream::operator=(CudaStream&& other) noexcept
	{
		if (stream != nullptr)
		{
			cudaStreamDestroy(stream);
			stream = nullptr;
		}
		std::swap(stream, other.stream);
		return *this;
	}

	CUDA_HOST_API bool CudaStream::isWorking()
	{
		return cudaStreamQuery(stream) == ::cudaErrorNotReady;
	}

	CUDA_HOST_API void CudaStream::sync()
	{
		if (isWorking())
		{
			cudaStreamSynchronize(stream);
		}
	}
}

