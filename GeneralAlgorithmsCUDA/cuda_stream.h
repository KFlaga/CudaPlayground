#pragma once

#include <GeneralAlgorithmsCUDA/cuda_interop.h>
#include <GeneralAlgorithmsCUDA/cpp_stuff.h>

namespace CudaPlayground
{
	// thin wrapper to allow stream allocation from native compiled code
	struct CudaStream
	{
		MOVE_ONLY_CLASS(CudaStream);

		cudaStream_t stream = nullptr;

		CUDA_HOST_API CudaStream();
		CUDA_HOST_API ~CudaStream();

		CUDA_HOST_API CudaStream(CudaStream&&) noexcept;
		CUDA_HOST_API CudaStream& operator=(CudaStream&&) noexcept;

		CUDA_HOST_API bool isWorking();
		CUDA_HOST_API void sync();

		operator cudaStream_t ()
		{
			return stream;
		}
	};
}
