#pragma once

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/cpp_stuff.h>
#include <memory>

namespace CudaPlayground
{
	struct EncodedStream
	{
		std::unique_ptr<uint8_t, void(*)(void*)> bytes;
		size_t size = 0;
	};
}
