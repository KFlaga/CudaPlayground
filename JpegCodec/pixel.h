#pragma once

#include <GeneralAlgorithmsCUDA/cuda_interop.h>

namespace CudaPlayground
{
	namespace Pixels
	{
		struct alignas(4) u8_RGBA
		{
			union
			{
				struct
				{
					uint8_t R;
					uint8_t G;
					uint8_t B;
					uint8_t A;
				} rgba;

				uint32_t mem;
			};
		};

		struct f32_RGBA
		{
			float R;
			float G;
			float B;
			float A;
		};
	}
}
