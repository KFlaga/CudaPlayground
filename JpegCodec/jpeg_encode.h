#pragma once

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <JpegCodec/encoded_stream.h>
#include <JpegCodec/pixel.h>

namespace CudaPlayground
{
	namespace JPEG
	{
		EncodedStream encodeReference(Matrix<Pixels::u8_RGBA> In);
	}
}
