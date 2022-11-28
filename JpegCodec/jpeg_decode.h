#pragma once

#include <GeneralAlgorithmsCUDA/matrix.h>
#include <GeneralAlgorithmsCUDA/matrix_host.h>
#include <JpegCodec/encoded_stream.h>

namespace CudaPlayground
{
	namespace JPEG
	{
		MatrixExtMem<Matrix<Pixels::u8_RGBA>> decodeReference(EncodedStream In);
	}
}
