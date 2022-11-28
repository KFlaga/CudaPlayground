#include "jpeg_encode.h"
#include "jpeg_decode.h"
#include <jpeg/jconfig.h>
#include <jpeg/jmorecfg.h>
#include <jpeg/turbojpeg.h>
#include <memory>
#include <cstdlib>

namespace CudaPlayground
{
namespace JPEG
{

EncodedStream encodeReference(Matrix<Pixels::u8_RGBA> In)
{
    unsigned char* outJpegBuf = nullptr;
    unsigned long outSize = 0;

    std::unique_ptr<void, void(*)(void*)> tjHandle(tjInitCompress(), [](void* p) { tjDestroy(p); });

    int res = tjCompress2(
        tjHandle.get(),
        (unsigned char*)In.elements, // pointer to an image buffer containing RGB, grayscale, or CMYK pixels to be compressed
        In.cols,  // width (in pixels) of the source image
        0, // Setting this parameter to 0 is the equivalent of setting it to width * tjPixelSize[pixelFormat]
        In.rows, // height (in pixels) of the source image
        TJPF_RGBA, // pixel format of the source image
        &outJpegBuf, // address of a pointer to an image buffer that will receive the JPEG image
        &outSize,
        TJSAMP_444,
        95, // 1 - 100
        0
    //    | TJFLAG_ACCURATEDCT 
        | TJFLAG_FASTDCT
    //    | TJFLAG_PROGRESSIVE // better but slower
        | TJFLAG_FASTUPSAMPLE
     //   | TJFLAG_LOSSLESS
        | TJFLAG_BOTTOMUP // for win format
    );

    if (res == 0)
    {
        return EncodedStream{
            {outJpegBuf, [](void* p) { tjFree((unsigned char*)p); }},
            outSize
        };
    }
    else
    {
        throw std::runtime_error(tjGetErrorStr2(tjHandle.get()));
    }
}

MatrixExtMem<Matrix<Pixels::u8_RGBA>> decodeReference(EncodedStream In)
{
    std::unique_ptr<void, void(*)(void*)> tjHandle(tjInitDecompress(), [](void* p) { tjDestroy(p); });

    int jpegRows;
    int jpegCols;
    int jpegSubsampling;
    int jpegColorspace;

    int res = tjDecompressHeader3(
        tjHandle.get(),
        (unsigned char*)In.bytes.get(),
        (unsigned long)In.size,
        &jpegCols,
        &jpegRows,
        &jpegSubsampling,
        &jpegColorspace
    );

    if (res != 0)
    {
        throw std::runtime_error(tjGetErrorStr2(tjHandle.get()));
    }

    int outPitch = jpegCols * tjPixelSize[TJPF_RGBA];
    uint8_t* bufAligned = (uint8_t*)_aligned_malloc(outPitch * jpegRows, 4);
    std::unique_ptr<uint8_t, void(*)(void*)> outBuf(bufAligned, [](void* p) { _aligned_free(p); });
    MatrixExtMem<Matrix<Pixels::u8_RGBA>> Out(jpegRows, jpegCols, outPitch / jpegCols, std::move(outBuf));

    res = tjDecompress2(
        tjHandle.get(),
        (unsigned char*)In.bytes.get(), // pointer to a buffer containing the JPEG image to decompress
        (unsigned long)In.size, // size of the JPEG image (in bytes)
        (unsigned char*)bufAligned, // pointer to an image buffer that will receive the decompressed image
        jpegCols,
        outPitch,
        jpegRows,
        TJPF_RGBA, // pixel format of the destination image
        0
        // | TJFLAG_ACCURATEDCT 
        | TJFLAG_FASTDCT
        //    | TJFLAG_PROGRESSIVE // better but slower
        | TJFLAG_FASTUPSAMPLE
    );

    if (res == 0)
    {
        return Out;
    }
    else
    {
        throw std::runtime_error(tjGetErrorStr2(tjHandle.get()));
    }
}


}
}
