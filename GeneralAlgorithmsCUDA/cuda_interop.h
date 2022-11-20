#pragma once

#ifdef __NVCC__

#include <GeneralAlgorithmsCUDA/cuda_all.h>

#define CUDA_HOST_API __host__
#define CUDA_DEVICE_API __device__
#define CUDA_COMMON_API __host__ __device__

#define CONCEPT(x) typename
#define CONCEPT_DECLARE(x)

#define OMS_ASSERT(x, y) 

#define KERNEL_ARGS(...) <<< __VA_ARGS__ >>>

#else

#include <builtin_types.h> // CUDA

#include <type_traits>
#include <concepts>
#include <stdexcept>

#define CUDA_HOST_API
#define CUDA_DEVICE_API
#define CUDA_COMMON_API

#define CONCEPT(x) x
#define CONCEPT_DECLARE(x) x

#ifdef NDEBUG
#define OMS_ASSERT(x, y)
#else
#define OMS_ASSERT(cond, msg) if(!(cond)) throw std::runtime_error(msg)
#endif

// This is for Visual Studio to quiet intelisense complaints
// CUDA code is always compiled with __NVCC__, so its not a problem
#define __syncthreads()

#define KERNEL_ARGS(...)

#endif

#ifdef __CUDA_RUNTIME_H__
#include <stdio.h>

template <typename T>
void check(T result, char const* const func, const char* const file,
    int const line) {
    if (result) {
        printf("CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

// This will output the proper CUDA error strings in the event  that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#endif
