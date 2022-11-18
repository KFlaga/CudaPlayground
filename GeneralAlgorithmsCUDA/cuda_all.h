#pragma once

// Quites CUDA warning

#pragma warning(disable:26812)   // The enum type '...' is unscoped. Prefer 'enum class' over 'enum'
#pragma warning(disable:26439)   // This kind of function may not throw. Declare it 'noexcept'
#pragma warning(disable:26495)   // Variable '...' is uninitialized. Always initialize a member variabl
#pragma warning(disable:26478)   // Don't use std::move on constant variables

#include <cuda_runtime.h>
#include <cublas.h>

#pragma warning(default:26812)
#pragma warning(default:26439)
#pragma warning(default:26495)
#pragma warning(default:26478)
