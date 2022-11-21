#include "pinned_allocator.h"
#include "cuda_all.h"


namespace CudaPlayground
{
    namespace detail
    {
        void* allocate_pinned(size_t size)
        {
            void* ptr;
            cudaMallocHost(&ptr, size);
            return ptr;
        }

        void deallocate_pinned(void* ptr)
        {
            cudaFreeHost(ptr);
        }
    }
}
