#pragma once

#include <memory>

namespace CudaPlayground
{
    namespace detail
    {
        void* allocate_pinned(size_t size);
        void deallocate_pinned(void* ptr);
    }

    template <class T>
    class PinnedAllocator
    {
    public:
        typedef T value_type;
        typedef value_type* pointer;
        typedef const value_type* const_pointer;
        typedef value_type& reference;
        typedef const value_type& const_reference;
        typedef typename std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        template <class tTarget>
        struct rebind
        {
            typedef PinnedAllocator<tTarget> other;
        };

        PinnedAllocator() {}
        ~PinnedAllocator() {}
        template <class T2>
        PinnedAllocator(const PinnedAllocator<T2>&)
        {
        }

        pointer address(reference ref)
        {
            return &ref;
        }
        const_pointer address(const_reference ref)
        {
            return &ref;
        }

        pointer allocate(size_type count, const void* = 0)
        {
            return reinterpret_cast<pointer>(detail::allocate_pinned(count * sizeof(T)));
        }

        void deallocate(pointer ptr, size_type)
        {
            detail::deallocate_pinned(ptr);
        }

        size_type max_size() const
        {
            return 0xffffffffUL / sizeof(T);
        }

        void construct(pointer ptr, const T& t)
        {
            new(ptr) T(t);
        }

        void destroy(pointer ptr)
        {
            ptr->~T();
        }

        template <class T2> bool operator==(PinnedAllocator<T2> const&) const
        {
            return true;
        }
        template <class T2> bool operator!=(PinnedAllocator<T2> const&) const
        {
            return false;
        }
    };
}