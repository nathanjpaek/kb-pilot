#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

template<typename T>
__global__ void matrix_transpose_kernel(T* a, T* c, int m, int n) {
    // 2D block configuration
    auto block_shape = make_shape(Int<64>{}, Int<64>{});
    auto block_threads = make_shape(Int<8>{}, Int<8>{}); // 64 threads
    
    // Thread layout for 2D
    auto thr_layout = make_layout(
        make_shape(Int<8>{}, Int<8>{}),
        make_stride(Int<8>{}, Int<1>{})
    );
    
    // Global memory layout (row-major)
    auto gmem_layout = make_layout(
        make_shape(Int<64>{}, Int<64>{}),
        make_stride(Int<64>{}, Int<1>{})  // row-major
    );
    
    // Shared memory layout (column-major for transpose)
    auto smem_layout = make_layout(
        make_shape(Int<64>{}, Int<64>{}),
        make_stride(Int<1>{}, Int<64>{})  // column-major (transposed!)
    );
    
    auto gmem_tensor = make_tensor(a, gmem_layout);
    auto smem_tensor = make_tensor(shared_memory, smem_layout);
    
    // Load with coalesced access
    auto gmem_tiled = composition(gmem_tensor, thr_layout);
    auto my_data = gmem_tiled(threadIdx.x, threadIdx.y);
    
    // Store to shared memory with transpose
    auto smem_tiled = composition(smem_tensor, thr_layout);
    smem_tiled(threadIdx.x, threadIdx.y) = my_data;
    __syncthreads();
    
    // Load from shared memory (already transposed)
    auto transposed_data = smem_tiled(threadIdx.y, threadIdx.x);
    
    // Store to global memory
    make_tensor(c, gmem_layout)(threadIdx.x, threadIdx.y) = transposed_data;
}