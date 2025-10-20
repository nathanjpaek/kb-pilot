#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

template<typename T>
__global__ void small_gemm_kernel(T* a, T* b, T* c, int m, int n, int k) {
    // 3-level hierarchy: gmem -> smem -> rmem
    auto block_shape = make_shape(Int<128>{}, Int<128>{});
    auto block_threads = make_shape(Int<16>{}, Int<8>{}); // 128 threads
    
    auto thr_layout = make_layout(
        make_shape(Int<16>{}, Int<8>{}),
        make_stride(Int<8>{}, Int<1>{})
    );
    
    // Shared memory tiles
    auto smem_a_layout = make_layout(make_shape(Int<128>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto smem_b_layout = make_layout(make_shape(Int<32>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}));
    
    auto smem_a = make_tensor(shared_memory, smem_a_layout);
    auto smem_b = make_tensor(shared_memory + 128*32*sizeof(T), smem_b_layout);
    
    // Register memory for accumulation
    auto rmem_c = make_fragment_like(make_layout(make_shape(Int<8>{}, Int<8>{})));
    clear(rmem_c);
    
    // Tiling over K dimension
    for (int k_tile = 0; k_tile < k; k_tile += 32) {
        // Load A tile to shared memory
        auto gmem_a = make_tensor(a, make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
        auto gmem_a_tiled = composition(gmem_a, thr_layout);
        auto my_a_data = gmem_a_tiled(threadIdx.x, threadIdx.y);
        
        // Load B tile to shared memory  
        auto gmem_b = make_tensor(b, make_layout(make_shape(k, n), make_stride(n, Int<1>{})));
        auto gmem_b_tiled = composition(gmem_b, thr_layout);
        auto my_b_data = gmem_b_tiled(threadIdx.x, threadIdx.y);
        
        __syncthreads();
        
        // Naive GEMM with register memory
        for (int kk = 0; kk < 32; ++kk) {
            auto a_val = smem_a(threadIdx.x, kk);
            auto b_val = smem_b(kk, threadIdx.y);
            rmem_c += a_val * b_val;
        }
        __syncthreads();
    }
    
    // Store result
    auto gmem_c = make_tensor(c, make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
    auto gmem_c_tiled = composition(gmem_c, thr_layout);
    gmem_c_tiled(threadIdx.x, threadIdx.y) = rmem_c;
}