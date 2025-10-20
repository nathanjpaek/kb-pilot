#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/mma_traits.hpp>

using namespace cute;

template<typename T>
__global__ void gemm_ampere_kernel(T* a, T* b, T* c, int m, int n, int k) {
    // Ampere MMA atoms for A100
    using MMA = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
    auto block_shape = make_shape(Int<512>{}, Int<512>{});
    auto block_threads = make_shape(Int<16>{}, Int<32>{}); // 512 threads
    
    auto thr_layout = make_layout(
        make_shape(Int<16>{}, Int<32>{}),
        make_stride(Int<32>{}, Int<1>{})
    );
    
    auto thr_mma = thr_layout(MMA{}, threadIdx.x);
    
    // Larger shared memory tiles for Ampere
    auto smem_a_layout = make_layout(make_shape(Int<512>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}));
    auto smem_b_layout = make_layout(make_shape(Int<128>{}, Int<512>{}), make_stride(Int<512>{}, Int<1>{}));
    
    auto smem_a = make_tensor(shared_memory, smem_a_layout);
    auto smem_b = make_tensor(shared_memory + 512*128*sizeof(T), smem_b_layout);
    
    auto tCrA = make_fragment_like(thr_mma(0_c));
    auto tCrB = make_fragment_like(thr_mma(0_c));
    auto tCrC = make_fragment_like(thr_mma(0_c));
    
    clear(tCrC);
    
    for (int k_tile = 0; k_tile < k; k_tile += 128) {
        auto gmem_a = make_tensor(a, make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
        auto gmem_b = make_tensor(b, make_layout(make_shape(k, n), make_stride(n, Int<1>{})));
        
        copy(gmem_a, smem_a);
        copy(gmem_b, smem_b);
        __syncthreads();
        
        for (int kk = 0; kk < 128; kk += 16) {
            auto tArA = smem_a(_, make_coord(kk, kk+16));
            auto tBrB = smem_b(make_coord(kk, kk+16), _);
            
            copy(tArA, tCrA);
            copy(tBrB, tCrB);
            
            gemm(tCrA, tCrB, tCrC);
        }
        __syncthreads();
    }
    
    auto gmem_c = make_tensor(c, make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
    copy(tCrC, gmem_c);
}