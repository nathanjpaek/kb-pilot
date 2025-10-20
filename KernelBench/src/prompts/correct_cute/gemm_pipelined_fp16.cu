#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/mma_traits.hpp>

using namespace cute;

template<typename T>
__global__ void gemm_pipelined_kernel(T* a, T* b, T* c, int m, int n, int k) {
    using MMA = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
    constexpr int kStages = 3; // Triple buffering
    
    auto block_shape = make_shape(Int<1024>{}, Int<1024>{});
    auto block_threads = make_shape(Int<16>{}, Int<64>{}); // 1024 threads
    
    auto thr_layout = make_layout(
        make_shape(Int<16>{}, Int<64>{}),
        make_stride(Int<64>{}, Int<1>{})
    );
    
    auto thr_mma = thr_layout(MMA{}, threadIdx.x);
    
    // Multi-stage shared memory
    auto smem_a_layout = make_layout(make_shape(Int<1024>{}, Int<256>{}), make_stride(Int<256>{}, Int<1>{}));
    auto smem_b_layout = make_layout(make_shape(Int<256>{}, Int<1024>{}), make_stride(Int<1024>{}, Int<1>{}));
    
    // Triple buffering
    auto smem_a_stages = make_tensor(shared_memory, make_layout(make_shape(kStages, 1024, 256)));
    auto smem_b_stages = make_tensor(shared_memory + kStages*1024*256*sizeof(T), make_layout(make_shape(kStages, 256, 1024)));
    
    auto tCrA = make_fragment_like(thr_mma(0_c));
    auto tCrB = make_fragment_like(thr_mma(0_c));
    auto tCrC = make_fragment_like(thr_mma(0_c));
    
    clear(tCrC);
    
    // Pipelined computation with stages
    for (int k_tile = 0; k_tile < k; k_tile += 256) {
        int stage = (k_tile / 256) % kStages;
        
        // Load current stage
        auto gmem_a = make_tensor(a, make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
        auto gmem_b = make_tensor(b, make_layout(make_shape(k, n), make_stride(n, Int<1>{})));
        
        auto smem_a = smem_a_stages(stage, _, _);
        auto smem_b = smem_b_stages(stage, _, _);
        
        // Async copy for pipelining
        copy_async(gmem_a, smem_a);
        copy_async(gmem_b, smem_b);
        
        // Compute on previous stage
        if (k_tile > 0) {
            int prev_stage = (stage - 1 + kStages) % kStages;
            auto prev_smem_a = smem_a_stages(prev_stage, _, _);
            auto prev_smem_b = smem_b_stages(prev_stage, _, _);
            
            for (int kk = 0; kk < 256; kk += 16) {
                auto tArA = prev_smem_a(_, make_coord(kk, kk+16));
                auto tBrB = prev_smem_b(make_coord(kk, kk+16), _);
                
                copy(tArA, tCrA);
                copy(tBrB, tCrB);
                
                gemm(tCrA, tCrB, tCrC);
            }
        }
        
        __syncthreads();
    }
    
    auto gmem_c = make_tensor(c, make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
    copy(tCrC, gmem_c);
}