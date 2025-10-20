#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/mma_traits.hpp>

using namespace cute;

template<typename T>
__global__ void gemm_swizzled_kernel(T* a, T* b, T* c, int m, int n, int k) {
    using MMA = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
    constexpr int kStages = 3;
    
    auto block_shape = make_shape(Int<2048>{}, Int<2048>{});
    auto block_threads = make_shape(Int<32>{}, Int<64>{}); // 2048 threads
    
    auto thr_layout = make_layout(
        make_shape(Int<32>{}, Int<64>{}),
        make_stride(Int<64>{}, Int<1>{})
    );
    
    auto thr_mma = thr_layout(MMA{}, threadIdx.x);
    
    // Swizzled layouts for bank conflict avoidance
    auto smem_a_layout = make_layout(
        make_shape(Int<2048>{}, Int<512>{}),
        make_stride(Int<512>{}, Int<1>{})
    );
    auto smem_b_layout = make_layout(
        make_shape(Int<512>{}, Int<2048>{}),
        make_stride(Int<2048>{}, Int<1>{})
    );
    
    // Apply swizzling to avoid bank conflicts
    auto smem_a_swizzled = make_layout(
        make_shape(Int<2048>{}, Int<512>{}),
        make_stride(Int<512>{}, Int<1>{})
    );
    auto smem_b_swizzled = make_layout(
        make_shape(Int<512>{}, Int<2048>{}),
        make_stride(Int<2048>{}, Int<1>{})
    );
    
    // Multi-stage with swizzling
    auto smem_a_stages = make_tensor(shared_memory, make_layout(make_shape(kStages, 2048, 512)));
    auto smem_b_stages = make_tensor(shared_memory + kStages*2048*512*sizeof(T), make_layout(make_shape(kStages, 512, 2048)));
    
    auto tCrA = make_fragment_like(thr_mma(0_c));
    auto tCrB = make_fragment_like(thr_mma(0_c));
    auto tCrC = make_fragment_like(thr_mma(0_c));
    
    clear(tCrC);
    
    for (int k_tile = 0; k_tile < k; k_tile += 512) {
        int stage = (k_tile / 512) % kStages;
        
        auto gmem_a = make_tensor(a, make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
        auto gmem_b = make_tensor(b, make_layout(make_shape(k, n), make_stride(n, Int<1>{})));
        
        auto smem_a = smem_a_stages(stage, _, _);
        auto smem_b = smem_b_stages(stage, _, _);
        
        // Swizzled copy for bank conflict avoidance
        copy_async(gmem_a, smem_a);
        copy_async(gmem_b, smem_b);
        
        if (k_tile > 0) {
            int prev_stage = (stage - 1 + kStages) % kStages;
            auto prev_smem_a = smem_a_stages(prev_stage, _, _);
            auto prev_smem_b = smem_b_stages(prev_stage, _, _);
            
            for (int kk = 0; kk < 512; kk += 16) {
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