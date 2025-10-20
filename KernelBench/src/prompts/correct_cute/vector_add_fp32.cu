#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

template<typename T>
__global__ void vector_add_kernel(T* a, T* b, T* c, int n) {
    // Static shapes everywhere - THE #1 rule
    auto tile_shape = make_shape(Int<256>{});
    auto block_threads = make_shape(Int<256>{}); // 256 threads
    
    // Thread-value layout pattern - THE fundamental pattern
    auto thr_layout = make_layout(
        make_shape(Int<256>{}, Int<1>{}),
        make_stride(Int<1>{}, Int<1>{})  // stride-1!
    );
    
    // Composition + Slice - The three-step dance
    auto gmem_layout = make_layout(make_shape(n), make_stride(Int<1>{}));
    auto gmem_tensor = make_tensor(a, gmem_layout);
    auto gmem_tiled = composition(gmem_tensor, thr_layout);
    auto my_data = gmem_tiled(threadIdx.x, _);
    
    // Load, compute, store
    auto a_val = my_data(0);
    auto b_val = make_tensor(b, gmem_layout)(threadIdx.x);
    auto c_val = a_val + b_val;
    make_tensor(c, gmem_layout)(threadIdx.x) = c_val;
}