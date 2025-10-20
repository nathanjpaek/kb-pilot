#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

template<typename T>
__global__ void vector_scale_kernel(T* a, T* b, T* c, T scale, int n) {
    // Static shapes with larger tile for FP16
    auto tile_shape = make_shape(Int<512>{});
    auto block_threads = make_shape(Int<256>{}, Int<2>{}); // 512 threads
    
    // Thread-value layout with vectorization
    auto thr_layout = make_layout(
        make_shape(Int<256>{}, Int<2>{}),
        make_stride(Int<2>{}, Int<1>{})  // stride-1 for vectorization
    );
    
    // Broadcasting: zero-stride for scalar
    auto scale_layout = make_layout(make_shape(Int<1>{}), make_stride(Int<0>{}));
    auto scale_tensor = make_tensor(&scale, scale_layout);
    
    auto gmem_layout = make_layout(make_shape(n), make_stride(Int<1>{}));
    auto gmem_tensor = make_tensor(a, gmem_layout);
    auto gmem_tiled = composition(gmem_tensor, thr_layout);
    auto my_data = gmem_tiled(threadIdx.x, _);
    
    // Vectorized load and compute
    auto a_vec = my_data(0);
    auto scale_vec = scale_tensor(0); // Broadcasts to all elements
    auto c_vec = a_vec * scale_vec;
    
    make_tensor(c, gmem_layout)(threadIdx.x) = c_vec;
}