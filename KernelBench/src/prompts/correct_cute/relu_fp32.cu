#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

template<typename T>
__global__ void relu_kernel(T* a, T* c, int n) {
    auto tile_shape = make_shape(Int<256>{});
    auto block_threads = make_shape(Int<256>{});
    
    auto thr_layout = make_layout(
        make_shape(Int<256>{}, Int<1>{}),
        make_stride(Int<1>{}, Int<1>{})
    );
    
    auto gmem_layout = make_layout(make_shape(n), make_stride(Int<1>{}));
    auto gmem_tensor = make_tensor(a, gmem_layout);
    auto gmem_tiled = composition(gmem_tensor, thr_layout);
    auto my_data = gmem_tiled(threadIdx.x, _);
    
    // Conditional operations with predication
    auto a_val = my_data(0);
    auto zero = T(0);
    auto c_val = max(a_val, zero); // ReLU: max(0, x)
    
    make_tensor(c, gmem_layout)(threadIdx.x) = c_val;
}