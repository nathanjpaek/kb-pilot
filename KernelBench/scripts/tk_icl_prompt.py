TK_ICL_PROMPT = """
THUNDERKITTENS ATTENTION EXAMPLE: 

#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
using namespace kittens;

#define NUM_WORKERS  (1)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define D_QK (16)
#define D_VO (64)

static constexpr int dv = 64;
static constexpr int fd = 16;

using q_tile = st<bf16, fd, fd>;
using k_tile = st<bf16, fd, fd>;
using v_tile = st<bf16, fd, dv>;
using o_tile = st<bf16, fd, dv>;
using q_gl     = gl<bf16,  -1, -1, -1, fd, q_tile>;
using k_gl     = gl<bf16,  -1, -1, -1, fd, k_tile>;
using v_gl     = gl<bf16,  -1, -1, -1, dv, v_tile>;
using o_gl     = gl<bf16,  -1, -1, -1, dv, o_tile>;

struct micro_globals { 
    // pointers
    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;
    int n;

    dim3 grid()  { return dim3(o.depth, o.batch); } // parallelize over (heads=depth, batch)
    dim3 block() { return dim3(NUM_THREADS); } // number of threads per threadblock
    size_t dynamic_shared_memory() { return 100000; } 
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    const int batch = blockIdx.y;
    const int head  = blockIdx.x;

    int laneid = kittens::laneid(); 
    int warpid = kittens::warpid(); 

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);

    // create shared tiles for qkvo    
    st<bf16, fd,fd> (&q_s)  = al.allocate<st<bf16, fd,fd>>();
    st<bf16, fd,fd> (&k_s)  = al.allocate<st<bf16, fd,fd>>();
    st<bf16, fd,dv> (&v_s)  = al.allocate<st<bf16, fd,dv>>();
    st<bf16, fd,dv> (&o_s)  = al.allocate<st<bf16, fd,dv>>();

    // zero the kv state
    int total_block_idx = 0; 

    // loop along sequence dimension
    int n_blocks = g.n / (kittens::TILE_ROW_DIM<bf16>);
    for (int block = 0; block < n_blocks; block++) {
        // create register tiles
        rt_bf<fd, fd> q, k, local_attn_bf; 
        rt_fl<fd, fd> local_attn, temp_attn_accum;
        rt_bf<fd, dv> v; 
        rt_fl<fd, dv> o, accum; 

        // load from global to shared
        int cur_idx;
        cur_idx = block + warpid;
        load(q_s, g.q, {batch, head, cur_idx, 0});
        load(k_s, g.k, {batch, head, cur_idx, 0});

        cur_idx = block + warpid;
        load(v_s, g.v, {batch, head, cur_idx, 0});
        __syncthreads();

        // load shared to register
        load(q, q_s[warpid]);
        load(k, k_s[warpid])    
        // QK^T in linear attention
        zero(local_attn);
        mma_ABt(local_attn, q, k, local_attn)   
        // convert back to bf16 type
        copy(local_attn_bf, local_attn) 
        // load V to registers
        // then multiply (QK^T) with V in linear attention
        load(v, v_s[warpid]);
        auto &v_col = swap_layout_inplace(v);
        zero(o);
        
        mma_AB(o, local_attn_bf, v_col, o)  
    }
    store(g.o, o_s[warpid], {batch, head, cur_idx, 0});
    __syncthreads();
}


void dispatch_micro(micro_globals g) {
    // launch
    unsigned long mem_size = 100000; 
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(
        micro_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(ATTN_H, ATTN_B);
    micro_tk<<<grid,NUM_THREADS,mem_size>>>(g);
    cudaDeviceSynchronize();
}

THUNDERKITTENS ELEMENTWISE ADD EXAMPLE (IN THE CORRECT PROBLEM FORMAT WITH MODELNEW AND 2 CODE BLOCKS):

```cpp
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_THREADS (kittens::WARP_THREADS) // use 1 warp

#define _row 16
#define _col 32

// define global layout
using _gl  = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;

struct micro_globals {
    _gl x, o;
    // grid - number of thread blocks we are launching
    dim3 grid()  { return dim3(x.batch, x.depth, x.rows); } 
    // block - number of threads in a thread block
    dim3 block() { return dim3(x.cols); } 
    // Safe shared memory size for H100
    size_t dynamic_shared_memory() { return 224000; } 
};

// define kernel
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {

    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_fl<_row, _col> (&x_s) = al.allocate<st_fl<_row, _col>>();
    st_fl<_row, _col> (&o_s) = al.allocate<st_fl<_row, _col>>();

    // register memory 
    rt_fl<_row, _col> x_reg_fl;

    // load from HBM to shared
    load(x_s, g.x, {0, 0, 0, 0});
    __syncthreads();

    // load from shared to register
    load(x_reg_fl, x_s);
    __syncthreads();

    // x (dst) = x (src b) + x (src a)
    add(x_reg_fl, x_reg_fl, x_reg_fl);
    __syncthreads();

    // store from register to shared
    store(o_s, x_reg_fl);
    __syncthreads();

    // store from shared to HBM
    store(g.o, o_s, {0, 0, 0, 0});
    __syncthreads();
}

// Launch Kernel
void dispatch_micro(micro_globals g) {
    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(
        micro_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    cudaDeviceSynchronize();
}

// the module to build will be called tk_kernels
PYBIND11_MODULE(tk_kernels, m) {
    m.doc() = "tk_kernels python module";
    // For wrapping kernels directly.
    BIND_KERNEL(m, "micro_tk", micro_tk, micro_globals, x, o); 
    // For host functions that wrap the kernel, this will be called from Python
    BIND_FUNCTION(m, "dispatch_micro", dispatch_micro, micro_globals, x, o); 
}
```

```python
# import compiled ThunderKitten kernels
import tk_kernels # compiled kernel module would also be named tk_kernels


import torch
import torch.nn as nn
import torch.nn.functional as F

B = 1 
N = 16
D = 32
DTYPE = torch.float32


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        output = torch.zeros_like(x, dtype=DTYPE)
        tk_kernels.dispatch_micro(x, output)
        return output
```
"""