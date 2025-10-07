// ThunderKittens warpgroup-level fp16 matmul for H100 (half inputs/outputs)
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define BLOCK_SIZE 64
#define NUM_WORKERS 8
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

struct micro_globals {
    using sub_tile = kittens::st_hf<BLOCK_SIZE, BLOCK_SIZE>;
    using tile_gl = kittens::gl<kittens::half, 1, 1, -1, -1, sub_tile>;
    
    tile_gl A, B, C;
    int M, K, N;
    
    dim3 grid() const { 
        return dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                    (M + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    }
    dim3 block() const { return dim3(NUM_THREADS, 1, 1); }
    size_t dynamic_shared_memory() const { return 100000; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);
    
    kittens::st_hf<BLOCK_SIZE, BLOCK_SIZE> (&As)[2] = al.allocate<kittens::st_hf<BLOCK_SIZE, BLOCK_SIZE>, 2>();
    kittens::st_hf<BLOCK_SIZE, BLOCK_SIZE> (&Bs)[2] = al.allocate<kittens::st_hf<BLOCK_SIZE, BLOCK_SIZE>, 2>();
    
    int tic = 0;
    int toc = 1;
    
    kittens::rt_fl<16, BLOCK_SIZE> C_accum;
    
    int row = blockIdx.y;
    int col = blockIdx.x;
    
    const int warpid = kittens::warpid();
    const int warpgroupid = warpid / 4;
    
    __shared__ kittens::semaphore bar;
    if (threadIdx.x == 0) {
        kittens::init_semaphore(bar, 0, 1);
        kittens::tma::expect_bytes(
            bar,
            kittens::size_bytes<typeof(As[0])> +
            kittens::size_bytes<typeof(Bs[0])>
        );
        kittens::tma::load_async(As[tic], g.A, {0, 0, row, 0}, bar);
        kittens::tma::load_async(Bs[tic], g.B, {0, 0, 0, col}, bar);
    }
    __syncthreads();
    
    kittens::warp::zero(C_accum);
    int num_tiles = (g.K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile, tic ^= 1, toc ^= 1) {
        // Arrive memory
        kittens::wait(bar, tic);
        __syncthreads();
        
        // Load next
        if (warpgroupid == 0) {
            kittens::warpgroup::decrease_registers<32>();
            if (threadIdx.x == 0 && tile + 1 < num_tiles) {
                kittens::tma::expect_bytes(bar,
                    kittens::size_bytes<typeof(As[0])> +
                    kittens::size_bytes<typeof(Bs[0])>
                );
                kittens::tma::load_async(As[toc], g.A, {0, 0, row, tile + 1}, bar);
                kittens::tma::load_async(Bs[toc], g.B, {0, 0, tile + 1, col}, bar);
            }
        } else {
            kittens::warpgroup::increase_registers<256>();
            kittens::warpgroup::mma_AB(C_accum, As[tic], Bs[tic]);
            kittens::warpgroup::mma_async_wait();
        }
        __syncthreads();
    }
    
    if (warpgroupid == 1) {
        kittens::warpgroup::store(g.C, C_accum, {0, 0, row, col});
    }
}

void dispatch_micro(micro_globals g) {
    size_t smem = g.dynamic_shared_memory();
    cudaFuncSetAttribute(micro_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    micro_tk<<<g.grid(), g.block(), smem>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    kittens::py::bind_kernel<micro_tk, micro_globals>(
        m, "micro_tk",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::M, &micro_globals::K, &micro_globals::N
    );
    kittens::py::bind_function<dispatch_micro, micro_globals>(
        m, "dispatch_micro",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::M, &micro_globals::K, &micro_globals::N
    );
}