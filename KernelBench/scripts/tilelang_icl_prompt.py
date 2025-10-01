ICL_PROMPT = """
TILELANG RMS NORM EXAMPLE: 

import torch
import tilelang
import tilelang.language as T


def rms_norm_splitk(M, N, blk_m, blk_k):
    dtype = "float"

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, blk_k), dtype)
            A_local = T.alloc_fragment((blk_m, blk_k), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            num_k_step = T.ceildiv(N, blk_k)
            T.clear(A_local)
            for k in range(num_k_step):
                T.copy(A[bx * blk_m, k * blk_k], A_shared)
                for i, j in T.Parallel(blk_m, blk_k):
                    A_local[i, j] += A_shared[i, j] * A_shared[i, j]
            T.reduce_sum(A_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N) + 1e-12

            for k in range(num_k_step):
                # reverse, better cache hit rate
                T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_k], A_shared)
                for i, j in T.Parallel(blk_m, blk_k):
                    A_shared[i, j] *= A_powsum[i]
                T.copy(A_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])

    return main


def rms_norm(M, N, blk_m):
    dtype = "float"

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_pow_local = T.alloc_fragment((blk_m, N), dtype)
            A_local = T.alloc_fragment((blk_m, N), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            T.copy(A[bx * blk_m:(bx + 1) * blk_m, :], A_shared)
            T.copy(A_shared, A_local)
            for i, j in T.Parallel(blk_m, N):
                A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
            T.reduce_sum(A_pow_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N) + 1e-12
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= A_powsum[i]
            T.copy(A_local, B[bx * blk_m:(bx + 1) * blk_m, :])

    return main


def ref_program(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12)


if __name__ == "__main__":
    M, N, blk_m, blk_k = 8192, 8192, 1, 512
    program = rms_norm(M, N, blk_m)
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target="cuda",
        execution_backend="cython",
        pass_configs={"tl.disable_tma_lower": True})
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks pass.")

    latency = profiler.do_bench(ref_program, warmup=500)
    print("Ref: {:.2f} ms".format(latency))
    latency = profiler.do_bench(warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))


TILELANG FLASH ATTENTION EXAMPLE:

@T.prim_func
def flash_attention(
    Q: T.Tensor(shape, dtype),
    K: T.Tensor(shape, dtype),
    V: T.Tensor(shape, dtype),
    Output: T.Tensor(shape, dtype),
):
    # Launch a specialized T.Kernel with 3D mapping: (bx, by, bz)
    #   bx: block index in sequence dimension
    #   by: block index in "heads" dimension
    #   bz: block index in "batch" dimension
    # threads=thread_num means how many threads per block
    with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
        # Allocate shared memory for Q, K, V to reduce global memory accesses
        Q_shared = T.alloc_shared([block_M, dim], dtype)
        K_shared = T.alloc_shared([block_N, dim], dtype)
        V_shared = T.alloc_shared([block_N, dim], dtype)
        # Allocate buffers on register
        # acc_s: buffer to hold intermediate attention scores
        acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
        # acc_s_cast: buffer for storing casted/adjusted scores
        acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
        # acc_o: partial accumulation of output
        acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
        # Buffers to track per-row maximum score and related stats
        scores_max = T.alloc_fragment([block_M], accum_dtype)
        scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
        scores_scale = T.alloc_fragment([block_M], accum_dtype)
        scores_sum = T.alloc_fragment([block_M], accum_dtype)
        logsum = T.alloc_fragment([block_M], accum_dtype)

        # Annotate layout for Q_shared, e.g., use a swizzled layout to optimize memory access
        T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})

        # Copy a block of Q from global memory to Q_shared
        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)

        # Initialize accumulators
        T.fill(acc_o, 0)
        T.fill(logsum, 0)
        T.fill(scores_max, -T.infinity(accum_dtype))
        loop_range = (
            T.ceildiv((bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)
        )

        # Pipeline the loop to overlap copies/gemm stages
        for k in T.Pipelined(loop_range, num_stages=num_stages):
            # Copy K block into shared memory
            T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)

            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
                    )
            else:
                T.clear(acc_s)

            # Perform the Q*K^T multiplication, Here, transpose_B=True indicates that K_shared is transposed,
            # policy=T.GemmWarpPolicy.FullRow means each warp is responsible for computing an entire row
            # of acc_s, and the resulting acc_s is retained in registers.
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            # Copy V block into shared memory
            T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
            for i, j in T.Parallel(block_M, dim):
                acc_s[i, j] *= scale

            # Save old scores_max, then reset scores_max
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # Compute the maximum value per row on dimension 1 (block_N)
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)

            # Compute the factor by which we need to rescale previous partial sums
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])

            # Rescale the partial output accumulation to keep exponents consistent
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

            # Exponentiate (scores - max) for the new block
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])

            # Make a cast of acc_s to fp16 for the next GEMM
            T.copy(acc_s, acc_s_cast)

            # Multiply the attention acc_s_cast by V and add to partial output (acc_o)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            # Update the "logsum" tracker with the newly accumulated sum
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

        # Final step: divide each partial output by logsum (completing the softmax)
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] /= logsum[i]

        # Write back the final output block from acc_o to the Output buffer
        T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

        
TILELANG ELEMENTWISE ADD EXAMPLE (IN THE CORRECT PROBLEM FORMAT WITH MODELNEW):

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def elementwise_add(M, N, block_M, block_N, in_dtype="float16", out_dtype="float", threads=256):
    @tilelang.jit(
        out_idx=-1, # create the output tensor during runtime
    )
    @T.prim_func
    def main(A: T.Tensor((M, N), in_dtype), B: T.Tensor((M, N), in_dtype), C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M
            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                C[y, x] = A[y, x] + B[y, x]
    return main


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.block_M_cfg = 128
        self.block_N_cfg = 128
        self.threads_cfg = 256 

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:

        # TileLang only supports float16 on CUDA
        A = A.to(device="cuda", dtype=torch.float16)
        B = B.to(device="cuda", dtype=torch.float16)

        M, N = A.shape

        sum_kernel = elementwise_add(
            M, N,
            self.block_M_cfg, self.block_N_cfg,
            self.input_dtype, self.output_dtype, self.threads_cfg
        )

        sum_kernel = elementwise_add(M, N)
        return sum_kernel(A, B).to(torch.float32)
"""