General matrix-vector multiplication (GEMV) can be viewed as a specialized case of general matrix-matrix multiplication (GEMM). It plays a critical role in deep learning, especially during the inference phase of large language models. In this tutorial, we will optimize GEMV from a thread-level perspective step by step using `TileLang`.

In this tutorial, we implement a simple GEMV kernel and learn that `TileLang` exposes low level control to user such as thread-level programming and CUDA primitives.

## Naive Implementation in TileLang
If you have a basic understanding of CUDA C, it is natural to start with a naive GEMV kernel by adapting a GEMM tiling strategy. You can think of GEMV as a `(1, k) * (k, n)` GEMM. Below is a simple example:

```python
def naive_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N)) as bn:
            tn = T.get_thread_binding(0)  # tn = threadIdx.x
            A_shared = T.alloc_shared((BLOCK_K,), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_reg = T.alloc_local((1,), accum_dtype)
            T.clear(C_reg)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for tk in T.serial(BLOCK_K):
                    A_shared[tk] = A[bk * BLOCK_K + tk]
                    B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                for tk in T.serial(BLOCK_K):
                    C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[tn,
                                                                            tk].astype(accum_dtype)
            C[bn * BLOCK_N + tn] = C_reg[0]

    return main
```

In this design, the first 128 threads act as the data producer and the last 128 threads as the consumer within a block (assuming a 1D block).

At this level, we only gain very little computation power from our GPU with around **~0.17 ms** compared to torch/cuBLAS's **~0.008 ms**, which is around 20x slower.

## More Concurrency

To further increase the concurrency of our kernel, we can exploit finer thread-level parallelism. Instead of assigning each thread to compute a single output element in C, you can introduce parallelism along the K dimension. Each thread computes a partial accumulation, and you then combine these partial results. This approach requires primitives like `atomicAdd` in CUDA.

Here’s a simplified version:
```python
def naive_splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, BLOCK_K)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((1,), dtype)
            B_local = T.alloc_local((1,), dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                A_local[0] = A[bk * BLOCK_K + tk]
                B_local[0] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                C_accum[0] += A_local[0].astype(accum_dtype) * B_local[0].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main
```

By introducing parallelism along K dimension, our kernel now achieves **~0.024 ms**, an improvement, but still not on par with torch/cuBLAS.

### Customizing Parallelism in K Dimension
If your K dimension is large, you can further customize how many elements each thread processes by introducing a `reduce_threads` parameter. This way, each thread handles multiple elements per iteration:

```python
def splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    TILE_K = T.ceildiv(BLOCK_K, reduce_threads)

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.serial(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main
```


## Vectorized Reads

GEMV is less computation intensive than GEMM as the computation intensity and memory throughput will be the optimization bottleneck. One effective strategy is to use vectorized load/store operations (e.g., `float2`, `float4`). In `TileLang`, you can specify vectorized operations via `T.vectorized`:

```python
def splitk_gemv_vectorized(
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main
```

With vectorized read, now the kernel finishs in **~0.0084 ms**, which is getting close to cuBLAS performance.


## `tvm_thread_allreduce` Instead of `atomicAdd`

`tvm_thread_allreduce` has implemented optimization when making an all-reduce across a number of threads, which should outperfrom out plain smem + `atomidAdd`:

```python
def splitk_gemv_vectorized_tvm(
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_accum = T.alloc_local((1,), accum_dtype)

            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            C_reduced = T.alloc_local((1,), accum_dtype)
            with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        C_accum[0],
                        True,
                        C_reduced[0],
                        tk,
                        dtype="handle",
                    ))

            C[bn * BLOCK_N + tn] = C_reduced[0]

    return main
```

With this optimization, the kernel latency now reduces from **~0.0084 ms** to **~0.0069 ms**, which is faster than torch/cuBLAS!

## Autotune

`BLOCK_N`, `BLOCK_K`, `reduce_threads` are hyperparameters in our kernel, which can be tuned to improve performance. We can use the `tilelang.autotune` feature to automatically search for optimal configurations:

```python
def get_best_config(N, K):

    def get_configs():
        BLOCK_N = [2, 4, 8, 32, 64, 128]
        reduce_threads = [4, 8, 32]
        _configs = list(itertools.product(
            BLOCK_N,
            reduce_threads,
        ))
        configs = [{
            "BLOCK_N": c[0],
            "reduce_threads": c[1],
        } for c in _configs]
        return configs

    @autotune(
        configs=get_configs(),
        keys=[
            "BLOCK_N",
            "reduce_threads",
        ],
        warmup=3,
        rep=20,
    )
    @jit(
        out_idx=[-1],
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=ref_program,
        skip_check=False,
        target="auto",
    )
    def kernel(
        BLOCK_N=None,
        reduce_threads=None,
    ):
        dtype = "float16"
        accum_dtype = "float"
        MAX_TRANSACTION_SIZE_IN_BITS = 128
        TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_K = reduce_threads * TILE_K

        @T.prim_func
        def main(
                A: T.Buffer((K,), dtype),
                B: T.Buffer((N, K), dtype),
                C: T.Buffer((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
                tn = T.get_thread_binding(0)
                tk = T.get_thread_binding(1)
                A_local = T.alloc_local((TILE_K,), dtype)
                B_local = T.alloc_local((TILE_K,), dtype)
                C_accum = T.alloc_local((1,), accum_dtype)

                T.clear(C_accum)
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    for k in T.vectorized(TILE_K):
                        A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                        B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                    for k in T.serial(TILE_K):
                        C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
                C_reduced = T.alloc_local((1,), accum_dtype)
                with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            C_accum[0],
                            True,
                            C_reduced[0],
                            tk,
                            dtype="handle",
                        ))

                C[bn * BLOCK_N + tn] = C_reduced[0]

        return main

    return kernel()
```

After autotuning, now our kernel gets **~0.0067 ms**.