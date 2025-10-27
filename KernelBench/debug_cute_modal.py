#!/usr/bin/env python3
"""
Debug CuTe kernel on Modal with detailed output
"""
import modal

app = modal.App("debug-cute-kernel")

cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install(
        "torch==2.5.0",
        "nvidia-cutlass-dsl",
        "numpy",
    )
    .add_local_python_source("scripts", "src")
    .add_local_dir("KernelBench", "/root/KernelBench")
)

@app.function(image=image, gpu="H100")
def debug_kernel():
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    
    print("="*80)
    print("Debugging CuTe kernel on Modal H100")
    print("="*80)
    
    # Copy kernel definition
    @cute.kernel
    def _linear_bias_relu_kernel(
        gA     : cute.Tensor,           # (B, K)
        gBTv   : cute.Tensor,           # ((1,V), (K, O/V))
        gBiasv : cute.Tensor,           # ((1,V), (O/V))
        gCv    : cute.Tensor,           # ((1,V), (B, O/V))
        K      : cutlass.Int32,
    ):
        tidx, _, _    = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        bdimx, _, _   = cute.arch.block_dim()

        bi = bidy
        og = bidx * bdimx + tidx

        B  = gCv.shape[1][0]
        Og = gCv.shape[1][1]

        if bi < B and og < Og:
            c_out = gCv[(None, (bi, og))]

            b_frag       = cute.make_fragment_like(c_out, gA.element_type)
            bias_frag    = cute.make_fragment_like(c_out, gA.element_type)
            acc_f32_frag = cute.make_fragment_like(c_out, cutlass.Float32)
            acc_f32_frag.fill(0.0)
            acc = acc_f32_frag.load()

            # GEMM reduction
            for k in range(K):
                a_val_f32 = cutlass.Float32(gA[bi, k])
                b_vec_gmem = gBTv[(None, (k, og))]
                cute.autovec_copy(b_vec_gmem, b_frag)
                b_vec_f32 = b_frag.load().to(cutlass.Float32)
                acc = acc + a_val_f32 * b_vec_f32

            # Add bias
            bias_vec_gmem = gBiasv[(None, og)]
            cute.autovec_copy(bias_vec_gmem, bias_frag)
            bias_f32 = bias_frag.load().to(cutlass.Float32)
            acc = acc + bias_f32

            # ReLU
            zero = cutlass.Float32(0.0)
            acc = cute.where(acc > zero, acc, zero)

            # Store
            out_frag = cute.make_fragment_like(c_out, gA.element_type)
            out_frag.store(acc.to(gA.element_type))
            cute.autovec_copy(out_frag, c_out)

    @cute.jit
    def _linear_bias_relu_host(
        mA    : cute.Tensor,
        mBT   : cute.Tensor,
        mBias : cute.Tensor,
        mC    : cute.Tensor,
        V     : cutlass.Constexpr,
        K     : cutlass.Constexpr,
    ):
        B, O = mA.shape[0], mBT.shape[1]

        gBTv   = cute.zipped_divide(mBT,   (1, V))
        gBiasv = cute.zipped_divide(mBias, (V,))
        gCv    = cute.zipped_divide(mC,    (1, V))

        threads_per_block = 256
        O_groups = O // V
        grid_x = cute.ceil_div(O_groups, threads_per_block)
        grid_y = B

        _linear_bias_relu_kernel(
            mA, gBTv, gBiasv, gCv, cutlass.Int32(K)
        ).launch(
            grid  = (grid_x, grid_y, 1),
            block = (threads_per_block, 1, 1)
        )
    
    # Small test case
    B, K, O = 4, 8, 16
    V = 8
    
    torch.manual_seed(42)
    
    x = torch.randn(B, K, dtype=torch.float16, device='cuda')
    W = torch.randn(O, K, dtype=torch.float16, device='cuda')
    bias = torch.randn(O, dtype=torch.float16, device='cuda')
    
    # Reference
    with torch.no_grad():
        out_ref = torch.nn.functional.linear(x, W, bias)
        out_ref = torch.relu(out_ref)
    
    print(f"Reference output[0,:4]: {out_ref[0, :4]}")
    
    # CuTe
    y = torch.empty((B, O), dtype=torch.float16, device='cuda')
    WT = W.t().contiguous()
    
    mA    = from_dlpack(x,    assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    mBT   = from_dlpack(WT,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
    mC    = from_dlpack(y,    assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    
    kernel = cute.compile(_linear_bias_relu_host, mA, mBT, mBias, mC, V, K)
    kernel(mA, mBT, mBias, mC)
    
    print(f"CuTe output[0,:4]: {y[0, :4]}")
    
    diff = (out_ref - y).abs()
    print(f"\nMax diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")
    print(f"Match: {torch.allclose(out_ref, y, rtol=1e-2, atol=1e-2)}")
    
    return {
        'matches': bool(torch.allclose(out_ref, y, rtol=1e-2, atol=1e-2)),
        'max_diff': float(diff.max().item()),
        'ref_sample': out_ref[0, :4].cpu().tolist(),
        'cute_sample': y[0, :4].cpu().tolist()
    }

@app.local_entrypoint()
def main():
    result = debug_kernel.remote()
    print("\n" + "="*80)
    print("RESULT:")
    print(result)

if __name__ == "__main__":
    main()

