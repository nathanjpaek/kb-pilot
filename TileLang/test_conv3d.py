import torch
import torch.nn as nn
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import numpy as np


def check_hopper():
    """Check if we're running on Hopper architecture"""
    try:
        # Simple check - in real implementation this would check GPU architecture
        return torch.cuda.get_device_capability()[0] >= 9
    except:
        return False


def convolution3d(N, C, D, H, W, F, K, S, Dil, P,
                  block_M, block_N, block_K, num_stages, threads,
                  dtype="float16", accum_dtype="float"):
    KD, KH, KW = K, K, K  # Assuming cubic kernel
    OD = (D + 2 * P - Dil * (K - 1) - 1) // S + 1
    OH = (H + 2 * P - Dil * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - Dil * (K - 1) - 1) // S + 1
    is_hopper = check_hopper()

    @T.prim_func
    def main(
            data: T.Tensor((N, D, H, W, C), dtype),
            kernel: T.Tensor((KD, KH, KW, C, F), dtype),
            out: T.Tensor((N, OD, OH, OW, F), dtype),
    ):
        with T.Kernel(
                T.ceildiv(F, block_N), T.ceildiv(N * OD * OH * OW, block_M),
                threads=threads) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            kernel_flat = T.Tensor((KD * KH * KW * C, F), dtype, kernel.data)
            out_flat = T.Tensor((N * OD * OH * OW, F), dtype, out.data)

            T.annotate_layout({
                out_shared: tilelang.layout.make_swizzled_layout(out_shared),
                data_shared: tilelang.layout.make_swizzled_layout(data_shared),
                kernel_shared: tilelang.layout.make_swizzled_layout(kernel_shared),
            })

            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(KD * KH * KW * C, block_K), num_stages=num_stages):
                # Load data using im2col for 3D
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i
                    
                    # Calculate indices for 3D convolution
                    out_idx = m % (OD * OH * OW)
                    out_d = out_idx // (OH * OW)
                    out_h = (out_idx % (OH * OW)) // OW
                    out_w = out_idx % OW
                    
                    # Kernel indices
                    kernel_idx = k % (KD * KH * KW * C)
                    kernel_d = kernel_idx // (KH * KW * C)
                    kernel_h = (kernel_idx % (KH * KW * C)) // (KW * C)
                    kernel_w = (kernel_idx % (KW * C)) // C
                    kernel_c = kernel_idx % C
                    
                    # Input access with stride, dilation, and padding
                    access_d = out_d * S + kernel_d * Dil - P
                    access_h = out_h * S + kernel_h * Dil - P
                    access_w = out_w * S + kernel_w * Dil - P
                    
                    in_bound = ((access_d >= 0) and (access_d < D) and
                               (access_h >= 0) and (access_h < H) and
                               (access_w >= 0) and (access_w < W))
                    
                    batch_idx = m // (OD * OH * OW)
                    data_shared[i, j] = T.if_then_else(
                        in_bound, 
                        data[batch_idx, access_d, access_h, access_w, kernel_c], 
                        0)
                
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    return main


def tilelang_conv3d(data, kernel, stride=1, dilation=1, padding=0):
    """
    TileLang 3D convolution wrapper function
    
    Args:
        data: Input tensor of shape (N, D, H, W, C)
        kernel: Kernel tensor of shape (KD, KH, KW, C, F)
        stride: Stride for convolution
        dilation: Dilation for convolution
        padding: Padding for convolution
    """
    # Move to GPU and ensure correct dtype
    data = data.cuda().half()
    kernel = kernel.cuda().half()
    
    N, D, H, W, C = data.shape
    KD, KH, KW, C_k, F = kernel.shape
    
    assert C == C_k, f"Input channels mismatch: {C} vs {C_k}"
    assert KD == KH == KW, "Only cubic kernels supported"
    
    K = KD
    S = stride
    Dil = dilation
    P = padding
    
    # Calculate output dimensions
    OD = (D + 2 * P - Dil * (K - 1) - 1) // S + 1
    OH = (H + 2 * P - Dil * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - Dil * (K - 1) - 1) // S + 1
    
    # Block sizes that work with TileLang GEMM constraints
    # These are based on the working TileLang gemm.py example
    # The constraint is that warp_m and warp_n must be divisible by 16
    block_M = 128   # Larger block to ensure proper warp alignment
    block_N = 128   # Larger block to ensure proper warp alignment  
    block_K = 32    # Keep K dimension smaller
    num_stages = 2  # Reduce to 2 to avoid memory issues
    threads = 128   # Use 128 threads like in the working GEMM example
    
    # Compile the kernel
    func = convolution3d(N, C, D, H, W, F, K, S, Dil, P,
                        block_M, block_N, block_K, num_stages, threads)
    compiled_kernel = tilelang.compile(func, out_idx=[2], target="cuda")
    
    # Execute
    return compiled_kernel(data, kernel)


def torch_conv3d_reference(data, kernel, stride=1, dilation=1, padding=0):
    """
    PyTorch reference implementation
    
    Args:
        data: Input tensor of shape (N, D, H, W, C) - will convert to (N, C, D, H, W)
        kernel: Kernel tensor of shape (KD, KH, KW, C, F) - will convert to (F, C, KD, KH, KW)
        stride: Stride for convolution
        dilation: Dilation for convolution
        padding: Padding for convolution
    """
    # Convert from TileLang format to PyTorch format
    # TileLang: (N, D, H, W, C) -> PyTorch: (N, C, D, H, W)
    data_torch = data.permute(0, 4, 1, 2, 3)
    
    # TileLang: (KD, KH, KW, C, F) -> PyTorch: (F, C, KD, KH, KW)
    kernel_torch = kernel.permute(4, 3, 0, 1, 2)
    
    # Perform convolution
    output_torch = F.conv3d(data_torch, kernel_torch, 
                           stride=stride, dilation=dilation, padding=padding)
    
    # Convert back to TileLang format: (N, C, D, H, W) -> (N, D, H, W, C)
    return output_torch.permute(0, 2, 3, 4, 1)


class ModelNew(nn.Module):
    """
    Model using TileLang 3D convolution
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, data, kernel, stride=1, dilation=1, padding=0):
        return tilelang_conv3d(data, kernel, stride, dilation, padding)


def test_conv3d():
    """Test 3D convolution implementation"""
    print("Testing TileLang 3D Convolution...")
    
    # Test parameters
    N = 2        # Batch size
    D, H, W = 8, 8, 8   # Input spatial dimensions
    C = 4        # Input channels
    F = 8        # Output channels (filters)
    K = 3        # Kernel size (cubic)
    S = 1        # Stride
    Dil = 1      # Dilation
    P = 1        # Padding
    
    print(f"Input shape: ({N}, {D}, {H}, {W}, {C})")
    print(f"Kernel shape: ({K}, {K}, {K}, {C}, {F})")
    print(f"Stride: {S}, Dilation: {Dil}, Padding: {P}")
    
    # Create test data
    torch.manual_seed(42)
    data = torch.randn(N, D, H, W, C, device="cuda", dtype=torch.float16)
    kernel = torch.randn(K, K, K, C, F, device="cuda", dtype=torch.float16)
    
    # Calculate expected output shape
    OD = (D + 2 * P - Dil * (K - 1) - 1) // S + 1
    OH = (H + 2 * P - Dil * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - Dil * (K - 1) - 1) // S + 1
    print(f"Expected output shape: ({N}, {OD}, {OH}, {OW}, {F})")
    
    # Test TileLang implementation
    model = ModelNew()
    
    try:
        print("\nRunning TileLang 3D convolution...")
        with torch.no_grad():
            output_tilelang = model(data, kernel, stride=S, dilation=Dil, padding=P)
        print(f"TileLang output shape: {output_tilelang.shape}")
        
        # Test PyTorch reference
        print("Running PyTorch reference...")
        with torch.no_grad():
            output_torch = torch_conv3d_reference(data, kernel, stride=S, dilation=Dil, padding=P)
        print(f"PyTorch output shape: {output_torch.shape}")
        
        # Compare results
        assert output_tilelang.shape == output_torch.shape, f"Shape mismatch: {output_tilelang.shape} vs {output_torch.shape}"
        
        # Calculate differences
        max_diff = torch.max(torch.abs(output_tilelang - output_torch))
        mean_diff = torch.mean(torch.abs(output_tilelang - output_torch))
        rel_error = max_diff / (torch.max(torch.abs(output_torch)) + 1e-7)
        
        print(f"\nComparison Results:")
        print(f"Maximum absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Relative error: {rel_error:.6f}")
        
        # Check if results are close enough
        tolerance = 1e-2  # Adjust based on expected precision
        success = max_diff < tolerance
        
        if success:
            print(f"✓ Test PASSED! (max_diff < {tolerance})")
        else:
            print(f"✗ Test FAILED! (max_diff >= {tolerance})")
            
        return success
        
    except Exception as e:
        print(f"✗ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_configs():
    """Test with different configurations"""
    configs = [
        # (N, D, H, W, C, F, K, S, Dil, P)
        (1, 4, 4, 4, 2, 4, 3, 1, 1, 1),    # Small test
        (2, 8, 8, 8, 4, 8, 3, 1, 1, 1),    # Medium test
        (1, 6, 6, 6, 3, 6, 3, 2, 1, 0),    # Stride 2, no padding
    ]
    
    print("\n" + "="*60)
    print("Testing different configurations...")
    
    success_count = 0
    for i, (N, D, H, W, C, F, K, S, Dil, P) in enumerate(configs):
        print(f"\n--- Configuration {i+1} ---")
        print(f"N={N}, D={D}, H={H}, W={W}, C={C}, F={F}, K={K}, S={S}, Dil={Dil}, P={P}")
        
        try:
            # Create test data
            torch.manual_seed(42 + i)
            data = torch.randn(N, D, H, W, C, device="cuda", dtype=torch.float16)
            kernel = torch.randn(K, K, K, C, F, device="cuda", dtype=torch.float16)
            
            # Test
            model = ModelNew()
            with torch.no_grad():
                output_tilelang = model(data, kernel, stride=S, dilation=Dil, padding=P)
                output_torch = torch_conv3d_reference(data, kernel, stride=S, dilation=Dil, padding=P)
            
            max_diff = torch.max(torch.abs(output_tilelang - output_torch))
            print(f"Max difference: {max_diff:.6f}")
            
            if max_diff < 1e-2:
                print("✓ PASSED")
                success_count += 1
            else:
                print("✗ FAILED")
                
        except Exception as e:
            print(f"✗ FAILED with error: {e}")
    
    print(f"\n{success_count}/{len(configs)} configurations passed")


if __name__ == "__main__":
    print("TileLang 3D Convolution Test")
    print("="*60)
    
    # Run main test
    success = test_conv3d()
    
    # Run additional configuration tests
    if success:
        test_different_configs()
    
    print("\n" + "="*60)
    print("Test completed!") 