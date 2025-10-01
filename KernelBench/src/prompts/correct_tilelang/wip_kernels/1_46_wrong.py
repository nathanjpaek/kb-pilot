import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _avg_pool3d_kernel(N, C, D, H, W, K, S, P, dtype="float16", accum_dtype="float32"):
    """
    Create a TileLang kernel for 3D average pooling.
    
    Args:
        N: Batch size
        C: Number of channels
        D, H, W: Input depth, height, width
        K: Kernel size (assumes cubic kernel)
        S: Stride
        P: Padding
        dtype: Data type for input/output
        accum_dtype: Accumulator data type for numerical stability
    """
    # Calculate output dimensions
    D_out = (D + 2 * P - K) // S + 1
    H_out = (H + 2 * P - K) // S + 1
    W_out = (W + 2 * P - K) // S + 1
    
    # Block configuration
    block_size = 128
    
    @T.prim_func
    def main(
        X: T.Tensor((N, C, D, H, W), dtype),
        Out: T.Tensor((N, C, D_out, H_out, W_out), dtype),
    ):
        # Launch kernel with 2D grid: (output_positions, batch*channels)
        with T.Kernel(
            T.ceildiv(D_out * H_out * W_out, block_size), 
            N * C,
            threads=block_size
        ) as (bx, by):
            # Use T.Parallel to handle thread-level computation
            for tid in T.Parallel(block_size):
                # Calculate global linear index
                lin_idx = bx * block_size + tid
                
                # Check bounds
                if lin_idx < D_out * H_out * W_out:
                    # Get batch and channel from by
                    n = by // C
                    c = by % C
                    
                    # Decompose linear index into 3D output coordinates
                    do_ = lin_idx // (H_out * W_out)
                    rem = lin_idx % (H_out * W_out)
                    ho_ = rem // W_out
                    wo_ = rem % W_out
                    
                    # Calculate input region
                    d_start = do_ * S - P
                    h_start = ho_ * S - P
                    w_start = wo_ * S - P
                    
                    # Initialize accumulator and counter
                    acc_val = T.Cast(accum_dtype, 0)
                    count = 0
                    
                    # Iterate over pooling kernel
                    for kd in range(K):
                        id_ = d_start + kd
                        if id_ >= 0 and id_ < D:
                            for kh in range(K):
                                ih_ = h_start + kh
                                if ih_ >= 0 and ih_ < H:
                                    for kw in range(K):
                                        iw_ = w_start + kw
                                        if iw_ >= 0 and iw_ < W:
                                            acc_val = acc_val + T.Cast(accum_dtype, X[n, c, id_, ih_, iw_])
                                            count = count + 1
                    
                    # Compute average and store
                    if count > 0:
                        Out[n, c, do_, ho_, wo_] = T.Cast(dtype, acc_val / T.Cast(accum_dtype, count))
                    else:
                        Out[n, c, do_, ho_, wo_] = T.Cast(dtype, 0)
    
    return main


class ModelNew(nn.Module):
    """
    TileLang-accelerated 3D Average Pooling layer.
    
    Args:
        kernel_size: Size of the pooling kernel (assumes cubic kernel)
        stride: Stride for the pooling operation (defaults to kernel_size)
        padding: Padding to apply to input
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self._cached_kernels = {}

    def _get_kernel(self, shape, dtype_str):
        """Get or compile kernel for given input shape."""
        N, C, D, H, W = shape
        key = (N, C, D, H, W, self.kernel_size, self.stride, self.padding, dtype_str)
        
        if key not in self._cached_kernels:
            # Create the kernel function
            kernel_func = _avg_pool3d_kernel(
                N, C, D, H, W,
                self.kernel_size,
                self.stride,
                self.padding,
                dtype=dtype_str,
                accum_dtype="float32"
            )
            # Apply JIT compilation with automatic output allocation
            self._cached_kernels[key] = tilelang.jit(out_idx=-1)(kernel_func)
        
        return self._cached_kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 3D average pooling.
        
        Args:
            x: Input tensor of shape (N, C, D, H, W)
            
        Returns:
            Output tensor after average pooling
        """
        # Ensure input is on CUDA and in the right format
        original_dtype = x.dtype
        x_cuda = x.cuda().contiguous()
        
        # Convert to float16 for kernel execution
        x_fp16 = x_cuda.to(dtype=torch.float16)
        
        # Get or compile kernel
        kernel = self._get_kernel(x_fp16.shape, "float16")
        
        # Execute kernel (output is automatically allocated by tilelang.jit)
        output = kernel(x_fp16)
        
        # Convert back to original dtype
        return output.to(dtype=original_dtype)