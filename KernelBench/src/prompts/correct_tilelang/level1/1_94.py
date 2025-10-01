"""
Problem Name: 94_MSELoss
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0981 runtime_stats={'mean': 0.0981, 'std': 0.00154, 'min': 0.096, 'max': 0.106, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0365, 'std': 0.00237, 'min': 0.0345, 'max': 0.0486, 'num_trials': 100}, 'speedup_ratio': 0.372}}
"""

import torch
import torch.nn as nn
import math

import tilelang
import tilelang.language as T


def mse_sum_kernel(M, N, threads=256, dtype="float16", accum_dtype="float32"):
    """
    TileLang kernel for computing the sum of squared differences.
    Uses efficient parallel reduction with atomic operations.
    """
    @T.prim_func
    def main(
        Pred: T.Tensor((M, N), dtype),
        Target: T.Tensor((M, N), dtype),
        Out: T.Tensor((1,), accum_dtype),
    ):
        with T.Kernel(1, threads=threads):
            # Allocate thread-local accumulator
            local_sum = T.alloc_fragment((1,), accum_dtype)
            T.clear(local_sum)
            
            # Calculate total number of elements
            numel = M * N
            
            # Each thread processes multiple elements
            for idx in T.Parallel(numel):
                row = idx // N
                col = idx % N
                
                # Load values and cast to accumulation dtype
                pred_val = T.Cast(accum_dtype, Pred[row, col])
                target_val = T.Cast(accum_dtype, Target[row, col])
                
                # Compute squared difference
                diff = pred_val - target_val
                local_sum[0] = local_sum[0] + (diff * diff)
            
            # Atomically add local sum to global output
            T.atomic_add(Out[0], local_sum[0])
    
    return main


class ModelNew(nn.Module):
    """
    Optimized model using TileLang to compute Mean Squared Error loss.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self._cached_kernels = {}

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA
        predictions = predictions.cuda()
        targets = targets.cuda()
        
        # Get dimensions
        M, N = predictions.shape
        
        # Convert to float16 for kernel computation
        pred_fp16 = predictions.to(dtype=torch.float16).contiguous()
        target_fp16 = targets.to(dtype=torch.float16).contiguous()
        
        # Cache key includes dimensions and dtype
        key = (M, N, pred_fp16.dtype)
        
        if key not in self._cached_kernels:
            # Create kernel function for these specific dimensions
            kernel_func = mse_sum_kernel(M, N, threads=256)
            # Use tilelang.jit decorator for compilation with automatic output allocation
            self._cached_kernels[key] = tilelang.jit(out_idx=-1)(kernel_func)
        
        kernel = self._cached_kernels[key]
        
        # Call kernel - it will automatically allocate the output tensor
        # The @tilelang.jit decorator handles output allocation when out_idx=-1
        sum_sq_tensor = kernel(pred_fp16, target_fp16)
        
        # Convert result to float32 and compute mean
        total_sum = sum_sq_tensor.item()
        mean_value = total_sum / (M * N)
        
        # Return as a scalar tensor on the same device
        return torch.tensor(mean_value, dtype=torch.float32, device=predictions.device)