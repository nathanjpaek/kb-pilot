"""
Problem Name: 95_CrossEntropyLoss
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.052 runtime_stats={'mean': 0.052, 'std': 0.0131, 'min': 0.0481, 'max': 0.181, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0715, 'std': 0.000903, 'min': 0.0704, 'max': 0.0768, 'num_trials': 100}, 'speedup_ratio': 1.38}}
"""

"""
Problem Name: 95_CrossEntropyLoss
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0492 runtime_stats={'mean': 0.0492, 'std': 0.00602, 'min': 0.0439, 'max': 0.0728, 'num_trials': 100}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def cross_entropy_kernel(B, C, block_B=128, dtype="float16", accum_dtype="float32"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        preds: T.Tensor((B, C), dtype),
        targets: T.Tensor((B,), "int32"),
        losses: T.Tensor((B,), accum_dtype),
    ):
        # One-dimensional grid along the batch dimension
        with T.Kernel(T.ceildiv(B, block_B), threads=block_B) as bx:
            # Per-thread scratch buffers
            max_buf = T.alloc_fragment((block_B,), accum_dtype)
            sum_buf = T.alloc_fragment((block_B,), accum_dtype)

            # Initialize buffers
            T.fill(max_buf, -T.infinity(accum_dtype))
            T.fill(sum_buf, 0)

            # Compute per-sample max
            for cls in range(C):
                for tx in T.Parallel(block_B):
                    sample = bx * block_B + tx
                    if sample < B:
                        val = T.Cast(accum_dtype, preds[sample, cls])
                        max_buf[tx] = T.if_then_else(
                            val > max_buf[tx], val, max_buf[tx]
                        )

            # Compute exp sum
            for cls in range(C):
                for tx in T.Parallel(block_B):
                    sample = bx * block_B + tx
                    if sample < B:
                        val = T.Cast(accum_dtype, preds[sample, cls])
                        sum_buf[tx] += T.exp(val - max_buf[tx])

            # Final loss for each sample
            for tx in T.Parallel(block_B):
                sample = bx * block_B + tx
                if sample < B:
                    log_sum = T.log(sum_buf[tx]) + max_buf[tx]
                    tgt = targets[sample]
                    pred_tgt = T.Cast(accum_dtype, preds[sample, tgt])
                    losses[sample] = log_sum - pred_tgt

    return main


class ModelNew(nn.Module):
    """
    Optimized model using TileLang to compute Cross Entropy Loss
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self._kernel_cache = {}

    def _get_kernel(self, B, C, dtype):
        key = (B, C, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = cross_entropy_kernel(B, C, dtype=dtype)
        return self._kernel_cache[key]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move tensors to CUDA and appropriate dtypes
        preds = predictions.to(device="cuda", dtype=torch.float16)
        targs = targets.to(device="cuda", dtype=torch.int32)

        B, C = preds.shape
        kernel = self._get_kernel(B, C, dtype="float16")

        # Execute kernel to obtain per-sample losses
        losses = kernel(preds, targs)

        # Return mean loss (default reduction in torch.nn.functional.cross_entropy)
        return losses.mean()