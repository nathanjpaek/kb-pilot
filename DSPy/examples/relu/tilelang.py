import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def relu_kernel(batch_size, dim, block_size=128, dtype="float16"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        x: T.Tensor((batch_size, dim), dtype),
        output: T.Tensor((batch_size, dim), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size * dim, block_size), threads=block_size) as bx:
            # Allocate fragment memory for input and output
            x_frag = T.alloc_fragment((block_size,), dtype)

            # Calculate global index
            start_idx = bx * block_size

            # Load data from global memory to fragment
            for i in T.Parallel(block_size):
                global_idx = start_idx + i
                if global_idx < batch_size * dim:
                    batch_idx = global_idx // dim
                    dim_idx = global_idx % dim
                    x_frag[i] = x[batch_idx, dim_idx]
                else:
                    x_frag[i] = T.Cast(dtype, 0)

            # Apply ReLU: max(0, x)
            for i in T.Parallel(block_size):
                x_frag[i] = T.max(x_frag[i], T.Cast(dtype, 0))

            # Store result back to global memory
            for i in T.Parallel(block_size):
                global_idx = start_idx + i
                if global_idx < batch_size * dim:
                    batch_idx = global_idx // dim
                    dim_idx = global_idx % dim
                    output[batch_idx, dim_idx] = x_frag[i]

    return main


def tilelang_relu(x):
    """
    Performs ReLU activation using TileLang for tensors of any shape.

    Args:
        x: Input tensor of any shape

    Returns:
        Result of ReLU activation max(0, x), same shape as input
    """
    x = x.cuda().half()

    # Store original shape
    batch_size, dim = x.shape

    # Create and JIT compile the TileLang kernel
    kernel = relu_kernel(batch_size, dim)

    # Execute the kernel
    return kernel(x)


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation using TileLang.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor using TileLang.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return tilelang_relu(x)
