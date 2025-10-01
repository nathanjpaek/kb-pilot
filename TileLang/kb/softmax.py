import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def softmax_kernel(batch_size, dim, dtype="float16", accum_dtype="float32"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        x: T.Tensor((batch_size, dim), dtype),
        output: T.Tensor((batch_size, dim), dtype),
    ):
        # One block per row, with 'dim' threads to cover all features
        with T.Kernel(batch_size, threads=dim) as by:
            # Fragment to hold one full row
            row = T.alloc_fragment((dim,), dtype)
            # Scalar accumulators for max and sum
            max_val = T.alloc_fragment((1,), accum_dtype)
            sum_val = T.alloc_fragment((1,), accum_dtype)

            # Load the row into the fragment
            for i in T.Parallel(dim):
                row[i] = x[by, i]

            # Compute maximum for numerical stability
            # Initialize
            max_val[0] = T.Cast(accum_dtype, row[0])
            # Reduction over the rest
            for i in T.Serial(1, dim):
                max_val[0] = T.max(max_val[0], T.Cast(accum_dtype, row[i]))

            # Subtract max and exponentiate
            for i in T.Parallel(dim):
                # Cast to accum_dtype for arithmetic
                temp = T.Cast(accum_dtype, row[i]) - max_val[0]
                row[i] = T.exp(temp)  # exp returns accum_dtype or dtype? assume accum_dtype

            # Sum of exponentials
            sum_val[0] = T.Cast(accum_dtype, 0)
            for i in T.Serial(dim):
                sum_val[0] = sum_val[0] + T.Cast(accum_dtype, row[i])

            # Normalize and write back, casting to storage dtype
            for i in T.Parallel(dim):
                out_val = row[i] / sum_val[0]
                output[by, i] = T.Cast(dtype, out_val)

    return main


def tilelang_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Softmax activation using TileLang in a numerically-stable way.
    """
    # Move to GPU and convert to half precision
    x = x.cuda().half()
    batch_size, dim = x.shape

    # JIT-compile the kernel (out_idx=-1 lets it allocate the output tensor)
    kernel = softmax_kernel(batch_size, dim)

    # Execute and return
    return kernel(x)


class ModelNew(nn.Module):
    """
    Model performing Softmax via the TileLang kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tilelang_softmax(x)


def test_softmax():
    # Create test inputs
    batch_size = 1024
    dim = 128
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float16)

    # Create models
    model_new = ModelNew()
    model_base = nn.Softmax(dim=1)

    # Run both implementations
    with torch.no_grad():
        output_new = model_new(x)
        output_base = model_base(x)

    # Compare results
    max_diff = torch.max(torch.abs(output_new - output_base))
    print(f"Maximum difference between implementations: {max_diff}")

    # Verify shapes match
    assert output_new.shape == output_base.shape, "Output shapes don't match"

    return max_diff < 1e-3  # Return True if results are close enough


if __name__ == "__main__":
    test_softmax()
