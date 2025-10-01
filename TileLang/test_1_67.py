import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def conv1d_gemm_kernel(
    batch: int,
    cin: int,
    lin: int,
    cout: int,
    ksize: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    lout = (lin + 2 * padding - dilation * (ksize - 1) - 1) // stride + 1
    m_total = batch * lout
    k_total = cin * ksize

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv1d_gemm(
        X: T.Tensor((batch, cin, lin), dtype),
        W: T.Tensor((cout, cin, ksize), dtype),
        O: T.Tensor((batch, cout, lout), dtype),
    ):
        with T.Kernel(
            T.ceildiv(cout, block_N),
            T.ceildiv(m_total, block_M),
            threads=128,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            O_shared = T.alloc_shared((block_M, block_N), dtype)

            O_flat = T.Tensor((m_total, cout), dtype, O.data)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(k_total, block_K), num_stages=3):
                # Load input tiles to shared memory (im2col on-the-fly)
                for i, j in T.Parallel(block_M, block_K):
                    m_idx = by * block_M + i
                    k_idx = ko * block_K + j
                    valid = (m_idx < m_total) and (k_idx < k_total)
                    n_idx = m_idx // lout
                    l_out = m_idx % lout
                    cin_idx = k_idx % cin
                    kpos = k_idx // cin
                    l_in = l_out * stride - padding + kpos * dilation
                    in_range = (l_in >= 0) and (l_in < lin)
                    A_shared[i, j] = T.if_then_else(
                        valid and in_range, X[n_idx, cin_idx, l_in], 0
                    )

                # Load weight tiles to shared memory
                for i, j in T.Parallel(block_K, block_N):
                    k_idx = ko * block_K + i
                    cout_idx = bx * block_N + j
                    valid = (k_idx < k_total) and (cout_idx < cout)
                    cin_idx = k_idx % cin
                    kpos = k_idx // cin
                    B_shared[i, j] = T.if_then_else(
                        valid, W[cout_idx, cin_idx, kpos], 0
                    )

                # GEMM
                T.gemm(A_shared, B_shared, C_local)

            # Write back results
            T.copy(C_local, O_shared)
            T.copy(O_shared, O_flat[by * block_M, bx * block_N])

    return conv1d_gemm


class ModelNew(nn.Module):
    """
    Optimized 1D Convolution using TileLang.
    Currently supports stride=1, padding=0, dilation=1, groups=1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        assert groups == 1, "Grouped conv not supported in this kernel"
        assert stride == 1 and padding == 0 and dilation == 1, "Only stride=1, padding=0, dilation=1 supported"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            bound = 1 / math.sqrt(in_channels * kernel_size)
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        self._kernel_cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, in_channels, length)
        Returns:
            Tensor of shape (batch, out_channels, length_out)
        """
        batch, _, lin = x.shape
        ksize = self.kernel_size
        lout = lin - ksize + 1  # given stride=1, padding=0, dilation=1

        key = (batch, lin, x.dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = conv1d_gemm_kernel(
                batch=batch,
                cin=self.in_channels,
                lin=lin,
                cout=self.out_channels,
                ksize=ksize,
            )

        kernel = self._kernel_cache[key]

        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)

        out_fp16 = kernel(x_fp16, w_fp16)

        if self.bias is not None:
            out_fp16 += self.bias.to(out_fp16.device, out_fp16.dtype).view(1, -1, 1)

        return out_fp16.to(x.dtype)


class Model(nn.Module):
    """
    Performs a standard 1D convolution operation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d(x)


def test_correctness():
    # Test parameters
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    length = 512
    stride = 1
    padding = 0
    dilation = 1

    # Create input tensor
    x = torch.randn(batch_size, in_channels, length, device='cuda')

    # Initialize both models
    baseline_model = Model(in_channels, out_channels, kernel_size, stride, padding, dilation)
    baseline_model = baseline_model.cuda()
    
    tilelang_model = ModelNew(in_channels, out_channels, kernel_size, stride, padding, dilation)
    tilelang_model = tilelang_model.cuda()

    # Copy weights from baseline to TileLang model to ensure fair comparison
    tilelang_model.weight.data.copy_(baseline_model.conv1d.weight.data)
    if baseline_model.conv1d.bias is not None:
        tilelang_model.bias.data.copy_(baseline_model.conv1d.bias.data)

    # Run both models
    with torch.no_grad():
        baseline_out = baseline_model(x)
        tilelang_out = tilelang_model(x)

    # Convert both outputs to float32 for comparison
    baseline_out = baseline_out.float()
    tilelang_out = tilelang_out.float()

    # Calculate differences
    abs_diff = torch.abs(baseline_out - tilelang_out)
    max_diff = abs_diff.max().item()
    avg_diff = abs_diff.mean().item()
    std_diff = abs_diff.std().item()
    
    # Calculate percentage of elements that differ by more than 1e-3
    threshold = 1e-3
    num_different = (abs_diff > threshold).sum().item()
    total_elements = baseline_out.numel()
    percent_different = (num_different / total_elements) * 100

    # Print statistics
    print("\nCorrectness Test Results:")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Average absolute difference: {avg_diff:.6f}")
    print(f"Standard deviation of differences: {std_diff:.6f}")
    print(f"Percentage of elements differing by > {threshold}: {percent_different:.2f}%")
    
    # Print shape information
    print("\nShape Information:")
    print(f"Baseline output shape: {baseline_out.shape}")
    print(f"TileLang output shape: {tilelang_out.shape}")
    
    # Print value ranges
    print("\nValue Ranges:")
    print(f"Baseline output range: [{baseline_out.min().item():.6f}, {baseline_out.max().item():.6f}]")
    print(f"TileLang output range: [{tilelang_out.min().item():.6f}, {tilelang_out.max().item():.6f}]")

    # Additional analysis for 1D conv specific patterns
    print("\nPosition-wise Analysis:")
    # Check if differences are concentrated at the start/end of the sequence
    start_diff = abs_diff[:, :, :10].mean().item()
    end_diff = abs_diff[:, :, -10:].mean().item()
    middle_diff = abs_diff[:, :, 10:-10].mean().item()
    print(f"Average difference at start (first 10 positions): {start_diff:.6f}")
    print(f"Average difference at end (last 10 positions): {end_diff:.6f}")
    print(f"Average difference in middle: {middle_diff:.6f}")


if __name__ == "__main__":
    test_correctness()