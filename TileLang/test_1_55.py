import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


def conv2d_kernel(
    N,
    C,
    H,
    W,
    F,
    K,
    S,
    D,
    P,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    KH = KW = K
    OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        data: T.Tensor((N, H, W, C), dtype),
        kernel: T.Tensor((KH, KW, C, F), dtype),
        out: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N),
            T.ceildiv(N * OH * OW, block_M),
            threads=threads,
        ) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            kernel_flat = T.Tensor((KH * KW * C, F), dtype, kernel.data)
            out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)

            T.clear(out_local)

            k_tiles = T.ceildiv(KH * KW * C, block_K)
            for k_iter in T.Pipelined(k_tiles, num_stages=num_stages):
                # ---------------- Im2Col copy ----------------
                for i, j in T.Parallel(block_M, block_K):
                    k_idx = k_iter * block_K + j
                    m_idx = by * block_M + i

                    valid = (m_idx < N * OH * OW) and (k_idx < KH * KW * C)

                    n_idx = m_idx // (OH * OW)
                    oh_idx = (m_idx % (OH * OW)) // OW
                    ow_idx = m_idx % OW

                    kh_idx = k_idx // (KW * C)
                    kw_idx = (k_idx // C) % KW
                    c_idx = k_idx % C

                    h_in = oh_idx * S + kh_idx * D - P
                    w_in = ow_idx * S + kw_idx * D - P

                    in_bound = (
                        (h_in >= 0)
                        and (h_in < H)
                        and (w_in >= 0)
                        and (w_in < W)
                    )

                    data_shared[i, j] = T.if_then_else(
                        valid and in_bound,
                        data[n_idx, h_in, w_in, c_idx],
                        T.Cast(dtype, 0),
                    )
                # -------------- Kernel copy ------------------
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                # -------------- GEMM -------------------------
                T.gemm(data_shared, kernel_shared, out_local)

            # Store results
            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    """
    Optimized Conv2d implementation using TileLang kernels.
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
        if groups != 1:
            raise NotImplementedError("Grouped convolution not supported.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias

        # Weight initialization identical to nn.Conv2d
        weight = torch.empty(
            out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32
        )
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        if bias:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            bias_param = torch.empty(out_channels, dtype=torch.float32)
            torch.nn.init.uniform_(bias_param, -bound, bound)
            self.bias = nn.Parameter(bias_param)
        else:
            self.register_parameter("bias", None)

        # Cache to store compiled kernels for different input shapes
        self._kernel_cache = {}

    def _fetch_kernel(self, N: int, H: int, W: int, dtype=torch.float16):
        key = (N, H, W, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = conv2d_kernel(
                N,
                self.in_channels,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.dilation,
                self.padding,
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prepare data
        x = x.to(device="cuda", dtype=torch.float16)
        w = self.weight.to(device="cuda", dtype=torch.float16)

        N, C, H, W = x.shape

        # Get / compile kernel
        kernel_fn = self._fetch_kernel(N, H, W)

        # Reorder tensors to NHWC / HWIO
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_hwio = w.permute(2, 3, 1, 0).contiguous()

        # Execute kernel
        out_nhwc = kernel_fn(x_nhwc, w_hwio)

        # Convert back to NCHW and float32
        out_nchw = out_nhwc.to(torch.float32).permute(0, 3, 1, 2).contiguous()
        return out_nchw

def test_correctness():
    # Test parameters
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    width = 256
    height = 128  # Asymmetric input
    stride = 1
    padding = 1
    dilation = 1

    # Create input tensor
    x = torch.randn(batch_size, in_channels, height, width, device='cuda')

    # Initialize both models
    baseline_model = Model(in_channels, out_channels, kernel_size, stride, padding, dilation)
    baseline_model = baseline_model.cuda()
    
    tilelang_model = ModelNew(in_channels, out_channels, kernel_size, stride, padding, dilation)
    tilelang_model = tilelang_model.cuda()

    # Copy weights from baseline to TileLang model to ensure fair comparison
    tilelang_model.weight.data.copy_(baseline_model.conv2d.weight.data)
    if baseline_model.conv2d.bias is not None:
        tilelang_model.bias.data.copy_(baseline_model.conv2d.bias.data)

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

if __name__ == "__main__":
    test_correctness()
    

