import torch
import torch.nn as nn
import torch.nn.functional as F
import tilelang
import tilelang.language as T


def conv_transpose3d_kernel(N, Cin, Cout, Din, Hin, Win, 
                           Kd, Kh, Kw, Sd, Sh, Sw, Pd, Ph, Pw, Od, Oh, Ow, groups,
                           block_M=128, block_N=128, block_K=32,
                           dtype="float16", accum_dtype="float"):
    # Corrected output dimension calculations
    Dout = (Din - 1) * Sd - 2 * Pd + Kd + Od
    Hout = (Hin - 1) * Sh - 2 * Ph + Kh + Oh
    Wout = (Win - 1) * Sw - 2 * Pw + Kw + Ow
    
    Cin_per_g = Cin // groups
    Cout_per_g = Cout // groups

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        data: T.Tensor((N, Din, Hin, Win, Cin), dtype),
        kernel: T.Tensor((Cin, Cout_per_g, Kd, Kh, Kw), dtype),
        bias: T.Tensor((Cout,), dtype),
        out: T.Tensor((N, Dout, Hout, Wout, Cout), dtype),
    ):
        # 2D grid: spatial tiles x output channels
        with T.Kernel(
            T.ceildiv(N * Dout * Hout * Wout, block_M), 
            groups * T.ceildiv(Cout_per_g, block_N),  # Group-aware grid
            threads=128) as (bx, by):
            
            # Thread indices
            tx = T.get_thread_binding(0)
            ty = T.get_thread_binding(1)
            
            # Group handling - critical fix
            total_blocks_per_group = T.ceildiv(Cout_per_g, block_N)
            group_idx = by // total_blocks_per_group
            block_in_group = by % total_blocks_per_group
            cout_start = group_idx * Cout_per_g + block_in_group * block_N
            
            # Shared memory allocation
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            C_acc = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            # Layout optimization
            T.annotate_layout({
                data_shared: tilelang.layout.make_swizzled_layout(data_shared),
                kernel_shared: tilelang.layout.make_swizzled_layout(kernel_shared),
            })
            
            T.clear(C_acc)
            
            # K dimension (Cin_per_g * Kd * Kh * Kw)
            for k_iter in T.serial(T.ceildiv(Cin_per_g * Kd * Kh * Kw, block_K)):
                # Load data tile with correct group handling
                for i, j in T.Parallel(block_M, block_K):
                    m_idx = bx * block_M + i
                    k_idx = k_iter * block_K + j
                    
                    if m_idx < N * Dout * Hout * Wout and k_idx < Cin_per_g * Kd * Kh * Kw:
                        # Decode output position
                        batch_idx = m_idx // (Dout * Hout * Wout)
                        spatial_idx = m_idx % (Dout * Hout * Wout)
                        d_out = spatial_idx // (Hout * Wout)
                        h_out = (spatial_idx % (Hout * Wout)) // Wout
                        w_out = spatial_idx % Wout
                        
                        # Decode kernel position
                        cin_local_idx = k_idx // (Kd * Kh * Kw)
                        kernel_pos = k_idx % (Kd * Kh * Kw)
                        kd = kernel_pos // (Kh * Kw)
                        kh = (kernel_pos % (Kh * Kw)) // Kw
                        kw = kernel_pos % Kw
                        
                        # Correct input position calculation
                        d_in = (d_out - kd + Pd) // Sd
                        h_in = (h_out - kh + Ph) // Sh
                        w_in = (w_out - kw + Pw) // Sw
                        
                        # Check if input position is valid
                        valid = (
                            (d_out - kd + Pd) % Sd == 0 and
                            (h_out - kh + Ph) % Sh == 0 and
                            (w_out - kw + Pw) % Sw == 0 and
                            d_in >= 0 and d_in < Din and
                            h_in >= 0 and h_in < Hin and
                            w_in >= 0 and w_in < Win
                        )
                        
                        # Global input channel with group handling
                        cin_global = group_idx * Cin_per_g + cin_local_idx
                        
                        if valid and cin_global < Cin:
                            data_shared[i, j] = data[batch_idx, d_in, h_in, w_in, cin_global]
                        else:
                            data_shared[i, j] = 0.0
                    else:
                        data_shared[i, j] = 0.0
                
                # Load kernel weights with group handling
                for k, n in T.Parallel(block_K, block_N):
                    k_idx = k_iter * block_K + k
                    cout_idx = cout_start + n
                    
                    if k_idx < Cin_per_g * Kd * Kh * Kw and cout_idx < Cout:
                        # Decode kernel indices
                        cin_local_idx = k_idx // (Kd * Kh * Kw)
                        kernel_pos = k_idx % (Kd * Kh * Kw)
                        kd = kernel_pos // (Kh * Kw)
                        kh = (kernel_pos % (Kh * Kw)) // Kw
                        kw = kernel_pos % Kw
                        
                        # Global input channel with group handling
                        cin_global = group_idx * Cin_per_g + cin_local_idx
                        cout_in_group = cout_idx - group_idx * Cout_per_g
                        
                        if cin_global < Cin and cout_in_group < Cout_per_g:
                            kernel_shared[k, n] = kernel[cin_global, cout_in_group, kd, kh, kw]
                        else:
                            kernel_shared[k, n] = 0.0
                    else:
                        kernel_shared[k, n] = 0.0
                
                # Synchronize before GEMM
                T.tvm_storage_sync("shared")
                
                # Compute partial GEMM
                C_partial = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_partial)
                T.gemm(data_shared, kernel_shared, C_partial)
                
                # Accumulate results
                for i, j in T.Parallel(block_M, block_N):
                    C_acc[i, j] += C_partial[i, j]
                
                # Synchronize after GEMM
                T.tvm_storage_sync("shared")
            
            # Write results back with bias addition
            for i, j in T.Parallel(block_M, block_N):
                m_idx = bx * block_M + i
                cout_idx = cout_start + j
                
                if m_idx < N * Dout * Hout * Wout and cout_idx < Cout:
                    # Decode output position
                    batch_idx = m_idx // (Dout * Hout * Wout)
                    spatial_idx = m_idx % (Dout * Hout * Wout)
                    d_out = spatial_idx // (Hout * Wout)
                    h_out = (spatial_idx % (Hout * Wout)) // Wout
                    w_out = spatial_idx % Wout
                    
                    # Add bias and store
                    result = C_acc[i, j] + T.Cast(accum_dtype, bias[cout_idx])
                    out[batch_idx, d_out, h_out, w_out, cout_idx] = T.Cast(dtype, result)

    return main


class ModelNew(nn.Module):
    """
    Corrected 3D transposed convolution using TileLang with proper accumulation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize parameters exactly like PyTorch ConvTranspose3d
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        
        # Use PyTorch's default initialization
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            fan_in = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)
        
        self._kernel_cache = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert from PyTorch format (N, C, D, H, W) to TileLang format (N, D, H, W, C)
        x_tl = x.permute(0, 2, 3, 4, 1).contiguous()
        
        # Move to CUDA and convert to fp16
        x16 = x_tl.to(device="cuda", dtype=torch.float16)
        w16 = self.weight.to(device="cuda", dtype=torch.float16)
        
        if self.bias is None:
            b16 = torch.zeros(self.out_channels, device="cuda", dtype=torch.float16)
        else:
            b16 = self.bias.to(device="cuda", dtype=torch.float16)
        
        N, Din, Hin, Win, Cin = x16.shape
        Kd, Kh, Kw = self.kernel_size
        Sd, Sh, Sw = self.stride
        Pd, Ph, Pw = self.padding
        Od, Oh, Ow = self.output_padding
        
        # Cache kernels by input shape
        key = (N, Din, Hin, Win)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = conv_transpose3d_kernel(
                N, Cin, self.out_channels, Din, Hin, Win, 
                Kd, Kh, Kw, Sd, Sh, Sw, Pd, Ph, Pw, Od, Oh, Ow, self.groups
            )
        
        # Execute kernel
        out16 = self._kernel_cache[key](x16, w16, b16)
        
        # Convert back to PyTorch format (N, D, H, W, C) -> (N, C, D, H, W)
        out_torch = out16.permute(0, 4, 1, 2, 3)
        return out_torch.to(torch.float32)
