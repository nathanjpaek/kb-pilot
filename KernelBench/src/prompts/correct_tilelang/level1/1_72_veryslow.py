"""
Problem Name: 72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1920.0 runtime_stats={'mean': 1920.0, 'std': 0.142, 'min': 1920.0, 'max': 1920.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.62, 'std': 0.0321, 'min': 2.55, 'max': 2.72, 'num_trials': 100}, 'speedup_ratio': 0.00136}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), 
                 padding=(0,0,0), output_padding=(0,0,0), groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weights and bias like PyTorch
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            fan_in = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.compiled_kernels = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to CUDA fp16
        x = x.contiguous().to(device="cuda", dtype=torch.float16)
        weight = self.weight.contiguous().to(device="cuda", dtype=torch.float16)
        if self.bias is None:
            bias = torch.zeros(self.out_channels, device="cuda", dtype=torch.float16)
        else:
            bias = self.bias.contiguous().to(device="cuda", dtype=torch.float16)
        
        # Calculate output dimensions
        B, C_in, D_in, H_in, W_in = x.shape
        Kd, Kh, Kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        opd, oph, opw = self.output_padding
        
        D_out = (D_in - 1) * sd + Kd - 2 * pd + opd
        H_out = (H_in - 1) * sh + Kh - 2 * ph + oph
        W_out = (W_in - 1) * sw + Kw - 2 * pw + opw
        
        # Get or compile kernel
        key = (B, C_in, self.out_channels, D_in, H_in, W_in, Kd, Kh, Kw)
        if key not in self.compiled_kernels:
            self.compiled_kernels[key] = self._build_kernel(B, C_in, self.out_channels, 
                                                          D_in, H_in, W_in, D_out, H_out, W_out,
                                                          Kd, Kh, Kw, sd, sh, sw, pd, ph, pw)
            
        result = self.compiled_kernels[key](x, weight, bias)
        return result.to(torch.float32)

    def _build_kernel(self, B, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
                     Kd, Kh, Kw, sd, sh, sw, pd, ph, pw):
        
        groups = self.groups
        C_in_per_group = C_in // groups
        C_out_per_group = C_out // groups
        
        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def conv_transpose_3d(
            input: T.Tensor((B, C_in, D_in, H_in, W_in), "float16"),
            weight: T.Tensor((C_in, C_out_per_group, Kd, Kh, Kw), "float16"),
            bias: T.Tensor((C_out,), "float16"),
            output: T.Tensor((B, C_out, D_out, H_out, W_out), "float16")
        ):
            with T.Kernel(D_out * H_out * W_out, C_out, B, threads=64) as (bx, by, bz):
                # Decode spatial coordinates
                d_out = bx // (H_out * W_out)
                temp = bx % (H_out * W_out)
                h_out = temp // W_out
                w_out = temp % W_out
                
                c_out = by
                batch = bz
                
                # Group calculations
                group_id = c_out // C_out_per_group
                c_out_in_group = c_out % C_out_per_group
                
                # Use local memory instead of fragment to avoid layout inference issues
                accum = T.alloc_local((1,), "float32")
                T.clear(accum)
                
                # Iterate through input channels in this group
                for c_in_offset in T.serial(C_in_per_group):
                    c_in = group_id * C_in_per_group + c_in_offset
                    
                    # Iterate through kernel dimensions
                    for kd in T.serial(Kd):
                        for kh in T.serial(Kh):
                            for kw in T.serial(Kw):
                                # Calculate input coordinates for transposed conv
                                d_in_coord = d_out + pd - kd
                                h_in_coord = h_out + ph - kh
                                w_in_coord = w_out + pw - kw
                                
                                # Check stride alignment and bounds
                                stride_aligned = (d_in_coord % sd == 0 and h_in_coord % sh == 0 and w_in_coord % sw == 0)
                                
                                if stride_aligned:
                                    d_in = d_in_coord // sd
                                    h_in = h_in_coord // sh
                                    w_in = w_in_coord // sw
                                    
                                    bounds_check = (d_in >= 0 and d_in < D_in and 
                                                  h_in >= 0 and h_in < H_in and 
                                                  w_in >= 0 and w_in < W_in)
                                    
                                    if bounds_check:
                                        input_val = input[batch, c_in, d_in, h_in, w_in].astype("float32")
                                        weight_val = weight[c_in, c_out_in_group, kd, kh, kw].astype("float32")
                                        accum[0] += input_val * weight_val
                
                # Add bias and store result
                final_val = accum[0] + bias[c_out].astype("float32")
                output[batch, c_out, d_out, h_out, w_out] = final_val.astype("float16")

        return conv_transpose_3d