import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import modal

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

#from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_generate_custom_tilelang_from_prompt_template
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets
from scripts.debug_suggestor_plan import PLAN_2_12, PLAN_3_9, PLAN_3_2
from scripts.tilelang_icl_prompt import ICL_PROMPT
from scripts.tilelang_guideline_prompt import GUIDELINE_PROMPT
from scripts.tilelang_paperinfo_prompt import PAPER_PROMPT


app = modal.App("debug_conv3d")

"""
Debug Conv3D kernel
Simple debugging script for Conv3D TileLang kernel
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}

class DebugConfig(Config):
    def __init__(self):
        # Evaluation
        # local (requires a GPU), modal (cloud GPU) coming soon
        self.eval_mode = "modal"
        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu = "H100"
        self.gpu_arch = ['Hopper']
        
        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = True

        self.log = False
        self.log_prompt = False
        self.log_generated_kernel = False
        self.log_eval_result = False

    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"DebugConfig({self.to_dict()})"

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git",
                "gcc-10",
                "g++-10",
                "clang" # note i skip a step 
                )
    .pip_install(  # required to build flash-attn
        "anthropic",
        "numpy",
        "openai",
        "packaging",
        "pydra_config",
        "torch==2.5.0",
        "tqdm",
        "datasets",
        "transformers",
        "google-generativeai",
        "together",
        "pytest",
        "ninja",
        "utils",
        "python-dotenv", # NATHAN ADDED THIS LINE 
        "tilelang",
        "apache-tvm",
    )
    .add_local_python_source("scripts", "src")  # Add local Python modules
)

@app.cls(image=image)
class Conv3DDebugger:

    @modal.method()
    def debug_conv3d_kernel(self, verbose, gpu_arch):
        """Debug the Conv3D kernel by comparing with PyTorch implementation."""
        
        # SET DEFAULT DTYPE TO FLOAT16 AT THE VERY BEGINNING OF MODAL FUNCTION
        torch.set_default_dtype(torch.float16)
        
        # Import everything we need inside the Modal function
        import math
        import tilelang
        import tilelang.language as T
        import torch.nn as nn
        from src.utils import set_gpu_arch
        
        # Set GPU architecture like in the original script
        set_gpu_arch(gpu_arch)
        
        # Define the kernel code directly here - using the exact code from 1_54.py
        def conv3d_kernel(
            N,
            C,
            D,
            H,
            W,
            F,
            K,
            stride,
            padding,
            dilation,
            block_M=128,
            block_N=128,
            block_K=32,
            threads=128,
            dtype="float16",
            accum_dtype="float",
        ):
            OD = (D + 2 * padding - dilation * (K - 1) - 1) // stride + 1
            OH = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
            OW = (W + 2 * padding - dilation * (K - 1) - 1) // stride + 1
            KK = K * K * K * C

            @tilelang.jit(out_idx=-1)
            @T.prim_func
            def main(
                inp: T.Tensor((N, D, H, W, C), dtype),
                ker: T.Tensor((K, K, K, C, F), dtype),
                out: T.Tensor((N, OD, OH, OW, F), dtype),
            ):
                with T.Kernel(
                    T.ceildiv(F, block_N),
                    T.ceildiv(N * OD * OH * OW, block_M),
                    threads=threads,
                ) as (bx, by):
                    a_shared = T.alloc_shared((block_M, block_K), dtype)
                    b_shared = T.alloc_shared((block_K, block_N), dtype)
                    c_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                    ker_flat = T.Tensor((KK, F), dtype, ker.data)
                    out_flat = T.Tensor((N * OD * OH * OW, F), dtype, out.data)

                    T.clear(c_local)

                    num_k_tiles = T.ceildiv(KK, block_K)
                    for k_iter in T.Pipelined(num_k_tiles, num_stages=3):
                        # Clear shared memory first
                        for i, j in T.Parallel(block_M, block_K):
                            a_shared[i, j] = T.cast(0, dtype)
                        
                        # Load input tile with boundary checks
                        for i, j in T.Parallel(block_M, block_K):
                            k_idx = k_iter * block_K + j
                            m_idx = by * block_M + i

                            if m_idx < N * OD * OH * OW and k_idx < KK:
                                n_idx = m_idx // (OD * OH * OW)
                                rem_1 = m_idx % (OD * OH * OW)
                                od_idx = rem_1 // (OH * OW)
                                rem_2 = rem_1 % (OH * OW)
                                oh_idx = rem_2 // OW
                                ow_idx = rem_2 % OW

                                c_idx = k_idx % C
                                kw_idx = (k_idx // C) % K
                                kh_idx = (k_idx // (C * K)) % K
                                kd_idx = k_idx // (C * K * K)

                                in_d = od_idx * stride - padding + kd_idx * dilation
                                in_h = oh_idx * stride - padding + kh_idx * dilation
                                in_w = ow_idx * stride - padding + kw_idx * dilation

                                if (in_d >= 0) and (in_d < D) and (in_h >= 0) and (in_h < H) and (in_w >= 0) and (in_w < W):
                                    a_shared[i, j] = inp[n_idx, in_d, in_h, in_w, c_idx]

                        # Load kernel tile with boundary checks
                        for i, j in T.Parallel(block_K, block_N):
                            k_idx = k_iter * block_K + i
                            f_idx = bx * block_N + j
                            if k_idx < KK and f_idx < F:
                                b_shared[i, j] = ker_flat[k_idx, f_idx]
                            else:
                                b_shared[i, j] = T.cast(0, dtype)

                        T.gemm(a_shared, b_shared, c_local)

                    # Store output with boundary checks
                    for i, j in T.Parallel(block_M, block_N):
                        m_idx = by * block_M + i
                        f_idx = bx * block_N + j
                        if m_idx < N * OD * OH * OW and f_idx < F:
                            out_flat[m_idx, f_idx] = c_local[i, j]

            return main

        class ModelNew(nn.Module):
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
                assert groups == 1, "Grouped convolution not supported in this TileLang implementation."

                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.bias_flag = bias

                # Initialize weights exactly like PyTorch Conv3d
                weight_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
                self.weight = nn.Parameter(torch.empty(weight_shape))
                torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

                if bias:
                    # Initialize bias exactly like PyTorch Conv3d
                    fan_in = in_channels * kernel_size * kernel_size * kernel_size
                    bound = 1 / math.sqrt(fan_in)
                    self.bias = nn.Parameter(torch.empty(out_channels))
                    torch.nn.init.uniform_(self.bias, -bound, bound)
                else:
                    self.register_parameter("bias", None)

                self._kernel_cache = {}

            def _get_kernel(self, shapes, dtype):
                key = (*shapes, dtype)
                if key not in self._kernel_cache:
                    N, C, D, H, W, F, K, stride, padding, dilation = shapes
                    kernel = conv3d_kernel(
                        N,
                        C,
                        D,
                        H,
                        W,
                        F,
                        K,
                        stride,
                        padding,
                        dilation,
                    )
                    self._kernel_cache[key] = kernel
                return self._kernel_cache[key]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.to(device="cuda", dtype=torch.float16)
                N, C, D, H, W = x.shape
                K = self.kernel_size
                F = self.out_channels

                # Permute input from NCDHW to NDHWC
                inp_t = x.permute(0, 2, 3, 4, 1).contiguous()
                
                # Permute weight from (F, C, K, K, K) to (K, K, K, C, F)
                ker_t = (
                    self.weight.to(device="cuda", dtype=torch.float16)
                    .permute(2, 3, 4, 1, 0)
                    .contiguous()
                )

                kernel_fn = self._get_kernel(
                    (N, C, D, H, W, F, K, self.stride, self.padding, self.dilation),
                    inp_t.dtype,
                )

                out_t = kernel_fn(inp_t, ker_t)
                
                # Permute output from (N, OD, OH, OW, F) to (N, F, OD, OH, OW)
                out_t = out_t.permute(0, 4, 1, 2, 3).contiguous()

                if self.bias is not None:
                    bias_cuda = self.bias.to(device="cuda", dtype=torch.float16)
                    out_t = out_t + bias_cuda.view(1, -1, 1, 1, 1)

                return out_t.to(torch.float32)

        print("=== Conv3D Kernel Debug Session ===")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Test configuration - Intermediate size to find where it breaks
        batch_size = 16
        in_channels = 3
        out_channels = 64
        kernel_size = 3
        depth = 16
        height = 16
        width = 16
        stride = 1
        padding = 1
        dilation = 1
        
        print(f"\nTest Configuration:")
        print(f"Input shape: ({batch_size}, {in_channels}, {depth}, {height}, {width})")
        print(f"Kernel size: {kernel_size}, Stride: {stride}, Padding: {padding}, Dilation: {dilation}")
        print(f"Output channels: {out_channels}")
        
        # Create test inputs with fixed seed for reproducibility
        torch.manual_seed(42)
        x = torch.randn(
            batch_size, in_channels, depth, height, width,
            dtype=torch.float16, device="cuda"
        )
        weight = torch.randn(
            out_channels, in_channels, kernel_size, kernel_size, kernel_size,
            dtype=torch.float16, device="cuda"
        )
        
        print(f"\nInput tensor stats:")
        print(f"  Min: {x.min().item():.6f}, Max: {x.max().item():.6f}")
        print(f"  Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")
        
        print(f"\nWeight tensor stats:")
        print(f"  Min: {weight.min().item():.6f}, Max: {weight.max().item():.6f}")
        print(f"  Mean: {weight.mean().item():.6f}, Std: {weight.std().item():.6f}")
        
        # Create PyTorch reference
        torch_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        ).to(device="cuda", dtype=torch.float16)
        
        # Copy weights to PyTorch layer
        torch_conv.weight.data = weight.clone()
        
        # Run PyTorch implementation
        with torch.no_grad():
            torch_out = torch_conv(x)
        
        print(f"\nPyTorch output stats:")
        print(f"  Shape: {torch_out.shape}")
        print(f"  Min: {torch_out.min().item():.6f}, Max: {torch_out.max().item():.6f}")
        print(f"  Mean: {torch_out.mean().item():.6f}, Std: {torch_out.std().item():.6f}")
        
        # Create and run our implementation
        print(f"\n=== Creating TileLang Model ===")
        try:
            model = ModelNew(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False
            ).to(device="cuda")
            print("✓ TileLang model created successfully")
            
            # Copy weights to our model
            model.weight.data = weight.clone()
            print("✓ Weights copied to TileLang model")
            
            # Run our implementation
            print("🚀 Running TileLang forward pass...")
            with torch.no_grad():
                our_out = model(x)
            print("✓ TileLang forward pass completed")
            
        except Exception as e:
            print(f"❌ Error in TileLang execution: {str(e)}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            
            # Return early with error info
            return {
                "correct": False,
                "max_difference": -1,
                "avg_difference": -1,
                "max_rel_difference": -1,
                "avg_rel_difference": -1,
                "error": str(e),
                "error_type": type(e).__name__
            }

        # Convert both outputs to float32 for comparison to avoid dtype mismatch
        torch_out = torch_out.to(torch.float32)
        our_out = our_out.to(torch.float32)
        
        print(f"\nOur TileLang output stats:")
        print(f"  Shape: {our_out.shape}")
        print(f"  Min: {our_out.min().item():.6f}, Max: {our_out.max().item():.6f}")
        print(f"  Mean: {our_out.mean().item():.6f}, Std: {our_out.std().item():.6f}")
        
        # Calculate differences
        diff = torch.abs(torch_out - our_out)
        rel_diff = diff / (torch.abs(torch_out) + 1e-8)  # avoid division by zero
        
        max_diff = diff.max().item()
        avg_diff = diff.mean().item()
        max_rel_diff = rel_diff.max().item()
        avg_rel_diff = rel_diff.mean().item()
        
        print(f"\nDifference Analysis:")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Average absolute difference: {avg_diff:.6f}")
        print(f"  Max relative difference: {max_rel_diff:.6f}")
        print(f"  Average relative difference: {avg_rel_diff:.6f}")
        
        # Check correctness with TileLang tolerances
        atol = 1.0
        rtol = 0.1
        is_correct = torch.allclose(torch_out, our_out, atol=atol, rtol=rtol)
        print(f"  Correct (atol={atol}, rtol={rtol}): {is_correct}")
        
        # Print some sample values for manual inspection
        print(f"\nSample Values Comparison:")
        print("PyTorch vs TileLang vs Difference:")
        
        # Flatten for easier comparison
        torch_flat = torch_out.flatten()
        our_flat = our_out.flatten()
        diff_flat = diff.flatten()
        
        # Print first 10 values
        print("First 10 values:")
        for i in range(min(10, len(torch_flat))):
            print(f"  [{i:2d}]: {torch_flat[i].item():8.4f} vs {our_flat[i].item():8.4f} (diff: {diff_flat[i].item():8.4f})")
        
        # Print middle values
        middle_start = len(torch_flat) // 2
        print(f"\nMiddle 10 values (starting at index {middle_start}):")
        for i in range(middle_start, min(middle_start + 10, len(torch_flat))):
            print(f"  [{i:2d}]: {torch_flat[i].item():8.4f} vs {our_flat[i].item():8.4f} (diff: {diff_flat[i].item():8.4f})")
        
        # Print last 10 values
        print(f"\nLast 10 values:")
        for i in range(max(0, len(torch_flat) - 10), len(torch_flat)):
            print(f"  [{i:2d}]: {torch_flat[i].item():8.4f} vs {our_flat[i].item():8.4f} (diff: {diff_flat[i].item():8.4f})")
        
        # Find and print the largest differences
        diff_sorted_indices = torch.argsort(diff_flat, descending=True)
        print(f"\nTop 10 largest differences:")
        for i in range(min(10, len(diff_sorted_indices))):
            idx = diff_sorted_indices[i].item()
            if diff_flat[idx].item() > 0:  # Only print if there's actually a difference
                print(f"  [idx={idx:6d}]: {torch_flat[idx].item():8.4f} vs {our_flat[idx].item():8.4f} (diff: {diff_flat[idx].item():8.4f})")
        
        # Print statistics about differences
        nonzero_diffs = diff_flat[diff_flat > 1e-6]  # Only consider meaningful differences
        if len(nonzero_diffs) > 0:
            print(f"\nNon-zero differences statistics:")
            print(f"  Number of non-zero differences: {len(nonzero_diffs)}")
            print(f"  Percentage of different values: {100.0 * len(nonzero_diffs) / len(diff_flat):.2f}%")
            print(f"  Min non-zero diff: {nonzero_diffs.min().item():.6f}")
            print(f"  Max non-zero diff: {nonzero_diffs.max().item():.6f}")
            print(f"  Mean non-zero diff: {nonzero_diffs.mean().item():.6f}")
        else:
            print(f"\nNo significant differences found (all < 1e-6)")
        
        # Find locations of maximum differences
        max_diff_idx = torch.argmax(diff)
        max_diff_coords = torch.unravel_index(max_diff_idx, torch_out.shape)
        
        print(f"\nMax difference location: {max_diff_coords}")
        print(f"  PyTorch value: {torch_out[max_diff_coords].item():.6f}")
        print(f"  TileLang value: {our_out[max_diff_coords].item():.6f}")
        print(f"  Difference: {diff[max_diff_coords].item():.6f}")
        
        # Check if there are any NaN or inf values
        torch_has_nan = torch.isnan(torch_out).any()
        our_has_nan = torch.isnan(our_out).any()
        torch_has_inf = torch.isinf(torch_out).any()
        our_has_inf = torch.isinf(our_out).any()
        
        print(f"\nNaN/Inf Check:")
        print(f"  PyTorch has NaN: {torch_has_nan}, has Inf: {torch_has_inf}")
        print(f"  TileLang has NaN: {our_has_nan}, has Inf: {our_has_inf}")
        
        return {
            "correct": is_correct,
            "max_difference": max_diff,
            "avg_difference": avg_diff,
            "max_rel_difference": max_rel_diff,
            "avg_rel_difference": avg_rel_diff,
            "torch_stats": {
                "min": torch_out.min().item(),
                "max": torch_out.max().item(),
                "mean": torch_out.mean().item(),
                "std": torch_out.std().item()
            },
            "tilelang_stats": {
                "min": our_out.min().item(),
                "max": our_out.max().item(),
                "mean": our_out.mean().item(),
                "std": our_out.std().item()
            }
        }

@pydra.main(base=DebugConfig)
def main(config: DebugConfig):
    """
    Debug Conv3D kernel on Modal
    """
    print(f"Starting Conv3D debug with config: {config}")
    
    print(">>> Setting default dtype to float16 <<<")
    torch.set_default_dtype(torch.float16)

    with app.run():
        debugger = Conv3DDebugger.with_options(gpu=config.gpu)()
        result = debugger.debug_conv3d_kernel.remote(verbose=config.verbose, gpu_arch=gpu_arch_mapping[config.gpu])
        
        print("\n=== Final Summary ===")
        print(f"Kernel correctness: {result['correct']}")
        print(f"Max difference: {result['max_difference']:.6f}")
        print(f"Average difference: {result['avg_difference']:.6f}")
        
        if not result['correct']:
            print("\n🚨 KERNEL FAILED CORRECTNESS CHECK 🚨")
            print("Check the detailed output above for debugging information.")
        else:
            print("\n✅ KERNEL PASSED CORRECTNESS CHECK ✅")

if __name__ == "__main__":
    main() 