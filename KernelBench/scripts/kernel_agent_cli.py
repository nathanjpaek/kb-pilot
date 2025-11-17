"""
CLI-based CuTe kernel generation agent.

Workflow:
1. Gather user specification for the desired kernel.
2. Generate candidate DSL code via the enhanced RAG pipeline.
3. Evaluate correctness/performance using the Modal evaluator.
4. Optionally retry with alternative RAG settings.

Run:
    python scripts/kernel_agent_cli.py \
        --language cute \
        --pytorch_ref path/to/reference.py \
        --rag_k 5 \
        --gpu H100

For interactive mode, run:
python scripts/kernel_agent_cli.py --interactive

If arguments are omitted, the script will prompt interactively.
"""

import argparse
import ast
import json
import os
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
SRC_DIR = PARENT_DIR / "src"

for path in (CURRENT_DIR, PARENT_DIR, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

# Ensure latest compatible versions of h2/grpclib for Modal
def _ensure_compatible_versions():
    """Ensure h2 and grpclib are at compatible versions for Modal"""
    try:
        import h2
        import grpclib
        # Upgrade to latest compatible versions
        if h2.__version__ < "4.3.0" or grpclib.__version__ < "0.4.8":
            import subprocess
            print(f"⚠️  Upgrading h2 ({h2.__version__}) and grpclib ({grpclib.__version__}) to latest compatible versions...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", "h2>=4.3.0", "grpclib>=0.4.8"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Reload modules
            import importlib
            importlib.reload(h2)
            importlib.reload(grpclib)
            print(f"✅ h2: {h2.__version__}, grpclib: {grpclib.__version__}")
    except Exception:
        pass  # Silently fail - packages might not be installed yet

_ensure_compatible_versions()

def _patch_h2_connection():
    """
    Patch H2Connection._receive_frame to ensure _frame_dispatch_table is always present.
    
    This is CRITICAL for Modal compatibility when creating parallel connections.
    If this patch fails, Modal will crash with AttributeError.
    """
    try:
        from h2.connection import H2Connection
        from hyperframe.frame import (
            HeadersFrame, PushPromiseFrame, SettingsFrame,
            DataFrame, WindowUpdateFrame, PingFrame,
            RstStreamFrame, GoAwayFrame, ContinuationFrame,
            PriorityFrame, AltSvcFrame, ExtensionFrame
        )
        original_receive_frame = H2Connection._receive_frame
        
        def patched_receive_frame(self, frame):
            if not hasattr(self, '_frame_dispatch_table') or self._frame_dispatch_table is None:
                self._frame_dispatch_table = {
                    HeadersFrame: self._receive_headers_frame,
                    PushPromiseFrame: self._receive_push_promise_frame,
                    SettingsFrame: self._receive_settings_frame,
                    DataFrame: self._receive_data_frame,
                    WindowUpdateFrame: self._receive_window_update_frame,
                    PingFrame: self._receive_ping_frame,
                    RstStreamFrame: self._receive_rst_stream_frame,
                    PriorityFrame: self._receive_priority_frame,
                    GoAwayFrame: self._receive_goaway_frame,
                    ContinuationFrame: self._receive_naked_continuation,
                    AltSvcFrame: self._receive_alt_svc_frame,
                    ExtensionFrame: self._receive_unknown_frame,
                }
            return original_receive_frame(self, frame)
        
        H2Connection._receive_frame = patched_receive_frame
        print("✅ H2Connection patch applied successfully (Modal compatibility fix)")
    except ImportError as e:
        # h2/hyperframe not installed - this is OK if Modal isn't being used
        print(f"⚠️  Warning: Could not apply H2Connection patch (h2 not available): {e}")
        print("   This is OK if you're not using Modal. If using Modal, install: pip install h2>=4.3.0")
    except Exception as e:
        # Other errors - this is more serious
        print(f"❌ ERROR: Failed to apply H2Connection patch: {e}")
        print("   Modal connections may fail with AttributeError. Please report this issue.")
        import traceback
        traceback.print_exc()

_patch_h2_connection()

from generate_and_eval_rag_modal import (
    configure_dspy,
    app as eval_app,
    EvalFunc,
    gpu_arch_mapping,
)
from cute_paperinfo_prompt import CUTE_PAPER_PROMPT
from cute_guideline_prompt import CUTE_GUIDELINE_PROMPT
from tilelang_guideline_prompt import TILELANG_GUIDELINE_PROMPT
from tk_guideline_prompt import TK_GUIDELINE_PROMPT
from tilelang_paperinfo_prompt import TILELANG_PAPER_PROMPT
from tk_paperinfo_prompt import TK_PAPER_PROMPT

from src.prompt_constructor_rag import prompt_generate_custom_dsl_rag_enhanced
from src.utils import extract_all_code_blocks, extract_first_code, read_file
from src.eval import eval_kernel_against_ref
from src.agent_refiner import CuteKernelRefiner, CuteKernelRefinerConfig

GUIDELINE_BY_LANG = {
    "cute": (CUTE_PAPER_PROMPT, CUTE_GUIDELINE_PROMPT),
    "tilelang": (TILELANG_PAPER_PROMPT, TILELANG_GUIDELINE_PROMPT),
    "tk": (TK_PAPER_PROMPT, TK_GUIDELINE_PROMPT),
}

DEFAULT_RETRIES = [
    {"rag_k": 3, "temperature": 1.0},  # Reduced from 5 for faster generation
    {"rag_k": 5, "temperature": 0.2},
    {"rag_k": 8, "temperature": 0.0},
]

FORMAT_HINT = (
    "Output format requirement:\n"
    "- Return exactly one ```python code block containing the complete CuTe kernel.\n"
    "- Include the ModelNew class, any host wrapper, and necessary helpers.\n"
    "- Do not emit prose or explanation outside the fenced code block.\n"
)

CUTE_TEMPLATE_EXAMPLE = """
Reference CuTe kernel structure to mirror:
```python
import math
import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, K: cutlass.Int32):
    # TODO: add tiled GEMM + epilogue logic
    pass

@cute.jit
def fused_kernel_host(A, B, bias, clamp_lo, clamp_hi):
    C = torch.empty_like(A @ B, device=A.device, dtype=A.dtype)
    fused_kernel(from_dlpack(A), from_dlpack(B), from_dlpack(C), cutlass.Int32(A.shape[-1]))
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = fused_kernel_host(x, self.weight, self.bias, cutlass.Float32(-1.0), cutlass.Float32(1.0))
        return y
```
""".strip()


@dataclass
class KernelSpec:
    language: str = "cute"
    gpu: str = "H100"
    rag_k: int = 3  # Reduced from 5 for faster generation
    model_name: str = "openai/o3"
    fast_model: Optional[str] = "openai/gpt-4o"  # Faster model for first attempt
    temperature: float = 1.0
    pytorch_reference: str = ""
    paper_prompt: str = ""
    guideline_prompt: str = ""
    operation_description: str = ""
    ops_sequence: str = ""
    input_shapes: str = ""
    dtype: str = ""
    target_speedup: Optional[float] = None
    additional_constraints: str = ""
    problem_label: str = "custom_kernel"
    ops_list: List[str] = field(default_factory=list)
    current_level: Optional[int] = None
    current_problem_id: Optional[int] = None
    max_attempts: Optional[int] = 3
    measure_performance: bool = True
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    use_modal: bool = True
    enable_refiner: bool = True
    refiner_attempts: int = 2
    refiner_server: str = "openai"
    refiner_model: Optional[str] = "o3"
    refiner_temperature: float = 1.0
    test_time_scaling: bool = True  # Enable test-time scaling (generate multiple candidates)
    num_candidates: int = 4  # Number of candidates to generate (4-8 per proposal)
    syntax_only: bool = False  # Only check Python syntax, skip compilation/evaluation

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class KernelAgentCLI:
    def __init__(self, spec: KernelSpec, retries: Optional[List[dict]] = None):
        self.spec = spec
        self.retries = retries or DEFAULT_RETRIES
        # Don't configure DSPy here - we'll configure it per attempt to use fast_model first

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("\n================ CuTe Kernel Agent ================")
        print(f"Language: {self.spec.language.upper()}  |  GPU: {self.spec.gpu}")
        if self.spec.syntax_only:
            print("⚠️  Syntax-only mode: Will only check Python syntax (no compilation/evaluation)")
        if self.spec.operation_description:
            print(f"Operation: {self.spec.operation_description}")
        print("===================================================\n")

        pytorch_code = self.spec.pytorch_reference
        retries = [
            {"rag_k": self.spec.rag_k, "temperature": self.spec.temperature},
            *self.retries,
        ]
        if self.spec.max_attempts:
            retries = retries[:max(self.spec.max_attempts, 1)]

        best_result = None
        best_strategy = None

        for attempt, strategy in enumerate(retries, start=1):
            # Use fast_model for first attempt, then fall back to main model
            model_to_use = self.spec.fast_model if (attempt == 1 and self.spec.fast_model) else self.spec.model_name
            if attempt == 1 and self.spec.fast_model:
                print(f"⚡ Using fast model ({model_to_use}) for first attempt")
            configure_dspy(model_to_use, strategy.get("temperature", self.spec.temperature))
            
            print(f"Attempt {attempt}: rag_k={strategy['rag_k']} temp={strategy.get('temperature', self.spec.temperature)} model={model_to_use}")
            
            # Test-time scaling: generate multiple candidates with variations
            if self.spec.test_time_scaling and attempt == 1:
                candidates_with_results = self._test_time_scaling(pytorch_code, strategy)
                if candidates_with_results:
                    # Select best candidate: prioritize correctness, then speedup
                    correct_candidates = [c for c in candidates_with_results if c["evaluation"].get("correctness", False)]
                    if correct_candidates:
                        # Among correct candidates, pick highest speedup
                        best_candidate_result = max(
                            correct_candidates,
                            key=lambda x: x["evaluation"].get("runtime_stats", {}).get("performance_comparison", {}).get("speedup_ratio", 0.0)
                        )
                    else:
                        # If none correct, pick first compiled one (for debugging)
                        compiled_candidates = [c for c in candidates_with_results if c["evaluation"].get("compiled", False)]
                        if compiled_candidates:
                            best_candidate_result = compiled_candidates[0]
                        else:
                            best_candidate_result = candidates_with_results[0]  # Best we have
                    if best_candidate_result["evaluation"].get("correctness", False):
                        speedup = best_candidate_result["evaluation"].get("runtime_stats", {}).get("performance_comparison", {}).get("speedup_ratio")
                        speedup_str = f" (speedup={speedup:.2f}×)" if speedup else ""
                        print(f"  ✓ Test-time scaling: Selected best of {len(candidates_with_results)} candidates{speedup_str}")
                        best_result = best_candidate_result
                        best_strategy = strategy
                        break
                    else:
                        print(f"  ✗ Test-time scaling: None of {len(candidates_with_results)} candidates passed correctness")
                        # Continue to fallback single generation
                else:
                    print("  ✗ Test-time scaling: Failed to generate any valid candidates")
            
            # Fallback to single candidate generation
            candidate = self._attempt_generation(pytorch_code, strategy)
            if candidate is None:
                print("  ✗ Generation failed (no valid code extracted)\n")
                continue

            eval_result = self._evaluate_candidate(candidate)
            if eval_result.get("correctness", False):
                print("  ✓ Kernel passes correctness checks!")
                best_result = {"code": candidate, "evaluation": eval_result}
                best_strategy = strategy
                break
            else:
                print("  ✗ Kernel failed correctness. See logs above.")
                refined = self._maybe_refine(candidate, eval_result)
                if refined:
                    best_result = refined
                    best_strategy = strategy
                    break

        if best_result:
            self._summarize_success(best_result, best_strategy or strategy)
            self._offer_save(best_result["code"])
        else:
            print("\n❌ All attempts failed. Consider refining specification or using multiturn optimizer." )

    # ------------------------------------------------------------------
    # Generation & Evaluation
    # ------------------------------------------------------------------

    def _get_variation_hints(self, variation_id: int, total: int) -> str:
        """
        Generate configuration hints for test-time scaling variations.
        Returns a string that will be added to the prompt to guide the LLM.
        """
        # Common tile sizes for different operations
        tile_sizes = [16, 32, 64, 128]
        vector_widths = [4, 8, 16]
        thread_configs = [
            "(256, 1, 1)",  # 1D block
            "(16, 16, 1)",  # 2D square block
            "(32, 8, 1)",   # 2D rectangular block
        ]
        
        # Cycle through variations
        tile_idx = variation_id % len(tile_sizes)
        vec_idx = (variation_id // len(tile_sizes)) % len(vector_widths)
        thread_idx = (variation_id // (len(tile_sizes) * len(vector_widths))) % len(thread_configs)
        
        tile_size = tile_sizes[tile_idx]
        vector_width = vector_widths[vec_idx]
        thread_config = thread_configs[thread_idx]
        
        hints = f"""
Configuration Variation {variation_id + 1} of {total}:

IMPORTANT: Use these specific configuration values in your CuTe kernel:
- Tile size: Use TILE_SIZE = {tile_size} (for 2D tiling) or similar power-of-2 values
- Vector width: Use V = {vector_width} for vectorized memory operations (cute.autovec_copy)
- Thread block dimensions: Use block_dim = {thread_config} when launching the kernel

Example pattern:
```python
TILE_SIZE = {tile_size}
V = {vector_width}

# In kernel launch:
threads_per_block = {thread_config}
num_blocks = ((N + TILE_SIZE - 1) // TILE_SIZE, (M + TILE_SIZE - 1) // TILE_SIZE, 1)
kernel[grid_dim, block_dim](...)
```

Ensure all code is valid Python/CuTe syntax. Include proper imports, ModelNew class, and host wrapper.
"""
        return hints.strip()

    def _test_time_scaling(self, pytorch_code: str, strategy: dict) -> List[Dict]:
        """
        Generate multiple candidates with systematic variations (test-time scaling).
        Returns list of {code, evaluation} dicts.
        """
        num_candidates = min(self.spec.num_candidates, 8)  # Cap at 8 per proposal
        print(f"\n🔄 Test-time scaling: Generating {num_candidates} candidates with variations...")
        
        candidates = []
        
        # Generate candidates with variations
        for i in range(num_candidates):
            variation_hints = self._get_variation_hints(i, num_candidates)
            # Calculate actual values for display
            tile_sizes = [16, 32, 64, 128]
            vector_widths = [4, 8, 16]
            tile_idx = i % len(tile_sizes)
            vec_idx = (i // len(tile_sizes)) % len(vector_widths)
            print(f"  Generating candidate {i+1}/{num_candidates} (tile={tile_sizes[tile_idx]}, vector={vector_widths[vec_idx]})...")
            
            candidate = self._attempt_generation_with_variation(pytorch_code, strategy, variation_hints)
            if candidate:
                candidates.append(candidate)
            else:
                print(f"    ✗ Candidate {i+1} generation failed")
        
        if not candidates:
            return []
        
        print(f"\n📊 Evaluating {len(candidates)} candidates in parallel...")
        
        # Evaluate all candidates in parallel
        candidates_with_results = []
        with ThreadPoolExecutor(max_workers=min(len(candidates), 4)) as executor:
            future_to_candidate = {
                executor.submit(self._evaluate_candidate, code, quiet=True): code
                for code in candidates
            }
            
            for idx, future in enumerate(as_completed(future_to_candidate), 1):
                candidate_code = future_to_candidate[future]
                try:
                    eval_result = future.result()
                    candidates_with_results.append({
                        "code": candidate_code,
                        "evaluation": eval_result,
                        "candidate_id": idx
                    })
                    
                    compiled = eval_result.get("compiled", False)
                    correct = eval_result.get("correctness", False)
                    speedup = eval_result.get("runtime_stats", {}).get("performance_comparison", {}).get("speedup_ratio")
                    
                    # Handle syntax-only mode
                    if self.spec.syntax_only:
                        compiled = eval_result.get("compiled", False)
                        correct = eval_result.get("correctness")
                        metadata = eval_result.get("metadata", {})
                        if compiled:
                            status = "✓"
                            print(f"    {status} Candidate {idx}: syntax valid (syntax-only mode)")
                        else:
                            status = "✗"
                            error_msg = eval_result.get("error", "Unknown syntax error")
                            if len(error_msg) > 100:
                                error_msg = error_msg[:100] + "..."
                            print(f"    {status} Candidate {idx}: syntax invalid - {error_msg}")
                    else:
                        status = "✓" if correct else ("⚠" if compiled else "✗")
                        speedup_str = f" speedup={speedup:.2f}×" if speedup else ""
                        
                        # Show compilation errors if available
                        error_info = ""
                        metadata = eval_result.get("metadata", {})
                        error_msg = eval_result.get("error", "")
                        
                        if not compiled:
                            # Check multiple possible error fields
                            if "compilation_error" in metadata:
                                error_msg = str(metadata["compilation_error"])
                            elif "error" in metadata:
                                error_msg = str(metadata["error"])
                            elif error_msg:
                                pass  # Already have it
                            elif "other_error" in metadata:
                                error_msg = str(metadata["other_error"])
                            
                            if error_msg:
                                # Show more of the error (first 200 chars)
                                if len(error_msg) > 200:
                                    error_info = f" (error: {error_msg[:200]}...)"
                                else:
                                    error_info = f" (error: {error_msg})"
                            else:
                                error_info = " (no error details available)"
                        elif not correct and "runtime_error" in metadata:
                            error_msg = str(metadata["runtime_error"])
                            if len(error_msg) > 200:
                                error_msg = error_msg[:200] + "..."
                            error_info = f" (runtime: {error_msg})"
                        
                        print(f"    {status} Candidate {idx}: compiled={compiled}, correct={correct}{speedup_str}{error_info}")
                except Exception as e:
                    print(f"    ✗ Candidate {idx} evaluation error: {e}")
        
        return candidates_with_results

    def _attempt_generation_with_variation(self, pytorch_code: str, strategy: dict, variation_hints: str) -> Optional[str]:
        """Generate a single candidate with specific variation hints."""
        try:
            guideline_parts = [FORMAT_HINT.strip()]
            if self.spec.language.lower() == "cute":
                guideline_parts.append(CUTE_TEMPLATE_EXAMPLE)
            if self.spec.guideline_prompt:
                guideline_parts.append(self.spec.guideline_prompt.strip())
            
            # Add variation hints to guide the LLM
            guideline_parts.append(f"\n{variation_hints}")
            
            guideline_with_format = "\n\n".join(part for part in guideline_parts if part).strip()
            generated = prompt_generate_custom_dsl_rag_enhanced(
                ref_arch_src=pytorch_code,
                language=self.spec.language,
                paper_prompt=self.spec.paper_prompt,
                guideline_prompt=guideline_with_format,
                problem_description=self.spec.operation_description or "Optimize this kernel",
                k=strategy["rag_k"],
                current_level=self.spec.current_level,
                current_problem_id=self.spec.current_problem_id,
                extra_ops=self.spec.ops_list,
            )
        except Exception as e:
            return None

        clean_code = extract_first_code(generated, ["python"])
        if not clean_code:
            forced = self._force_python_block(generated)
            if forced:
                return forced
            return None
        return clean_code

    def _attempt_generation(self, pytorch_code: str, strategy: dict) -> Optional[str]:
        try:
            guideline_parts = [FORMAT_HINT.strip()]
            if self.spec.language.lower() == "cute":
                guideline_parts.append(CUTE_TEMPLATE_EXAMPLE)
            if self.spec.guideline_prompt:
                guideline_parts.append(self.spec.guideline_prompt.strip())
            guideline_with_format = "\n\n".join(part for part in guideline_parts if part).strip()
            generated = prompt_generate_custom_dsl_rag_enhanced(
                ref_arch_src=pytorch_code,
                language=self.spec.language,
                paper_prompt=self.spec.paper_prompt,
                guideline_prompt=guideline_with_format,
                problem_description=self.spec.operation_description or "Optimize this kernel",
                k=strategy["rag_k"],
                current_level=self.spec.current_level,
                current_problem_id=self.spec.current_problem_id,
                extra_ops=self.spec.ops_list,
            )
        except Exception as e:
            print(f"  Generation error: {e}")
            return None

        clean_code = extract_first_code(generated, ["python"])
        if not clean_code:
            preview = generated.strip()
            preview = preview[:3000] + ("..." if len(preview) > 3000 else "")
            print("  LLM response did not contain a valid python code block.")
            print("  --- Raw LLM output (truncated) ---")
            print(preview)
            print("  ---------------------------------")
            forced = self._force_python_block(generated)
            if forced:
                print("  ✓ Formatting fallback produced a python code block.")
                return forced
            else:
                print("  ⚠️ Formatting fallback failed to produce code.")
            return None
        return clean_code

    def _check_python_syntax(self, code: str) -> dict:
        """Check if code is valid Python syntax (no CUDA/GPU required)."""
        try:
            ast.parse(code)
            return {
                "compiled": True,  # Syntax is valid
                "correctness": None,  # Can't check without running
                "metadata": {"syntax_check": "passed", "note": "Syntax-only mode: no compilation or runtime checks"}
            }
        except SyntaxError as e:
            return {
                "compiled": False,
                "correctness": False,
                "error": f"Python syntax error: {e}",
                "metadata": {"syntax_error": str(e), "line": e.lineno, "offset": e.offset}
            }
        except Exception as e:
            return {
                "compiled": False,
                "correctness": False,
                "error": f"Syntax check failed: {e}",
                "metadata": {"syntax_check_error": str(e)}
            }

    def _evaluate_candidate(
        self,
        code: str,
        *,
        measure_performance: Optional[bool] = None,
        quiet: bool = False,
    ) -> dict:
        # Syntax-only mode: just check Python syntax
        if self.spec.syntax_only:
            return self._check_python_syntax(code)

        reference = self.spec.pytorch_reference
        language = self.spec.language
        measure = self.spec.measure_performance if measure_performance is None else measure_performance

        if not self.spec.use_modal:
            try:
                local_result = eval_kernel_against_ref(
                    original_model_src=reference,
                    custom_model_src=code,
                    num_correct_trials=self.spec.num_correct_trials,
                    num_perf_trials=self.spec.num_perf_trials,
                    verbose=not quiet,
                    measure_performance=measure,
                    language=language,
                )
            except Exception as e:
                error_msg = str(e)
                if "CUDA is not available" in error_msg:
                    if not quiet:
                        print(f"  ⚠️ CUDA not available for local evaluation.")
                        print(f"  💡 Tip: Use --use_modal (or remove --local_eval) to evaluate on Modal's GPUs")
                    return {
                        "correctness": False, 
                        "compiled": False, 
                        "error": "CUDA not available - cannot run local evaluation. Use Modal for GPU evaluation.",
                        "metadata": {"cuda_error": True}
                    }
                if not quiet:
                    print(f"  Local evaluation error: {e}")
                return {"correctness": False, "compiled": False, "error": error_msg}

            if hasattr(local_result, "dict"):
                result_dict = local_result.dict()
            elif hasattr(local_result, "model_dump"):
                result_dict = local_result.model_dump()
            else:
                result_dict = dict(local_result)
        else:
            try:
                with eval_app.run():
                    try:
                        result = EvalFunc.with_options(gpu=self.spec.gpu)().eval_single_sample_modal.remote(
                            reference,
                            code,
                            False,
                            gpu_arch_mapping[self.spec.gpu],
                            language,
                            None,
                            None,
                            self.spec.current_problem_id or 0,
                            self.spec.current_level or 0,
                            measure_performance=measure,
                            num_correct_trials=self.spec.num_correct_trials,
                            num_perf_trials=self.spec.num_perf_trials,
                        )
                    except Exception as e:
                        error_str = str(e)
                        if "billing" in error_str.lower() or "spend limit" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
                            if not quiet:
                                print(f"  ⚠️ Modal billing limit reached.")
                                print(f"  💡 Tip: Use --syntax_only to check code without evaluation")
                                print(f"  💡 Or wait for billing cycle to reset, or upgrade Modal plan")
                            return {
                                "correctness": False,
                                "compiled": False,
                                "error": "Modal billing limit reached - cannot evaluate. Use --syntax_only for syntax checks.",
                                "metadata": {"modal_billing_error": True}
                            }
                        if not quiet:
                            print(f"  Evaluation error: {e}")
                        return {"correctness": False, "compiled": False, "error": error_str}
            except Exception as e:
                error_str = str(e)
                if "billing" in error_str.lower() or "spend limit" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
                    if not quiet:
                        print(f"  ⚠️ Modal billing limit reached.")
                        print(f"  💡 Tip: Use --syntax_only to check code without evaluation")
                    return {
                        "correctness": False,
                        "compiled": False,
                        "error": "Modal billing limit reached - cannot evaluate. Use --syntax_only for syntax checks.",
                        "metadata": {"modal_billing_error": True}
                    }
                if not quiet:
                    print(f"  Modal setup error: {e}")
                    if "image builder versions" in error_str or "2025.06" in error_str:
                        print(f"  💡 Modal server is using a newer image builder. Try: python -m pip install --upgrade modal")
                return {"correctness": False, "compiled": False, "error": error_str, "metadata": {"modal_setup_error": True}}

            # Modal may return a pydantic model or plain dict; normalize.
            if hasattr(result, "dict"):
                result_dict = result.dict()
            elif hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            else:
                result_dict = dict(result)

        if not quiet:
            self._display_eval_summary(result_dict)
        return result_dict

    def _force_python_block(self, raw_response: str) -> Optional[str]:
        # Handle double fences and other common formatting issues
        sanitized = raw_response
        
        # Remove duplicate python fences
        sanitized = sanitized.replace("```python\n```python", "```python")
        sanitized = sanitized.replace("```python\r\n```python", "```python")
        sanitized = sanitized.replace("```python```python", "```python")
        
        # Remove duplicate generic fences
        sanitized = sanitized.replace("```\n```", "```")
        sanitized = sanitized.replace("```\r\n```", "```")
        sanitized = sanitized.replace("``````", "```")
        
        # Try to extract code
        code = extract_first_code(sanitized, ["python"])
        if code:
            return code

        # Try extracting all blocks and getting the largest python block
        blocks = extract_all_code_blocks(sanitized)
        if "python" in blocks and blocks["python"].strip():
            return blocks["python"]
        
        # If still no code, try to find any code block
        if blocks:
            # Return the first non-empty block
            for lang, content in blocks.items():
                if content.strip():
                    return content

        if not self.spec.enable_refiner:
            return None

        config = CuteKernelRefinerConfig(
            max_attempts=max(1, self.spec.refiner_attempts or 1),
            server_type=self.spec.refiner_server,
            model_name=self.spec.refiner_model or "o3",
            temperature=self.spec.refiner_temperature,
        )
        refiner = CuteKernelRefiner(self.spec.guideline_prompt, config)

        def eval_stub(_code: str) -> Dict:
            return {"compiled": True, "correctness": True}

        initial_result = {
            "compiled": False,
            "correctness": False,
            "metadata": {"format_issue": "missing_python_block"},
        }

        try:
            formatted_code, _ = refiner.refine(
                initial_code=sanitized,
                evaluate_fn=eval_stub,
                problem_label=self.spec.problem_label,
                initial_result=initial_result,
            )
            return formatted_code
        except Exception as exc:
            print(f"  Formatting fallback failed: {exc}")
            return None

    def _maybe_refine(self, code: str, initial_result: dict) -> Optional[dict]:
        if not self.spec.enable_refiner or self.spec.refiner_attempts <= 0:
            return None

        print("  🔁 Invoking automatic refinement loop...")
        ref_config = CuteKernelRefinerConfig(
            max_attempts=self.spec.refiner_attempts,
            server_type=self.spec.refiner_server,
            model_name=self.spec.refiner_model or "o3",
            temperature=self.spec.refiner_temperature,
        )
        refiner = CuteKernelRefiner(self.spec.guideline_prompt, ref_config)

        def eval_fn(updated_code: str) -> Dict:
            return self._evaluate_candidate(
                updated_code,
                measure_performance=self.spec.measure_performance,
                quiet=True,
            )

        refined_code, refined_result = refiner.refine(
            initial_code=code,
            evaluate_fn=eval_fn,
            problem_label=self.spec.problem_label,
            initial_result=initial_result,
        )

        if refined_code and refined_result and refined_result.get("correctness"):
            print("  ✓ Refiner produced a passing kernel.")
            self._display_eval_summary(refined_result)
            return {"code": refined_code, "evaluation": refined_result}

        if refined_result and refined_result.get("error"):
            print(f"  Refiner error: {refined_result['error']}")
        else:
            print("  Refiner attempts exhausted without success.")
        return None

    def _display_eval_summary(self, result: dict) -> None:
        compiled = result.get("compiled", False)
        correctness = result.get("correctness", False)
        runtime_stats = result.get("runtime_stats", {}) or {}
        runtime_mean = runtime_stats.get("mean")
        speedup = None

        perf = runtime_stats.get("performance_comparison", {})
        if isinstance(perf, dict):
            speedup = perf.get("speedup_ratio")

        print("  ── Eval Summary ──")
        print(f"    compiled   : {'yes' if compiled else 'no'}")
        print(f"    correctness: {'yes' if correctness else 'no'}")
        if not self.spec.measure_performance:
            print("    performance: skipped")
        else:
            if runtime_mean is not None:
                print(f"    mean runtime (us): {runtime_mean:.4f}")
            if speedup is not None:
                print(f"    speedup vs PyTorch: {speedup}×")
        if "metadata" in result and result["metadata"]:
            meta = result["metadata"]
            hardware = meta.get("hardware")
            if hardware:
                print(f"    hardware  : {hardware}")
        if not compiled or not correctness:
            error_keys = [k for k in ("compilation_error", "runtime_error") if k in result.get("metadata", {})]
            for key in error_keys:
                print(f"    {key}: {result['metadata'][key]}")
        print()

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _summarize_success(self, record: dict, strategy: dict) -> None:
        eval_result = record["evaluation"]
        speedup = None
        if eval_result and "runtime_stats" in eval_result:
            perf = eval_result["runtime_stats"]
            if perf and "performance_comparison" in perf:
                speedup = perf["performance_comparison"].get("speedup_ratio")

        print("\n================ SUCCESS ================")
        print(f"Strategy: rag_k={strategy['rag_k']} temp={strategy['temperature']}")
        if self.spec.measure_performance:
            if speedup:
                print(f"Speedup Ratio: {speedup:.2f}× vs PyTorch")
        else:
            print("Performance measurement skipped (correctness only).")
        print("Generated CuTe kernel:\n")
        print(record["code"])
        print("========================================\n")

    def _offer_save(self, code: str) -> None:
        save = input("Save kernel to file? [y/N]: ").strip().lower()
        if save != "y":
            return
        path = input("Enter output filepath (default: generated_kernel.py): ").strip()
        if not path:
            path = "generated_kernel.py"
        with open(path, "w") as f:
            f.write(code)
        print(f"Kernel saved to {path}")


# ----------------------------------------------------------------------
# CLI utilities
# ----------------------------------------------------------------------
def prompt_with_default(message: str, default: Optional[str] = None) -> str:
    prompt = f"{message}" + (f" [{default}]" if default is not None else "") + ": "
    response = input(prompt).strip()
    if not response and default is not None:
        return default
    return response


def prompt_multiline(terminator: str = "END") -> str:
    print(f"Enter text (finish with a line containing only {terminator}):")
    lines: List[str] = []
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        if line.strip() == terminator:
            break
        lines.append(line)
    return "".join(lines)


def load_pytorch_reference(path: Optional[str]) -> str:
    if path:
        return read_file(path)
    print("No PyTorch reference path provided.")
    choice = prompt_with_default("Do you have a path to a reference file? (y/N)", "n").lower()
    if choice == "y":
        while True:
            file_path = input("Enter absolute path to PyTorch reference: ").strip()
            if file_path and os.path.exists(file_path):
                return read_file(file_path)
            print("Path not found. Try again or press Enter to paste the code.")
            retry = input("Paste instead? (y/N): ").strip().lower()
            if retry == "y":
                break
    print("Paste the PyTorch reference implementation now.")
    return prompt_multiline()


def interactive_collect_spec(default_language: str = "cute",
                              default_gpu: str = "H100",
                              default_model: str = "openai/o3",
                              default_rag_k: int = 3,  # Reduced from 5
                              default_temperature: float = 1.0) -> KernelSpec:
    print("\n=== Interactive CuTe Kernel Agent ===")
    language = prompt_with_default("Target DSL (cute/tilelang/tk)", default_language).lower()
    gpu = prompt_with_default("Target GPU", default_gpu)
    model = prompt_with_default("LLM model name", default_model)
    fast_model = prompt_with_default("Fast model for first attempt (faster, optional)", "openai/gpt-4o")
    rag_k_str = prompt_with_default("How many RAG examples to fetch?", str(default_rag_k))
    temperature_str = prompt_with_default("Sampling temperature", f"{default_temperature}")
    description = input("Describe the kernel you need (high-level overview): ").strip()
    ops_sequence = prompt_with_default("Operation pipeline (comma-separated, e.g. gemm,bias,hardtanh,gelu)", "")
    ops_list = [op.strip().lower() for op in ops_sequence.split(",") if op.strip()]
    input_shapes = prompt_with_default("Key tensor shapes (e.g. A[B, M, K], W[K, N])", "")
    dtype = prompt_with_default("Primary dtype (e.g. fp16)", "fp16")
    target_speedup_input = prompt_with_default("Target speedup vs PyTorch (e.g. 1.2)", "")
    constraints = prompt_with_default("Additional constraints (tiling, memory, etc.)", "")
    problem_label = prompt_with_default("Short problem label", "custom_kernel")

    level_input = prompt_with_default("KernelBench level (optional)", "")
    problem_input = prompt_with_default("KernelBench problem id (optional)", "")
    max_attempts_input = prompt_with_default("Max generation attempts", "3")
    skip_perf = prompt_with_default("Skip performance measurement? (y/N)", "n").lower() == "y"
    num_correct_input = ""
    num_perf_input = ""
    if not skip_perf:
        num_correct_input = prompt_with_default("Correctness trials", "5")
        num_perf_input = prompt_with_default("Performance trials", "100")
    use_modal = prompt_with_default("Use Modal for evaluation? (Y/n)", "y").lower() != "n"
    enable_test_time_scaling = prompt_with_default("Enable test-time scaling (generate 4-8 candidates)? (Y/n)", "y").lower() != "n"
    num_candidates_input = "4"
    if enable_test_time_scaling:
        num_candidates_input = prompt_with_default("Number of candidates to generate (4-8)", "4")
    enable_refiner = prompt_with_default("Enable automatic refinement if evaluation fails? (Y/n)", "y").lower() != "n"
    refiner_attempts_input = "0"
    refiner_model = ""
    refiner_temperature_input = ""
    if enable_refiner:
        refiner_attempts_input = prompt_with_default("Refiner attempts", "2")
        refiner_model = prompt_with_default("Refiner model name", "o3")
        refiner_temperature_input = prompt_with_default("Refiner temperature", "1.0")

    pytorch_code = load_pytorch_reference(None)

    paper_prompt, guideline_prompt = GUIDELINE_BY_LANG.get(language, GUIDELINE_BY_LANG[default_language])

    # Compose structured operation description
    description_parts = []
    if description:
        description_parts.append(description)
    if ops_sequence:
        description_parts.append(f"Operation pipeline: {ops_sequence}")
    if input_shapes:
        description_parts.append(f"Shapes: {input_shapes}")
    if dtype:
        description_parts.append(f"Dtype: {dtype}")
    if target_speedup_input:
        description_parts.append(f"Target speedup vs PyTorch: {target_speedup_input}×")
    if constraints:
        description_parts.append(f"Constraints: {constraints}")
    structured_description = "\n".join(description_parts)

    spec = KernelSpec(
        language=language or default_language,
        gpu=gpu or default_gpu,
        rag_k=int(rag_k_str) if rag_k_str else default_rag_k,
        model_name=model or default_model,
        fast_model=fast_model or "openai/gpt-4o",
        temperature=float(temperature_str) if temperature_str else default_temperature,
        pytorch_reference=pytorch_code,
        operation_description=structured_description,
        ops_sequence=ops_sequence,
        input_shapes=input_shapes,
        dtype=dtype,
        target_speedup=float(target_speedup_input) if target_speedup_input else None,
        additional_constraints=constraints,
        problem_label=problem_label,
        ops_list=ops_list,
        paper_prompt=paper_prompt,
        guideline_prompt=guideline_prompt,
        current_level=int(level_input) if level_input else None,
        current_problem_id=int(problem_input) if problem_input else None,
        max_attempts=int(max_attempts_input) if max_attempts_input else 3,
        measure_performance=not skip_perf,
        num_correct_trials=int(num_correct_input) if num_correct_input else 5,
        num_perf_trials=int(num_perf_input) if num_perf_input else 100,
        use_modal=use_modal,
        test_time_scaling=enable_test_time_scaling,
        num_candidates=int(num_candidates_input) if num_candidates_input else 4,
        syntax_only=False,  # Interactive mode doesn't prompt for this, can add later if needed
        enable_refiner=enable_refiner,
        refiner_attempts=int(refiner_attempts_input) if refiner_attempts_input else 0,
        refiner_server="openai",
        refiner_model=refiner_model or None,
        refiner_temperature=float(refiner_temperature_input) if refiner_temperature_input else 1.0,
    )
    return spec


def gather_spec_from_flags(args: argparse.Namespace) -> KernelSpec:
    language = args.language.lower()
    if language not in GUIDELINE_BY_LANG:
        raise ValueError(f"Unsupported language: {language}")

    paper_prompt, guideline_prompt = GUIDELINE_BY_LANG[language]

    pytorch_code = load_pytorch_reference(args.pytorch_ref)
    ops_sequence = args.ops or ""
    ops_list = [op.strip().lower() for op in ops_sequence.split(",") if op.strip()]
    input_shapes = args.shapes or ""
    dtype = args.dtype or ""
    target_speedup = args.target_speedup
    constraints = args.constraints or ""
    problem_label = args.problem_label or "custom_kernel"

    description_parts = []
    if args.description:
        description_parts.append(args.description)
    if ops_sequence:
        description_parts.append(f"Operation pipeline: {ops_sequence}")
    if input_shapes:
        description_parts.append(f"Shapes: {input_shapes}")
    if dtype:
        description_parts.append(f"Dtype: {dtype}")
    if target_speedup:
        description_parts.append(f"Target speedup vs PyTorch: {target_speedup}×")
    if constraints:
        description_parts.append(f"Constraints: {constraints}")
    operation_description = "\n".join(description_parts)

    current_level = args.level
    current_problem_id = args.problem_id
    enable_refiner = not args.disable_refiner
    default_refiner_attempts = 2
    refiner_attempts = (
        args.refiner_attempts
        if args.refiner_attempts is not None
        else (default_refiner_attempts if enable_refiner else 0)
    )
    refiner_server = args.refiner_server or "openai"
    refiner_model = args.refiner_model
    refiner_temperature = args.refiner_temperature if args.refiner_temperature is not None else 1.0

    return KernelSpec(
        language=language,
        gpu=args.gpu,
        rag_k=args.rag_k,
        model_name=args.model,
        fast_model=args.fast_model,
        temperature=args.temperature,
        pytorch_reference=pytorch_code,
        paper_prompt=paper_prompt,
        guideline_prompt=guideline_prompt,
        operation_description=operation_description,
        ops_sequence=ops_sequence,
        input_shapes=input_shapes,
        dtype=dtype,
        target_speedup=target_speedup,
        additional_constraints=constraints,
        problem_label=problem_label,
        ops_list=ops_list,
        current_level=current_level,
        current_problem_id=current_problem_id,
        max_attempts=args.max_attempts if args.max_attempts else 3,
        measure_performance=not args.skip_perf,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        use_modal=not args.local_eval,
        test_time_scaling=getattr(args, 'test_time_scaling', True),
        num_candidates=getattr(args, 'num_candidates', 4),
        syntax_only=getattr(args, 'syntax_only', False),
        enable_refiner=enable_refiner,
        refiner_attempts=refiner_attempts,
        refiner_server=refiner_server,
        refiner_model=refiner_model or None,
        refiner_temperature=refiner_temperature,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive CuTe kernel generation agent")
    parser.add_argument("--language", default="cute", help="Target DSL (cute, tilelang, tk)")
    parser.add_argument("--gpu", default="H100", help="Target GPU architecture")
    parser.add_argument("--rag_k", type=int, default=3, help="Number of RAG examples to retrieve")
    parser.add_argument("--model", default="openai/o3", help="LLM model name")
    parser.add_argument("--fast_model", default="openai/gpt-4o", help="Faster model for first attempt")
    parser.add_argument("--temperature", type=float, default=1.0, help="LLM sampling temperature")
    parser.add_argument("--pytorch_ref", help="Path to PyTorch reference implementation")
    parser.add_argument("--description", help="Short natural language description of the operation")
    parser.add_argument("--ops", help="Comma-separated operation pipeline")
    parser.add_argument("--shapes", help="Tensor shapes description")
    parser.add_argument("--dtype", help="Primary dtype")
    parser.add_argument("--target_speedup", type=float, help="Target speedup vs PyTorch")
    parser.add_argument("--constraints", help="Additional constraints or notes")
    parser.add_argument("--problem_label", help="Short problem label")
    parser.add_argument("--level", type=int, help="Optional KernelBench level for context exclusion")
    parser.add_argument("--problem_id", type=int, help="Optional KernelBench problem id for exclusion")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive conversational mode")
    parser.add_argument("--max_attempts", type=int, help="Maximum number of generation attempts")
    parser.add_argument("--skip_perf", action="store_true", help="Skip performance measurement during evaluation")
    parser.add_argument("--num_correct_trials", type=int, default=5, help="Correctness trials when measuring performance")
    parser.add_argument("--num_perf_trials", type=int, default=100, help="Performance timing trials")
    parser.add_argument("--local_eval", action="store_true", help="Run evaluation locally instead of Modal")
    parser.add_argument("--syntax_only", action="store_true", help="Only check Python syntax, skip compilation/evaluation (useful when Modal billing limit reached or no CUDA)")
    parser.add_argument("--disable_refiner", action="store_true", help="Disable automatic refinement")
    parser.add_argument("--refiner_attempts", type=int, help="Maximum refinement attempts")
    parser.add_argument("--refiner_server", default="openai", help="Server type for refinement loop")
    parser.add_argument("--refiner_model", help="Model name for refinement loop")
    parser.add_argument("--refiner_temperature", type=float, help="Temperature for refinement loop")
    parser.add_argument("--test_time_scaling", action="store_true", default=True, help="Enable test-time scaling (generate multiple candidates)")
    parser.add_argument("--no_test_time_scaling", dest="test_time_scaling", action="store_false", help="Disable test-time scaling")
    parser.add_argument("--num_candidates", type=int, default=4, help="Number of candidates for test-time scaling (4-8)")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    use_interactive = args.interactive or (not args.description and not args.pytorch_ref)

    if use_interactive:
        while True:
            spec = interactive_collect_spec()
            agent = KernelAgentCLI(spec)
            agent.run()
            again = prompt_with_default("Generate another kernel? (y/N)", "n").lower()
            if again != "y":
                break
        return

    try:
        spec = gather_spec_from_flags(args)
    except Exception as e:
        print(f"Spec error: {e}")
        sys.exit(1)

    agent = KernelAgentCLI(spec)
    agent.run()


if __name__ == "__main__":
    main()
