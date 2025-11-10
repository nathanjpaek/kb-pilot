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
import json
import os
import sys
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
SRC_DIR = PARENT_DIR / "src"

for path in (CURRENT_DIR, PARENT_DIR, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

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
    {"rag_k": 5, "temperature": 1.0},
    {"rag_k": 8, "temperature": 0.2},
    {"rag_k": 10, "temperature": 0.0},
]

FORMAT_HINT = (
    "Output format requirement:\n"
    "- Return exactly one ```python code block containing the complete CuTe kernel.\n"
    "- Include the ModelNew class, any host wrapper, and necessary helpers.\n"
    "- Do not emit prose or explanation outside the fenced code block.\n"
)


@dataclass
class KernelSpec:
    language: str = "cute"
    gpu: str = "H100"
    rag_k: int = 5
    model_name: str = "openai/o3"
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

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class KernelAgentCLI:
    def __init__(self, spec: KernelSpec, retries: Optional[List[dict]] = None):
        self.spec = spec
        self.retries = retries or DEFAULT_RETRIES
        configure_dspy(spec.model_name, spec.temperature)

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("\n================ CuTe Kernel Agent ================")
        print(f"Language: {self.spec.language.upper()}  |  GPU: {self.spec.gpu}")
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
            print(f"Attempt {attempt}: rag_k={strategy['rag_k']} temp={strategy['temperature']}")
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

    def _attempt_generation(self, pytorch_code: str, strategy: dict) -> Optional[str]:
        try:
            guideline_with_format = f"{self.spec.guideline_prompt}\n\n{FORMAT_HINT}"
            generated = prompt_generate_custom_dsl_rag_enhanced(
                ref_arch_src=pytorch_code,
                language=self.spec.language,
                paper_prompt=self.spec.paper_prompt,
                guideline_prompt=guideline_with_format,
                problem_description=self.spec.operation_description or "Optimize this kernel",
                k=strategy["rag_k"],
                current_level=self.spec.current_level,
                current_problem_id=self.spec.current_problem_id,
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

    def _evaluate_candidate(
        self,
        code: str,
        *,
        measure_performance: Optional[bool] = None,
        quiet: bool = False,
    ) -> dict:
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
                if not quiet:
                    print(f"  Local evaluation error: {e}")
                return {"correctness": False, "compiled": False, "error": str(e)}

            if hasattr(local_result, "dict"):
                result_dict = local_result.dict()
            elif hasattr(local_result, "model_dump"):
                result_dict = local_result.model_dump()
            else:
                result_dict = dict(local_result)
        else:
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
                    if not quiet:
                        print(f"  Evaluation error: {e}")
                    return {"correctness": False, "compiled": False, "error": str(e)}

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
        sanitized = raw_response.replace("```python\n```python", "```python")
        sanitized = sanitized.replace("```python\r\n```python", "```python")
        sanitized = sanitized.replace("```\n```", "```")
        sanitized = sanitized.replace("```\r\n```", "```")

        code = extract_first_code(sanitized, ["python"])
        if code:
            return code

        blocks = extract_all_code_blocks(sanitized)
        if "python" in blocks and blocks["python"].strip():
            return blocks["python"]

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
                              default_rag_k: int = 5,
                              default_temperature: float = 1.0) -> KernelSpec:
    print("\n=== Interactive CuTe Kernel Agent ===")
    language = prompt_with_default("Target DSL (cute/tilelang/tk)", default_language).lower()
    gpu = prompt_with_default("Target GPU", default_gpu)
    model = prompt_with_default("LLM model name", default_model)
    rag_k_str = prompt_with_default("How many RAG examples to fetch?", str(default_rag_k))
    temperature_str = prompt_with_default("Sampling temperature", f"{default_temperature}")
    description = input("Describe the kernel you need (high-level overview): ").strip()
    ops_sequence = prompt_with_default("Operation pipeline (comma-separated, e.g. gemm,bias,hardtanh,gelu)", "")
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
        temperature=float(temperature_str) if temperature_str else default_temperature,
        pytorch_reference=pytorch_code,
        operation_description=structured_description,
        ops_sequence=ops_sequence,
        input_shapes=input_shapes,
        dtype=dtype,
        target_speedup=float(target_speedup_input) if target_speedup_input else None,
        additional_constraints=constraints,
        problem_label=problem_label,
        paper_prompt=paper_prompt,
        guideline_prompt=guideline_prompt,
        current_level=int(level_input) if level_input else None,
        current_problem_id=int(problem_input) if problem_input else None,
        max_attempts=int(max_attempts_input) if max_attempts_input else 3,
        measure_performance=not skip_perf,
        num_correct_trials=int(num_correct_input) if num_correct_input else 5,
        num_perf_trials=int(num_perf_input) if num_perf_input else 100,
        use_modal=use_modal,
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
        current_level=current_level,
        current_problem_id=current_problem_id,
        max_attempts=args.max_attempts if args.max_attempts else 3,
        measure_performance=not args.skip_perf,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        use_modal=not args.local_eval,
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
    parser.add_argument("--rag_k", type=int, default=5, help="Number of RAG examples to retrieve")
    parser.add_argument("--model", default="openai/o3", help="LLM model name")
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
    parser.add_argument("--disable_refiner", action="store_true", help="Disable automatic refinement")
    parser.add_argument("--refiner_attempts", type=int, help="Maximum refinement attempts")
    parser.add_argument("--refiner_server", default="openai", help="Server type for refinement loop")
    parser.add_argument("--refiner_model", help="Model name for refinement loop")
    parser.add_argument("--refiner_temperature", type=float, help="Temperature for refinement loop")
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
