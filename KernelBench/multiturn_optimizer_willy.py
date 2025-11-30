"""
SIMPLE KERNEL FIXER

Supports ThunderKittens and CuTe

- Loads an existing kernel for a given level/problem from the repo
- ThunderKittens: expects .py wrapper and .cu kernel under
  /Users/willychan/Desktop/projects/kb-pilot/KernelBench/src/prompts/correct_thunderkittens/level{level}/
- CuTe: expects a single .py file under
  /Users/willychan/Desktop/projects/kb-pilot/KernelBench/src/prompts/correct_cute/level{level}/
- Evaluates on Modal; captures full error output (compile/runtime) or result payload.
- Prompts an LLM with: current kernel (both .cu and .py), full error output, and strict instructions:
  "Fix this kernel so that it compiles correctly, has correctness, and is also as performant as possible."
- Repeats up to N attempts (default 3) or stops early once correctness is achieved. If still failing, prints not fixable.

Notes:
- Reuses Modal evaluation class and image from scripts/generate_and_eval_rag_modal.py.
- Uses util helpers from src.utils for code extraction and LLM invocation.


ThunderKittens (default):
Example: python KernelBench/multiturn_optimizer_willy.py --level 1 --problem_id 93 --language thunderkittens
CuTe:
Example: python KernelBench/multiturn_optimizer_willy.py --level 1 --problem_id 1 --language cute
"""

import os
import sys
import traceback
from typing import Optional, Tuple

import pydra
from pydra import REQUIRED, Config
from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset  # used if dataset_src == "huggingface"

from src.dataset import construct_kernelbench_dataset
from src.utils import (
    extract_all_code_blocks,
    create_tk_makefile,
    read_file,
    create_inference_server_from_presets,
)

# Reuse Modal app and evaluator for ThunderKittens from the RAG eval script
from scripts.generate_and_eval_rag_modal import (
    app as eval_app,
    EvalFunc,
    gpu_arch_mapping,
    REPO_TOP_DIR,
)

# Prompts for TK/CuTe generation
from scripts.tk_guideline_prompt import TK_GUIDELINE_PROMPT
from scripts.cute_guideline_prompt import CUTE_GUIDELINE_PROMPT

# IMPORTANT: THESE ARE THE KERNEL FIXING LOOP CONFIGURATIONS
class TKFixLoopConfig(Config):
    def __init__(self):
        # Dataset source and target problem
        self.dataset_src = "local"  # "huggingface" or "local"
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = REQUIRED
        self.problem_id = REQUIRED

        # Modal / GPU
        self.gpu = "H100"
        self.gpu_arch = ["Hopper"]
        self.verbose = True

        # Language/DSL to fix: "thunderkittens" or "cute"
        self.language = "thunderkittens"

        # LLM server presets (uses src.utils SERVER_PRESETS)
        self.server_type = "openai"  # e.g., "openai", "anthropic", "together"
        # Use valid OpenAI model IDs here; "o3" is a reasoning model
        self.model_name = "o3"  # optional override; see SERVER_PRESETS for defaults
        self.temperature = 1.0  # optional override
        self.max_tokens = 30000  # optional override
        # Reasoning models require enabling reasoning flags in our utils
        self.is_reasoning_model = True
        self.reasoning_effort = "high"  # "low" | "medium" | "high"

        # Loop settings
        self.max_attempts = 5

        # Logging/output
        self.logdir = os.path.join(REPO_TOP_DIR, "results", "tk_fix_loop_logs")
        # Save directory is derived per-language in _save_kernel
        self.save_dir = None

    def __repr__(self):
        return f"TKFixLoopConfig({self.to_dict()})"


# This function just returns a tuple of (the pytorch solution code, problem name)
def _load_problem(config: TKFixLoopConfig) -> Tuple[str, str]:
    """Return (ref_arch_src, problem_name)."""
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    else:
        curr_level_dataset = construct_kernelbench_dataset(config.level)
        idx = config.problem_id - 1
        ref_arch_path = curr_level_dataset[idx]
        problem_name = os.path.basename(ref_arch_path).replace(".py", "")
        ref_arch_src = read_file(ref_arch_path)

    # Validate filename number vs problem_id if local
    problem_number = int(problem_name.split("_")[0])
    assert (
        problem_number == config.problem_id
    ), f"Problem number in name ({problem_number}) != problem_id ({config.problem_id})"
    return ref_arch_src, problem_name


def _load_existing_tk_code(config: TKFixLoopConfig) -> Tuple[str, str]:
    """Load existing ThunderKittens .py and .cu for the given level/problem from the repo.

    Expected paths:
      /Users/willychan/Desktop/projects/kb-pilot/KernelBench/src/prompts/correct_thunderkittens/level{level}/{level}_{problem_id}.py
      /Users/willychan/Desktop/projects/kb-pilot/KernelBench/src/prompts/correct_thunderkittens/level{level}/{level}_{problem_id}.cu
    """
    tk_dir = f"/Users/willychan/Desktop/projects/kb-pilot/KernelBench/src/prompts/correct_thunderkittens/level{config.level}"
    stem = f"{config.level}_{config.problem_id}"
    py_path = os.path.join(tk_dir, f"{stem}.py")
    cu_path = os.path.join(tk_dir, f"{stem}.cu")

    if not os.path.exists(cu_path):
        raise FileNotFoundError(f"Missing .cu file: {cu_path}")
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Missing .py wrapper: {py_path}")

    py_code = read_file(py_path)
    cu_code = read_file(cu_path)
    return py_code, cu_code


def _load_existing_cute_code(config: TKFixLoopConfig) -> str:
    """Load existing CuTe .py for the given level/problem from the repo.

    Expected path:
      /Users/willychan/Desktop/projects/kb-pilot/KernelBench/src/prompts/correct_cute/level{level}/{level}_{problem_id}.py
    """
    cute_dir = f"/Users/willychan/Desktop/projects/kb-pilot/KernelBench/src/prompts/correct_cute/level{config.level}"
    stem = f"{config.level}_{config.problem_id}"
    py_path = os.path.join(cute_dir, f"{stem}.py")
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Missing CuTe .py file: {py_path}")
    return read_file(py_path)


def _eval_tk_on_modal(
    py_code: str,
    cu_code: str,
    ref_arch_src: str,
    problem_name: str,
    config: TKFixLoopConfig,
) -> Tuple[bool, Optional[str], Optional[float]]:
    """Evaluate TK kernel on Modal. Returns (success, error_text, speedup).

    success means correctness True. If failure, error_text is a best-effort capture of
    compile/runtime logs or exception string. speedup is parsed on success if possible.
    """
    entry_point = None
    if config.level == 9:
        first_underscore = problem_name.find("_")
        if first_underscore != -1:
            entry_point = problem_name[first_underscore + 1 :]

    try:
        with eval_app.run():
            result = (
                EvalFunc.with_options(gpu=config.gpu)()
                .eval_single_sample_modal
                .remote(
                    ref_arch_src,
                    py_code,
                    config.verbose,
                    gpu_arch_mapping[config.gpu],
                    "thunderkittens",
                    entry_point,
                    cu_code,
                    config.problem_id,
                    config.level,
                    True,  # return_logs_on_failure
                )
            )
        # Success path
        try:
            if getattr(result, "correctness", False):
                # Try to parse speedup from string form
                result_str = str(result)
                speedup = None
                try:
                    # crude parse: look for "speedup_ratio': <num>"
                    if "speedup_ratio" in result_str:
                        speedup = float(result_str.split("speedup_ratio': ")[1].split("}")[0])
                except Exception:
                    speedup = None
                return True, None, speedup
        except Exception:
            # If result object is unexpected, treat as failure
            return False, f"Unexpected result object: {repr(result)}", None

        # Handle dict error payload (compile logs)
        if isinstance(result, dict) and not result.get("correctness", True):
            # Prefer stderr+stdout
            stderr = result.get("stderr", "")
            stdout = result.get("stdout", "")
            stage = result.get("stage", "unknown")
            joined = (
                f"[Modal Failure Stage: {stage}]\n\nSTDERR:\n{stderr}\n\nSTDOUT:\n{stdout}"
            )
            return False, joined, None

        # Correctness false (non-dict): include full textual payload
        return False, str(result), None

    except Exception as e:
        # Best-effort capture of remote compile or runtime error logs
        err_text = (
            f"Modal evaluation failed with exception:\n{repr(e)}\n\nTraceback (local):\n{traceback.format_exc()}"
        )
        return False, err_text, None


def _eval_cute_on_modal(
    py_code: str,
    ref_arch_src: str,
    problem_name: str,
    config: TKFixLoopConfig,
) -> Tuple[bool, Optional[str], Optional[float]]:
    """Evaluate CuTe kernel on Modal. Returns (success, error_text, speedup)."""
    entry_point = None
    if config.level == 9:
        first_underscore = problem_name.find("_")
        if first_underscore != -1:
            entry_point = problem_name[first_underscore + 1 :]

    try:
        with eval_app.run():
            result = (
                EvalFunc.with_options(gpu=config.gpu)()
                .eval_single_sample_modal
                .remote(
                    ref_arch_src,
                    py_code,
                    config.verbose,
                    gpu_arch_mapping[config.gpu],
                    "cute",
                    entry_point,
                    None,  # cu_code not used for CuTe
                    config.problem_id,
                    config.level,
                    True,  # return_logs_on_failure
                )
            )

        try:
            if getattr(result, "correctness", False):
                result_str = str(result)
                speedup = None
                try:
                    if "speedup_ratio" in result_str:
                        speedup = float(result_str.split("speedup_ratio': ")[1].split("}")[0])
                except Exception:
                    speedup = None
                return True, None, speedup
        except Exception:
            return False, f"Unexpected result object: {repr(result)}", None

        if isinstance(result, dict) and not result.get("correctness", True):
            stderr = result.get("stderr", "")
            stdout = result.get("stdout", "")
            stage = result.get("stage", "unknown")
            joined = f"[Modal Failure Stage: {stage}]\n\nSTDERR:\n{stderr}\n\nSTDOUT:\n{stdout}"
            return False, joined, None

        return False, str(result), None

    except Exception as e:
        err_text = (
            f"Modal evaluation failed with exception:\n{repr(e)}\n\nTraceback (local):\n{traceback.format_exc()}"
        )
        return False, err_text, None


def _build_fix_prompt(
    py_code: str,
    cu_code: str,
    error_text: str,
    problem_name: str,
    attempt_idx: int,
) -> str:
    """Compose the instruction to the LLM, demanding two code blocks (python, cpp)."""
    return (
        f"You are an expert ThunderKittens engineer. You MUST use the kittens:: API to do this task.\n"
        f"Task: Fix this ThunderKittens kernel so that it (1) compiles, (2) passes correctness, and (3) is as performant as possible.\n"
        f"THUNDERKITTENS GUIDELINES:\n"
        f"{TK_GUIDELINE_PROMPT}\n\n"
        f"Problem: {problem_name}\n"
        f"Attempt: {attempt_idx}\n\n"
        f"Current Python wrapper (thunderkittens):\n"  # require python fenced block
        f"```python\n{py_code}\n```\n\n"
        f"Current CUDA (.cu) kernel:\n"  # require cpp fenced block
        f"```cpp\n{cu_code}\n```\n\n"
        f"Full error output/logs from evaluation (compile/runtime):\n"
        f"{error_text}\n\n"
        f"Output requirements:\n"
        f"- Return ONLY TWO code blocks:\n"
        f"  1) the full updated Python wrapper in a ```python block\n"
        f"  2) the full updated .cu kernel in a ```cpp block\n"
        f"- Do NOT include any other text.\n"
        f"- Ensure both files are self-contained and importable/compilable.\n"
    )


def _extract_tk_blocks(llm_response: str) -> Tuple[Optional[str], Optional[str]]:
    blocks = extract_all_code_blocks(llm_response)
    py_code = blocks.get("python")
    cu_code = blocks.get("cpp") or blocks.get("c++") or blocks.get("cuda")
    return py_code, cu_code


def _build_cute_fix_prompt(
    py_code: str,
    error_text: str,
    problem_name: str,
    attempt_idx: int,
) -> str:
    """Compose the instruction to the LLM, demanding one python code block (CuTe)."""
    return (
        f"You are an expert CuTe engineer. You MUST use the CuTe Python DSL to do this task.\n"
        f"Task: Fix this CuTe kernel so that it (1) compiles, (2) passes correctness, and (3) is as performant as possible.\n"
        f"CUTE GUIDELINES:\n"
        f"{CUTE_GUIDELINE_PROMPT}\n\n"
        f"Problem: {problem_name}\n"
        f"Attempt: {attempt_idx}\n\n"
        f"Current CuTe Python code:\n"
        f"```python\n{py_code}\n```\n\n"
        f"Full error output/logs from evaluation (compile/runtime):\n"
        f"{error_text}\n\n"
        f"Output requirements:\n"
        f"- Return ONLY ONE code block:\n"
        f"  1) the full updated CuTe Python code in a ```python block\n"
        f"- Do NOT include any other text.\n"
        f"- Ensure the code is self-contained and importable/compilable.\n"
    )


def _extract_cute_block(llm_response: str) -> Optional[str]:
    blocks = extract_all_code_blocks(llm_response)
    return blocks.get("python")


def _save_kernel(
    py_code: str,
    cu_code: Optional[str],
    config: TKFixLoopConfig,
    suffix: str = "",
    speedup: Optional[float] = None,
) -> str:
    """Save kernel to the appropriate directory based on language and return base path."""
    if config.language == "thunderkittens":
        save_dir = os.path.join(REPO_TOP_DIR, "src", "prompts", "correct_thunderkittens", f"level{config.level}")
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, f"{config.level}_{config.problem_id}")
        if suffix:
            base = f"{base}_{suffix}"
        py_path = f"{base}.py"
        cu_path = f"{base}.cu"
        with open(py_path, "w") as f:
            f.write(py_code)
        with open(cu_path, "w") as f:
            if speedup is not None:
                f.write(f"// Speedup ratio: {speedup:.3f}x\n\n")
            f.write(cu_code or "")
        create_tk_makefile(save_dir, gpu=config.gpu, cu_file=os.path.basename(cu_path))
        return base
    elif config.language == "cute":
        save_dir = os.path.join(REPO_TOP_DIR, "src", "prompts", "correct_cute", f"level{config.level}")
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, f"{config.level}_{config.problem_id}")
        if suffix:
            base = f"{base}_{suffix}"
        py_path = f"{base}.py"
        with open(py_path, "w") as f:
            if speedup is not None:
                f.write(f"# Speedup ratio: {speedup:.3f}x\n\n")
            f.write(py_code)
        return base
    else:
        raise ValueError(f"Unsupported language for saving: {config.language}")


@pydra.main(base=TKFixLoopConfig)
def main(config: TKFixLoopConfig):
    print(f"Starting simplified kernel fix loop with config: {config}")

    # Prepare outputs/logs
    os.makedirs(config.logdir, exist_ok=True)

    # Prepare the LLM caller
    llm_kwargs = {}
    if config.model_name is not None:
        llm_kwargs["model_name"] = config.model_name
    if config.temperature is not None:
        llm_kwargs["temperature"] = config.temperature
    if config.max_tokens is not None:
        llm_kwargs["max_tokens"] = config.max_tokens
    # Pass reasoning flags through for o3/o1/o4-mini
    if getattr(config, "is_reasoning_model", None) is not None:
        llm_kwargs["is_reasoning_model"] = config.is_reasoning_model
    if getattr(config, "reasoning_effort", None) is not None:
        llm_kwargs["reasoning_effort"] = config.reasoning_effort
    query_llm = create_inference_server_from_presets(
        server_type=config.server_type,
        greedy_sample=False,
        verbose=False,
        time_generation=False,
        **llm_kwargs,
    )

    # Load problem
    ref_arch_src, problem_name = _load_problem(config)
    print(f"Level {config.level} Problem {config.problem_id}: {problem_name}")

    # Load existing files from repo based on language
    if config.language == "thunderkittens":
        print("Loading existing ThunderKittens kernel (.py and .cu) from repo...")
        try:
            py_code, cu_code = _load_existing_tk_code(config)
        except Exception as e:
            print(f"Failed to load TK files: {e}")
            sys.exit(1)
    elif config.language == "cute":
        print("Loading existing CuTe kernel (.py) from repo...")
        try:
            py_code = _load_existing_cute_code(config)
            cu_code = None
        except Exception as e:
            print(f"Failed to load CuTe file: {e}")
            sys.exit(1)
    else:
        print(f"Unsupported language: {config.language}. Choose 'thunderkittens' or 'cute'.")
        sys.exit(1)

    # Track intermediate files for cleanup
    intermediate_files = []
    
    # Iterative loop
    for attempt in range(1, config.max_attempts + 1):
        print(f"\n=== Attempt {attempt}/{config.max_attempts} ===")

        # Evaluate current kernel
        if config.language == "thunderkittens":
            success, error_text, speedup = _eval_tk_on_modal(
                py_code, cu_code, ref_arch_src, problem_name, config
            )
        else:
            success, error_text, speedup = _eval_cute_on_modal(
                py_code, ref_arch_src, problem_name, config
            )

        if success:
            print("Success: Kernel compiled and passed correctness.")
            if speedup is not None:
                print(f"Reported speedup ratio: {speedup:.3f}")
            # Clean up any intermediate files
            for file_path in intermediate_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Cleaned up intermediate file: {file_path}")
            base = _save_kernel(py_code, cu_code, config, speedup=speedup)
            if config.language == "thunderkittens":
                print(f"Saved working kernel to: {base}.py and {base}.cu")
            else:
                print(f"Saved working kernel to: {base}.py")
            return

        # If this is the last attempt, save final version and exit
        if attempt == config.max_attempts:
            print(f"\nFinal attempt {attempt} failed. Saving final version.")
            # Clean up any intermediate files
            for file_path in intermediate_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Cleaned up intermediate file: {file_path}")
            final_base = _save_kernel(py_code, cu_code, config, "final", speedup=None)
            if config.language == "thunderkittens":
                print(f"Saved final kernel to: {final_base}.py and {final_base}.cu")
            else:
                print(f"Saved final kernel to: {final_base}.py")
            print("\nNot fixable within the specified number of attempts.")
            return

        # Build and send fix prompt to LLM
        assert error_text is not None, "Failure without error text; cannot proceed."
        if config.language == "thunderkittens":
            prompt = _build_fix_prompt(py_code, cu_code, error_text, problem_name, attempt)
        else:
            prompt = _build_cute_fix_prompt(py_code, error_text, problem_name, attempt)
        print(prompt)
        try:
            llm_output = query_llm(prompt)
        except Exception as e:
            print(f"LLM call failed: {e}")
            print("Stopping.")
            sys.exit(1)

        # Extract updated blocks based on language
        if config.language == "thunderkittens":
            new_py, new_cu = _extract_tk_blocks(llm_output)
            if not new_py or not new_cu:
                print("LLM did not return both python and cpp code blocks; stopping.")
                print("Raw LLM output (truncated):\n" + llm_output[:1000])
                sys.exit(1)
            py_code, cu_code = new_py, new_cu
        else:
            new_py = _extract_cute_block(llm_output)
            if not new_py:
                print("LLM did not return a python code block; stopping.")
                print("Raw LLM output (truncated):\n" + llm_output[:1000])
                sys.exit(1)
            py_code = new_py
            cu_code = None


if __name__ == "__main__":
    main()


