"""
Generate and Evaluate DSL Kernels using DSPy RAG + Modal

This script uses the high-performance DSPy RAG system for DSL generation,
then evaluates the generated kernels on Modal infrastructure.

Supported DSLs: TileLang, ThunderKittens, CuTe
Optimized for use with OpenAI O3 and resource-unconstrained environments.
"""

try:
    import pydra
    from pydra import REQUIRED, Config
except (ImportError, AttributeError):
    REQUIRED = object()

    class Config:
        """Minimal stand-in for pydra.Config when pydra is unavailable."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _PydraStub:
        def main(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def validate(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    pydra = _PydraStub()

import os, sys
import torch
import json
import modal
import dspy
import textwrap

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor_rag import prompt_generate_custom_dsl_rag_enhanced
from src.utils import extract_first_code, extract_code_block, extract_all_code_blocks, create_tk_makefile, set_gpu_arch, read_file, strip_docstring_from_code

# TileLang-specific prompts
from scripts.tilelang_paperinfo_prompt import TILELANG_PAPER_PROMPT
from scripts.tilelang_guideline_prompt import TILELANG_GUIDELINE_PROMPT

# ThunderKittens-specific prompts
from scripts.tk_paperinfo_prompt import TK_PAPER_PROMPT
from scripts.tk_guideline_prompt import TK_GUIDELINE_PROMPT

# CuTe-specific prompts
from scripts.cute_paperinfo_prompt import CUTE_PAPER_PROMPT
from scripts.cute_guideline_prompt import CUTE_GUIDELINE_PROMPT


app = modal.App("eval_rag_dsl")

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}

def configure_dspy(model_name: str, temperature: float = 0.0):
    """Configure DSPy with the specified model"""
    print(f"🤖 Configuring DSPy with model: {model_name}")
    
    # Configure the language model
    if (model_name.lower() in ['gpt-5']):
        lm = dspy.LM(model_name, temperature=temperature, max_completion_tokens=20000)
    else:  
        lm = dspy.LM(model_name, temperature=temperature, max_tokens=20000)
    dspy.configure(lm=lm)
    
    print(f"✅ DSPy configured successfully with {model_name}")
    return lm

class RAGEvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Language/DSL
        self.language = "tilelang"  # Options: "tilelang", "thunderkittens", "cuda", "cute"

        # Problem Specification
        self.level = REQUIRED
        self.problem_id = REQUIRED

        # Evaluation
        self.eval_mode = "modal"
        self.gpu = "H100"
        self.gpu_arch = ['Hopper']

        # DSPy Model Configuration
        self.dspy_model = "openai/o3"  # Options: "openai/o3-mini", "openai/o1-preview", "openai/gpt-4o"
        self.dspy_temperature = 1.0
        
        # RAG Configuration
        self.rag_k = 5  # Number of examples to retrieve
        
        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/rag_eval_logs")
        self.verbose = False

        self.log = True
        self.log_generated_kernel = True
        self.log_eval_result = True
        
        self.eval_only = False
        self.eval_file_path = None

    def verbose_logging(self):
        self.log = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"RAGEvalConfig({self.to_dict()})"


cuda_version = "12.8.0"  # Updated from 12.4.0 to 12.8.0
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10").apt_install(
    "git",
    "gcc-10",
    "g++-10",
    "clang",
).pip_install(
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
    "pyutils",
    "ninja",
    "utils",
    "pybind11",
    "python-dotenv",
    "tilelang",
    "apache-tvm",
    "dspy-ai",
    "nvidia-cutlass-dsl",
)

image = image.env({"THUNDERKITTENS_ROOT": "/root/ThunderKittens"})
image = image.add_local_python_source("scripts", "src")
image = image.add_local_dir("KernelBench", "/root/KernelBench")

correct_tk_local = os.path.join(REPO_TOP_DIR, "correct_tk")
if os.path.isdir(correct_tk_local):
    image = image.add_local_dir(correct_tk_local, "/root/correct_tk")
else:
    print(f"⚠️ Skipping mount; directory not found: {correct_tk_local}")

thunder_local = os.path.join(REPO_TOP_DIR, "ThunderKittens")
if os.path.isdir(thunder_local):
    image = image.add_local_dir(thunder_local, "/root/ThunderKittens")
else:
    print(f"⚠️ Skipping mount; directory not found: {thunder_local}")

@app.cls(image=image)
class EvalFunc:
    @modal.method()
    def eval_single_sample_modal(
        self,
        ref_arch_src,
        custom_cuda,
        verbose,
        gpu_arch,
        language,
        entry_point=None,
        cu_code: str | None = None,
        problem_id: int = None,
        level: int = None,
        return_logs_on_failure: bool = False,
        measure_performance: bool = True,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
    ):
        # SET DEFAULT DTYPE TO FLOAT16 ONLY FOR TILELANG
        if language == "tilelang":
            torch.set_default_dtype(torch.float16)
        
        from src.eval import eval_kernel_against_ref
        from src.utils import set_gpu_arch, create_tk_makefile
        import os, subprocess, sys

        # If ThunderKittens, write and compile the kernel inside Modal container
        if language == "thunderkittens" and cu_code is not None:
            # Determine TK kernel dir consistently with src.eval
            # src.eval computes TK dir as parent_of_src/correct_tk
            src_dir = os.path.dirname(__import__('src').__file__)
            repo_top = os.path.abspath(os.path.join(src_dir, ".."))
            tk_kernel_dir = os.path.join(repo_top, "correct_tk")
            os.makedirs(tk_kernel_dir, exist_ok=True)

            # Use specific problem filename instead of generic custom_tk.cu
            if problem_id is not None and level is not None:
                cu_file_path = os.path.join(tk_kernel_dir, f"{level}_{problem_id}.cu")
            with open(cu_file_path, "w") as f:
                f.write(cu_code)

            # Create Makefile and compile
            cu_filename = os.path.basename(cu_file_path)
            create_tk_makefile(tk_kernel_dir, gpu="H100", cu_file=cu_filename)
            try:
                subprocess.run(["make", "clean"], cwd=tk_kernel_dir, check=False, capture_output=True, text=True)
                result = subprocess.run(["make"], cwd=tk_kernel_dir, check=True, capture_output=True, text=True, env={**os.environ})
            except subprocess.CalledProcessError as e:
                print("[Modal TK Compile] Failed to compile ThunderKittens kernel")
                print("STDOUT:\n" + (e.stdout or ""))
                print("STDERR:\n" + (e.stderr or ""))
                if return_logs_on_failure:
                    # Return rich error info instead of raising to allow callers to consume logs
                    return {
                        "correctness": False,
                        "stage": "compile",
                        "stdout": e.stdout or "",
                        "stderr": e.stderr or "",
                    }
                raise
            # Ensure path is importable
            if tk_kernel_dir not in sys.path:
                sys.path.append(tk_kernel_dir)
                
        set_gpu_arch(gpu_arch)
        # Sanitize custom CUDA/Python wrapper code to avoid indentation errors
        if isinstance(custom_cuda, str):
            custom_cuda = textwrap.dedent(custom_cuda).lstrip()
        return eval_kernel_against_ref(
            ref_arch_src,
            custom_cuda,
            verbose=verbose,
            measure_performance=measure_performance,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            language=language,
            entry_point=entry_point,
        )


@pydra.main(base=RAGEvalConfig)
def main(config: RAGEvalConfig):
    """
    Generate kernels using DSPy RAG and evaluate on Modal
    """

    ########################################################
    ##### PART 0: Fetch Problem from KernelBench
    ########################################################
    print(f"Starting RAG Evaluation with config: {config}")
    
    if config.language == "tilelang":
        print(">>> Setting default dtype to float16 (TileLang only) <<<")
        torch.set_default_dtype(torch.float16)

    # Configure DSPy with the specified model
    if (config.dspy_model.lower() in ['gpt-5']):
        lm = dspy.LM(config.dspy_model, temperature=config.dspy_temperature, max_completion_tokens=20000)
    else:  
        lm = dspy.LM(config.dspy_model, temperature=config.dspy_temperature, max_tokens=20000)
    dspy.configure(lm=lm)
    
    print(f"✅ DSPy configured successfully with {config.dspy_model}")

    # Load dataset
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)

    num_problems = len(curr_level_dataset)
    # print(f"Number of problems in Level {config.level}: {num_problems}")
    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"


    ########################################################
    ##### PART 1: Fetch Problem from KernelBench
    ########################################################
    if config.dataset_src == "huggingface":
        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]
        problem_name = os.path.basename(ref_arch_path)
        problem_name = problem_name.replace(".py", "")
        ref_arch_src = read_file(ref_arch_path)
    print(f"Start RAG Generation + Evaluation for Level {config.level} Problem {config.problem_id}: {problem_name}")

    # Extract problem number from problem name
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"
    





    #################################################################
    ##### PART 2: Load (and generate if needed?) the DSL Code using RAG
    ##################################################################

    print(f"config: {config.eval_only}")
    ### CASE 1: Eval Only
    if config.eval_only:
        # This is the eval_only=true case: we don't generate DSPy code, we just use existing files
        print(">>> USING CODE WITH EVAL ONLY <<<")

        # In TileLang or CuTe case, we just need to get a string of the python code
        if config.language in ["tilelang", "cute"]:
            if config.eval_file_path:
                path = config.eval_file_path
            else:
                path = f"src/prompts/correct_{config.language}/level{config.level}/{config.level}_{config.problem_id}.py"

            raw_code = open(path).read()
            clean_code = strip_docstring_from_code(raw_code)
            tilelang_code = "```python\n" + clean_code + "\n```"
            print(f">>> Stripped old docstring, using clean code for evaluation <<<")
        
        # For ThunderKittens eval_only, load the .py and .cu from the specified absolute directory and run Modal
        if config.language == "thunderkittens":
            tk_dir = f"/Users/willychan/Desktop/projects/kb-pilot/KernelBench/src/prompts/correct_thunderkittens/level{config.level}"
            stem = f"{config.level}_{config.problem_id}"
            py_path = os.path.join(tk_dir, f"{stem}.py")
            cu_path = os.path.join(tk_dir, f"{stem}.cu")

            if not os.path.exists(py_path) or not os.path.exists(cu_path):
                raise FileNotFoundError(
                    f"Required files not found for eval_only: {py_path} or {cu_path}. Please ensure both exist."
                )

            with open(py_path, "r") as _f:
                custom_cuda = textwrap.dedent(_f.read()).lstrip()
            with open(cu_path, "r") as _f:
                cu_code = _f.read()

            entry_point = None
            if config.level == 9:
                first_underscore = problem_name.find("_")
                entry_point = problem_name[first_underscore+1:]

            print(f"🚀 Evaluating ThunderKittens kernel on Modal...\nPY: {py_path}\nCU: {cu_path}")

            with app.run():
                kernel_exec_result = EvalFunc.with_options(gpu=config.gpu)().eval_single_sample_modal.remote(
                    ref_arch_src, custom_cuda, config.verbose, gpu_arch_mapping[config.gpu], config.language, entry_point, cu_code, config.problem_id, config.level
                )

                print(f"\n{'='*60}")
                print(f"🎯 EVALUATION RESULT for Level {config.level} Problem {config.problem_id}")
                print(f"{'='*60}")
                print(f"Problem: {problem_name}")
                print(f"Result: {kernel_exec_result}")
                print(f"{'='*60}")

                if config.log_eval_result:
                    with open(os.path.join(config.logdir, f"rag_eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
                        f.write(f"Problem Name: {problem_name}\n")
                        f.write(str(kernel_exec_result))

            return
    ### CASE 2: NOT Eval Only - i.e. we need to generate with RAG
    else:
        print(f">>> GENERATING {config.language.upper()} CODE USING DSPY RAG <<<")
        if config.language == "tilelang":
            PAPER_PROMPT = TILELANG_PAPER_PROMPT
            GUIDELINE_PROMPT = TILELANG_GUIDELINE_PROMPT
        elif config.language == "thunderkittens":
            PAPER_PROMPT = TK_PAPER_PROMPT
            GUIDELINE_PROMPT = TK_GUIDELINE_PROMPT
        elif config.language == "cute":
            PAPER_PROMPT = CUTE_PAPER_PROMPT
            GUIDELINE_PROMPT = CUTE_GUIDELINE_PROMPT
        else:
            raise ValueError(f"Unsupported language: {config.language}. Use 'tilelang', 'thunderkittens', or 'cute'")
        
        try:
            # Use the high-performance RAG system
            # Note: variable is named 'tilelang_code' for historical reasons but holds code for any DSL
            tilelang_code = prompt_generate_custom_dsl_rag_enhanced(
                ref_arch_src=ref_arch_src,
                language=config.language,
                paper_prompt=PAPER_PROMPT,
                guideline_prompt=GUIDELINE_PROMPT,
                problem_description=f"Optimize {problem_name}",
                k=config.rag_k,
                current_level=config.level,
                current_problem_id=config.problem_id
            )
            print(f"✅ RAG generation successful for {problem_name}")
            # Log DSPy prompt history to file
            if config.log:
                try:
                    print("📝 Saving DSPy prompt history...")
                    history_file = os.path.join(config.logdir, f"dspy_history_level_{config.level}_problem_{config.problem_id}.txt")
                    
                    # Capture DSPy history
                    import io
                    import sys
                    from contextlib import redirect_stdout
                    
                    # Redirect stdout to capture inspect_history output
                    history_output = io.StringIO()
                    with redirect_stdout(history_output):
                        dspy.inspect_history(n=1)
                    
                    history_content = history_output.getvalue()
                    
                    with open(history_file, "w") as f:
                        f.write(f"DSPy Prompt History for Level {config.level} Problem {config.problem_id}\n")
                        f.write(f"Problem: {problem_name}\n")
                        f.write(f"Model: {config.dspy_model}\n")
                        f.write(f"Temperature: {config.dspy_temperature}\n")
                        f.write(f"RAG Examples (k): {config.rag_k}\n")
                        f.write("="*80 + "\n\n")
                        f.write(history_content)
                    
                    print(f"💾 DSPy history saved to {history_file}")
                    
                except Exception as e:
                    print(f"⚠️ Could not save DSPy history: {e}")
            
        except Exception as e:
            print(f"❌ RAG generation failed: {e}")
            print("This indicates a configuration or system issue")
            return

        # Print out costs (only for generation runs)
        if not config.eval_only:
            history = lm.history
            total_cost = 0
            for entry in history:
                if entry.get("cost") is not None:
                    total_cost += entry["cost"]
            print(f"Number of interactions: {len(history)}")
            print(f"Total cost: ${total_cost}")





    #################################################################
    ##### PART 2.5: In the thunderkittens case, we need to get the .cu and .py code blocks
    ##################################################################
    cu_code = None  # Initialize for ThunderKittens

    print(f"LANGUAGE: {config.language}, EVAL ONLY: {config.eval_only}")
    if config.language == "thunderkittens" and not config.eval_only:
        print(">>> PROCESSING THUNDERKITTENS CODE <<<")
        
        # ThunderKittens requires TWO code blocks: .cu kernel and .py wrapper
        all_blocks = extract_all_code_blocks(tilelang_code)
        
        # Extract .cu kernel (can be marked as cpp, c++, or cuda)
        cu_code = all_blocks.get("cpp") or all_blocks.get("c++") or all_blocks.get("cuda")
        py_code = all_blocks.get("python")
        
        if cu_code is None:
            raise ValueError(f"ThunderKittens requires .cu kernel code block (```cpp). Found blocks: {list(all_blocks.keys())}")
        if py_code is None:
            raise ValueError(f"ThunderKittens requires Python wrapper code block (```python). Found blocks: {list(all_blocks.keys())}")
        
        print(f"✅ Extracted both .cu kernel ({len(cu_code)} chars) and .py wrapper ({len(py_code)} chars)")
        
        import re
        tk_kernel_dir = os.path.join(REPO_TOP_DIR, "src", "prompts", "correct_thunderkittens", f"level{config.level}")
        os.makedirs(tk_kernel_dir, exist_ok=True)
        
        print(f"📁 REPO_TOP_DIR: {REPO_TOP_DIR}")
        print(f"📁 Saving files to: {tk_kernel_dir}")

        cu_file_path = os.path.join(tk_kernel_dir, f"{config.level}_{config.problem_id}.cu")
        print(f"📝 Writing .cu kernel to: {os.path.abspath(cu_file_path)}")
        with open(cu_file_path, "w") as f:
            f.write(cu_code)
        print(f"✅ Wrote .cu kernel ({len(cu_code)} bytes)")

        # Create Makefile for reference only; do not run make locally
        create_tk_makefile(tk_kernel_dir, gpu=config.gpu)
        print(f"📝 Created Makefile in {tk_kernel_dir}")
        
        py_file_path = os.path.join(tk_kernel_dir, f"{config.level}_{config.problem_id}.py")
        print(f"📝 Writing .py wrapper to: {os.path.abspath(py_file_path)}")
        with open(py_file_path, "w") as f:
            f.write(py_code)
        print(f"✅ Wrote .py wrapper ({len(py_code)} bytes)")

        # Use Python wrapper code for execution; pass ORIGINAL cu_code to Modal for remote compile
        custom_cuda = textwrap.dedent(py_code).lstrip()
        
    else:
        # Standard code extraction for other languages (TileLang, CuTe, CUDA)
        # These are single-file Python or C++ implementations
        custom_cuda = extract_first_code(tilelang_code, ["python", "cpp"])
        cu_code = custom_cuda
    
    # Validate generation
    assert custom_cuda is not None, f"{config.language} code generation failed"





    #################################################################
    ##### PART 3: Evaluate results on Modal
    ##################################################################
    entry_point = None
    if config.level == 9:
        first_underscore = problem_name.find("_")
        entry_point = problem_name[first_underscore+1:]
        
    print(f"🚀 Evaluating generated kernel on Modal...")
    print(f"Entry point: {entry_point}")
    
    # Debug: verify cu_code is set for ThunderKittens
    if config.language == "thunderkittens":
        if cu_code is not None:
            print(f"✅ Passing cu_code to Modal ({len(cu_code)} bytes)")
        else:
            print(f"❌ WARNING: cu_code is None for ThunderKittens!")
    
    with app.run():
        kernel_exec_result = EvalFunc.with_options(gpu=config.gpu)().eval_single_sample_modal.remote(
            ref_arch_src, custom_cuda, config.verbose, gpu_arch_mapping[config.gpu], config.language, entry_point, cu_code if config.language == "thunderkittens" else None, config.problem_id, config.level
        )
        
        print(f"\n{'='*60}")
        print(f"INITIAL EVALUATION RESULT for Level {config.level} Problem {config.problem_id}")
        print(f"{'='*60}")
        print(f"Problem: {problem_name}")
        print(f"DSPy Model: {config.dspy_model}")
        print(f"RAG Examples Used: {config.rag_k}")
        print(f"Result: {kernel_exec_result}")
        print(f"{'='*60}")
        
        # Save successful kernels
        if kernel_exec_result.correctness:
            if config.eval_file_path is None:
                # Map language to correct directory (tilelang, cute, thunderkittens, cuda, etc.)
                base_dir = f"src/prompts/correct_{config.language}/level{config.level}"
                
                os.makedirs(base_dir, exist_ok=True)
                base_path = f"{base_dir}/{config.level}_{config.problem_id}.py"
                path = base_path
                
                # Extract speedup ratio from current result
                eval_str = str(kernel_exec_result)
                current_ratio = float(eval_str.split("speedup_ratio': ")[1].split("}")[0])
                
                # If file exists (and we are generating new kernels), compare speedup ratios
                if not config.eval_only and os.path.exists(path):
                    with open(path, 'r') as f:
                        existing_content = f.read()
                        existing_eval = existing_content.split('Evaluation Result:\n')[1].split('\n"""')[0]
                        existing_ratio = float(existing_eval.split("speedup_ratio': ")[1].split("}")[0])
                        
                        if current_ratio <= existing_ratio:
                            print(f"💾 Discarding kernel - existing speedup ratio {existing_ratio:.2f} better than current {current_ratio:.2f}")
                            return
                        else:
                            print(f"💾 Replacing kernel - current speedup ratio {current_ratio:.2f} better than existing {existing_ratio:.2f}")
                else:
                    print(f"💾 Writing new {config.language} kernel to {path}")
            else:
                path = config.eval_file_path

            with open(path, "w") as f:
                f.write(f'''"""
Problem Name: {problem_name}
Generated using DSPy RAG with {config.dspy_model}
RAG Examples: {config.rag_k}
Evaluation Result:
{str(kernel_exec_result)}
"""

{custom_cuda}''')
            
            # For ThunderKittens, also save the .cu file in the same directory
            if config.language == "thunderkittens" and not config.eval_only:
                cu_save_path = path.replace(".py", ".cu")
                # cu_code was extracted earlier, save it
                if 'cu_code' in locals():
                    with open(cu_save_path, "w") as f:
                        f.write(cu_code)
                    print(f"💾 Also saved .cu kernel to {cu_save_path}")
                else:
                    # Copy from correct_tk/custom_tk.cu if it exists
                    tk_cu_source = os.path.join(REPO_TOP_DIR, f"src/prompts/correct_{config.language}/level{config.level}")
                    if os.path.exists(tk_cu_source):
                        import shutil
                        shutil.copy(tk_cu_source, cu_save_path)
                        print(f"💾 Copied .cu kernel to {cu_save_path}")
        
        if config.log_eval_result:
            with open(os.path.join(config.logdir, f"rag_eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(f"DSPy Model: {config.dspy_model}\n")
                f.write(f"RAG Examples: {config.rag_k}\n")
                f.write(str(kernel_exec_result))


if __name__ == "__main__":
    main() 