"""
Generate and Evaluate TileLang Kernels using DSPy RAG + Modal

This script uses the high-performance DSPy RAG system for TileLang generation,
then evaluates the generated kernels on Modal infrastructure.

Optimized for use with OpenAI O3 and resource-unconstrained environments.
"""

import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import modal
import dspy

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor_rag import prompt_generate_custom_tilelang_rag_enhanced
from src.utils import extract_first_code, set_gpu_arch, read_file
from scripts.tilelang_paperinfo_prompt import PAPER_PROMPT
from scripts.tilelang_guideline_prompt import GUIDELINE_PROMPT


def strip_docstring_from_code(code: str) -> str:
    """
    Remove the docstring from Python code if it exists at the beginning.
    This is useful for eval_only mode to remove old evaluation results.
    """
    lines = code.split('\n')
    
    # Find first non-empty, non-comment line
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            start_idx = i
            break
    
    # Check if it starts with a docstring
    if start_idx < len(lines):
        stripped_line = lines[start_idx].strip()
        if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
            quote_type = '"""' if stripped_line.startswith('"""') else "'''"
            
            # Find the end of the docstring
            if stripped_line.count(quote_type) >= 2:
                # Single line docstring
                return '\n'.join(lines[start_idx + 1:])
            else:
                # Multi-line docstring
                for i in range(start_idx + 1, len(lines)):
                    if quote_type in lines[i]:
                        return '\n'.join(lines[i + 1:])
    
    return code


app = modal.App("eval_rag_tilelang")

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}


class RAGEvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

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


cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git",
                "gcc-10",
                "g++-10",
                "clang"
                )
    .pip_install(
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
        "python-dotenv",
        "tilelang",
        "apache-tvm",
        "dspy-ai"
    )
    .add_local_python_source("scripts", "src")
    .add_local_dir("KernelBench", "/root/KernelBench")
)


@app.cls(image=image)
class EvalFunc:

    @modal.method()
    def eval_single_sample_modal(self, ref_arch_src, custom_cuda, verbose, gpu_arch, language, entry_point=None):
        # SET DEFAULT DTYPE TO FLOAT16 AT THE VERY BEGINNING OF MODAL FUNCTION
        torch.set_default_dtype(torch.float16)
        
        from src.eval import eval_kernel_against_ref
        from src.utils import set_gpu_arch
        set_gpu_arch(gpu_arch)
        return eval_kernel_against_ref(
            ref_arch_src, custom_cuda, verbose=verbose, measure_performance=True, 
            num_correct_trials=5, num_perf_trials=100, language=language, entry_point=entry_point
        )


def configure_dspy(model_name: str, temperature: float = 0.0):
    """Configure DSPy with the specified model"""
    print(f"🤖 Configuring DSPy with model: {model_name}")
    
    # Configure the language model
    lm = dspy.LM(model_name, temperature=temperature, max_tokens=20000)
    dspy.configure(lm=lm)
    
    print(f"✅ DSPy configured successfully with {model_name}")
    return lm


@pydra.main(base=RAGEvalConfig)
def main(config: RAGEvalConfig):
    """
    Generate TileLang kernels using DSPy RAG and evaluate on Modal
    """
    print(f"Starting RAG Evaluation with config: {config}")
    
    print(">>> Setting default dtype to float16 <<<")
    torch.set_default_dtype(torch.float16)

    # Configure DSPy with the specified model
    lm = configure_dspy(config.dspy_model, config.dspy_temperature)

    # Load dataset
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)
        
    # Problem validation
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"

    # 1. Fetch Problem
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
    
    # 2. Generate TileLang Code using DSPy RAG
    if not config.eval_only:
        print(">>> GENERATING TILELANG CODE USING DSPY RAG <<<")
        
        try:
            # Use the high-performance RAG system
            tilelang_code = prompt_generate_custom_tilelang_rag_enhanced(
                ref_arch_src=ref_arch_src,
                paper_prompt=PAPER_PROMPT,
                guideline_prompt=GUIDELINE_PROMPT,
                problem_description=f"Optimize {problem_name}",
                k=config.rag_k,
                current_level=config.level,
                current_problem_id=config.problem_id
            )
            
            # print(f"TileLang Code: {tilelang_code}")
            
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
            
    else:
        print(">>> USING CODE WITH EVAL ONLY <<<")
        if config.eval_file_path:
            path = config.eval_file_path
        else:
            path = f"src/prompts/correct_tilelang/level{config.level}/{config.level}_{config.problem_id}.py"

        raw_code = open(path).read()
        clean_code = strip_docstring_from_code(raw_code)
        tilelang_code = "```python\n" + clean_code + "\n```"
        print(f">>> Stripped old docstring, using clean code for evaluation <<<")

    # Extract code from markdown
    custom_cuda = extract_first_code(tilelang_code, ["python", "cpp"])
    
    # Validate generation
    assert custom_cuda is not None, "TileLang code generation failed"
    
    # Print out costs
    history = lm.history
    total_cost = 0
    for entry in history:
        total_cost += entry["cost"]
    print(f"Number of interactions: {len(history)}")
    print(f"Total cost: ${total_cost}")

    if config.log_generated_kernel:
        with open(os.path.join(config.logdir, f"rag_generated_kernel_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
            f.write(custom_cuda)
    
    # 3. Evaluate on Modal
    entry_point = None
    if config.level == 9:
        first_underscore = problem_name.find("_")
        entry_point = problem_name[first_underscore+1:]
        
    print(f"🚀 Evaluating generated kernel on Modal...")
    print(f"Entry point: {entry_point}")
    
    with app.run():
        kernel_exec_result = EvalFunc.with_options(gpu=config.gpu)().eval_single_sample_modal.remote(
            ref_arch_src, custom_cuda, config.verbose, gpu_arch_mapping[config.gpu], "tilelang", entry_point
        )
        
        print(f"\n{'='*60}")
        print(f"🎯 EVALUATION RESULT for Level {config.level} Problem {config.problem_id}")
        print(f"{'='*60}")
        print(f"Problem: {problem_name}")
        print(f"DSPy Model: {config.dspy_model}")
        print(f"RAG Examples Used: {config.rag_k}")
        print(f"Result: {kernel_exec_result}")
        print(f"{'='*60}")
        
        # Save successful kernels
        if kernel_exec_result.correctness:
            if config.eval_file_path is None:
                base_path = f"src/prompts/correct_tilelang/level{config.level}/{config.level}_{config.problem_id}.py"
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
                    print(f"💾 Writing new TileLang kernel to {path}")
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
        
        if config.log_eval_result:
            with open(os.path.join(config.logdir, f"rag_eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(f"DSPy Model: {config.dspy_model}\n")
                f.write(f"RAG Examples: {config.rag_k}\n")
                f.write(str(kernel_exec_result))


if __name__ == "__main__":
    main() 