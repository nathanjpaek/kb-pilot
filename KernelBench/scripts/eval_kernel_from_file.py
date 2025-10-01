import pydra
from pydra import REQUIRED, Config
import os, sys
import torch

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.utils import set_gpu_arch, read_file

"""
Evaluate a pre-generated kernel from a file
Useful for testing manually written kernels or kernels generated outside the main pipeline
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}

class EvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        self.language = "cuda"  # cuda or tilelang
        
        # Problem Specification
        self.level = REQUIRED
        # NOTE: this is the logical index (problem id the problem_name)
        self.problem_id = REQUIRED
        
        # Kernel file to evaluate
        self.kernel_file = REQUIRED  # path to the kernel file to evaluate

        # Evaluation
        # local (requires a GPU)
        self.eval_mode = "local"
        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu = "H100"
        self.gpu_arch = ['Hopper']

        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = False

        self.log = False
        self.log_eval_result = False

    def verbose_logging(self):
        self.log = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    Evaluate a pre-generated kernel from a file
    """
    print(f"Starting Kernel Evaluation with config: {config}")

    # Configurations
    
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    # Update GPU architecture based on GPU name if available
    if hasattr(config, 'gpu') and config.gpu in gpu_arch_mapping:
        config.gpu_arch = gpu_arch_mapping[config.gpu]

    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)  # otherwise build for all architectures

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)
        
    # Problem Checks
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Evaluating Level {config.level} Problem {config.problem_id}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"

    # 1. Fetch Problem
    if config.dataset_src == "huggingface":
        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"
    
    # 2. Load Kernel from File
    if not os.path.exists(config.kernel_file):
        raise FileNotFoundError(f"Kernel file not found: {config.kernel_file}")
    
    print(f"Loading kernel from: {config.kernel_file}")
    kernel_code = read_file(config.kernel_file)
    
    # Wrap the kernel code in appropriate format (like the original script does)
    if config.language == "tilelang":
        custom_cuda = f"```python\n{kernel_code}\n```"
    else:  # cuda
        custom_cuda = f"```cpp\n{kernel_code}\n```"

    print("KERNEL CODE TO EVALUATE:")
    print(custom_cuda)
    print("\n" + "="*50 + "\n")

    # 3. Evaluate Kernel
    print("Starting kernel evaluation...")
    kernel_exec_result = eval_kernel_against_ref(
        ref_arch_src, custom_cuda, verbose=config.verbose, measure_performance=True, 
        num_correct_trials=5, num_perf_trials=100, language=config.language
    )
    
    print(f"Evaluation result for level {config.level} problem {config.problem_id}:")
    print(kernel_exec_result)
    
    if config.log:
        with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}_from_file.txt"), "w") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(f"Kernel File: {config.kernel_file}\n")
            f.write(f"Language: {config.language}\n")
            f.write("="*50 + "\n")
            f.write(str(kernel_exec_result))

if __name__ == "__main__":
    main() 