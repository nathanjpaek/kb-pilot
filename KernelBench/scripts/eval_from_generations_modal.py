from dataclasses import dataclass
import time
import pydra
from pydra import REQUIRED, Config
import json
from tqdm import tqdm
import torch
import os
import multiprocessing as mp
import modal

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.utils import set_gpu_arch, read_file

"""
Batch Evaluation from Existing Generations using Modal

This expects you have generated the kernels and stored them in the runs/{run_name} directory
This eval script will evaluate the kernels against the reference architecture using Modal cloud GPUs
and store the results in the runs/{run_name}/eval_results.json file
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

# Modal app and image configuration
app = modal.App("eval_from_generations")

# GPU architecture mapping
gpu_arch_mapping = {
    "L40S": ["Ada"], 
    "H100": ["Hopper"], 
    "A100": ["Ampere"], 
    "A100-80GB": ["Ampere"], 
    "L4": ["Ada"], 
    "T4": ["Turing"], 
    "A10G": ["Ampere"]
}

# Configure the Docker image for Modal
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "git",
        "gcc-10",
        "g++-10",
        "clang",
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
    )
)


class EvalConfig(Config):
    def __init__(self):
        self.run_name = REQUIRED  # name of the run to evaluate

        self.dataset_src = REQUIRED  # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED

        # subset of problems to evaluate
        self.subset = (None, None)  # (start_id, end_id), these are the logical index

        # Modal GPU configuration
        self.gpu = "L40S"  # GPU type for Modal
        self.gpu_arch = None  # Will be derived from gpu param
        self.batch_size = 10  # Number of parallel evaluations to run per batch

        # Logging
        # Top Directory to Store Runs
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        self.verbose = False

        # Eval settings
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 300  # in seconds
        self.measure_performance = True
        self.language = "cuda"  # can be "cuda" or "tilelang"

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@dataclass
class WorkItem:
    problem_id: int
    sample_id: int


def fetch_ref_arch_from_problem_id(dataset, problem_id: int, dataset_src: str) -> str | None:
    """
    Fetch reference architecture from problem directory
    Either from Hugging Face or Local Dataset
    """
    if dataset_src == "huggingface":
        curr_problem_row = dataset.filter(lambda x: x["problem_id"] == problem_id, num_proc=1, desc=None)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    
    elif dataset_src == "local":
        problem_idx_in_dataset = problem_id - 1  # due to dataset list being 0-indexed locally
        ref_arch_path = dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # verify
    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({problem_id})"
    
    return ref_arch_src


def fetch_kernel_from_disk(run_dir: str, level: int, problem_id: int, sample_id: int) -> str | None:
    """
    Fetch kernel file from disk (stored in runs/{run_name})
    """
    kernel_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
    
    if os.path.exists(kernel_path):
        return read_file(kernel_path)
    else:
        return None


def check_if_eval_exists_local(problem_id: int, sample_id: int, eval_file_path: str) -> bool:
    """
    Check if evaluation result already exists in eval results file
    """
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as f:
            eval_results = json.load(f)
        return str(problem_id) in eval_results
    return False


def add_to_eval_results_file(problem_id: int, sample_id: int, eval_result, eval_file_path: str):
    """
    Add evaluation result to eval results file
    """
    # Load existing results if file exists
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = {}
    
    # Add new result
    eval_results[str(problem_id)] = {
        'sample_id': sample_id,
        'compiled': eval_result.get('compiled', False),
        'correctness': eval_result.get('correctness', False),
        'metadata': eval_result.get('metadata', {}),
        'runtime': eval_result.get('runtime', -1.0),
        'runtime_stats': eval_result.get('runtime_stats', {}),
    }
    
    # Write updated results back to file
    if not os.path.exists(eval_file_path):
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        
    with open(eval_file_path, "w") as f:
        json.dump(eval_results, f)


@app.cls(image=image)
class EvalFunc:
    @modal.method()
    def batch_eval_modal(self, work_items, config_dict, dataset_refs, run_dir):
        """
        Execute batch evaluation on Modal GPU
        
        Args:
            work_items: List of (problem_id, sample_id) tuples
            config_dict: Configuration dictionary
            dataset_refs: Dataset to use for reference architectures
            run_dir: Directory where the generated kernels are stored
            
        Returns:
            List of evaluation results
        """
        from src.eval import eval_kernel_against_ref, KernelExecResult
        from src.utils import set_gpu_arch
        from src.eval import check_metadata_serializable_all_types
        
        # Set GPU architecture
        set_gpu_arch(gpu_arch_mapping[config_dict["gpu"]])
        
        results = []
        
        for problem_id, sample_id in work_items:
            try:
                # Fetch reference architecture
                ref_arch_src = fetch_ref_arch_from_problem_id(dataset_refs, problem_id, config_dict["dataset_src"])
                
                # Fetch kernel
                kernel_src = fetch_kernel_from_disk(run_dir, config_dict["level"], problem_id, sample_id)
                
                if kernel_src is None:
                    print(f"[ERROR] Kernel not found for problem {problem_id} sample {sample_id}")
                    continue
                
                # Evaluate kernel
                eval_result = eval_kernel_against_ref(
                    original_model_src=ref_arch_src,
                    custom_model_src=kernel_src,
                    measure_performance=config_dict["measure_performance"],
                    verbose=config_dict["verbose"],
                    num_correct_trials=config_dict["num_correct_trials"],
                    num_perf_trials=config_dict["num_perf_trials"],
                    language=config_dict["language"]
                )
                
                # Convert to dictionary for JSON serialization
                eval_result_dict = {
                    "compiled": eval_result.compiled,
                    "correctness": eval_result.correctness,
                    "metadata": check_metadata_serializable_all_types(eval_result.metadata),
                    "runtime": eval_result.runtime,
                    "runtime_stats": eval_result.runtime_stats
                }
                
                results.append((problem_id, sample_id, eval_result_dict))
                print(f"[Eval Result] Problem ID: {problem_id}, Sample ID: {sample_id}: {eval_result_dict}")
                
            except Exception as e:
                print(f"[ERROR] Evaluation failed for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}")
                # Return failure result
                metadata = {"error": str(e)}
                eval_result_dict = {
                    "compiled": False,
                    "correctness": False,
                    "metadata": metadata,
                    "runtime": -1.0,
                    "runtime_stats": {}
                }
                results.append((problem_id, sample_id, eval_result_dict))
        
        return results


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    Batch Eval Samples from Particular Run using Modal
    Store Eval Results in specified eval results file
    """
    print(f"Starting Batch Eval with config: {config}")
    
    # Dataset Configurations
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)
    
    num_problems_in_level = len(curr_level_dataset)

    if config.subset == (None, None):
        problem_id_range = range(1, num_problems_in_level)
    else:
        assert config.subset[0] >= 1 and config.subset[1] <= num_problems_in_level, f"Subset range {config.subset} out of range for Level {config.level}"
        problem_id_range = range(config.subset[0], config.subset[1])

    print(f"Evaluating 1 sample each for level {config.level} problems: {problem_id_range}")

    run_dir = os.path.join(config.runs_dir, config.run_name)
    eval_file_path = os.path.join(run_dir, f"eval_results.json")

    # Set GPU architecture from the provided GPU type
    if config.gpu_arch is None:
        config.gpu_arch = gpu_arch_mapping[config.gpu]

    # Identify work to be done
    total_work = []
    for problem_id in range(problem_id_range.start, problem_id_range.stop + 1):  # end index is inclusive
        sample_id = 0  # only evaluate 1 sample for now
        if not check_if_eval_exists_local(problem_id, sample_id, eval_file_path):
            total_work.append((problem_id, sample_id))

    print(f"Starting evaluation on {len(total_work)} unevaluated samples in range: {problem_id_range}")
    
    # Process batches
    with app.run():
        # Process work in batches
        with tqdm(total=len(total_work), desc="Processing batches") as pbar:
            for i in range(0, len(total_work), config.batch_size):
                batch = total_work[i:i+config.batch_size]
                
                # Execute batch evaluation on Modal
                batch_results = EvalFunc.with_options(gpu=config.gpu)().batch_eval_modal.remote(
                    batch, 
                    config.to_dict(), 
                    curr_level_dataset, 
                    run_dir
                )
                
                # Process and store results
                for problem_id, sample_id, eval_result in batch_results:
                    add_to_eval_results_file(problem_id, sample_id, eval_result, eval_file_path)
                
                pbar.update(len(batch))
                print(f"Completed batch {i//config.batch_size + 1}/{(len(total_work) + config.batch_size - 1)//config.batch_size}")


if __name__ == "__main__":
    main()