import os
import subprocess
from itertools import product
import concurrent.futures
from typing import List, Tuple
import glob

def run_single_job(eval_file_path: str, base_cmd: List[str]) -> None:
    """Run a single evaluation job."""
    print(f"Running evaluation for {eval_file_path}")

    # Extract level and problem_id from filename (e.g. "2_3.py" -> level=2, problem_id=3)
    filename = os.path.basename(eval_file_path)
    filename = filename.replace(".py", "")

    split = filename.split("_")
    level = int(split[0])
    problem_id = int(split[1])

    cmd = base_cmd + [
        f"eval_file_path={eval_file_path}",
        f"level={level}",
        f"problem_id={problem_id}"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for {eval_file_path}: {e}")

def run_generation_and_eval():
    # Base command template
    base_cmd = [
        "python", "scripts/generate_and_eval_single_sample_modal.py",
        "dataset_src=huggingface",
        "server_type=openai",
        "model_name=o3",
        "verbose=true",
        "language=tilelang",
        "log=true",
        "log_prompt=true",
        "log_generated_kernel=true",
        "gpu=H100",
        "eval_only=true"
    ]

    # Run eval on all files in src/prompts/correct_tilelang level directories
    eval_files = []
    
    # Search through all level directories
    correct_tilelang_path = "src/prompts/correct_tilelang"
    for level_dir in os.listdir(correct_tilelang_path):
        level_path = os.path.join(correct_tilelang_path, level_dir)
        if os.path.isdir(level_path) and level_dir.startswith("level"):
            # Add all .py files from this level directory
            for f in os.listdir(level_path):
                if f.endswith(".py"):
                    eval_files.append(os.path.join(level_path, f))
    
    eval_files = set(eval_files)
    
    print(f"Evaluating {len(eval_files)} TileLang kernels")

    # Run jobs in parallel with max 16 concurrent processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(run_single_job, eval_file, base_cmd)
            for eval_file in eval_files
        ]
        
        # Wait for all jobs to complete
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    run_generation_and_eval()
