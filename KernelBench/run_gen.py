import os
import subprocess
from itertools import product
import concurrent.futures
from typing import List, Tuple
import glob

def should_generate_kernel(level: int, problem_id: int) -> bool:
    """Check if we need to generate a kernel for this problem."""
    # Look for existing kernel files
    pattern = f"src/prompts/correct_tilelang/level{level}/{level}_{problem_id}.py"
    existing_files = glob.glob(pattern)
    
    # If no files exist, we need to generate
    if not existing_files:
        return True
        
    # If file exists but is wrong or cheating, we need to regenerate
    if any("wrong" in f or "cheat" in f for f in existing_files):
        return True
        
    # Otherwise, kernel exists and is correct
    return False

def run_single_job(level: int, problem_id: int, base_cmd: List[str]) -> None:
    """Run a single generation and evaluation job."""
    # Skip if kernel already exists and is correct
    print(f"Running Level {level}, Problem {problem_id}")

    cmd = base_cmd + [f"level={level}", f"problem_id={problem_id}"]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running level {level} problem {problem_id}: {e}")

def run_generation_and_eval():
    # Define the levels and problems per level
    problems_per_level = {
        # 1: range(1, 101),  # Level 1 has 100 problems
        # 2: range(1, 101),  # Level 2 has 100 problems
        # 3: range(1, 51),  # Level 3 has 50 problems
        6: range(1, 101),  # Level 6 has 100 problems
    }

    # Base command template
    base_cmd = [
        "python", "scripts/generate_and_eval_single_sample_modal.py",
        "dataset_src=local",
        "server_type=openai",
        "model_name=o3",
        "verbose=true",
        "language=tilelang",
        "log=true",
        "log_prompt=true",
        "log_generated_kernel=true",
        "gpu=H100",
        "use_tilelang_fewshot=true"
    ]

    # Create list of all (level, problem_id) pairs
    jobs: List[Tuple[int, int]] = []
    for level in problems_per_level.keys():
        for problem_id in problems_per_level[level]:
            jobs.append((level, problem_id))

    # Run jobs in parallel with max 5 concurrent processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(run_single_job, level, problem_id, base_cmd)
            for level, problem_id in jobs
        ]
        
        # Wait for all jobs to complete
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    run_generation_and_eval()
