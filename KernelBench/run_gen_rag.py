import os
import subprocess
from itertools import product
import concurrent.futures
from typing import List, Tuple
import glob
import argparse


def should_generate_kernel(level: int, problem_id: int, language: str = "tilelang") -> bool:
    """Check if we need to generate a kernel for this problem."""
    # Look for existing kernel files
    pattern = f"src/prompts/correct_{language}/level{level}/{level}_{problem_id}*.py"
    existing_files = glob.glob(pattern)

    # If no files exist, we need to generate
    if not existing_files:
        return True

    # If file exists but is wrong or cheating, we need to regenerate
    if any("wrong" in f or "cheat" in f for f in existing_files):
        return True

    # # Check speedup ratio in existing file
    # if existing_files:
    #     with open(existing_files[0], 'r') as f:
    #         content = f.read()
    #         try:
    #             eval_result = content.split('Evaluation Result:\n')[1].split('\n"""')[0]
    #             speedup_ratio = float(eval_result.split("speedup_ratio': ")[1].split("}")[0])
    #             # Regenerate if speedup ratio is less than 0.5x
    #             print(f"Regenerating kernel {existing_files[0]} because speedup ratio is less than 0.5x")
    #             if speedup_ratio < 0.5:
    #                 return True
    #         except (IndexError, ValueError):
    #             # If we can't parse the speedup ratio, regenerate to be safe
    #             return True

    # Otherwise, kernel exists and is correct with good performance
    return False


def run_single_job(level: int, problem_id: int, base_cmd: List[str]) -> None:
    """Run a single generation and evaluation job."""

    print(f"Running Level {level}, Problem {problem_id}")

    cmd = base_cmd + [f"level={level}", f"problem_id={problem_id}"]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running level {level} problem {problem_id}: {e}")


def run_generation_and_eval(language: str, levels: List[int]):
    # Define the levels and problems per level
    all_problems_per_level = {
        1: range(1, 101),  # Level 1 has 100 problems
        2: range(1, 101),  # Level 2 has 100 problems
        3: range(1, 51),  # Level 3 has 50 problems
        4: range(1, 21),  # Level 4 has 20 problems
    }
    
    # Filter to only requested levels
    problems_per_level = {level: all_problems_per_level[level] for level in levels if level in all_problems_per_level}
    
    if not problems_per_level:
        print(f"No valid levels specified. Valid levels are: {list(all_problems_per_level.keys())}")
        return

    # Base command template
    base_cmd = [
        "python",
        "scripts/generate_and_eval_rag_modal.py",
        "dataset_src=huggingface",
        f"language={language}",
        "rag_k=7",
        "gpu=H100",
    ]

    # Create list of all (level, problem_id) pairs
    jobs: List[Tuple[int, int]] = []
    for level in problems_per_level.keys():
        for problem_id in problems_per_level[level]:

            # Skip if kernel already exists and is correct
            if should_generate_kernel(level, problem_id, language):
                jobs.append((level, problem_id))
                # print(f"Adding job: Level {level}, Problem {problem_id}")
                
    print(f"Language: {language}")
    print(f"Levels: {levels}")
    print(f"Total jobs: {len(jobs)}")

    # Run jobs in parallel with max 16 concurrent processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(run_single_job, level, problem_id, base_cmd) for level, problem_id in jobs]

        # Wait for all jobs to complete
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and evaluate kernels for specified levels and language"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="cute",
        choices=["cute", "thunderkittens", "tilelang"],
        help="Target language for kernel generation (default: cute)"
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="List of levels to generate (default: 1 2 3 4). Example: --levels 1 2"
    )
    
    args = parser.parse_args()
    
    run_generation_and_eval(args.language, args.levels)
