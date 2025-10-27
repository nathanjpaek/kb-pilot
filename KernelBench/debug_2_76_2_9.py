#!/usr/bin/env python3
"""
Debug script to test problems 76 and 9 with verbose output
"""
import sys
import os

# Ensure we can import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.eval import eval_kernel_against_ref

def test_problem_76():
    print("=" * 80)
    print("Testing Problem 76: Gemm_Add_ReLU")
    print("=" * 80)
    
    with open('KernelBench/level2/76_Gemm_Add_ReLU.py', 'r') as f:
        ref_src = f.read()
    
    with open('src/prompts/correct_cute/level2/2_76.py', 'r') as f:
        custom_src = f.read()
    
    result = eval_kernel_against_ref(
        ref_src,
        custom_src,
        verbose=True,
        language='cute',
        num_correct_trials=1,
        measure_performance=False
    )
    
    print("\n" + "=" * 80)
    print("RESULT for Problem 76:")
    print(f"  Compiled: {result.compiled}")
    print(f"  Correctness: {result.correctness}")
    print(f"  Runtime: {result.runtime}")
    print(f"  Metadata: {result.metadata}")
    print("=" * 80)
    return result

def test_problem_9():
    print("\n\n" + "=" * 80)
    print("Testing Problem 9: Matmul_Subtract_Multiply_ReLU")
    print("=" * 80)
    
    with open('KernelBench/level2/9_Matmul_Subtract_Multiply_ReLU.py', 'r') as f:
        ref_src = f.read()
    
    with open('src/prompts/correct_cute/level2/2_9.py', 'r') as f:
        custom_src = f.read()
    
    result = eval_kernel_against_ref(
        ref_src,
        custom_src,
        verbose=True,
        language='cute',
        num_correct_trials=1,
        measure_performance=False
    )
    
    print("\n" + "=" * 80)
    print("RESULT for Problem 9:")
    print(f"  Compiled: {result.compiled}")
    print(f"  Correctness: {result.correctness}")
    print(f"  Runtime: {result.runtime}")
    print(f"  Metadata: {result.metadata}")
    print("=" * 80)
    return result

if __name__ == "__main__":
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a CUDA GPU.")
        sys.exit(1)
    
    result_76 = test_problem_76()
    result_9 = test_problem_9()
    
    print("\n\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Problem 76: {'✓ PASS' if result_76.correctness else '✗ FAIL'}")
    print(f"Problem 9:  {'✓ PASS' if result_9.correctness else '✗ FAIL'}")
    print("=" * 80)


