import os
import glob

for level in [1, 2, 3, 6]:
    level_dir = f'level{level}'
    if not os.path.exists(level_dir):
        continue
    
    count = 0
    for problem_id in range(1, 101):
        # Use word boundary or specific pattern to avoid partial matches
        pattern = f'{level_dir}/{level}_{problem_id}_*.py'  # Matches 3_1_2.py but not 3_10.py
        base_pattern = f'{level_dir}/{level}_{problem_id}.py'  # Matches exact 3_1.py
        
        solution_files = glob.glob(pattern) + glob.glob(base_pattern)
        if solution_files:
            count += 1
    
    print(f"Level {level}: {count} kernels")
