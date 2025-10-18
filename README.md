# TilePilot: Writing Correct and Fast DSL Kernels for the GPU

COMMANDS:

```bash
# Generate and evaluate RAG for a specific problem, in a specific DSL
python scripts/generate_and_eval_rag_modal.py dataset_src="huggingface" language=thunderkittens level=1 problem_id=64
python scripts/generate_and_eval_rag_modal.py dataset_src="local" language=cute level=6 problem_id=1

# Evaluate a kernel that you are manually debugging
python scripts/generate_and_eval_rag_modal.py dataset_src="huggingface" language=tilelang level=2 problem_id=29 eval_only=true eval_file_path={PATH_TO_KERNEL_FILE}

# Generate and evaluate a few shot for a specific problem
python scripts/generate_and_eval_single_sample_modal.py dataset_src="huggingface" server_type="openai" model_name="o3" verbose=true language=tilelang log=true log_prompt=true log_generated_kernel=true gpu=H100 level=1 problem_id=95

# Multiturn optimization for a specific problem
python scripts/multiturn_optimization.py kernel_file=src/prompts/correct_tilelang/level2/2_3.py level=2 problem_id=3

# Bulk generate RAG for all problems in level 1-3 in a specific DSL
python run_gen_rag.py --language=cute --levels=1 2 3
 
# Bulk evaluate the generated kernels
python run_eval.py

# Bulk generate few shot for all problems
python run_gen.py
```