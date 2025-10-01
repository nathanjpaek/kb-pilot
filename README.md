# TilePilot: A Lightweight Framework for Optimizing GPU Kernels

COMMANDS:

```bash
# Generate and evaluate RAG for a specific problem
python scripts/generate_and_eval_rag_modal.py dataset_src="huggingface" level=1 problem_id=64
python scripts/generate_and_eval_rag_modal.py dataset_src="local" level=6 problem_id=1

# Evaluate the generated kernel for a specific problem
python scripts/generate_and_eval_rag_modal.py dataset_src="huggingface" level=2 problem_id=29 eval_only=true

# Generate and evaluate a few shot for a specific problem
python scripts/generate_and_eval_single_sample_modal.py dataset_src="huggingface" server_type="openai" model_name="o3" verbose=true language=tilelang log=true log_prompt=true log_generated_kernel=true gpu=H100 level=1 problem_id=95

# Multiturn optimization for a specific problem
python scripts/multiturn_optimization.py kernel_file=src/prompts/correct_tilelang/level2/2_3.py level=2 problem_id=3

# Bulk generate RAG for all problems
python run_gen_rag.py

# Bulk evaluate the generated kernels
python run_eval.py

# Bulk generate few shot for all problems
python run_gen.py
```