# CuTe-DSL RAG Agent

This codebase is heavily based on the KernelBench codebase!

As such it requires:
1. A logged-in Modal account
2. An OpenAI API Key with sufficient credits
3. A conda environment with the sufficient prerequisite packages installed.

We have instructions below but if you have trouble please contact Willy or Mafer for assistance! We are more than happy to demo the project and prove that it does indeed compile/work (there is just a lot of dependencies and annoying technical details to get setup in the first place)

**Prerequisites:**
- Activate conda environment: `conda activate kb-pilot` (or your environment name)
- Set OpenAI API key (for kernel generation): `export OPENAI_API_KEY=your_key_here`
- Ensure Modal is set up (for evaluation): `modal token new`

If you would like to run this code for yourself, please contact us for an OpenAI key/Modal account, or you can also just provide one yourself. 

**Relevant Commands**
### 1. Generate Kernel using RAG for a Specific Problem

Generate and evaluate a kernel using RAG for a specific level and problem ID:

```bash
python scripts/generate_and_eval_rag_modal.py dataset_src=huggingface language=cute level=1 problem_id=5 eval_only=False
```

Generated kernels are saved to `src/prompts/correct_{language}/level{level}/{level}_{problem_id}.py`

**Parameters:**
- `language`: `cute`, you can also try tilelang or thunderkittens but this is not fully functional.
- `level`: Problem level (1-4)
- `problem_id`: Specific problem ID within that level
- `rag_k`: Number of RAG examples to retrieve (default: 5)
- `gpu`: GPU type for evaluation (`H100`, `A100`, `L40S`, etc.)

### 2. Evaluate a generated kernel

If you already have a generated kernel saved to `src/prompts/correct_{language}/level{level}/{level}_{problem_id}.py` and want to just evaluate it on model (compare to the PyTorch reference), then you can run something like:

```bash
python scripts/generate_and_eval_rag_modal.py dataset_src=huggingface language=cute level=1 problem_id=5 eval_only=True
```

Basically just set eval_only=True. This will run on the modal backend.


### 3. Multiturn Optimization

If your initially generated kernel has syntax/compiler errors or is not that performant, you can use our multiturn optimizer script. In a nutshell it essentially concatenates the Modal output and retries with the same RAG-based system (we found this to improve correctness and performance significantly). You can run it with something like this (make sure the kernel in question exists!):
```bash
# For CuTe kernels
python multiturn_optimizer_willy.py level=1 problem_id=5 language=cute
```

Optimized kernels overwrite the original files in `src/prompts/correct_{language}/level{level}/`

**IMPORTANT:** Note that the kernel being optimized MUST be in the format `{level}_{problem_id}.py` so that the script parses it correctly.

### 4. Document Chunking
Run hierarchical summarization to create comprehensive CuTe guidelines from documentation:
```bash
python -m scripts.cute_guideline_prompt
```

**Output location:** Final guideline prompt saved to `scripts/.cute_summary_cache/CUTE_GUIDELINE_PROMPT_FINAL.txt`

This uses the chinking method we learned in class to get the final guideline prompt. This assumes you have all the documentation for the DSL in question, in this case we provided all the ones for CuTe-DSL that we could find.

Now you have all the necessary commands to make your own bash script and start evaluating kernels!
