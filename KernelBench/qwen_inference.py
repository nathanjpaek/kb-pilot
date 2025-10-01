MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

KB_LEVEL = 1
KB_PROBLEM = 6

# ! pip install vllm datasets

import os
import datasets
from datetime import datetime
from vllm import LLM, SamplingParams

from scripts.tilelang_paperinfo_prompt import PAPER_PROMPT
from scripts.tilelang_icl_prompt import ICL_PROMPT

# Load the model
# max_model_len: Maximum sequence length the model can handle (context window size)
#   For coding tasks, a larger context length (4096-8192) is recommended to fit more code
# gpu_memory_utilization: Fraction of GPU memory to use (0.9 = 90% of available memory)
#   This controls the trade-off between memory usage and performance
llm = LLM(model=MODEL_NAME, max_model_len=8192, gpu_memory_utilization=0.7, enforce_eager=True)
SOURCE = open("../KernelBench/KernelBench/level1/6_Matmul_with_large_K_dimension_.py").read()
TILELANG_SOURCE = open("../KernelBench/src/prompts/correct_tilelang/level1/1_6.py").read()

ICL_EXAMPLES = ICL_PROMPT
TILELANG_INFO = PAPER_PROMPT

problem = "level2/12_Gemm_Multiply_LeakyReLU"
PYTORCH_MODEL = open(f"../KernelBench/KernelBench/{problem}.py").read()


SYSTEM_PROMPT = """You are an expert kernel engineer specializing in TileLang.
1. First, analyze the PyTorch model inside <think> and </think> tags. Here you can reason about the computation, identify optimization opportunities, and plan your kernel implementation.
2. When confident, output your optimized TileLang implementation inside <code> and </code> tags."""

USER_PROMPT = """Task: Optimize the given PyTorch model by implementing custom TileLang kernels.

Rules:
- You must create a ModelNew class that inherits from nn.Module
- Define and compile TileLang kernels in the __init__ method
- Call the compiled kernel in the forward method
- Focus on replacing PyTorch operators with efficient TileLang implementations
- Ensure your implementation maintains the same functionality as the original model
- Use proper tilelang.language (T) constructs for parallelism and memory access patterns

This is an example PyTorch model that you would receive:

<pytorch_source>
{SOURCE}
</pytorch_source>

This is an example of expected TileLang kernel implementation to speed up the matmul operation:

<tilelang_source>
{TILELANG_SOURCE}
</tilelang_source>

The PyTorch model that you must optimize using TileLang is:

<pytorch_model>
{PYTORCH_MODEL}
</pytorch_model>

Here are additional examples of kernels written in TileLang:

<icl_examples>
{ICL_EXAMPLES}
</icl_examples>

Here is additional information about TileLang: 

<tilelang_info>
{TILELANG_INFO}
</tilelang_info>

You must use this format:

<think>
[Brief analysis and planning - identify key operations to optimize or fuse]
</think>

<code>
[Your TileLang implementation]
</code>
"""

# Retrieve the problem from the dataset
ds = datasets.load_dataset("ScalingIntelligence/KernelBench")

curr_level_ds = ds[f"level_{KB_LEVEL}"]
curr_problem_row = curr_level_ds.filter(lambda x: x["problem_id"] == KB_PROBLEM)
ref_arch_src = curr_problem_row["code"][0]
problem_name = curr_problem_row["name"][0]

user_prompt = USER_PROMPT.format(
    SOURCE=SOURCE,
    TILELANG_SOURCE=TILELANG_SOURCE,
    PYTORCH_MODEL=PYTORCH_MODEL,
    ICL_EXAMPLES=ICL_EXAMPLES,
    TILELANG_INFO=TILELANG_INFO,
)

# Create conversations with the correct prompt field
conversations = [
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
]

# Perform inference and save the results
outputs = llm.chat(conversations, sampling_params=SamplingParams(max_tokens=8192))

# Create timestamp for filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{problem_name}_{timestamp}.txt"

# Create results directory if it doesn't exist
path = f"qwen_results/{MODEL_NAME.split('/')[-1]}"
os.makedirs(path, exist_ok=True)

# Save output with problem name and timestamp
for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text
    with open(f"{path}/{filename}", "w") as f:
        f.write(generated_text)