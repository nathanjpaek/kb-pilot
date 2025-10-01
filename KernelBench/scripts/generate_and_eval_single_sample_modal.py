import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import modal
import re

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_generate_custom_tilelang_from_prompt_template, prompt_generate_custom_tilelang_fewshot_and_template
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets
from scripts.debug_suggestor_plan import PLAN_2_12, PLAN_3_9, PLAN_3_2
from scripts.tilelang_icl_prompt import ICL_PROMPT
from scripts.tilelang_guideline_prompt import GUIDELINE_PROMPT
from scripts.tilelang_paperinfo_prompt import PAPER_PROMPT
from scripts.tilelang_elemdocs_prompt import ELEMDOCS_PROMPT
from scripts.tilelang_flashmladocs_prompt import FLASHMLADOCS_PROMPT
from scripts.tilelang_cumsum_prompt import CUMSUM_PROMPT
from scripts.tilelang_conv_prompt import CONV_PROMPT


def strip_docstring_from_code(code: str) -> str:
    """
    Remove the docstring from Python code if it exists at the beginning.
    This is useful for eval_only mode to remove old evaluation results.
    """
    # Remove leading whitespace and find the start
    lines = code.split('\n')
    
    # Find first non-empty, non-comment line
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            start_idx = i
            break
    
    # Check if it starts with a docstring
    if start_idx < len(lines):
        stripped_line = lines[start_idx].strip()
        if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
            quote_type = '"""' if stripped_line.startswith('"""') else "'''"
            
            # Find the end of the docstring
            if stripped_line.count(quote_type) >= 2:
                # Single line docstring
                return '\n'.join(lines[start_idx + 1:])
            else:
                # Multi-line docstring
                for i in range(start_idx + 1, len(lines)):
                    if quote_type in lines[i]:
                        return '\n'.join(lines[i + 1:])
    
    return code


app = modal.App("eval_single_sample")

"""
Generate and evaluate a single sample
Easiest way to get started, to test a single problem for experimentation or debugging
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}

class EvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        self.language = "cuda"
        # Problem Specification
        self.level = REQUIRED
        # NOTE: this is the logical index (problem id the problem_name)\
        self.problem_id = REQUIRED

        # Evaluation
        # local (requires a GPU), modal (cloud GPU) coming soon
        self.eval_mode = "modal"
        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu = "L40S"
        self.gpu_arch = ['Ada']


        # Inference config
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        self.max_tokens = 4096
        self.temperature = 0.0
        
        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = False

        self.log = False
        self.log_prompt = False
        self.log_generated_kernel = False
        self.log_eval_result = False
        
        self.eval_only = False
        self.eval_file_path = None
        
        # TileLang few-shot configuration
        self.use_tilelang_fewshot = False

    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git",
                "gcc-10",
                "g++-10",
                "clang" # note i skip a step 
                )
    .pip_install(  # required to build flash-attn
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
        "python-dotenv", # NATHAN ADDED THIS LINE 
        "tilelang",
        "apache-tvm",
    )
    .add_local_python_source("scripts", "src")  # Add local Python modules
    .add_local_dir("KernelBench", "/root/KernelBench")  # Add the dataset files
)

@app.cls(image=image)
class EvalFunc:

    @modal.method()
    def eval_single_sample_modal(self, ref_arch_src, custom_cuda, verbose, gpu_arch, language, entry_point=None):
        # SET DEFAULT DTYPE TO FLOAT16 AT THE VERY BEGINNING OF MODAL FUNCTION
        torch.set_default_dtype(torch.float16)
        
        # 3. Evaluate Kernel
        # NOTE: no need to wrap around process here as only a single sample
        # see batch eval for examples of process isolation
        from src.eval import eval_kernel_against_ref
        from src.utils import set_gpu_arch
        set_gpu_arch(gpu_arch)
        return eval_kernel_against_ref(
            ref_arch_src, custom_cuda, verbose=verbose, measure_performance=True, num_correct_trials=5, num_perf_trials=100, language=language, entry_point=entry_point
        )

@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    
    """
    Keep it simple: Generate and evaluate a single sample
    """
    print(f"Starting Eval with config: {config}")
    
    print(">>> Setting default dtype to float16 <<<")
    torch.set_default_dtype(torch.float16)

    # Configurations
    
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)
        
    # Problem Checks
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Start Generation + Evaluation for Level {config.level} Problem {config.problem_id}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"


    # 1. Fetch Problem
    if config.dataset_src == "huggingface":

        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        problem_name = problem_name.replace(".py", "")
        ref_arch_src = read_file(ref_arch_path)

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"
    
    entry_point = None
    if config.level == 9:
        first_underscore = problem_name.find("_")
        entry_point = problem_name[first_underscore+1:]
        print(f"entry_point: {entry_point}")
    
    
    # 2. Generate Sample
    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                        model_name=config.model_name,
                                                        temperature=config.temperature,
                                                        max_tokens=config.max_tokens, 
                                                        verbose=config.verbose, 
                                                        time_generation=True)
    
    if config.language == "cuda":
        custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
    elif config.language == "tilelang":
        if config.use_tilelang_fewshot:
            # Use few-shot prompting with examples from correct_tilelang
            print(">>> Using TileLang few-shot examples (all available) <<<")
            custom_cuda_prompt = f"""
You are given information about TileLang here: \n{PAPER_PROMPT}\n
You are given additional tips about TileLang here: \n{GUIDELINE_PROMPT}\n

IMPORTANT GUIDELINES:
- DO NOT USE torch.nn (except for Parameter, containers, and init). This means you cannot use torch.nn.functional, F, torch.nn.Conv3d, or any other torch.nn modules.
- When giving your output PLEASE remember to not add additional text within your code block!
- Focus on generating efficient TileLang implementations using TileLang-specific optimizations
- Optimize for performance on NVIDIA H100 (e.g. shared memory, fusion of operations, warp primitives, vectorization,...)

"""
            # Use few-shot prompt with all available examples
            custom_cuda_prompt += prompt_generate_custom_tilelang_fewshot_and_template(ref_arch_src)
        else:
            # Use original prompting approach
            print(">>> Using original TileLang prompting (no few-shot) <<<")
            # First add general TileLang information and guidelines
            custom_cuda_prompt = f"""
You are given information about TileLang here: \n{PAPER_PROMPT}\n
You are given additional tips about TileLang here: \n{GUIDELINE_PROMPT}\n
You are given in context examples of TileLang kernels here: \n{ICL_PROMPT}\n

IMPORTANT GUIDELINES:
- DO NOT USE torch.nn (except for Parameter, containers, and init). This means you cannot use torch.nn.functional, F, torch.nn.Conv3d, or any other torch.nn modules.
- When giving your output PLEASE remember to not add additional text within your code block!
- Focus on generating efficient TileLang implementations using TileLang-specific optimizations
- Optimize for performance on NVIDIA H100 (e.g. shared memory, kernel fusion, warp primitives, vectorization,...)

Now, here is your task:
"""
            # Then add the specific task prompt
            custom_cuda_prompt += prompt_generate_custom_tilelang_from_prompt_template(ref_arch_src)

            # # Add error information if available
            # ERR = """
            # compiled=True correctness=False metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'runtime_error': 'Kernel call failed: TMA Desc Addr:   0x7fc24e6b5e80\nformat         6\ndim            2\ngmem_address   0x2b97e2000000\nglobalDim      0x7fc24e6b5e40\nglobalStrides  0x7fc24e6b5e58\nboxDim         0x7fc24e6b5e30\nelementStrides 0x7fc24e6b5e38\ninterleave     0\nswizzle        2\nl2Promotion    2\noobFill        0\nError: Failed to initialize the TMA descriptor A_desc'} runtime=-1.0 runtime_stats={}
            # """
            # custom_cuda_prompt += f"\nYou are given a previous error message here: {ERR}\n"
    else:
        raise ValueError(f"Unsupported language specified: {config.language}. Choose 'cuda' or 'tilelang'.")

    if config.log_prompt:
        with open(os.path.join(config.logdir, f"prompt_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
            f.write(custom_cuda_prompt)
    
    if not config.eval_only:
        print(">>> PROMPTING LLM TO GENERATE CODE <<<")
        # print(f"CUSTOM PROMPT: {custom_cuda_prompt}" + "\n")
        custom_cuda = inference_server(custom_cuda_prompt)
    else:
        print(">>> USING CODE WITH EVAL ONLY <<<")
        if config.eval_file_path:
            path = config.eval_file_path
        else:
            path = f"src/prompts/correct_tilelang/level{config.level}/{config.level}_{config.problem_id}.py"

        # Read the file and strip the old docstring to get fresh evaluation results
        raw_code = open(path).read()
        clean_code = strip_docstring_from_code(raw_code)
        custom_cuda = "```python\n" + clean_code + "\n```"
        print(f">>> Stripped old docstring, using clean code for evaluation <<<")

    custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
    # print("GENERATED CODE: " + custom_cuda)

    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"

    #print(custom_cuda)
    
    # this should be optional
    if config.log:
        with open(os.path.join(config.logdir, f"generated_kernel_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
            f.write(custom_cuda)

    print(f"ref_arch_src: {ref_arch_src}\n\n")
    print(f"custom_cuda: {custom_cuda}\n\n")

    with app.run():
        kernel_exec_result = EvalFunc.with_options(gpu=config.gpu)().eval_single_sample_modal.remote(ref_arch_src, custom_cuda, config.verbose, gpu_arch_mapping[config.gpu], config.language, entry_point)
        
        print(f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}")
        
        if kernel_exec_result.correctness:
            if config.eval_file_path is None:
                os.makedirs("src/prompts/correct_tilelang2", exist_ok=True)
                path = f"src/prompts/correct_tilelang2/{config.level}_{config.problem_id}.py"
            else:
                path = config.eval_file_path

            with open(path, "w") as f:
                print(f">>> Writing correct TileLang kernel to {f.name} <<<")
                f.write(f'''"""
Problem Name: {problem_name}
Evaluation Result:
{str(kernel_exec_result)}
"""

{custom_cuda}''')
        
        if config.log:
            with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(str(kernel_exec_result))

if __name__ == "__main__":
    main()