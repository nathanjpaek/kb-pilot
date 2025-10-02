"""
Generate and Evaluate DSL Kernels using DSPy RAG + Modal

This script uses the high-performance DSPy RAG system for DSL generation,
then evaluates the generated kernels on Modal infrastructure.

Optimized for use with OpenAI O3 and resouce-unconstrained environments.
"""

import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import modal
import dspy
import textwrap

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor_rag import prompt_generate_custom_dsl_rag_enhanced
from src.utils import extract_first_code, extract_code_block, extract_all_code_blocks, create_tk_makefile, set_gpu_arch, read_file

# TileLang-specific prompts
from scripts.tilelang_paperinfo_prompt import TILELANG_PAPER_PROMPT
from scripts.tilelang_guideline_prompt import TILELANG_GUIDELINE_PROMPT

# ThunderKittens-specific prompts
from scripts.tk_paperinfo_prompt import TK_PAPER_PROMPT
from scripts.tk_guideline_prompt import TK_GUIDELINE_PROMPT



def strip_docstring_from_code(code: str) -> str:
    """
    Remove the docstring from Python code if it exists at the beginning.
    This is useful for eval_only mode to remove old evaluation results.
    """
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


app = modal.App("eval_rag_dsl")

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}


class RAGEvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Language/DSL
        self.language = "tilelang"  # Options: "tilelang", "tk", "cuda"

        # Problem Specification
        self.level = REQUIRED
        self.problem_id = REQUIRED

        # Evaluation
        self.eval_mode = "modal"
        self.gpu = "H100"
        self.gpu_arch = ['Hopper']

        # DSPy Model Configuration
        self.dspy_model = "openai/o3"  # Options: "openai/o3-mini", "openai/o1-preview", "openai/gpt-4o"
        self.dspy_temperature = 1.0
        
        # RAG Configuration
        self.rag_k = 5  # Number of examples to retrieve
        
        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/rag_eval_logs")
        self.verbose = False

        self.log = True
        self.log_generated_kernel = True
        self.log_eval_result = True
        
        self.eval_only = False
        self.eval_file_path = None

    def verbose_logging(self):
        self.log = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"RAGEvalConfig({self.to_dict()})"


cuda_version = "12.8.0"  # Updated from 12.4.0 to 12.8.0
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git",
                "gcc-10",
                "g++-10",
                "clang"
                )
    .pip_install(
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
        "pyutils",
        "ninja",
        "utils",
        "pybind11",
        "python-dotenv",
        "tilelang",
        "apache-tvm",
        "dspy-ai"
    )
    .env({"THUNDERKITTENS_ROOT": "/root/ThunderKittens"})
    .add_local_python_source("scripts", "src")
    .add_local_dir("KernelBench", "/root/KernelBench")
    .add_local_dir("correct_tk", "/root/correct_tk")
    .add_local_dir("ThunderKittens", "/root/ThunderKittens")
)

@app.cls(image=image)
class EvalFunc:

    @modal.method()
    def eval_single_sample_modal(self, ref_arch_src, custom_cuda, verbose, gpu_arch, language, entry_point=None, cu_code: str | None = None):
        # SET DEFAULT DTYPE TO FLOAT16 ONLY FOR TILELANG
        if language == "tilelang":
            torch.set_default_dtype(torch.float16)
        
        from src.eval import eval_kernel_against_ref
        from src.utils import set_gpu_arch, create_tk_makefile
        import os, subprocess, sys

        py_code = """
import tk_kernels
import torch
import torch.nn as nn

INPUT_DTYPE = torch.bfloat16
OUTPUT_DTYPE = torch.float

M = 16
N = 16384
K = N

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        output = torch.zeros(M, N, dtype=OUTPUT_DTYPE, device='cuda')

        A = A.cuda()
        B = B.cuda()
        tk_kernels.dispatch_micro(A, B, output, K)
        
        return output


A = torch.rand(M, K, dtype=INPUT_DTYPE) / K  # [16, 16384]
B = torch.rand(K, N, dtype=INPUT_DTYPE) / K  # [16384, 16384]

A = A.cuda()
B = B.cuda()

output_ref = torch.matmul(A, B).to(OUTPUT_DTYPE)
print("Ref output shape:", output_ref.shape)
print("Ref output mean:", output_ref.mean())


model = ModelNew().cuda()
output = model(A, B)
print("TK Output shape:", output.shape)
print("TK Output mean:", output.mean())

# import pdb; pdb.set_trace()

assert torch.allclose(output, output_ref, atol=1e-2)
        """

        cu_code = """
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_WORKERS (1)
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

// Tile dimensions
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// Global memory descriptors
using a_gl = gl<bf16, -1, -1, -1, -1, st<bf16, TILE_M, TILE_K>>;
using b_gl = gl<bf16, -1, -1, -1, -1, st<bf16, TILE_K, TILE_N>>;
using c_gl = gl<float, -1, -1, -1, -1, st<float, TILE_M, TILE_N>>;

struct micro_globals {
    a_gl a;
    b_gl b;
    c_gl c;
    int K; // Total K dimension to loop over
    
    dim3 grid() const { 
        return dim3((c.cols() + TILE_N - 1) / TILE_N, 
                   (c.rows() + TILE_M - 1) / TILE_M); 
    }
    dim3 block() const { return dim3(NUM_THREADS); }
    int dynamic_shared_memory() const { return 224000; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Allocate shared memory tiles
    st<bf16, TILE_M, TILE_K> (&a_s) = al.allocate<st<bf16, TILE_M, TILE_K>>();
    st<bf16, TILE_K, TILE_N> (&b_s) = al.allocate<st<bf16, TILE_K, TILE_N>>();
    st<float, TILE_M, TILE_N> (&c_s) = al.allocate<st<float, TILE_M, TILE_N>>();

    // Register tiles
    rt<bf16, TILE_M, TILE_K, ducks::rt_layout::row> a_reg;
    rt<bf16, TILE_K, TILE_N, ducks::rt_layout::col> b_reg;
    rt<float, TILE_M, TILE_N, ducks::rt_layout::row> accum;
    kittens::warp::zero(accum);

    // Loop over K dimension in tiles
    for (int k = 0; k < g.K; k += TILE_K) {
        // Load tiles from global memory using block indices
        kittens::warpgroup::load(a_s, g.a, {tile_row * TILE_M, k});
        kittens::warpgroup::load(b_s, g.b, {k, tile_col * TILE_N});
        __syncthreads();

        // Load to registers
        kittens::warp::load(a_reg, a_s);
        kittens::warp::load(b_reg, b_s);
        __syncthreads();

        // Perform matrix multiplication on tiles
        kittens::warp::mma_AB(accum, a_reg, b_reg, accum);

        __syncthreads();
    }

    // Store result using block indices
    kittens::warp::store(c_s, accum);
    __syncthreads();
    kittens::warpgroup::store(g.c, c_s, {tile_row * TILE_M, tile_col * TILE_N});
    __syncthreads();

}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = 50480;
    cudaFuncSetAttribute(
        micro_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    m.doc() = "tk_kernels python module";
    kittens::py::bind_kernel<micro_tk, micro_globals>(m, "micro_tk", 
        &micro_globals::a, 
        &micro_globals::b, 
        &micro_globals::c, 
        &micro_globals::K);
    kittens::py::bind_function<dispatch_micro, micro_globals>(m, "dispatch_micro", 
        &micro_globals::a, 
        &micro_globals::b, 
        &micro_globals::c, 
        &micro_globals::K);
}
        """
        
        # If ThunderKittens, write and compile the kernel inside Modal container
        if language == "thunderkittens" and cu_code is not None:
            # Determine TK kernel dir consistently with src.eval
            # src.eval computes TK dir as parent_of_src/correct_tk
            src_dir = os.path.dirname(__import__('src').__file__)
            repo_top = os.path.abspath(os.path.join(src_dir, ".."))
            tk_kernel_dir = os.path.join(repo_top, "correct_tk")
            os.makedirs(tk_kernel_dir, exist_ok=True)

            cu_file_path = os.path.join(tk_kernel_dir, "custom_tk.cu")
            with open(cu_file_path, "w") as f:
                f.write(cu_code)

            # Create Makefile and compile
            create_tk_makefile(tk_kernel_dir, gpu="H100")
            try:
                subprocess.run(["make", "clean"], cwd=tk_kernel_dir, check=False, capture_output=True, text=True)
                result = subprocess.run(["make"], cwd=tk_kernel_dir, check=True, capture_output=True, text=True, env={**os.environ})
            except subprocess.CalledProcessError as e:
                print("[Modal TK Compile] Failed to compile ThunderKittens kernel")
                print("STDOUT:\n" + (e.stdout or ""))
                print("STDERR:\n" + (e.stderr or ""))
                raise
            # Ensure path is importable
            if tk_kernel_dir not in sys.path:
                sys.path.append(tk_kernel_dir)
                
        set_gpu_arch(gpu_arch)
        # Sanitize custom CUDA/Python wrapper code to avoid indentation errors
        if isinstance(custom_cuda, str):
            custom_cuda = textwrap.dedent(custom_cuda).lstrip()
        return eval_kernel_against_ref(
            ref_arch_src, custom_cuda, verbose=verbose, measure_performance=True, 
            num_correct_trials=5, num_perf_trials=100, language=language, entry_point=entry_point
        )


def configure_dspy(model_name: str, temperature: float = 0.0):
    """Configure DSPy with the specified model"""
    print(f"🤖 Configuring DSPy with model: {model_name}")
    
    # Configure the language model
    lm = dspy.LM(model_name, temperature=temperature, max_tokens=20000)
    dspy.configure(lm=lm)
    
    print(f"✅ DSPy configured successfully with {model_name}")
    return lm


@pydra.main(base=RAGEvalConfig)
def main(config: RAGEvalConfig):
    """
    Generate TileLang kernels using DSPy RAG and evaluate on Modal
    """
    print(f"Starting RAG Evaluation with config: {config}")
    
    
    if config.language == "tilelang":
        print(">>> Setting default dtype to float16 (TileLang only) <<<")
        torch.set_default_dtype(torch.float16)

    # Configure DSPy with the specified model
    lm = configure_dspy(config.dspy_model, config.dspy_temperature)

    # Load dataset
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)
        
    # Problem validation
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"

    # 1. Fetch Problem
    if config.dataset_src == "huggingface":
        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]
        problem_name = os.path.basename(ref_arch_path)
        problem_name = problem_name.replace(".py", "")
        ref_arch_src = read_file(ref_arch_path)
        
    print(f"Start RAG Generation + Evaluation for Level {config.level} Problem {config.problem_id}: {problem_name}")

    # Extract problem number from problem name
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"
    
    # 2. Generate DSL Code using DSPy RAG
    if not config.eval_only:
        print(f">>> GENERATING {config.language.upper()} CODE USING DSPY RAG <<<")

        if config.language == "tilelang":
            PAPER_PROMPT = TILELANG_PAPER_PROMPT
            GUIDELINE_PROMPT = TILELANG_GUIDELINE_PROMPT
        elif config.language == "thunderkittens":
            PAPER_PROMPT = TK_PAPER_PROMPT
            GUIDELINE_PROMPT = TK_GUIDELINE_PROMPT
        else:
            raise ValueError(f"Unsupported language: {config.language}. Use 'tilelang' or 'thunderkittens'")
        
        try:
            # Use the high-performance RAG system
            tilelang_code = prompt_generate_custom_dsl_rag_enhanced(
                ref_arch_src=ref_arch_src,
                language=config.language,
                paper_prompt=PAPER_PROMPT,
                guideline_prompt=GUIDELINE_PROMPT,
                problem_description=f"Optimize {problem_name}",
                k=config.rag_k,
                current_level=config.level,
                current_problem_id=config.problem_id
            )
            
            # print(f"TileLang Code: {tilelang_code}")
            
            print(f"✅ RAG generation successful for {problem_name}")
            
            # Log DSPy prompt history to file
            if config.log:
                try:
                    print("📝 Saving DSPy prompt history...")
                    history_file = os.path.join(config.logdir, f"dspy_history_level_{config.level}_problem_{config.problem_id}.txt")
                    
                    # Capture DSPy history
                    import io
                    import sys
                    from contextlib import redirect_stdout
                    
                    # Redirect stdout to capture inspect_history output
                    history_output = io.StringIO()
                    with redirect_stdout(history_output):
                        dspy.inspect_history(n=1)
                    
                    history_content = history_output.getvalue()
                    
                    with open(history_file, "w") as f:
                        f.write(f"DSPy Prompt History for Level {config.level} Problem {config.problem_id}\n")
                        f.write(f"Problem: {problem_name}\n")
                        f.write(f"Model: {config.dspy_model}\n")
                        f.write(f"Temperature: {config.dspy_temperature}\n")
                        f.write(f"RAG Examples (k): {config.rag_k}\n")
                        f.write("="*80 + "\n\n")
                        f.write(history_content)
                    
                    print(f"💾 DSPy history saved to {history_file}")
                    
                except Exception as e:
                    print(f"⚠️ Could not save DSPy history: {e}")
            
        except Exception as e:
            print(f"❌ RAG generation failed: {e}")
            print("This indicates a configuration or system issue")
            return
            
    else:
        print(">>> USING CODE WITH EVAL ONLY <<<")
        if config.eval_file_path:
            path = config.eval_file_path
        else:
            path = f"src/prompts/correct_{config.language}/level{config.level}/{config.level}_{config.problem_id}.py"

        raw_code = open(path).read()
        clean_code = strip_docstring_from_code(raw_code)
        tilelang_code = "```python\n" + clean_code + "\n```"
        print(f">>> Stripped old docstring, using clean code for evaluation <<<")
        
        # For ThunderKittens eval_only, ensure kernel is compiled
        if config.language == "thunderkittens":
            tk_kernel_dir = os.path.join(REPO_TOP_DIR, "correct_tk")
            cu_file_path = os.path.join(tk_kernel_dir, "custom_tk.cu")
            
            # Check if .cu file exists
            if not os.path.exists(cu_file_path):
                raise FileNotFoundError(f"ThunderKittens kernel file not found: {cu_file_path}. Please ensure custom_tk.cu exists in correct_tk/")
            
            # Check if compiled .so exists
            so_files = [f for f in os.listdir(tk_kernel_dir) if f.startswith("tk_kernels") and f.endswith(".so")]
            
            if not so_files:
                print(f"⚠️ No compiled .so found, compiling {cu_file_path}...")
                create_tk_makefile(tk_kernel_dir, gpu=config.gpu)
                
                import subprocess
                compile_result = subprocess.run(
                    ["make", "clean"],
                    cwd=tk_kernel_dir,
                    capture_output=True,
                    text=True
                )
                compile_result = subprocess.run(
                    ["make"],
                    cwd=tk_kernel_dir,
                    capture_output=True,
                    text=True,
                    env={**os.environ}
                )
                
                if compile_result.returncode != 0:
                    raise RuntimeError(f"ThunderKittens compilation failed: {compile_result.stderr}")
                
                print(f"✅ Compiled ThunderKittens kernel")
            else:
                print(f"✅ Using existing compiled kernel: {so_files[0]}")

    # Extract and process code based on language
    if config.language == "thunderkittens":
        print(">>> PROCESSING THUNDERKITTENS CODE <<<")
        
        # ThunderKittens requires TWO code blocks: .cu kernel and .py wrapper
        all_blocks = extract_all_code_blocks(tilelang_code)
        
        # Extract .cu kernel (can be marked as cpp, c++, or cuda)
        cu_code = all_blocks.get("cpp") or all_blocks.get("c++") or all_blocks.get("cuda")
        py_code = all_blocks.get("python")
        
        if cu_code is None:
            raise ValueError(f"ThunderKittens requires .cu kernel code block (```cpp). Found blocks: {list(all_blocks.keys())}")
        if py_code is None:
            raise ValueError(f"ThunderKittens requires Python wrapper code block (```python). Found blocks: {list(all_blocks.keys())}")
        
        print(f"✅ Extracted both .cu kernel ({len(cu_code)} chars) and .py wrapper ({len(py_code)} chars)")
        
        import re
        tk_kernel_dir = os.path.join(REPO_TOP_DIR, "correct_tk")
        os.makedirs(tk_kernel_dir, exist_ok=True)

        cu_file_path = os.path.join(tk_kernel_dir, "custom_tk.cu")
        with open(cu_file_path, "w") as f:
            f.write(cu_code)
        print(f"📝 Wrote .cu kernel to {cu_file_path}")

        # Create Makefile for reference only; do not run make locally
        create_tk_makefile(tk_kernel_dir, gpu=config.gpu)
        print(f"📝 Created Makefile in {tk_kernel_dir}")


        py_code = """
import tk_kernels
import torch
import torch.nn as nn

INPUT_DTYPE = torch.bfloat16
OUTPUT_DTYPE = torch.float

M = 16
N = 16384
K = N

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        output = torch.zeros(M, N, dtype=OUTPUT_DTYPE, device='cuda')

        A = A.cuda()
        B = B.cuda()
        tk_kernels.dispatch_micro(A, B, output, K)
        
        return output


A = torch.rand(M, K, dtype=INPUT_DTYPE) / K  # [16, 16384]
B = torch.rand(K, N, dtype=INPUT_DTYPE) / K  # [16384, 16384]

A = A.cuda()
B = B.cuda()

output_ref = torch.matmul(A, B).to(OUTPUT_DTYPE)
print("Ref output shape:", output_ref.shape)
print("Ref output mean:", output_ref.mean())


model = ModelNew().cuda()
output = model(A, B)
print("TK Output shape:", output.shape)
print("TK Output mean:", output.mean())

# import pdb; pdb.set_trace()

assert torch.allclose(output, output_ref, atol=1e-2)
        """

        # Use Python wrapper for execution; pass ORIGINAL cu_code to Modal for remote compile
        # (Modal has ThunderKittens installed, so includes are needed)
        custom_cuda = textwrap.dedent(py_code).lstrip()
        # Keep cu_code as original with includes intact for Modal compilation
        
    else:
        # Standard code extraction for other languages
        custom_cuda = extract_first_code(tilelang_code, ["python", "cpp"])
    
    # Validate generation
    assert custom_cuda is not None, f"{config.language} code generation failed"
    
    # Print out costs
    history = lm.history
    total_cost = 0
    for entry in history:
        if entry.get("cost") is not None:
            total_cost += entry["cost"]
    print(f"Number of interactions: {len(history)}")
    print(f"Total cost: ${total_cost}")

    if config.log_generated_kernel:
        with open(os.path.join(config.logdir, f"rag_generated_kernel_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
            f.write(custom_cuda)
    
    # 3. Evaluate on Modal
    entry_point = None
    if config.level == 9:
        first_underscore = problem_name.find("_")
        entry_point = problem_name[first_underscore+1:]
        
    print(f"🚀 Evaluating generated kernel on Modal...")
    print(f"Entry point: {entry_point}")
    
    with app.run():
        kernel_exec_result = EvalFunc.with_options(gpu=config.gpu)().eval_single_sample_modal.remote(
            ref_arch_src, custom_cuda, config.verbose, gpu_arch_mapping[config.gpu], config.language, entry_point, cu_code if config.language == "thunderkittens" else None
        )
        
        print(f"\n{'='*60}")
        print(f"🎯 EVALUATION RESULT for Level {config.level} Problem {config.problem_id}")
        print(f"{'='*60}")
        print(f"Problem: {problem_name}")
        print(f"DSPy Model: {config.dspy_model}")
        print(f"RAG Examples Used: {config.rag_k}")
        print(f"Result: {kernel_exec_result}")
        print(f"{'='*60}")
        
        # Save successful kernels
        if kernel_exec_result.correctness:
            if config.eval_file_path is None:
                # Map language to correct directory
                base_dir = f"src/prompts/correct_{config.language}/level{config.level}"
                
                os.makedirs(base_dir, exist_ok=True)
                base_path = f"{base_dir}/{config.level}_{config.problem_id}.py"
                path = base_path
                
                # Extract speedup ratio from current result
                eval_str = str(kernel_exec_result)
                current_ratio = float(eval_str.split("speedup_ratio': ")[1].split("}")[0])
                
                # If file exists (and we are generating new kernels), compare speedup ratios
                if not config.eval_only and os.path.exists(path):
                    with open(path, 'r') as f:
                        existing_content = f.read()
                        existing_eval = existing_content.split('Evaluation Result:\n')[1].split('\n"""')[0]
                        existing_ratio = float(existing_eval.split("speedup_ratio': ")[1].split("}")[0])
                        
                        if current_ratio <= existing_ratio:
                            print(f"💾 Discarding kernel - existing speedup ratio {existing_ratio:.2f} better than current {current_ratio:.2f}")
                            return
                        else:
                            print(f"💾 Replacing kernel - current speedup ratio {current_ratio:.2f} better than existing {existing_ratio:.2f}")
                else:
                    print(f"💾 Writing new TileLang kernel to {path}")
            else:
                path = config.eval_file_path

            with open(path, "w") as f:
                f.write(f'''"""
Problem Name: {problem_name}
Generated using DSPy RAG with {config.dspy_model}
RAG Examples: {config.rag_k}
Evaluation Result:
{str(kernel_exec_result)}
"""

{custom_cuda}''')
            
            # For ThunderKittens, also save the .cu file in the same directory
            if config.language == "thunderkittens" and not config.eval_only:
                cu_save_path = path.replace(".py", ".cu")
                # cu_code was extracted earlier, save it
                if 'cu_code' in locals():
                    with open(cu_save_path, "w") as f:
                        f.write(cu_code)
                    print(f"💾 Also saved .cu kernel to {cu_save_path}")
                else:
                    # Copy from correct_tk/custom_tk.cu if it exists
                    tk_cu_source = os.path.join(REPO_TOP_DIR, "correct_tk/custom_tk.cu")
                    if os.path.exists(tk_cu_source):
                        import shutil
                        shutil.copy(tk_cu_source, cu_save_path)
                        print(f"💾 Copied .cu kernel to {cu_save_path}")
        
        if config.log_eval_result:
            with open(os.path.join(config.logdir, f"rag_eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(f"DSPy Model: {config.dspy_model}\n")
                f.write(f"RAG Examples: {config.rag_k}\n")
                f.write(str(kernel_exec_result))


if __name__ == "__main__":
    main() 