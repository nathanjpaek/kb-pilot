"""
Multiturn Optimization for TileLang Kernels using DSPy

This script takes existing TileLang kernels and iteratively optimizes them
using sophisticated performance analysis and targeted optimization strategies.

Optimized for use with OpenAI O3 and resource-unconstrained environments.
"""

import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import modal
import dspy
import re
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()

from src.eval import eval_kernel_against_ref
from src.utils import extract_first_code, set_gpu_arch, read_file
from src.prompt_constructor_rag import prompt_generate_custom_tilelang_rag_enhanced
from scripts.tilelang_paperinfo_prompt import PAPER_PROMPT
from scripts.tilelang_guideline_prompt import GUIDELINE_PROMPT
from src.dataset import construct_kernelbench_dataset
from datasets import load_dataset


def extract_speedup_ratio(eval_result: str) -> float:
    """Extract speedup ratio from evaluation result string"""
    try:
        if isinstance(eval_result, str):
            # Try different patterns for speedup ratio
            patterns = [
                r"speedup_ratio['\"]?\s*:\s*([0-9.]+)",
                r"speedup_ratio['\"]?\s*=\s*([0-9.]+)",
                r"Speedup:\s*([0-9.]+)",
                r"speedup.*?([0-9.]+)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, eval_result, re.IGNORECASE)
                if match:
                    return float(match.group(1))
        
        # If it's a file path, parse the file
        elif os.path.exists(eval_result):
            with open(eval_result, 'r') as f:
                content = f.read()
            return extract_speedup_ratio(content)
    except Exception as e:
        print(f"Warning: Failed to extract speedup ratio: {e}")
        
    return 0.0


def parse_existing_kernel(file_path: str) -> Tuple[str, float, str]:
    """Parse an existing kernel file and extract the evaluation result"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract problem name
    problem_name_match = re.search(r'Problem Name: (.+)', content)
    problem_name = problem_name_match.group(1) if problem_name_match else "Unknown"
    
    # Extract evaluation result section
    eval_start = content.find('Evaluation Result:')
    if eval_start == -1:
        # Try alternative patterns
        eval_patterns = [
            'speedup',
            'Speedup',
            'performance',
            'Performance'
        ]
        for pattern in eval_patterns:
            idx = content.lower().find(pattern.lower())
            if idx != -1:
                eval_start = idx
                break
    
    if eval_start != -1:
        eval_end = content.find('"""', eval_start)
        if eval_end == -1:
            eval_end = len(content)
        eval_result = content[eval_start:eval_end].strip()
    else:
        eval_result = content  # Use entire content if no specific section found
    
    # Extract speedup ratio
    speedup_ratio = extract_speedup_ratio(eval_result)
    
    # If still no speedup found, try extracting from filename or content
    if speedup_ratio == 0.0:
        # Try extracting from filename patterns like "optimized_1.234x.py"
        filename = os.path.basename(file_path)
        filename_match = re.search(r'([0-9.]+)x\.py', filename)
        if filename_match:
            speedup_ratio = float(filename_match.group(1))
        else:
            # Default to a very small value to indicate needs optimization
            speedup_ratio = 0.001
            print(f"Warning: Could not extract speedup from {file_path}, using default {speedup_ratio}")
    
    # Extract code (everything after the docstring)
    code_start = content.find('"""', content.find('"""') + 3) + 3
    if code_start < 3:
        # No docstring found, use entire content
        kernel_code = content.strip()
    else:
        kernel_code = content[code_start:].strip()
    
    return problem_name, speedup_ratio, kernel_code


class PerformanceAnalyzer(dspy.Module):
    """DSPy module for analyzing TileLang kernel performance bottlenecks"""
    
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(
            "kernel_code, current_speedup, target_speedup, optimization_context -> bottleneck_analysis, optimization_suggestions"
        )
    
    def forward(self, kernel_code: str, current_speedup: float, target_speedup: float = 1.5, 
                optimization_context: str = "") -> dspy.Prediction:
        """Analyze kernel performance and suggest optimizations"""
        
        prompt_context = f"""
PERFORMANCE ANALYSIS TASK:
Analyze this TileLang kernel that currently has a speedup of {current_speedup:.3f}x.
Target speedup is {target_speedup:.3f}x.

Current Kernel Code:
{kernel_code}

Optimization Examples for Context:
{optimization_context}

Please provide:
1. A detailed bottleneck analysis identifying the main performance limitations
2. Specific optimization suggestions that could improve performance

Focus on TileLang-specific optimizations like shared memory usage, register tiling, 
tensor core utilization, memory coalescing, and parallelization strategies.
"""
        
        return self.analyze(
            kernel_code=prompt_context,
            current_speedup=current_speedup,
            target_speedup=target_speedup,
            optimization_context=optimization_context
        )


class KernelOptimizer(dspy.Module):
    """DSPy module for optimizing TileLang kernels based on analysis"""
    
    def __init__(self):
        super().__init__()
        self.optimize = dspy.ChainOfThought(
            "original_kernel, bottleneck_analysis, optimization_suggestions, tilelang_guidelines -> optimized_kernel"
        )
    
    def forward(self, original_kernel: str, bottleneck_analysis: str, 
                optimization_suggestions: str, tilelang_guidelines: str = "") -> dspy.Prediction:
        """Generate optimized kernel based on analysis"""
        
        prompt = f"""
KERNEL OPTIMIZATION TASK:

You must generate an OPTIMIZED version of the TileLang kernel below. DO NOT just copy the original code.
Apply the bottleneck analysis and optimization suggestions to create a meaningfully improved kernel.

ORIGINAL KERNEL:
```python
{original_kernel}
```

BOTTLENECK ANALYSIS:
{bottleneck_analysis}

OPTIMIZATION SUGGESTIONS:
{optimization_suggestions}

TILELANG GUIDELINES:
{tilelang_guidelines}

REQUIREMENTS:
1. Generate ONLY the optimized ModelNew class code (no explanations)
2. Apply the optimization suggestions from the analysis
3. Make meaningful changes to improve performance
4. Ensure the code is syntactically correct TileLang
5. Focus on the most impactful optimizations identified

OUTPUT FORMAT: Only provide the complete optimized Python class code.
"""
        
        return self.optimize(
            original_kernel=prompt,
            bottleneck_analysis=bottleneck_analysis,
            optimization_suggestions=optimization_suggestions,
            tilelang_guidelines=tilelang_guidelines
        )


class MultiturnOptimizationRAG(dspy.Module):
    """High-performance RAG module for multiturn TileLang optimization"""
    
    def __init__(self, correct_tilelang_dir: str, k: int = 3, 
                 exclude_current_problem: bool = True, current_level: int = None, current_problem_id: int = None):
        super().__init__()
        
        # Store exclusion parameters
        self.exclude_current_problem = exclude_current_problem
        self.current_level = current_level
        self.current_problem_id = current_problem_id
        
        # Load optimization examples
        self.optimization_examples = self._load_optimization_examples(correct_tilelang_dir)
        
        # Use best embedding model
        embedder = dspy.Embedder('openai/text-embedding-3-large', dimensions=3072)
        
        # Create corpus for retrieval
        corpus = []
        for example in self.optimization_examples:
            # Focus on operations and optimization patterns
            operations_str = ', '.join(example['operations']) if example['operations'] else 'general'
            text = f"Operations: {operations_str}\nOptimization: {example['optimization_type']}\nTechniques: {example['techniques']}\nSpeedup: {example['after_speedup']:.3f}"
            corpus.append(text)
        
        # Initialize retriever
        self.retriever = dspy.retrievers.Embeddings(
            embedder=embedder, 
            corpus=corpus, 
            k=k
        )
    
    def _extract_operations(self, code: str) -> List[str]:
        """Extract operations from TileLang kernel code"""
        operations = []
        
        # TileLang specific operations
        tilelang_ops = [
            'relu', 'softmax', 'matmul', 'conv2d', 'conv3d', 'linear', 'sigmoid',
            'tanh', 'gelu', 'dropout', 'batch_norm', 'layer_norm', 'max_pool',
            'avg_pool', 'transpose', 'permute', 'reshape', 'view', 'add', 'mul',
            'div', 'sub', 'sqrt', 'pow', 'exp', 'log', 'mean', 'sum', 'max', 'min',
            'gemm', 'gemv', 'reduce', 'elementwise', 'activation', 'normalization'
        ]
        
        # Also look for TileLang specific patterns
        tilelang_patterns = [
            'T.gemm', 'T.reduce', 'T.alloc_shared', 'T.alloc_fragment', 'T.alloc_local',
            'T.copy', 'T.fill', 'T.atomic', 'T.Parallel', 'T.Pipelined', 'T.vectorized'
        ]
        
        code_lower = code.lower()
        
        # Check for PyTorch/standard operations
        for op in tilelang_ops:
            if (op in code_lower or f'torch.{op}' in code_lower or 
                f'nn.{op}' in code_lower or f'F.{op}' in code_lower):
                operations.append(op)
        
        # Check for TileLang specific patterns
        for pattern in tilelang_patterns:
            if pattern.lower() in code_lower:
                # Extract the operation type
                op_name = pattern.split('.')[-1] if '.' in pattern else pattern
                operations.append(f"tilelang_{op_name}")
        
        # Look for specific operation signatures in comments, function names, and class usage
        if 'layernorm' in code_lower or 'layer_norm' in code_lower or 'nn.layernorm' in code_lower:
            operations.append('layer_norm')
        if 'batchnorm' in code_lower or 'batch_norm' in code_lower or 'nn.batchnorm' in code_lower:
            operations.append('batch_norm')
        if 'attention' in code_lower:
            operations.append('attention')
        if 'convolution' in code_lower or 'conv' in code_lower or 'nn.conv' in code_lower:
            operations.append('convolution')
        if 'avgpool' in code_lower or 'avg_pool' in code_lower or 'nn.avgpool' in code_lower:
            operations.append('avg_pool')
        if 'maxpool' in code_lower or 'max_pool' in code_lower or 'nn.maxpool' in code_lower:
            operations.append('max_pool')
        if 'convtranspose' in code_lower or 'nn.convtranspose' in code_lower:
            operations.append('conv_transpose')
        
        # Check for mathematical operations that might be in kernels
        if 'erf(' in code_lower:
            operations.append('erf')
        if 'rsqrt' in code_lower or 'sqrt' in code_lower:
            operations.append('sqrt')
        if 'exp(' in code_lower:
            operations.append('exp')
        
        return list(set(operations))
    
    def _load_optimization_examples(self, correct_tilelang_dir: str) -> List[Dict]:
        """Load examples of successful optimizations"""
        examples = []
        excluded_count = 0
        
        for level_dir in os.listdir(correct_tilelang_dir):
            if not level_dir.startswith('level'):
                continue
                
            if level_dir == 'level3':
                continue
                
            level_num = int(level_dir.replace('level', ''))
            level_path = os.path.join(correct_tilelang_dir, level_dir)
            
            for filename in os.listdir(level_path):
                if not filename.endswith('.py'):
                    continue
                
                # Parse problem number from filename  
                parts = filename.replace('.py', '').split('_')
                if len(parts) < 2:
                    continue
                    
                try:
                    problem_num = int(parts[1])
                except ValueError:
                    continue
                
                # Exclude current problem if specified
                if (self.exclude_current_problem and 
                    self.current_level is not None and 
                    self.current_problem_id is not None and
                    level_num == self.current_level and 
                    problem_num == self.current_problem_id):
                    
                    print(f"🚫 Excluding current problem from RAG: {filename}")
                    excluded_count += 1
                    continue
                    
                try:
                    file_path = os.path.join(level_path, filename)
                    problem_name, speedup_ratio, kernel_code = parse_existing_kernel(file_path)
                    
                    # Extract operations from the kernel
                    operations = self._extract_operations(kernel_code)
                    
                    # Analyze the kernel for optimization patterns
                    optimization_type = self._classify_optimization(kernel_code)
                    techniques = self._extract_techniques(kernel_code)
                    
                    examples.append({
                        'problem_name': problem_name,
                        'speedup_ratio': speedup_ratio,
                        'after_speedup': speedup_ratio,
                        'before_speedup': 1.0,  # Assume baseline is 1.0
                        'optimization_type': optimization_type,
                        'techniques': techniques,
                        'operations': operations,  # Add operations to examples
                        'kernel_code': kernel_code,
                        'level': level_num,
                        'problem_id': problem_num
                    })
                    
                except Exception:
                    continue
        
        print(f"Loaded {len(examples)} optimization examples for RAG (excluded {excluded_count} current problem examples)")
        return examples
    
    def _classify_optimization(self, kernel_code: str) -> str:
        """Classify the type of optimization used in the kernel"""
        optimizations = []
        
        if 'T.alloc_shared' in kernel_code:
            optimizations.append('shared_memory')
        if 'T.alloc_fragment' in kernel_code:
            optimizations.append('register_optimization')
        if 'T.Pipelined' in kernel_code:
            optimizations.append('pipelining')
        if 'T.gemm' in kernel_code:
            optimizations.append('tensor_cores')
        if 'T.Parallel' in kernel_code and kernel_code.count('T.Parallel') > 1:
            optimizations.append('multi_level_parallelism')
        if 'T.atomic' in kernel_code:
            optimizations.append('atomic_operations')
        
        return ', '.join(optimizations) if optimizations else 'basic_optimization'
    
    def _extract_techniques(self, kernel_code: str) -> str:
        """Extract specific optimization techniques used"""
        techniques = []
        
        if 'ceildiv' in kernel_code:
            techniques.append('grid_sizing')
        if 'block_M' in kernel_code or 'block_N' in kernel_code:
            techniques.append('tiling')
        if 'T.copy' in kernel_code:
            techniques.append('memory_coalescing')
        if 'accum_dtype' in kernel_code:
            techniques.append('mixed_precision')
        if 'threads=' in kernel_code:
            techniques.append('thread_tuning')
        
        return ', '.join(techniques) if techniques else 'standard_techniques'
    
    def forward(self, current_kernel: str, current_speedup: float) -> dspy.Prediction:
        """Retrieve relevant optimization examples"""
        
        # Extract operations from current kernel
        current_operations = self._extract_operations(current_kernel)
        
        # Create query based on operations in current kernel
        if current_operations:
            operations_str = ', '.join(current_operations)
            query = f"Operations: {operations_str}"
            
            # Add performance context to the query
            if current_speedup < 0.5:
                query += " slow performance memory bandwidth optimization"
            elif current_speedup < 0.8:
                query += " moderate performance shared memory tiling"
            elif current_speedup < 1.2:
                query += " good performance fine-tuning pipelining"
            else:
                query += " high performance advanced optimization"
        else:
            # Fallback to performance-based query if no operations detected
            if current_speedup < 0.5:
                query = "slow kernel optimization memory bandwidth register usage"
            elif current_speedup < 0.8:
                query = "moderate optimization shared memory tiling parallelism"
            elif current_speedup < 1.2:
                query = "fine-tuning optimization pipelining tensor cores"
            else:
                query = "advanced optimization techniques memory hierarchy"
        
        # Retrieve relevant examples
        retrieved = self.retriever(query).passages
        
        # Print retrieved document names for debugging
        print(f"\n🔍 Retrieved RAG Documents:")
        print(f"Current kernel operations: {current_operations}")
        print(f"Query: '{query}'")
        retrieved_names = []
        for i, passage in enumerate(retrieved):
            # Find corresponding example to get the problem name
            for example in self.optimization_examples:
                operations_str = ', '.join(example['operations']) if example['operations'] else 'general'
                example_text = f"Operations: {operations_str}\nOptimization: {example['optimization_type']}\nTechniques: {example['techniques']}\nSpeedup: {example['after_speedup']:.3f}"
                if passage.strip() == example_text.strip():
                    problem_name = example['problem_name']
                    speedup = example['speedup_ratio']
                    optimization_type = example['optimization_type']
                    example_operations = example['operations']
                    retrieved_names.append(problem_name)
                    print(f"  {i+1}. {problem_name} (speedup: {speedup:.3f}x, type: {optimization_type}, ops: {example_operations})")
                    break
            else:
                # If no exact match found, print what we can
                print(f"  {i+1}. [Unknown document] - {passage[:100]}...")
        
        if retrieved_names:
            print(f"📚 Total documents retrieved: {len(retrieved_names)}")
        else:
            print("⚠️ No matching documents found in optimization examples")
        
        # Format context
        context = self._format_context(retrieved)
        
        return dspy.Prediction(context=context)
    
    def _format_context(self, retrieved_passages: List[str]) -> str:
        """Format retrieved optimization examples as context"""
        context_parts = []
        
        for i, passage in enumerate(retrieved_passages):
            # Find corresponding example
            for example in self.optimization_examples:
                operations_str = ', '.join(example['operations']) if example['operations'] else 'general'
                example_text = f"Operations: {operations_str}\nOptimization: {example['optimization_type']}\nTechniques: {example['techniques']}\nSpeedup: {example['after_speedup']:.3f}"
                if passage.strip() == example_text.strip():
                    context_parts.append(f"""
Optimization Example {i+1} - {example['problem_name']}:
- Operations: {', '.join(example['operations']) if example['operations'] else 'general'}
- Speedup Achieved: {example['speedup_ratio']:.3f}x
- Optimization Type: {example['optimization_type']}
- Techniques Used: {example['techniques']}

Optimized Code Pattern:
```python
{example['kernel_code'][:500]}...
```
""")
                    break
        
        return "\n".join(context_parts)


app = modal.App("multiturn_optimization")

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}


class OptimizationConfig(Config):
    def __init__(self):
        # File to optimize
        self.kernel_file = REQUIRED  # Path to existing kernel file
        
        # Dataset configuration
        self.dataset_src = "local"  # "huggingface" or "local"
        self.dataset_name = "ScalingIntelligence/KernelBench"
        
        # Problem specification
        self.level = REQUIRED
        self.problem_id = REQUIRED
        
        # Optimization parameters
        self.max_iterations = 5
        self.target_speedup = 1.5
        self.min_improvement_threshold = 0.05  # Minimum improvement to continue
        
        # Evaluation
        self.eval_mode = "modal"
        self.gpu = "H100"
        self.gpu_arch = ['Hopper']
        
        # DSPy Model Configuration
        self.dspy_model = "openai/o3"
        self.dspy_temperature = 1.0
        
        # RAG Configuration
        self.rag_k = 5
        
        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/multiturn_logs")
        self.verbose = True
        
        # Output
        self.save_best_kernel = True
        self.save_optimization_history = True

    def __repr__(self):
        return f"OptimizationConfig({self.to_dict()})"


cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install(
        "anthropic", "numpy", "openai", "packaging", "pydra_config",
        "torch==2.5.0", "tqdm", "datasets", "transformers",
        "google-generativeai", "together", "pytest", "ninja", "utils",
        "python-dotenv", "tilelang", "apache-tvm", "dspy-ai"
    )
    .add_local_python_source("scripts", "src")
    .add_local_dir("KernelBench", "/root/KernelBench")
)


@app.cls(image=image)
class EvalFunc:
    @modal.method()
    def eval_single_sample_modal(self, ref_arch_src, custom_cuda, verbose, gpu_arch, language, entry_point=None):
        torch.set_default_dtype(torch.float16)
        
        from src.eval import eval_kernel_against_ref
        from src.utils import set_gpu_arch
        set_gpu_arch(gpu_arch)
        return eval_kernel_against_ref(
            ref_arch_src, custom_cuda, verbose=verbose, measure_performance=True, 
            num_correct_trials=5, num_perf_trials=100, language=language, entry_point=entry_point
        )


def configure_dspy(model_name: str, temperature: float = 1.0):
    """Configure DSPy with the specified model"""
    print(f"🤖 Configuring DSPy with model: {model_name}")
    
    lm = dspy.LM(model_name, temperature=1.0, max_tokens=30000)
    dspy.configure(lm=lm)
    
    print(f"✅ DSPy configured successfully with {model_name}")
    return lm


def create_optimization_guidelines(current_speedup: float, iteration: int) -> str:
    """Create targeted optimization guidelines based on current performance"""
    
    base_guidelines = f"""
{PAPER_PROMPT}

{GUIDELINE_PROMPT}

MULTITURN OPTIMIZATION GUIDELINES (Iteration {iteration}):

CURRENT PERFORMANCE ANALYSIS:
- Current Speedup Ratio: {current_speedup:.3f}
- Performance Category: {"SLOW" if current_speedup < 0.8 else "MODERATE" if current_speedup < 1.2 else "GOOD"}

OPTIMIZATION FOCUS AREAS:

1. MEMORY BANDWIDTH OPTIMIZATION (Priority: HIGH if speedup < 0.8):
   - Analyze memory access patterns for coalescing
   - Use shared memory for repeated data access
   - Minimize global memory transactions
   - Optimize data layout and alignment

2. COMPUTATIONAL EFFICIENCY:
   - Eliminate redundant computations
   - Use register-level optimizations (T.alloc_fragment)
   - Leverage tensor cores when applicable (T.gemm)
   - Optimize arithmetic intensity

3. PARALLELISM AND WORKLOAD BALANCE:
   - Analyze thread utilization and occupancy
   - Balance work across thread blocks
   - Use multi-level parallelism effectively
   - Optimize grid and block dimensions

4. ALGORITHM OPTIMIZATION:
   - Strongly consider fusion opportunities
   - Implement efficient reduction patterns
   - Use pipelining for overlapping computation/memory
   - Optimize loop structures and control flow

SPECIFIC OPTIMIZATION STRATEGIES:
"""
    
    if current_speedup < 0.5:
        base_guidelines += """
CRITICAL PERFORMANCE ISSUES DETECTED:
- Focus on memory bandwidth limitations
- Check for memory bank conflicts
- Implement aggressive shared memory usage
- Consider algorithmic changes for better locality
- Use mixed precision arithmetic strategically
"""
    elif current_speedup < 0.8:
        base_guidelines += """
MODERATE PERFORMANCE ISSUES:
- Optimize thread block configurations
- Implement memory coalescing patterns
- Use register tiling techniques
- Consider workload rebalancing
"""
    elif current_speedup < 1.2:
        base_guidelines += """
FINE-TUNING OPTIMIZATIONS:
- Implement advanced pipelining
- Optimize for specific GPU architecture features
- Fine-tune block sizes and thread counts
- Consider instruction-level optimizations
"""
    else:
        base_guidelines += """
ADVANCED OPTIMIZATIONS:
- Explore cutting-edge optimization techniques
- Consider architecture-specific optimizations
- Implement sophisticated memory hierarchies
- Push the boundaries of performance
"""
    
    base_guidelines += """

OUTPUT REQUIREMENTS:
- Generate ONLY the optimized ModelNew class code
- Include detailed comments explaining optimization choices
- Maintain correctness while maximizing performance
- Focus on the most impactful optimizations for this iteration
"""
    
    return base_guidelines


@pydra.main(base=OptimizationConfig)
def main(config: OptimizationConfig):
    """
    Multiturn optimization of TileLang kernels using DSPy
    """
    print(f"Starting Multiturn Optimization with config: {config}")
    
    torch.set_default_dtype(torch.float16)
    
    # Configure DSPy
    lm = configure_dspy(config.dspy_model, config.dspy_temperature)
    
    # Create log directory
    if config.verbose:
        os.makedirs(config.logdir, exist_ok=True)
    
    # Parse initial kernel
    print(f"📁 Loading kernel from: {config.kernel_file}")
    problem_name, initial_speedup, current_kernel = parse_existing_kernel(config.kernel_file)
    
    print(f"🎯 Optimizing: {problem_name}")
    print(f"📊 Initial speedup: {initial_speedup:.3f}")
    
    # Set current speedup to initial speedup
    current_speedup = initial_speedup
    
    # Load dataset and find reference architecture (same approach as generate_and_eval_rag_modal.py)
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        dataset_problem_name = curr_problem_row["name"][0]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)
        problem_idx_in_dataset = config.problem_id - 1
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]
        dataset_problem_name = os.path.basename(ref_arch_path)
        dataset_problem_name = dataset_problem_name.replace(".py", "")
        ref_arch_src = read_file(ref_arch_path)
    
    print(f"🔍 Level: {config.level}")
    print(f"🔍 Problem ID: {config.problem_id}")
    print(f"🔍 Dataset problem name: {dataset_problem_name}")
    
    # Validate problem number consistency
    problem_number = int(dataset_problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in dataset ({problem_number}) does not match config problem_id ({config.problem_id})"
    
    # Initialize optimization components
    correct_tilelang_dir = os.path.join(REPO_TOP_DIR, "src/prompts/correct_tilelang")
    print(f"🔧 Initializing RAG system (excluding current problem: level{config.level}/problem_{config.problem_id})")
    optimization_rag = MultiturnOptimizationRAG(correct_tilelang_dir, k=config.rag_k, exclude_current_problem=True, current_level=config.level, current_problem_id=config.problem_id)
    analyzer = PerformanceAnalyzer()
    optimizer = KernelOptimizer()
    
    # Optimization history
    optimization_history = []
    best_kernel = current_kernel
    best_speedup = current_speedup
    
    print(f"\n🚀 Starting {config.max_iterations} optimization iterations...")
    print(f"🎯 Target speedup: {config.target_speedup:.3f}")
    print(f"📏 Min improvement threshold: {config.min_improvement_threshold:.3f}")
    
    for iteration in range(config.max_iterations):
        print(f"\n{'='*60}")
        print(f"🔄 OPTIMIZATION ITERATION {iteration + 1}/{config.max_iterations}")
        print(f"📊 Current speedup: {current_speedup:.3f}")
        print(f"{'='*60}")
        
        # Get RAG context for optimization
        print("🧠 Retrieving optimization examples...")
        rag_context = optimization_rag(current_kernel, current_speedup)
        if config.verbose:
            print(f"📚 RAG Context Preview: {rag_context.context[:200]}...")
        
        # Analyze current kernel for bottlenecks
        print("🔍 Analyzing performance bottlenecks...")
        analysis = analyzer(
            kernel_code=current_kernel,
            current_speedup=current_speedup,
            target_speedup=config.target_speedup,
            optimization_context=rag_context.context
        )
        
        # Print analyzer output for debugging
        print("\n🔬 PERFORMANCE ANALYSIS:")
        print("=" * 40)
        print("BOTTLENECK ANALYSIS:")
        print(analysis.bottleneck_analysis)
        print("\nOPTIMIZATION SUGGESTIONS:")
        print(analysis.optimization_suggestions)
        print("=" * 40)
        
        # Generate optimization guidelines
        guidelines = create_optimization_guidelines(current_speedup, iteration + 1)
        
        # Generate optimized kernel
        print("⚡ Generating optimized kernel...")
        optimization = optimizer(
            original_kernel=current_kernel,
            bottleneck_analysis=analysis.bottleneck_analysis,
            optimization_suggestions=analysis.optimization_suggestions,
            tilelang_guidelines=guidelines
        )
        
        # Print optimization output for debugging
        if config.verbose:
            print("\n🔧 OPTIMIZATION OUTPUT:")
            print("=" * 40)
            print(optimization.optimized_kernel)
            print("=" * 40)
        
        # Extract optimized code
        optimized_code = extract_first_code(optimization.optimized_kernel, ["python"])
        if not optimized_code:
            print("❌ Failed to extract optimized code")
            optimized_code = optimization.optimized_kernel
            
        print(optimized_code)
        
        # Check if the optimized code is actually different
        if optimized_code.strip() == current_kernel.strip():
            print("⚠️ Generated code is identical to current kernel - no changes made")
        else:
            print("✅ Generated code is different from current kernel")
        
        # Evaluate optimized kernel
        print("📊 Evaluating optimized kernel...")
        
        entry_point = None
        if config.level == 9:
            first_underscore = dataset_problem_name.find("_")
            if first_underscore != -1:
                entry_point = dataset_problem_name[first_underscore+1:]
        
        try:
            with app.run():
                eval_result = EvalFunc.with_options(gpu=config.gpu)().eval_single_sample_modal.remote(
                    ref_arch_src, optimized_code, config.verbose, gpu_arch_mapping[config.gpu], "tilelang", entry_point
                )
            
            if eval_result.correctness:
                new_speedup = extract_speedup_ratio(str(eval_result))
                improvement = new_speedup - current_speedup
                
                print(f"✅ Optimization successful!")
                print(f"📈 Speedup: {current_speedup:.3f} → {new_speedup:.3f} (Δ: {improvement:+.3f})")
                
                # Save iteration result
                iteration_result = {
                    'iteration': iteration + 1,
                    'previous_speedup': current_speedup,
                    'new_speedup': new_speedup,
                    'improvement': improvement,
                    'bottleneck_analysis': analysis.bottleneck_analysis,
                    'optimization_suggestions': analysis.optimization_suggestions,
                    'optimized_code': optimized_code,
                    'eval_result': str(eval_result)
                }
                optimization_history.append(iteration_result)
                
                # Update best kernel if improved
                if new_speedup > best_speedup:
                    best_kernel = optimized_code
                    best_speedup = new_speedup
                    print(f"🏆 New best kernel! Speedup: {best_speedup:.3f}")
                
                # Check if we should continue
                if improvement < config.min_improvement_threshold:
                    print(f"⏹️ Improvement below threshold ({config.min_improvement_threshold:.3f}), stopping")
                    break
                
                if new_speedup >= config.target_speedup:
                    print(f"🎉 Target speedup achieved! ({new_speedup:.3f} >= {config.target_speedup:.3f})")
                    break
                
                # Update for next iteration
                current_kernel = optimized_code
                current_speedup = new_speedup
                
            else:
                print(f"❌ Optimization failed correctness check")
                print(f"Error: {eval_result}")
                
                # Save failed attempt
                iteration_result = {
                    'iteration': iteration + 1,
                    'previous_speedup': current_speedup,
                    'new_speedup': 0.0,
                    'improvement': 0.0,
                    'bottleneck_analysis': analysis.bottleneck_analysis,
                    'optimization_suggestions': analysis.optimization_suggestions,
                    'optimized_code': optimized_code,
                    'eval_result': str(eval_result),
                    'failed': True
                }
                optimization_history.append(iteration_result)
                
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Final results
    print(f"\n🏁 OPTIMIZATION COMPLETED")
    print(f"{'='*60}")
    print(f"Problem: {dataset_problem_name}")
    print(f"Initial speedup: {initial_speedup:.3f}")
    print(f"Final speedup: {best_speedup:.3f}")
    print(f"Total improvement: {best_speedup - initial_speedup:+.3f}")
    print(f"Iterations completed: {len(optimization_history)}")
    
    # Save results
    if config.save_optimization_history:
        history_file = os.path.join(config.logdir, f"optimization_history_{dataset_problem_name}.json")
        with open(history_file, 'w') as f:
            json.dump(optimization_history, f, indent=2)
        print(f"💾 Optimization history saved to: {history_file}")
    
    if config.save_best_kernel and best_speedup > initial_speedup:
        output_file = config.kernel_file.replace('.py', f'_optimized_{best_speedup:.3f}x.py')
        
        with open(output_file, 'w') as f:
            f.write(f'''"""
Problem Name: {dataset_problem_name}
Multiturn Optimization Result
Original speedup: {initial_speedup:.3f}
Optimized speedup: {best_speedup:.3f}
Improvement: {best_speedup - initial_speedup:+.3f}
Iterations: {len(optimization_history)}
"""

{best_kernel}''')
        
        print(f"💾 Best kernel saved to: {output_file}")
    
    # Print cost information
    history = lm.history
    total_cost = sum(entry.get("cost", 0) for entry in history)
    print(f"💰 Total optimization cost: ${total_cost:.2f}")
    print(f"🔄 Total LLM interactions: {len(history)}")


if __name__ == "__main__":
    main() 