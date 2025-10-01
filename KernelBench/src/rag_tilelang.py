"""
High-Performance RAG for TileLang Kernel Generation

Simplified RAG system using DSPy with best-in-class models for maximum performance.
Designed for use with OpenAI O3 and resource-unconstrained environments.
"""

import os
import pickle
from typing import List, Optional
from dataclasses import dataclass
import dspy

from .utils import read_file


@dataclass
class TileLangExample:
    """Simple TileLang example representation"""
    problem_name: str
    original_code: str
    tilelang_solution: str
    operations: List[str]


class TileLangRAG(dspy.Module):
    """High-performance RAG module for TileLang generation"""
    
    def __init__(self, correct_tilelang_dir: str, kernelbench_dir: str, k: int = 5, 
                 exclude_current_problem: bool = True, current_level: int = None, current_problem_id: int = None):
        super().__init__()
        
        # Store exclusion parameters
        self.exclude_current_problem = exclude_current_problem
        self.current_level = current_level
        self.current_problem_id = current_problem_id
        
        # Load and prepare examples
        self.examples = self._load_examples(correct_tilelang_dir, kernelbench_dir)
        
        # Use best embedding model
        embedder = dspy.Embedder('openai/text-embedding-3-large', dimensions=3072)
        
        # Create corpus for retrieval
        corpus = []
        for example in self.examples:
            # Combine problem description and operations for better retrieval
            text = f"Operations: {', '.join(example.operations)}\nCode: {example.original_code}"
            corpus.append(text)
        
        # Initialize retriever
        self.retriever = dspy.retrievers.Embeddings(
            embedder=embedder, 
            corpus=corpus, 
            k=k
        )
        
        # TileLang generation module with optimized signature
        self.generate = dspy.ChainOfThought(
            "tilelang_guidelines, context, original_code -> tilelang_code"
        )
    
    def _load_examples(self, correct_tilelang_dir: str, kernelbench_dir: str) -> List[TileLangExample]:
        """Load all TileLang examples efficiently"""
        examples = []
        excluded_count = 0
        
        for level_dir in os.listdir(correct_tilelang_dir):
            if not level_dir.startswith('level'):
                continue
            
            # exclude level3 from rag
            if level_dir.startswith('level3'):
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
                
                # Find original problem
                original_path = self._find_original_problem(kernelbench_dir, level_num, problem_num)
                if not original_path:
                    continue
                
                try:
                    original_code = read_file(original_path)
                    solution_code = read_file(os.path.join(level_path, filename))
                    
                    # Extract clean solution code
                    solution_code = self._extract_solution_code(solution_code)
                    
                    # Extract operations
                    operations = self._extract_operations(original_code)
                    
                    examples.append(TileLangExample(
                        problem_name=os.path.basename(original_path).replace('.py', ''),
                        original_code=original_code,
                        tilelang_solution=solution_code,
                        operations=operations
                    ))
                    
                except Exception:
                    continue
        
        print(f"Loaded {len(examples)} TileLang examples for RAG (excluded {excluded_count} current problem examples)")
        return examples
    
    def _find_original_problem(self, kernelbench_dir: str, level: int, problem_num: int) -> Optional[str]:
        """Find original problem file"""
        level_dir = os.path.join(kernelbench_dir, f"level{level}")
        if not os.path.exists(level_dir):
            return None
            
        for filename in os.listdir(level_dir):
            if filename.startswith(f"{problem_num}_") and filename.endswith('.py'):
                return os.path.join(level_dir, filename)
        return None
    
    def _extract_solution_code(self, content: str) -> str:
        """Extract solution code, skip evaluation headers"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('class '):
                return '\n'.join(lines[i:])
        return content
    
    def _extract_operations(self, code: str) -> List[str]:
        """Extract PyTorch operations from code"""
        operations = []
        torch_ops = [
            'relu', 'softmax', 'matmul', 'conv2d', 'conv3d', 'linear', 'sigmoid',
            'tanh', 'gelu', 'dropout', 'batch_norm', 'layer_norm', 'max_pool',
            'avg_pool', 'transpose', 'permute', 'reshape', 'view', 'add', 'mul',
            'div', 'sub', 'sqrt', 'pow', 'exp', 'log', 'mean', 'sum', 'max', 'min'
        ]
        
        code_lower = code.lower()
        for op in torch_ops:
            if op in code_lower or f'torch.{op}' in code_lower or f'nn.{op}' in code_lower:
                operations.append(op)
        
        return list(set(operations))
    
    def forward(self, original_code: str, tilelang_guidelines: str = "") -> dspy.Prediction:
        """Generate TileLang code using RAG"""
        
        # Create query from original code
        operations = self._extract_operations(original_code)
        query = f"Operations: {', '.join(operations)}\nCode: {original_code}"
        
        # Retrieve relevant examples
        retrieved = self.retriever(query).passages
        
        # Format context with examples
        context = self._format_context(retrieved)
        
        # Generate optimized TileLang code
        return self.generate(
            context=context,
            original_code=original_code,
            tilelang_guidelines=tilelang_guidelines
        )
    
    def _format_context(self, retrieved_passages: List[str]) -> str:
        """Format retrieved examples as context"""
        context_parts = []
        retrieved_examples = []
        
        for i, passage in enumerate(retrieved_passages):
            # Find corresponding example
            for example in self.examples:
                example_text = f"Operations: {', '.join(example.operations)}\nCode: {example.original_code}"
                if passage.strip() == example_text.strip():
                    retrieved_examples.append(example)
                    context_parts.append(f"""
Example {i+1} - {example.problem_name}:

Original PyTorch Code:
```python
{example.original_code}
```

Optimized TileLang Code:
```python
{example.tilelang_solution}
```
""")
                    break
        
        # Print retrieved examples info
        if retrieved_examples:
            print(f"\n📋 Retrieved {len(retrieved_examples)} RAG examples for Level {self.current_level} Problem {self.current_problem_id}:")
            for i, example in enumerate(retrieved_examples, 1):
                ops_str = ', '.join(example.operations) if example.operations else 'none detected'
                print(f"  {i}. {example.problem_name} (ops: {ops_str})")
            print()
        
        return "\n".join(context_parts)


def create_tilelang_rag(correct_tilelang_dir: str, kernelbench_dir: str, k: int = 5,
                       exclude_current_problem: bool = True, current_level: int = None, current_problem_id: int = None) -> TileLangRAG:
    """Create high-performance TileLang RAG system"""
    return TileLangRAG(correct_tilelang_dir, kernelbench_dir, k, exclude_current_problem, current_level, current_problem_id)


def generate_tilelang_with_rag(original_code: str, 
                              correct_tilelang_dir: str,
                              kernelbench_dir: str,
                              paper_prompt: str = "",
                              guideline_prompt: str = "",
                              k: int = 5,
                              exclude_current_problem: bool = True,
                              current_level: int = None,
                              current_problem_id: int = None) -> str:
    """
    Generate TileLang code using RAG - simplified interface
    
    This is the main function to use for TileLang generation.
    """
    
    # Load TileLang documentation
    docs_dir = os.path.join(os.path.dirname(correct_tilelang_dir), "tilelang_docs")
    tilelang_docs = ""
    
    if os.path.exists(docs_dir):
        try:
            # Load elementwise operations doc
            elementwise_doc_path = os.path.join(docs_dir, "elementwise_ops.md")
            if os.path.exists(elementwise_doc_path):
                elementwise_doc = read_file(elementwise_doc_path)
                tilelang_docs += f"\n## TileLang ElementWise Operations Documentation:\n{elementwise_doc}\n"
            
            # Load GEMV doc
            gemv_doc_path = os.path.join(docs_dir, "gemv.md")
            if os.path.exists(gemv_doc_path):
                gemv_doc = read_file(gemv_doc_path)
                tilelang_docs += f"\n## TileLang GEMV Documentation:\n{gemv_doc}\n"
                
        except Exception as e:
            print(f"Warning: Could not load TileLang docs: {e}")
    
    # Create RAG system
    rag = create_tilelang_rag(correct_tilelang_dir, kernelbench_dir, k, exclude_current_problem, current_level, current_problem_id)
    
    # Prepare comprehensive guidelines with speed focus
    guidelines = f"""
{paper_prompt}

{guideline_prompt}

{tilelang_docs}

CRITICAL GUIDELINES:
- PRIMARY GOAL: Generate a CORRECT and FAST TileLang implementation
- PERFORMANCE IS IMPORTANT: Your optimization decisions should prioritize performance
- DO NOT USE torch.nn (except for Parameter, containers, and init)
- Generate efficient TileLang implementations using @T.prim_func
- Use tilelang.jit(out_idx=-1) for output tensor creation
- Use (val > a and val < b) instead of (a < val < b) for boundary checks

SPEED OPTIMIZATION STRATEGIES:
1. MEMORY COALESCING: Ensure coalesced memory access patterns
2. REGISTER USAGE: Use T.alloc_local() for register-level computations
3. SHARED MEMORY: Use T.alloc_shared() for block-level data sharing
4. THREAD PARALLELISM: Maximize T.Parallel usage and thread utilization
5. MEMORY HIERARCHY: Optimize data movement between global/shared/register memory
6. BLOCK/GRID SIZING: Choose optimal block and grid dimensions for the target GPU
7. CONTEXT: You may see similar problems in the context, but they may not necessarily be efficient. You should use the context as a guide, but not as a strict rule, and try to generate a better implementation.

TARGET HARDWARE: NVIDIA H100 (Hopper architecture)
- Optimize for high memory bandwidth utilization
- Leverage tensor cores when applicable
- Design for maximum occupancy and warp efficiency

OUTPUT REQUIREMENTS:
- Generate ONLY the ModelNew class code
- No additional text, comments, or explanations
- Focus on the most performance-critical optimizations
- Focus on replacing and fusing all PyTorch operators with efficient TileLang implementations
- If a module has multiple operators, try to fuse them as much as possible into a single TileLang kernel or write multiple efficient TileLang kernels to replace all of those operators.
- Your highest priority should be correctness, then speed and finally code simplicity
"""
    
    # Generate TileLang code
    result = rag(original_code=original_code, tilelang_guidelines=guidelines.strip())
    
    return result.tilelang_code


# Example usage
if __name__ == "__main__":
    # Configure DSPy with best model
    lm = dspy.LM('openai/o1-preview')  # or gpt-4o for faster iteration
    dspy.configure(lm=lm)
    
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    correct_tilelang_dir = os.path.join(REPO_ROOT, "src/prompts/correct_tilelang")
    kernelbench_dir = os.path.join(REPO_ROOT, "KernelBench")
    
    test_code = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)
"""
    
    result = generate_tilelang_with_rag(
        original_code=test_code,
        correct_tilelang_dir=correct_tilelang_dir,
        kernelbench_dir=kernelbench_dir,
        guideline_prompt="Focus on efficient ReLU implementation"
    )
    
    print("Generated TileLang code:")
    print(result) 