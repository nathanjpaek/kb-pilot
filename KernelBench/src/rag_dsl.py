"""
High-Performance RAG for Kernel DSL Generation

Simplified RAG system using DSPy with best-in-class models for maximum performance.
Designed for use with OpenAI O3 and resource-unconstrained environments.
Supports multiple DSLs: TileLang, ThunderKittens, CUDA, etc.
"""

import os
import pickle
from typing import List, Optional
from dataclasses import dataclass
import dspy

from .utils import read_file


@dataclass
class DSLExample:
    """Language-agnostic DSL example representation"""
    problem_name: str
    original_code: str
    dsl_solution: str
    operations: List[str]
    language: str  # e.g., "tilelang", "tk", "cuda"


class KernelRAG(dspy.Module):
    """High-performance RAG module for kernel DSL generation"""
    
    def __init__(self, correct_dsl_dir: str, kernelbench_dir: str, language: str = "tilelang",
                 k: int = 5, exclude_current_problem: bool = True, 
                 current_level: int = None, current_problem_id: int = None):
        super().__init__()
        
        # Store language and exclusion parameters
        self.language = language
        self.exclude_current_problem = exclude_current_problem
        self.current_level = current_level
        self.current_problem_id = current_problem_id
        
        # Load and prepare examples
        self.examples = self._load_examples(correct_dsl_dir, kernelbench_dir)
        
        # If we have at least one example, set up embeddings-based retrieval.
        # Otherwise, skip retriever setup and fall back to generation without RAG context.
        self.retriever = None
        if len(self.examples) > 0:
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
        else:
            print(f"Loaded 0 {self.language.upper()} examples for RAG; proceeding without retrieval context.")
        
        # DSL generation module with optimized signature
        self.generate = dspy.ChainOfThought(
            "dsl_guidelines, context, original_code -> dsl_code"
        )
    
    def _load_examples(self, correct_dsl_dir: str, kernelbench_dir: str) -> List[DSLExample]:
        """Load all DSL examples efficiently"""
        examples = []
        excluded_count = 0
        
        for level_dir in os.listdir(correct_dsl_dir):
            if not level_dir.startswith('level'):
                continue
            
            # exclude level3 from rag
            if level_dir.startswith('level3'):
                continue
                
            level_num = int(level_dir.replace('level', ''))
            level_path = os.path.join(correct_dsl_dir, level_dir)
            
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
                    
                    examples.append(DSLExample(
                        problem_name=os.path.basename(original_path).replace('.py', ''),
                        original_code=original_code,
                        dsl_solution=solution_code,
                        operations=operations,
                        language=self.language
                    ))
                    
                except Exception:
                    continue
        
        print(f"Loaded {len(examples)} {self.language.upper()} examples for RAG (excluded {excluded_count} current problem examples)")
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
    
    def forward(self, original_code: str, dsl_guidelines: str = "") -> dspy.Prediction:
        """Generate DSL code using RAG"""
        
        # Create query from original code
        operations = self._extract_operations(original_code)
        query = f"Operations: {', '.join(operations)}\nCode: {original_code}"
        
        # Retrieve relevant examples when retriever is available
        if self.retriever is not None:
            retrieved = self.retriever(query).passages
            context = self._format_context(retrieved)
        else:
            context = ""
        
        # Generate optimized DSL code
        return self.generate(
            context=context,
            original_code=original_code,
            dsl_guidelines=dsl_guidelines
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

Optimized {self.language.upper()} Code:
```python
{example.dsl_solution}
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


def create_kernel_rag(correct_dsl_dir: str, kernelbench_dir: str, language: str = "tilelang",
                      k: int = 5, exclude_current_problem: bool = True, 
                      current_level: int = None, current_problem_id: int = None) -> KernelRAG:
    """Create high-performance kernel DSL RAG system"""
    return KernelRAG(correct_dsl_dir, kernelbench_dir, language, k, exclude_current_problem, current_level, current_problem_id)


def generate_dsl_with_rag(original_code: str, 
                          correct_dsl_dir: str,
                          kernelbench_dir: str,
                          language: str = "tilelang",
                          paper_prompt: str = "",
                          guideline_prompt: str = "",
                          k: int = 5,
                          exclude_current_problem: bool = True,
                          current_level: int = None,
                          current_problem_id: int = None) -> str:
    """
    Generate DSL code using RAG - simplified interface
    
    This is the main function to use for kernel DSL generation.
    
    Args:
        language: DSL to generate ("tilelang", "tk", "cuda", etc.)
    """
    
    # Create RAG system
    rag = create_kernel_rag(correct_dsl_dir, kernelbench_dir, language, k, exclude_current_problem, current_level, current_problem_id)
    
    # Prepare comprehensive guidelines with speed focus
    lang_upper = language.upper()
    guidelines = f"""
{paper_prompt}

{guideline_prompt}
"""
    
    # Generate DSL code
    result = rag(original_code=original_code, dsl_guidelines=guidelines.strip())
    
    return result.dsl_code


# Example usage
if __name__ == "__main__":
    # Configure DSPy with best model
    lm = dspy.LM('openai/o1-preview')  # or gpt-4o for faster iteration
    dspy.configure(lm=lm)
    
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    
    # Example 1: TileLang generation
    print("=" * 50)
    print("TileLang Generation Example")
    print("=" * 50)
    
    correct_tilelang_dir = os.path.join(REPO_ROOT, "src/prompts/correct_tilelang")
    result_tilelang = generate_dsl_with_rag(
        original_code=test_code,
        correct_dsl_dir=correct_tilelang_dir,
        kernelbench_dir=kernelbench_dir,
        language="tilelang",
        guideline_prompt="Focus on efficient ReLU implementation using TileLang",
        k=5
    )
    
    print("Generated TileLang code:")
    print(result_tilelang)
    
    # Example 2: ThunderKittens generation
    print("\n" + "=" * 50)
    print("ThunderKittens Generation Example")
    print("=" * 50)
    
    correct_tk_dir = os.path.join(REPO_ROOT, "correct_tk")
    result_tk = generate_dsl_with_rag(
        original_code=test_code,
        correct_dsl_dir=correct_tk_dir,
        kernelbench_dir=kernelbench_dir,
        language="tk",
        guideline_prompt="Focus on efficient ReLU implementation using ThunderKittens",
        k=5
    )
    
    print("Generated ThunderKittens code:")
    print(result_tk) 