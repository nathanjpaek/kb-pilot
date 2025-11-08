"""
High-Performance RAG for Kernel DSL Generation

Simplified RAG system using DSPy with best-in-class models for maximum performance.
Designed for use with OpenAI O3 and resource-unconstrained environments.
Supports multiple DSLs: TileLang, ThunderKittens, CUDA, etc.
"""

import os
from typing import Dict, List, Optional
import dspy

from .utils import read_file
from .example_selector_mafer import select_smart_examples


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
        self.correct_dsl_dir = correct_dsl_dir
        self.kernelbench_dir = kernelbench_dir
        self.k = k
        
        # DSL generation module with optimized signature
        self.generate = dspy.ChainOfThought(
            "dsl_guidelines, context, original_code -> dsl_code"
        )
    
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
        
        # Retrieve top examples using smart selector
        examples = select_smart_examples(
            problem_code=original_code,
            language=self.language,
            k=self.k,
            current_level=self.current_level,
            current_problem_id=self.current_problem_id,
        )
        
        if examples:
            context = self._format_smart_examples(examples)
        else:
            print(f"No RAG examples found for {self.language.upper()} – generating without context.")
            context = ""
        
        # Generate optimized DSL code
        return self.generate(
            context=context,
            original_code=original_code,
            dsl_guidelines=dsl_guidelines
        )
    
    def _format_smart_examples(self, examples: List[Dict]) -> str:
        """Format smart-selected examples into retrieval context"""
        context_parts = []
        print(f"\n📋 Retrieved {len(examples)} RAG examples for Level {self.current_level} Problem {self.current_problem_id}:")
        
        for idx, example in enumerate(examples, 1):
            problem_name = example.get("problem_name", "unknown")
            score = example.get("score", 0.0)
            speedup = example.get("speedup", 0.0)
            solution_content = example.get("solution_code") or example.get("code", "")
            solution_code = self._extract_solution_code(solution_content)
            reference_code = example.get("reference_code", "")
            ops = example.get("operations", [])
            ops_str = ', '.join(ops) if ops else 'none detected'
            score_str = f"{score:.1f}"
            speedup_str = f"{speedup:.2f}×" if speedup else "unknown"
            
            print(f"  {idx}. {problem_name} (ops: {ops_str}) | score={score_str} | speedup={speedup_str}")
            
            context_parts.append(f"""
Example {idx} - {problem_name} (score {score_str}, speedup {speedup_str}):

Original PyTorch Code:
```python
{reference_code}
```

Optimized {self.language.upper()} Code:
```python
{solution_code}
```
""")
        
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