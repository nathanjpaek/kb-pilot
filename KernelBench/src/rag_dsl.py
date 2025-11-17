"""
High-Performance RAG for Kernel DSL Generation

Simplified RAG system using DSPy with best-in-class models for maximum performance.
Designed for use with OpenAI O3 and resource-unconstrained environments.
Supports multiple DSLs: TileLang, ThunderKittens, CUDA, etc.
"""

import os
import pickle
from typing import List, Optional, Dict
from dataclasses import dataclass
import dspy

from .utils import read_file
from .example_selector_mafer import select_doc_chunks, select_smart_examples


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
                 current_level: int = None, current_problem_id: int = None,
                 extra_ops: Optional[List[str]] = None):
        super().__init__()
        
        # Store language and exclusion parameters
        self.language = language
        self.exclude_current_problem = exclude_current_problem
        self.current_level = current_level
        self.current_problem_id = current_problem_id
        self.k = k
        self.extra_ops = [op.lower() for op in extra_ops] if extra_ops else []
        
        # Load and prepare examples
        self.examples = self._load_examples(correct_dsl_dir, kernelbench_dir)
        
        # Embeddings retriever is expensive (15-30s for 50+ examples).
        # Disabled by default since smart selector provides better examples.
        # Only enable if explicitly requested via RAG_USE_EMBEDDINGS=1
        self.retriever = None
        self.use_embeddings = os.getenv("RAG_USE_EMBEDDINGS", "0").lower() in ("1", "true", "yes", "on")
        
        if self.use_embeddings and len(self.examples) > 0:
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
        elif len(self.examples) == 0:
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

        context_sections: List[str] = []

        combined_ops = list({op.lower() for op in operations})
        if self.extra_ops:
            combined_ops.extend(self.extra_ops)

        # Smart example selection (always enabled, uses caching for speed)
        try:
            smart_examples = select_smart_examples(
                problem_code=original_code,
                language=self.language,
                k=self.k,
                current_level=self.current_level,
                current_problem_id=self.current_problem_id,
                fast_mode=False,  # Use full scoring, but caching makes it fast
            )
        except Exception as exc:
            print(f"⚠️ Smart example selection failed: {exc}")
            smart_examples = []

        if smart_examples:
            context_sections.append(self._format_smart_example_context(smart_examples))

        # Documentation chunks (enabled by default, can be disabled via RAG_SKIP_DOCS=1)
        skip_docs = os.getenv("RAG_SKIP_DOCS", "0").lower() in ("1", "true", "yes", "on")
        doc_chunks = []
        if not skip_docs:
            try:
                doc_chunks = select_doc_chunks(
                    problem_code=original_code,
                    language=self.language,
                    current_level=self.current_level,
                    current_problem_id=self.current_problem_id,
                    extra_ops=combined_ops,
                )
            except Exception as exc:
                print(f"⚠️ Doc chunk selection failed: {exc}")
                doc_chunks = []

        if doc_chunks:
            context_sections.append(self._format_doc_context(doc_chunks))

        if self.retriever is not None:
            retrieved = self.retriever(query).passages
            embed_context = self._format_context(retrieved)
            if embed_context.strip():
                context_sections.append(embed_context)

        if context_sections:
            context = "\n\n".join(section for section in context_sections if section.strip())
        else:
            print(f"No RAG context found for {self.language.upper()} – generating without retrieval support.")
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

    def _format_smart_example_context(self, examples: List[Dict], max_lines: int = 150) -> str:
        """
        Format smart examples as context, truncating long code to reduce prompt size.
        
        Args:
            examples: List of example dicts with reference_code and code
            max_lines: Maximum lines to include per code block (default 150)
        """
        lines = []
        print(f"\n📋 Smart selector returned {len(examples)} examples for Level {self.current_level} Problem {self.current_problem_id}:")
        for idx, example in enumerate(examples, 1):
            problem_name = example.get("problem_name", "unknown")
            score = example.get("score", 0.0)
            speedup = example.get("speedup")
            ops = example.get("operations") or []
            reference_code = example.get("reference_code", "")
            solution_code = example.get("code", "")
            
            # Truncate long code examples to reduce prompt size
            ref_lines = reference_code.split('\n')
            sol_lines = solution_code.split('\n')
            
            ref_truncated = len(ref_lines) > max_lines
            sol_truncated = len(sol_lines) > max_lines
            
            ref_code_final = '\n'.join(ref_lines[:max_lines])
            sol_code_final = '\n'.join(sol_lines[:max_lines])
            
            if ref_truncated:
                ref_code_final += f"\n# ... (truncated, showing first {max_lines} of {len(ref_lines)} lines)"
            if sol_truncated:
                sol_code_final += f"\n# ... (truncated, showing first {max_lines} of {len(sol_lines)} lines)"
            
            print(f"  {idx}. {problem_name} | score={score:.1f} | ops={','.join(ops) or 'none'} | speedup={speedup}")
            lines.append(
                f"""
Smart Example {idx} - {problem_name} (score {score:.1f}):

Original PyTorch Code:
```python
{ref_code_final}
```

Optimized {self.language.upper()} Code:
```python
{sol_code_final}
```
"""
            )
        return "\n".join(lines)

    def _format_doc_context(self, doc_chunks: List[Dict]) -> str:
        """Format documentation summaries as context"""
        lines = ["CuTe Documentation Insights:"]
        print(f"📚 Retrieved {len(doc_chunks)} documentation chunks.")
        for idx, doc in enumerate(doc_chunks, 1):
            title = doc.get("title", "Unnamed")
            category = doc.get("category", "")
            summary = doc.get("compressed_summary", "")
            key_concepts = doc.get("key_concepts", [])[:5]
            speed = doc.get("score", 0.0)
            lines.append(
                f"\nDoc {idx}: {title} ({category}) [score {speed:.1f}]\n"
                f"Key Concepts: {', '.join(key_concepts)}\n"
                f"Summary:\n{summary}"
            )
        return "\n".join(lines)


def create_kernel_rag(correct_dsl_dir: str, kernelbench_dir: str, language: str = "tilelang",
                      k: int = 5, exclude_current_problem: bool = True, 
                      current_level: int = None, current_problem_id: int = None,
                      extra_ops: Optional[List[str]] = None) -> KernelRAG:
    """Create high-performance kernel DSL RAG system"""
    return KernelRAG(correct_dsl_dir, kernelbench_dir, language, k, exclude_current_problem, current_level, current_problem_id, extra_ops=extra_ops)


def generate_dsl_with_rag(original_code: str, 
                          correct_dsl_dir: str,
                          kernelbench_dir: str,
                          language: str = "tilelang",
                          paper_prompt: str = "",
                          guideline_prompt: str = "",
                          k: int = 5,
                          exclude_current_problem: bool = True,
                          current_level: int = None,
                          current_problem_id: int = None,
                          extra_ops: Optional[List[str]] = None) -> str:
    """
    Generate DSL code using RAG - simplified interface
    
    This is the main function to use for kernel DSL generation.
    
    Args:
        language: DSL to generate ("tilelang", "tk", "cuda", etc.)
    """
    
    # Create RAG system
    rag = create_kernel_rag(correct_dsl_dir, kernelbench_dir, language, k, exclude_current_problem, current_level, current_problem_id, extra_ops=extra_ops)
    
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