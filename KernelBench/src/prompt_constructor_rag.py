"""
Simplified RAG-based Prompt Constructor for Kernel DSLs

High-performance prompt generation using the simplified RAG system.
Supports multiple DSLs: TileLang, ThunderKittens, CUDA, etc.
"""

import os
from .rag_dsl import generate_dsl_with_rag


REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)


def prompt_generate_custom_dsl_rag_enhanced(
    ref_arch_src: str,
    language: str = "tilelang",
    paper_prompt: str = "",
    guideline_prompt: str = "",
    problem_description: str = "",
    k: int = 5,
    current_level: int = None,
    current_problem_id: int = None
) -> str:
    """
    Generate kernel DSL code using RAG with enhanced prompting.
    
    Args:
        ref_arch_src: Original PyTorch code to optimize
        language: Target DSL ("tilelang", "thunderkittens", "cuda", etc.)
        paper_prompt: Research paper context
        guideline_prompt: Language-specific guidelines  
        problem_description: Description of the problem being solved
        k: Number of RAG examples to retrieve
        current_level: Current problem level (for exclusion)
        current_problem_id: Current problem ID (for exclusion)
    
    Returns:
        Generated DSL code
    """
    
    # Paths to examples (language-specific)
    correct_dsl_dir = os.path.join(REPO_TOP_PATH, f"src/prompts/correct_{language}")
    kernelbench_dir = os.path.join(REPO_TOP_PATH, "KernelBench")
    
    try:
        # Use the high-performance RAG system
        result = generate_dsl_with_rag(
            original_code=ref_arch_src,
            correct_dsl_dir=correct_dsl_dir,
            kernelbench_dir=kernelbench_dir,
            language=language,
            paper_prompt=paper_prompt,
            guideline_prompt=guideline_prompt,
            k=k,
            exclude_current_problem=True,
            current_level=current_level,
            current_problem_id=current_problem_id
        )
        
        # For TileLang, wrap in a single python fence to aid simple extractors.
        # For ThunderKittens/CUDA, return raw so multiple fenced blocks (cpp + python) remain detectable.
        if language == "tilelang":
            return f"```python\n{result}\n```"
        return result
        
    except Exception as e:
        print(f"RAG generation failed for {language}: {e}")
        print("Falling back to template-based generation")
        
        # Simple fallback (only for TileLang for now)
        if language == "tilelang":
            from .prompt_constructor import prompt_generate_custom_tilelang_from_prompt_template
            return prompt_generate_custom_tilelang_from_prompt_template(ref_arch_src)
        else:
            raise Exception(f"No fallback available for {language}")


def prompt_generate_custom_tilelang_rag_enhanced(
    ref_arch_src: str,
    paper_prompt: str = "",
    guideline_prompt: str = "",
    problem_description: str = "",
    k: int = 5,
    current_level: int = None,
    current_problem_id: int = None
) -> str:
    """
    Generate TileLang code using RAG with enhanced prompting.
    
    Convenience wrapper for TileLang generation using the language-agnostic RAG system.
    
    Args:
        ref_arch_src: Original PyTorch code to optimize
        paper_prompt: Research paper context
        guideline_prompt: TileLang guidelines  
        problem_description: Description of the problem being solved
        k: Number of RAG examples to retrieve
        current_level: Current problem level (for exclusion)
        current_problem_id: Current problem ID (for exclusion)
    
    Returns:
        Generated TileLang code
    """
    return prompt_generate_custom_dsl_rag_enhanced(
        ref_arch_src=ref_arch_src,
        language="tilelang",
        paper_prompt=paper_prompt,
        guideline_prompt=guideline_prompt,
        problem_description=problem_description,
        k=k,
        current_level=current_level,
        current_problem_id=current_problem_id
    ) 