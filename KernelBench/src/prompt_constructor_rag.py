"""
Simplified RAG-based Prompt Constructor for TileLang

High-performance prompt generation using the simplified RAG system.
"""

import os
from .rag_tilelang import generate_tilelang_with_rag


REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)


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
    
    # Paths to examples
    correct_tilelang_dir = os.path.join(REPO_TOP_PATH, "src/prompts/correct_tilelang")
    kernelbench_dir = os.path.join(REPO_TOP_PATH, "KernelBench")
    
    try:
        # Use the high-performance RAG system
        result = generate_tilelang_with_rag(
            original_code=ref_arch_src,
            correct_tilelang_dir=correct_tilelang_dir,
            kernelbench_dir=kernelbench_dir,
            paper_prompt=paper_prompt,
            guideline_prompt=guideline_prompt,
            k=k,
            exclude_current_problem=True,
            current_level=current_level,
            current_problem_id=current_problem_id
        )
        
        # Format as code block for extraction
        return f"```python\n{result}\n```"
        
    except Exception as e:
        print(f"RAG generation failed: {e}")
        print("Falling back to template-based generation")
        
        # Simple fallback
        from .prompt_constructor import prompt_generate_custom_tilelang_from_prompt_template
        return prompt_generate_custom_tilelang_from_prompt_template(ref_arch_src) 