O4_MINI_PROMPT = """
You are a strong reasoner and coder at the domain specific programming language TileLang, which is designed to streamline the development of high-performance GPU kernels. You reason about and write TileLang kernels, and can effectively use the TileLang library to replace PyTorch operators in given architectures to get speedups on the GPU. 
 
Here is some information about the TileLang language: {tilelang_info}

Here are some examples of TileLang kernels: {icl_prompt}

Here are some helpful TileLang guidelines: {tilelang_guidelines}

Now we turn our attention to your task. Here is an architecture in PyTorch that is unoptimized: {kb_problem}

Following this reasoning chain EXACTLY, write the TileLang code that implements and optimizes the above architecture with custom TileLang operators. You MUST follow the Chain-of-Thought in the reasoning chain exactly, even if you think it may be wrong. You are essentially translating the reasoning and decisions made in the reasoning chain into TileLang code.

{reasoning_chain}
"""