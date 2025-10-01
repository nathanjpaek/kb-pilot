################################################################################
# These base prompts should accept either BAD_SYNTAX_PROMPT or SUBTLE_WRONG_REASONING_PROMPT as diversity_aspect input

PROMPT_BASE_BAD = """
You are a strong reasoner and coder at the domain specific programming language TileLang, which is designed to streamline the development of high-performance GPU kernels. You reason about and write TileLang kernels, and can effectively use the TileLang library to replace PyTorch operators in given architectures to get speedups on the GPU. 
 
Here is some information about the TileLang language: {tilelang_info}

Here are some examples of TileLang kernels: {icl_prompt}

Now we turn our attention to your task. Here is an architecture in PyTorch that is unoptimized: {kb_problem}

Here is TileLang code that implements and optimizes the above architecture with custom TileLang operators: {gold_kernel}


Pretend you are the one reasoning through writing this TileLang code. Output a reasoning chain STRICTLY following the guidelines below:

{diversity_aspect} And don't just reason at a high level but try to add detail about code choices (e.g. for loops, if statements, etc.).

Output this in a think aloud, chain-of-thought style reasoning chain ("Alright, so I..."). DO NOT talk about the provided TileLang code in your reasoning chain; remember, you are recreating a fresh chain of thought as if you had to solve the problem without knowledge of the final answer. Do not give away full lines of code in your reasoning. You may reference variables and dimensions but do not copy lines from the correct implementation in your reasoning. 
"""


PROMPT_BASE_BAD_NO_ICL = """
You are a strong reasoner and coder at the domain specific programming language TileLang, which is designed to streamline the development of high-performance GPU kernels. You reason about and write TileLang kernels, and can effectively use the TileLang library to replace PyTorch operators in given architectures to get speedups on the GPU. 
 
Here is some information about the TileLang language: {tilelang_info}

Now we turn our attention to your task. Here is an architecture in PyTorch that is unoptimized: {kb_problem}

Here is TileLang code that implements and optimizes the above architecture with custom TileLang operators: {gold_kernel}

Pretend you are the one reasoning through writing this TileLang code. Output a reasoning chain STRICTLY following the guidelines below:

{diversity_aspect} And don't just reason at a high level but try to add detail about code choices (e.g. for loops, if statements, etc.).

Output this in a think aloud, chain-of-thought style reasoning chain ("Okay, I..."). DO NOT talk about the provided TileLang code in your reasoning chain; remember, you are recreating a fresh chain of thought as if you had to solve the problem without knowledge of the final answer. Do not give away full lines of code in your reasoning. You may reference variables and dimensions but do not copy lines from the correct implementation in your reasoning. 
"""


BAD_SYNTAX_PROMPT = """
Generate a concise reasoning chain that is ONLY in natural language, not in ANY code, and does not reference any of the TileLang specific primitives. The goal is to be extremely vague about specific TileLang syntax but convey the conceptual choices the kernel makes (reason at the level of the concepts, i.e. tiles, threads, shared memory, accumulations, blocks, operations, loops, rather than T.alloc_shared, T.alloc_fragment, T.gemm, T.Pipelined, or other specific syntax). Make this as concise as possible while still ensuring that the general reasoning is thorough. Only use non-TileLang words.
"""


SUBTLE_WRONG_REASONING_PROMPT = """
However, do not recreate this reasoning chain entirely faithfully. Generate a reasoning chain that inserts a problematic reasoning error somewhere along the process that would derail someone from recreating this kernel correctly. Don't reveal the error, but rather seamlessly go into the error and continue on as if the error is correct. Up to some point follow the correct reasoning steps, then transition into the error. 
"""


RAMBLING_PROMPT = """
Generate a reasoning chain that rambles a lot and arrives at the point in a roundabout manner. 
"""


################################################################################
# These base prompts can accept BAD_SYNTAX_PROMPT, SUBTLE_WRONG_REASONING_PROMPT, 
# GENERAL_TO_SPECIFIC_PROMPT, TOP_TO_BOTTOM_PROMPT, EXPLAIN_WHY_PROMPT, CONSIDER_ALTERNATIVES_PROMPT_1, 
# CONSIDER_ALTERNATIVES_PROMPT_2, CONCISE_PROMPT, CONCEPTS_ONLY_PROMPT, or NORMAL_PROMPT as diversity_aspect input


PROMPT_BASE_BAD_NO_ICL_NO_INFO = """
You are a strong reasoner and coder at the domain specific programming language TileLang, which is designed to streamline the development of high-performance GPU kernels. You reason about and write TileLang kernels, and can effectively use the TileLang library to replace PyTorch operators in given architectures to get speedups on the GPU. 
 
Here is an architecture in PyTorch that is unoptimized: {kb_problem}

Here is TileLang code that implements and optimizes the above architecture with custom TileLang operators: {gold_kernel}

Pretend you are the one reasoning through writing this TileLang code. Output a reasoning chain STRICTLY following the guidelines below:

{diversity_aspect}

Output this in a think aloud, chain-of-thought style reasoning chain ("Okay, so I..."). DO NOT talk about the provided TileLang code in your reasoning chain; remember, you are recreating a fresh chain of thought as if you had to solve the problem without knowledge of the final answer. Do not give away full lines of code in your reasoning. You may reference variables and dimensions but do not copy lines from the correct implementation in your reasoning. 
"""


PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD = """
You are a strong reasoner and coder at the domain specific programming language TileLang, which is designed to streamline the development of high-performance GPU kernels. You reason about and write TileLang kernels, and can effectively use the TileLang library to replace PyTorch operators in given architectures to get speedups on the GPU. 
 
Here is an architecture in PyTorch that is unoptimized: {kb_problem}

Reason through how to write this TileLang code; don't actually write it. Output a reasoning chain STRICTLY following the guidelines below:

{diversity_aspect}

Output this in a think aloud, chain-of-thought style reasoning chain ("Alright, I..."). You are recreating a fresh chain of thought as if you had to solve the problem and arrive at the final answer. Reason specifically about TileLang primitives and syntax choices you make. And don't just reason at a high level but try to add detail about code choices (e.g. for loops, if statements, etc.).
"""


PROMPT_BASE_BAD_NO_ICL_NO_GOLD= """
You are a strong reasoner and coder at the domain specific programming language TileLang, which is designed to streamline the development of high-performance GPU kernels. You reason about and write TileLang kernels, and can effectively use the TileLang library to replace PyTorch operators in given architectures to get speedups on the GPU. 
 
Here is some information about the TileLang language: {tilelang_info}

Now we turn our attention to your task. Here is an architecture in PyTorch that is unoptimized: {kb_problem}

Reason through how to write this TileLang code; don't actually write it. Output a reasoning chain STRICTLY following the guidelines below:

{diversity_aspect}

Output this in a think aloud, chain-of-thought style reasoning chain ("Okay, so I'm..."). You are recreating a fresh chain of thought as if you had to solve the problem and arrive at the final answer. Don't just reason at a high level but try to add detail about code choices (e.g. for loops, if statements, etc.).
"""

################################################################################
