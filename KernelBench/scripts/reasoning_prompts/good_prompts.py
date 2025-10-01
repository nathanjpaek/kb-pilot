################################################################################
# This base prompt can accept GENERAL_TO_SPECIFIC_PROMPT, TOP_TO_BOTTOM_PROMPT, EXPLAIN_WHY_PROMPT, CONSIDER_ALTERNATIVES_PROMPT_1, 
# CONSIDER_ALTERNATIVES_PROMPT_2, CONCISE_PROMPT, CONCEPTS_ONLY_PROMPT, or NORMAL_PROMPT as diversity_aspect input


PROMPT_BASE_GOOD = """
You are a strong reasoner and coder at the domain specific programming language TileLang, which is designed to streamline the development of high-performance GPU kernels. You reason about and write TileLang kernels, and can effectively use the TileLang library to replace PyTorch operators in given architectures to get speedups on the GPU. 
 
Here is some information about the TileLang language: {tilelang_info}

Here are some examples of TileLang kernels: {icl_prompt}

Now we turn our attention to your task. Here is an architecture in PyTorch that is unoptimized: {kb_problem}

Here is TileLang code that implements and optimizes the above architecture with custom TileLang operators: {gold_kernel}

Your task is to generate a reasoning chain that, if given to another model with minimal knowledge of TileLang, would help the model to solve the above problem correctly. Specifically, look at the correct kernel and reverse engineer the process of writing it. Pretend you are the one reasoning through writing this TileLang code. Don't just reason at a high level but try to add detail about code choices (e.g. for loops, if statements, etc.). Output a reasoning chain STRICTLY following the guidelines below:

Guidelines (must follow): {diversity_aspect}

Output this in natural language, think aloud, chain-of-thought style reasoning chain ("Alright, I..."). DO NOT talk about the correct implementation in your reasoning chain; remember, you are recreating a fresh chain of thought as if you had to solve the problem without knowledge of the final answer. ONLY OUTPUT YOUR REASONING CHAIN AND NOTHING ELSE.
"""


GENERAL_TO_SPECIFIC_PROMPT = """
Generate a reasoning chain that progresses from the most important things in the kernel to the lesser important things, not necessarily in order of the top-to-bottom code implementation of the kernel, but rather from most important choices that the kernel makes to lesser important choices. Include TileLang primitive and syntax decisions. Be thorough! Make sure all important choices are conveyed. 
"""


TOP_TO_BOTTOM_PROMPT = """
Generate a reasoning chain that reasons from the top to bottom of the function, in that strict order. Start from the top of the correct implementation to the bottom. Don't mention the correct implementation though. Be thorough! Make sure all important choices are conveyed. 
"""


EXPLAIN_WHY_PROMPT = """
Generate a reasoning chain that excessively explains why it’s making the choices it is making. Justify everything. Note when specific syntax should be used in the places that it should used in. Remember to reason through this process as if you were coming up with it for the first time. 
"""


STRESS_SYNTAX_PROMPT = """
Generate a thorough reasoning chain that stresses the syntax usage (without giving full lines of code) and explains why specific syntax should be used in the places that it is used in. Remember to reason through this process as if you were coming up with it for the first time. 
"""


CONSIDER_ALTERNATIVES_PROMPT_1 = """
Generate a reasoning chain that briefly goes down the path of alternative choices at ONLY A FEW critical steps of reasoning through making this kernel, but backtracks and corrects them to be the right choice (as seen in the correct kernel implementation), and explains why you went with that choice. Remember to reason through this process as if you were coming up with it for the first time. 
"""


CONSIDER_ALTERNATIVES_PROMPT_2 = """
Generate a reasoning chain that considers many alternative choices at steps of reasoning through making this kernel but reasons why it is wrong and why the correct choice is correct. Include TileLang primitive and syntax decisions.
"""


CONCISE_PROMPT = """
Generate a reasoning chain that is as concise as possible while still ensuring that the reasoning is thorough. In other words, the reasoning should compress only the MOST important choices that matter to getting the correct kernel. Make sure that the correct kernel can be recreated though!
"""


CONCEPTS_ONLY_PROMPT = """
Generate a reasoning chain that is ONLY in natural language, not in ANY code, and do not reference any of the TileLang specific primitives. The goal is to not explicitly mention any specific TileLang syntax but still convey the conceptual choices the kernel makes (reason at the level of the concepts, i.e. tiles, threads, shared memory, accumulations, blocks, operations, loops, rather than T.alloc_shared, T.alloc_fragment, T.gemm, T.Pipelined, or other specific syntax). Another model, when given this reasoning, should still recreate the kernel correctly. 
""" 

NORMAL_PROMPT = """
Generate a reasoning chain that does not ramble too much, but is still thorough, outlining the process of reasoning through the problem from start to finish. Note when specific syntax should be used in the places that it should used in. 
"""


STEP_BY_STEP_PROMPT = """
Generate a reasoning chain that is very structured and detailed. Even though it is in natural language, sentences should be distinct, concise actions or deductions. Aim to convey a systematic and traceable approach to problem-solving. This is the opposite of being rambly and roundabout.
"""