DIVERSE_STRATEGY_PROMPT = """
You are a strong reasoner and coder at the domain specific programming language TileLang, which is designed to streamline the development of high-performance GPU kernels. You reason about and write TileLang kernels, and can effectively use the TileLang library to replace PyTorch operators in given architectures to get speedups on the GPU. 
 
Here is some information about the TileLang language: {tilelang_info}

Here are some examples of TileLang kernels: {icl_prompt}

Now we turn our attention to your task. Here is an architecture in PyTorch that is unoptimized: {kb_problem}

Here is TileLang code that implements and optimizes the above architecture with custom TileLang operators: {gold_kernel}

Your task is to generate a reasoning chain that, if given to another model with minimal knowledge of TileLang, would help the model to solve the above problem correctly. Specifically, look at the correct kernel and reverse engineer the process of writing it. Pretend you are the one reasoning through writing this TileLang code. Output a reasoning chain STRICTLY following the guidelines below:

Guidelines (must follow): Generate a reasoning chain that systematically explores a diverse set of optimization strategies for this TileLang kernel. At each major decision point, propose at least two alternative approaches—such as different tiling shapes, memory layouts (shared vs. register buffering), thread-to-data mappings, loop unrolling versus vectorization, or pipeline stages versus fused kernels—evaluate their potential performance benefits and drawbacks, and then explain which you would choose and why. Your chain of thought should read as if you’re surveying the design space for the first time, carefully comparing trade‐offs before settling on each design decision.

Output this in natural language, think aloud, chain-of-thought style reasoning chain. DO NOT talk about the correct implementation in your reasoning chain; remember, you are recreating a fresh chain of thought as if you had to solve the problem without knowledge of the final answer. You may reference variables and dimensions but do not give away full lines of code from the correct implementation in your reasoning. 

The output that you give will be given to an engineer that has exmaples of TileLang as well as documentation that you have been given. With your output, the engineer needs to be able to write correct TileLang code themselves.


"""
