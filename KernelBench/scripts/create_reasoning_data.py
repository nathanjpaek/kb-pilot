import argparse
import json
import os
import glob
import random
import datetime
from together import Together
from scripts.reasoning_prompts.bad_prompts import (
    PROMPT_BASE_BAD, PROMPT_BASE_BAD_NO_ICL, PROMPT_BASE_BAD_NO_ICL_NO_INFO, 
    PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD, PROMPT_BASE_BAD_NO_ICL_NO_GOLD, BAD_SYNTAX_PROMPT, 
    SUBTLE_WRONG_REASONING_PROMPT, RAMBLING_PROMPT
)
from scripts.reasoning_prompts.good_prompts import (
    PROMPT_BASE_GOOD, GENERAL_TO_SPECIFIC_PROMPT, TOP_TO_BOTTOM_PROMPT, 
    EXPLAIN_WHY_PROMPT, STRESS_SYNTAX_PROMPT, CONSIDER_ALTERNATIVES_PROMPT_1, 
    CONSIDER_ALTERNATIVES_PROMPT_2, CONCISE_PROMPT, CONCEPTS_ONLY_PROMPT, NORMAL_PROMPT, STEP_BY_STEP_PROMPT
)
from scripts.reasoning_prompts.diverse_strategy_prompts import DIVERSE_STRATEGY_PROMPT
from scripts.reasoning_prompts.qwen_base_prompt import QWEN_BASE_PROMPT
from scripts.reasoning_prompts.o4_mini_prompt import O4_MINI_PROMPT

# Import the existing query_server function from utils
import sys
sys.path.append("../src")
from src.utils import query_server, TOGETHER_KEY, OPENAI_KEY, read_file
from scripts.tilelang_paperinfo_prompt import PAPER_PROMPT
from scripts.tilelang_icl_prompt import ICL_PROMPT

from dotenv import load_dotenv 
load_dotenv()

# Define problems to skip (e.g., those in a predefined test set)
# Format: list of (level, problem_id) tuples
# Holdout indices: {129, 2, 9, 139, 149, 24, 159, 39, 169, 44, 50, 179, 59, 189, 62, 199, 89, 96, 109, 119}
TEST_SET_PROBLEMS = [
    (1, 2), (1, 9), (1, 24), (1, 39), (1, 44), (1, 50), (1, 59), (1, 62), (1, 89), (1, 96),
    (2, 9), (2, 19), (2, 29), (2, 39), (2, 49), (2, 59), (2, 69), (2, 79), (2, 89), (2, 99)
] 

# Keywords in filename that cause a problem to be skipped
FILENAME_SKIP_KEYWORDS = ["cheat", "veryslow", "slow"]

# Define all imported prompt constants for reverse name lookup
ALL_PROMPT_CONSTANTS_FOR_NAMING = {
    "PROMPT_BASE_BAD": PROMPT_BASE_BAD,
    "PROMPT_BASE_BAD_NO_ICL": PROMPT_BASE_BAD_NO_ICL,
    "PROMPT_BASE_BAD_NO_ICL_NO_GOLD": PROMPT_BASE_BAD_NO_ICL_NO_GOLD,
    "PROMPT_BASE_BAD_NO_ICL_NO_INFO": PROMPT_BASE_BAD_NO_ICL_NO_INFO,
    "PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD": PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD,
    "BAD_SYNTAX_PROMPT": BAD_SYNTAX_PROMPT,
    "SUBTLE_WRONG_REASONING_PROMPT": SUBTLE_WRONG_REASONING_PROMPT,
    "RAMBLING_PROMPT": RAMBLING_PROMPT,
    "PROMPT_BASE_GOOD": PROMPT_BASE_GOOD,
    "GENERAL_TO_SPECIFIC_PROMPT": GENERAL_TO_SPECIFIC_PROMPT,
    "TOP_TO_BOTTOM_PROMPT": TOP_TO_BOTTOM_PROMPT,
    "EXPLAIN_WHY_PROMPT": EXPLAIN_WHY_PROMPT,
    "STRESS_SYNTAX_PROMPT": STRESS_SYNTAX_PROMPT,
    "CONSIDER_ALTERNATIVES_PROMPT_1": CONSIDER_ALTERNATIVES_PROMPT_1,
    "CONSIDER_ALTERNATIVES_PROMPT_2": CONSIDER_ALTERNATIVES_PROMPT_2,
    "CONCISE_PROMPT": CONCISE_PROMPT,
    "CONCEPTS_ONLY_PROMPT": CONCEPTS_ONLY_PROMPT,
    "NORMAL_PROMPT": NORMAL_PROMPT,
    "DIVERSE_STRATEGY_PROMPT": DIVERSE_STRATEGY_PROMPT,
    "QWEN_BASE_PROMPT": QWEN_BASE_PROMPT,
    "O4_MINI_PROMPT": O4_MINI_PROMPT,
    "STEP_BY_STEP_PROMPT": STEP_BY_STEP_PROMPT,
}
VALUE_TO_NAME_LOOKUP = {v: k for k, v in ALL_PROMPT_CONSTANTS_FOR_NAMING.items() if isinstance(v, str)}

def _get_combined_prompt_template_name(base_value, aspect_value, lookup_map):
    base_name = lookup_map.get(base_value, "UNKNOWN_BASE_PROMPT")
    if aspect_value is not None:
        aspect_name = lookup_map.get(aspect_value, "UNKNOWN_ASPECT_PROMPT")
        return f"{base_name}_X_{aspect_name}"
    else:
        return f"{base_name}_ONLY"

# Model configurations for different prompt strategies
good_reasoning_models = ["gpt-4o", "gpt-4o-mini", 'gpt-4o', 'gpt-4o', 'gpt-4o', "gpt-4.1", "gpt-4.1-mini", "o4-mini", "o4-mini","o4-mini", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "lgai/exaone-deep-32b"]

#bad_models = ["gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini", "Qwen/Qwen2.5-72B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14"]
bad_models = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o-mini", "gpt-4o-mini", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"]


client = Together()

# for when we deploy the bad ones on together ai
qwen_bad_models = ["sweetkruts/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-d4d3c602", "sweetkruts/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-d4d3c602", "sweetkruts/deepcogito/cogito-v1-preview-llama-8B-87e38c36", "sweetkruts/deepcogito/cogito-v1-preview-llama-8B-87e38c36"]

diverse_strategy_models = ["o4-mini", "o3", "gpt-4o"]
tree_search_models = ["o3"]


# Prompt strategy configurations with proper base_prompt + diversity_aspect structure
PROMPT_STRATEGIES = {
    "diverse_strategy": {
        "models": diverse_strategy_models,
        "templates": [
            {"base_prompt": DIVERSE_STRATEGY_PROMPT, "diversity_aspect": None}  # No diversity aspect needed
        ]
    },
    "good_reasoning": {
        "models": good_reasoning_models,
        "templates": [
            {"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": GENERAL_TO_SPECIFIC_PROMPT},
            {"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": TOP_TO_BOTTOM_PROMPT},
            {"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": EXPLAIN_WHY_PROMPT},
            {"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": STRESS_SYNTAX_PROMPT},
            {"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": CONSIDER_ALTERNATIVES_PROMPT_1},
            {"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": CONSIDER_ALTERNATIVES_PROMPT_2},
            #{"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": CONCISE_PROMPT},
            {"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": CONCEPTS_ONLY_PROMPT},
            {"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": NORMAL_PROMPT},
            {"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": STEP_BY_STEP_PROMPT},
            {"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": STEP_BY_STEP_PROMPT},
            {"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": STEP_BY_STEP_PROMPT},
            {"base_prompt": PROMPT_BASE_GOOD, "diversity_aspect": STEP_BY_STEP_PROMPT},
        ]
    },
    "bad_reasoning": {
        "models": bad_models,
        "templates": [
            #{"base_prompt": PROMPT_BASE_BAD, "diversity_aspect": BAD_SYNTAX_PROMPT},
            #{"base_prompt": PROMPT_BASE_BAD, "diversity_aspect": SUBTLE_WRONG_REASONING_PROMPT},
            #{"base_prompt": PROMPT_BASE_BAD, "diversity_aspect": SUBTLE_WRONG_REASONING_PROMPT},
            #{"base_prompt": PROMPT_BASE_BAD, "diversity_aspect": SUBTLE_WRONG_REASONING_PROMPT},
            #{"base_prompt": PROMPT_BASE_BAD_NO_ICL, "diversity_aspect": BAD_SYNTAX_PROMPT},
            #{"base_prompt": PROMPT_BASE_BAD_NO_ICL, "diversity_aspect": SUBTLE_WRONG_REASONING_PROMPT},
            #{"base_prompt": PROMPT_BASE_BAD, "diversity_aspect": SUBTLE_WRONG_REASONING_PROMPT},
            #{"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO, "diversity_aspect": EXPLAIN_WHY_PROMPT},
            #{"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO, "diversity_aspect": CONSIDER_ALTERNATIVES_PROMPT_2},
            #{"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO, "diversity_aspect": CONCISE_PROMPT},
            #{"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO, "diversity_aspect": CONCEPTS_ONLY_PROMPT},
            #{"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO, "diversity_aspect": RAMBLING_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO, "diversity_aspect": SUBTLE_WRONG_REASONING_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO, "diversity_aspect": SUBTLE_WRONG_REASONING_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO, "diversity_aspect": SUBTLE_WRONG_REASONING_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO, "diversity_aspect": SUBTLE_WRONG_REASONING_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO, "diversity_aspect": SUBTLE_WRONG_REASONING_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_GOLD, "diversity_aspect": STRESS_SYNTAX_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_GOLD, "diversity_aspect": RAMBLING_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_GOLD, "diversity_aspect": RAMBLING_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_GOLD, "diversity_aspect": RAMBLING_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_GOLD, "diversity_aspect": CONCEPTS_ONLY_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD, "diversity_aspect": GENERAL_TO_SPECIFIC_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD, "diversity_aspect": TOP_TO_BOTTOM_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD, "diversity_aspect": EXPLAIN_WHY_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD, "diversity_aspect": STRESS_SYNTAX_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD, "diversity_aspect": CONSIDER_ALTERNATIVES_PROMPT_1},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD, "diversity_aspect": CONSIDER_ALTERNATIVES_PROMPT_2},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD, "diversity_aspect": CONCISE_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD, "diversity_aspect": CONCEPTS_ONLY_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD, "diversity_aspect": NORMAL_PROMPT},
            {"base_prompt": PROMPT_BASE_BAD_NO_ICL_NO_INFO_NO_GOLD, "diversity_aspect": RAMBLING_PROMPT}
        ]
    }
}


def get_server_type_for_model(model_name):
    """
    Map model names to their corresponding server types for the query_server function
    """
    if model_name.startswith("Qwen") or model_name.startswith("deepseek-ai") or model_name.startswith("lgai"):
        return "together" 
    else:
        return "openai" 


def is_reasoning_model(model_name):
    """
    Check if a model is a reasoning model that requires special handling
    """
    return model_name in ["o1", "o3"]


def generate_reasoning_chains(kernel_content, prompt_strategy, num_chains=8, tilelang_info="", icl_prompt="", kb_problem=""):
    """
    Generate reasoning chains for a given kernel using the specified prompt strategy.
    
    Args:
        kernel_content (str): The kernel content to generate reasoning for
        prompt_strategy (str): The prompting strategy to use
        num_chains (int): Number of reasoning chains to generate
        tilelang_info (str): TileLang documentation/info
        icl_prompt (str): In-context learning examples
        kb_problem (str): KernelBench problem description
        
    Returns:
        list: A list of generated reasoning chains with metadata
    """
    if prompt_strategy not in PROMPT_STRATEGIES:
        raise ValueError(f"Unknown prompt strategy: {prompt_strategy}")
    
    strategy_config = PROMPT_STRATEGIES[prompt_strategy]
    models = strategy_config["models"]
    templates = strategy_config["templates"]
    
    reasoning_chains = []
    
    for i in range(num_chains):
        # Randomly sample model and prompt template
        selected_model = random.choice(models)
        selected_template = random.choice(templates)
        
        print(f"Generating chain {i+1}/{num_chains} using model: {selected_model}")
        
        try:
            # Format the prompt template with the actual data
            base_prompt = selected_template["base_prompt"]
            diversity_aspect = selected_template["diversity_aspect"]
            
            # Determine which format parameters are needed by checking for placeholders in the template
            import re
            placeholders = re.findall(r'\{([^}]+)\}', base_prompt)
            
            # Prepare format parameters based on what's actually needed
            format_params = {}
            if 'tilelang_info' in placeholders:
                format_params['tilelang_info'] = tilelang_info
            if 'icl_prompt' in placeholders:
                format_params['icl_prompt'] = icl_prompt
            if 'kb_problem' in placeholders:
                format_params['kb_problem'] = kb_problem
            if 'gold_kernel' in placeholders:
                format_params['gold_kernel'] = kernel_content
            if 'diversity_aspect' in placeholders and diversity_aspect is not None:
                format_params['diversity_aspect'] = diversity_aspect
            
            # Format with all needed parameters in one step
            formatted_prompt = base_prompt.format(**format_params)
            
            # Determine server type and reasoning model settings
            server_type = get_server_type_for_model(selected_model)
            is_reasoning = is_reasoning_model(selected_model)

            # Conditionally append to prompt for OpenAI models
            if server_type == "openai":
                openai_specific_instruction = "\nChain-of-thought style means that when you are tackling a task, you might restate the goal in simple terms, then break it into its core operations, asking yourself clarifying questions as you go. You use words like 'wait' a lot, and you weigh alternative approaches by noting their pros and cons. As you refine your plan, you explicitly note what you need to figure out next and—if you discover a mistake—backtrack to correct earlier steps. You do this many times. You output fairly long text."
                formatted_prompt += openai_specific_instruction

            if selected_model == "lgai/exaone-deep-32b":
                lgai_specific_instruction = "\nMAKE YOUR RESPONSE BRIEF."
                formatted_prompt += lgai_specific_instruction

            #if selected_model == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B":
            #    deepseek_specific_instruction = "\nOnly output the reasoning chain, not the final solution (stop generating after your <think> tokens.)"
            #    formatted_prompt += deepseek_specific_instruction
            
            # Debug: Print the final prompt being sent to OpenAI
            prompt_combo_name = _get_combined_prompt_template_name(selected_template["base_prompt"], selected_template["diversity_aspect"], VALUE_TO_NAME_LOOKUP)
            print(f"\n{'='*80}")
            print(f"USING PROMPT COMBINATION: {prompt_combo_name} FOR MODEL: {selected_model.upper()}")
            print(f"{'='*80}")
            print(formatted_prompt)
            print(f"{'='*80}")
            print(f"Prompt length: {len(formatted_prompt)} characters")
            print(f"{'='*80}\n")
            
            # Set up model-specific parameters
            if is_reasoning:
                # Reasoning models like o1, o3 use different parameters
                reasoning_effort = "medium" if selected_model in ["o1", "o3"] else None
                response = query_server(
                    prompt=formatted_prompt,
                    server_type=server_type,
                    model_name=selected_model,
                    max_tokens=8192,  # Higher token limit for reasoning chains
                    temperature=0.7,
                    is_reasoning_model=True,
                    reasoning_effort=reasoning_effort
                )
            else:
                # Standard models
                response = query_server(
                    prompt=formatted_prompt,
                    system_prompt="You are an expert GPU kernel engineer skilled in TileLang optimization.",
                    server_type=server_type,
                    model_name=selected_model,
                    max_tokens=8192,
                    temperature=0.75,
                    top_p=0.9
                )
            
            # Extract the reasoning chain from the response
            reasoning_chain = response[0] if isinstance(response, list) and len(response) > 0 else str(response)

            if selected_model == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B":
                # First, remove a leading "<think>" if the reasoning chain starts with it.
                if reasoning_chain.startswith("<think>"):
                    reasoning_chain = reasoning_chain[len("<think>"):] 
                
                # Now, treat "</think>" as the primary stop token.
                # Take everything before the first occurrence of "</think>".
                stop_tag_index = reasoning_chain.find("</think>")
                if stop_tag_index != -1:
                    reasoning_chain = reasoning_chain[:stop_tag_index]
                # else: 
                #    # Optional: Fallback if </think> is not found but <think> might still be a relevant stop. 
                #    # For now, if </think> isn't there, we keep the rest of the string (after stripping leading <think>).
                #    if "<think>" in reasoning_chain: # This would be a <think> that wasn't at the very start
                #        reasoning_chain = reasoning_chain.split("<think>")[0]
            
            # Create the chain metadata
            chain = {
                "chain_id": i + 1,
                "model": selected_model,
                "server_type": server_type,
                "prompt_template": _get_combined_prompt_template_name(selected_template["base_prompt"], selected_template["diversity_aspect"], VALUE_TO_NAME_LOOKUP),
                "kernel_content": kernel_content,
                "reasoning_chain": reasoning_chain,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "prompt_strategy": prompt_strategy,
                "is_reasoning_model": is_reasoning,
                "generation_successful": True,
                "formatted_prompt_length": len(formatted_prompt)
            }
            reasoning_chains.append(chain)
            print(f"✓ Successfully generated chain {i+1}/{num_chains}")
            
        except Exception as e:
            print(f"✗ Error generating chain {i+1}/{num_chains} with {selected_model}: {str(e)}")
            # Add error information to the chain
            chain = {
                "chain_id": i + 1,
                "model": selected_model,
                "server_type": get_server_type_for_model(selected_model),
                "prompt_template": _get_combined_prompt_template_name(selected_template["base_prompt"], selected_template["diversity_aspect"], VALUE_TO_NAME_LOOKUP),
                "kernel_content": kernel_content,
                "reasoning_chain": f"[ERROR] Failed to generate reasoning chain: {str(e)}",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "prompt_strategy": prompt_strategy,
                "is_reasoning_model": is_reasoning_model(selected_model),
                "generation_successful": False,
                "error": str(e)
            }
            reasoning_chains.append(chain)
    
    return reasoning_chains


def process_file(filepath):
    print(f"Processing {filepath}...")
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        # Placeholder for actual processing
        processed_data = {"filename": os.path.basename(filepath), "content_length": len(content), "status": "processed"}
    except Exception as e:
        processed_data = {"filename": os.path.basename(filepath), "error": str(e), "status": "error"}
    return processed_data


def parse_custom_args():
    """
    Parse custom command line arguments in the format key=value
    """
    import sys
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Try to convert to int if possible
            try:
                value = int(value)
            except ValueError:
                pass
            args[key] = value
        else:
            # Handle traditional flags
            if arg.startswith('--'):
                args[arg[2:]] = True
    return args


def load_kernel_file(level, problem_id, correct_dir):
    """
    Load the kernel file for a given level and problem ID
    """
    # Files directly matching LEVEL_ID.py
    file_pattern_direct = os.path.join(correct_dir, f"level{level}", f"{level}_{problem_id}.py")
    # Files matching LEVEL_ID_*.py
    file_pattern_suffix = os.path.join(correct_dir, f"level{level}", f"{level}_{problem_id}_*.py")
    
    found_files = glob.glob(file_pattern_direct) + glob.glob(file_pattern_suffix)
    
    if not found_files:
        # This will be caught by the try-except in main and skip the problem
        raise FileNotFoundError(f"No gold kernel files found for level {level}, problem ID {problem_id} in {correct_dir}/level{level}")
    
    # Use the first found file for now. Could be extended to handle multiple matches if necessary.
    filepath = found_files[0]

    # Check for skip keywords in the filename
    for keyword in FILENAME_SKIP_KEYWORDS:
        if keyword in os.path.basename(filepath).lower(): # Case-insensitive check
            print(f"Skipping L{level} P{problem_id}: Filename '{os.path.basename(filepath)}' contains keyword '{keyword}'.")
            return None, None # Indicate that this file should be skipped
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    return filepath, content


def load_prompt_template_data(correct_dir):
    """
    Load the actual TileLang info and ICL prompt data from existing infrastructure
    """
    # Load the actual TileLang documentation from PAPER_PROMPT
    tilelang_info = PAPER_PROMPT
    
    # Load the actual in-context learning examples from ICL_PROMPT  
    icl_prompt = ICL_PROMPT
    
    return tilelang_info, icl_prompt, ""


def load_kernelbench_problem(level, problem_id):
    """
    Load the actual KernelBench problem description for the given level and problem ID
    This should integrate with your existing KernelBench data loading infrastructure
    """
    # Try to load from the problems directory structure
    problems_dir = "../KernelBench"
    level_dir_path = os.path.join(problems_dir, f"level{level}")
    file_pattern_to_search = os.path.join(level_dir_path, f"{problem_id}_*.py")

    problem_code_content = "[Problem code not found]"
    actual_file_loaded = "N/A"

    matching_files = glob.glob(file_pattern_to_search)

    if matching_files:
        file_to_read = matching_files[0] # Read the first match
        actual_file_loaded = os.path.basename(file_to_read)
        try:
            problem_code_content = read_file(file_to_read) # Assuming read_file is imported and works
        except Exception as e:
            print(f"Warning: Error reading problem file {file_to_read}: {e}")
            problem_code_content = f"[Error reading problem code from {actual_file_loaded}]"
    else:
        print(f"Warning: No problem file found matching pattern '{file_pattern_to_search}'")

    # Ensure the loaded content is not empty if a file was supposedly read and no error occurred during reading
    if problem_code_content != "[Problem code not found]" and not problem_code_content.startswith("[Error reading problem code"):
        assert len(problem_code_content) > 0, f"Problem code read from {actual_file_loaded} is empty."
    
    return f"KernelBench Level {level}, Problem {problem_id} (from file: {actual_file_loaded}):\n{problem_code_content}"


def generate_qwen_response_with_together(model_name, prompt_content, temperature=0.63, max_tokens=8192):
    """
    Generate a response using Together API client for Qwen models.
    Based on the working code provided by the user.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Collect the streamed response
        full_response = ""
        for token in response:
            if hasattr(token, 'choices') and token.choices and token.choices[0].delta.content:
                full_response += token.choices[0].delta.content
        
        return full_response
        
    except Exception as e:
        print(f"Error with Together API for {model_name}: {str(e)}")
        return f"[ERROR] Failed to generate response: {str(e)}"


def generate_additional_qwen_chains(kernel_content, tilelang_info="", icl_prompt="", kb_problem="", num_qwen_chains=5):
    """
    Generate additional reasoning chains using Qwen models for bad reasoning augmentation.
    This is called specifically for the bad_reasoning strategy.
    """
    print(f"\n🔄 Generating {num_qwen_chains} additional Qwen bad reasoning chains...")
    
    qwen_chains = []
    
    for i in range(num_qwen_chains):
        # Randomly select a Qwen model
        selected_qwen_model = random.choice(qwen_bad_models)
        
        print(f"Generating Qwen chain {i+1}/{num_qwen_chains} using model: {selected_qwen_model}")
        
        try:
            # Format the QWEN_BASE_PROMPT with the actual data
            import re
            placeholders = re.findall(r'\{([^}]+)\}', QWEN_BASE_PROMPT)
            
            # Prepare format parameters based on what's actually needed
            format_params = {}
            if 'tilelang_info' in placeholders:
                format_params['tilelang_info'] = tilelang_info
            if 'icl_prompt' in placeholders:
                format_params['icl_prompt'] = icl_prompt
            if 'kb_problem' in placeholders:
                format_params['kb_problem'] = kb_problem
            if 'gold_kernel' in placeholders:
                format_params['gold_kernel'] = kernel_content
            
            # Format the prompt
            formatted_prompt = QWEN_BASE_PROMPT.format(**format_params)
            
            print(f"Using QWEN_BASE_PROMPT for {selected_qwen_model}")
            print(f"Prompt length: {len(formatted_prompt)} characters")
            
            # Generate response using Together API with specific temperature and max_tokens
            reasoning_chain = generate_qwen_response_with_together(
                selected_qwen_model, 
                formatted_prompt, 
                temperature=0.8, 
                max_tokens=4096
            )

            if reasoning_chain.startswith("<think>"):
                    reasoning_chain = reasoning_chain[len("<think>"):] 

            stop_tag_index = reasoning_chain.find("</think>")
            if stop_tag_index != -1:
                reasoning_chain = reasoning_chain[:stop_tag_index]
            
            # Additional stop conditions for code blocks
            code_tag_index = reasoning_chain.find("<code>")
            if code_tag_index != -1:
                reasoning_chain = reasoning_chain[:code_tag_index]
            
            close_code_tag_index = reasoning_chain.find("</code>")
            if close_code_tag_index != -1:
                reasoning_chain = reasoning_chain[:close_code_tag_index]
            
            python_block_index = reasoning_chain.find("```")
            if python_block_index != -1:
                reasoning_chain = reasoning_chain[:python_block_index]
            
            # Create the chain metadata
            chain = {
                "chain_id": f"qwen_{i + 1}",  # Distinguish from regular chains
                "model": selected_qwen_model,
                "server_type": "together",
                "prompt_template": "QWEN_BASE_PROMPT_ONLY",
                "kernel_content": kernel_content,
                "reasoning_chain": reasoning_chain,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "prompt_strategy": "bad_reasoning_qwen_augmentation",
                "is_reasoning_model": False,
                "generation_successful": True,
                "formatted_prompt_length": len(formatted_prompt)
            }
            qwen_chains.append(chain)
            print(f"✓ Successfully generated Qwen chain {i+1}/{num_qwen_chains}")
            
        except Exception as e:
            print(f"✗ Error generating Qwen chain {i+1}/{num_qwen_chains} with {selected_qwen_model}: {str(e)}")
            # Add error information to the chain
            chain = {
                "chain_id": f"qwen_{i + 1}",
                "model": selected_qwen_model,
                "server_type": "together",
                "prompt_template": "QWEN_BASE_PROMPT_ONLY",
                "kernel_content": kernel_content,
                "reasoning_chain": f"[ERROR] Failed to generate reasoning chain: {str(e)}",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "prompt_strategy": "bad_reasoning_qwen_augmentation",
                "is_reasoning_model": False,
                "generation_successful": False,
                "error": str(e)
            }
            qwen_chains.append(chain)
    
    return qwen_chains


def main():
    args = parse_custom_args()
    
    # Check required arguments
    required_args = ['level', 'start_problem_id', 'end_problem_id', 'prompt']
    for req_arg in required_args:
        if req_arg not in args:
            print(f"Error: Missing required argument: {req_arg}")
            print("Usage: python create_reasoning_data.py level=1 start_problem_id=1 end_problem_id=10 prompt=diverse_strategy [num_chains=8] [prompts_dir=...] [reasoning_dir=...]")
            print(f"Available prompt strategies: {list(PROMPT_STRATEGIES.keys())}")
            return
    
    level = args['level']
    start_problem_id = args['start_problem_id']
    end_problem_id = args['end_problem_id']
    prompt_strategy = args['prompt']
    num_chains = args.get('num_chains', 8)
    
    # Validate prompt strategy
    if prompt_strategy not in PROMPT_STRATEGIES:
        print(f"Error: Unknown prompt strategy: {prompt_strategy}")
        print(f"Available strategies: {list(PROMPT_STRATEGIES.keys())}")
        return
    
    # Set up directories
    correct_dir = args.get('prompts_dir', "../src/prompts/correct_tilelang")
    base_reasoning_dir = args.get('reasoning_dir', "../reasoning_data")
    
    if not os.path.isdir(correct_dir):
        print(f"Error: Prompts directory not found: {correct_dir}")
        return
    
    # Create specific subdirectory based on prompt_strategy
    specific_reasoning_output_dir = os.path.join(base_reasoning_dir, prompt_strategy)
    os.makedirs(specific_reasoning_output_dir, exist_ok=True)

    for current_problem_id in range(start_problem_id, end_problem_id + 1):
        print(f"\n{'='*40}")
        print(f"Processing Level {level}, Problem ID: {current_problem_id}")
        print(f"Using prompt strategy: {prompt_strategy}")
        print(f"{'='*40}\n")

        # Check if the current problem is in the defined test set
        if (level, current_problem_id) in TEST_SET_PROBLEMS:
            print(f"Skipping L{level} P{current_problem_id}: Problem is in the predefined test set.")
            continue

        try:
            # Load the kernel file (gold solution)
            # This will also check for filename keywords like "cheat" or "veryslow"
            filepath, kernel_content = load_kernel_file(level, current_problem_id, correct_dir)
            
            if filepath is None and kernel_content is None: # Indicates a skip based on filename keywords
                continue # load_kernel_file already printed the reason
            
            print(f"Loaded gold kernel from: {filepath}")
            
            # Load prompt template data (TileLang info, ICL examples)
            print("Loading prompt template data...")
            tilelang_info, icl_prompt, _ = load_prompt_template_data(correct_dir)
            
            # Load the specific KernelBench problem (original problem code/description)
            print(f"Loading KernelBench problem description for L{level} P{current_problem_id}...")
            kb_problem = load_kernelbench_problem(level, current_problem_id)
            
            # Generate reasoning chains
            print(f"Generating {num_chains} reasoning chains using '{prompt_strategy}' strategy...")
            reasoning_chains = generate_reasoning_chains(
                kernel_content=kernel_content,
                prompt_strategy=prompt_strategy,
                num_chains=num_chains,
                tilelang_info=tilelang_info,
                icl_prompt=icl_prompt,
                kb_problem=kb_problem
            )
            
            # Generate additional Qwen chains if this is bad_reasoning strategy
            if prompt_strategy == "bad_reasoning":
                qwen_chains = generate_additional_qwen_chains(
                    kernel_content=kernel_content,
                    tilelang_info=tilelang_info,
                    icl_prompt=icl_prompt,
                    kb_problem=kb_problem,
                    num_qwen_chains=4
                )
                # Combine the regular reasoning chains with the Qwen chains
                reasoning_chains.extend(qwen_chains)
                print(f"✅ Added {len(qwen_chains)} Qwen augmentation chains to bad_reasoning output")
            
            successful_chains = [chain for chain in reasoning_chains if chain.get('generation_successful', False)]
            failed_chains = [chain for chain in reasoning_chains if not chain.get('generation_successful', False)]
            
            output_data = {
                "level": level,
                "problem_id": current_problem_id,
                "prompt_strategy": prompt_strategy,
                "source_file": filepath,
                "num_chains_requested": num_chains,
                "num_chains_total": len(reasoning_chains),
                "num_chains_successful": len(successful_chains),
                "num_chains_failed": len(failed_chains),
                "reasoning_chains": reasoning_chains,
                "generation_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "template_data": {
                    "tilelang_info_length": len(tilelang_info),
                    "icl_prompt_length": len(icl_prompt),
                    "kb_problem_length": len(kb_problem)
                }
            }
            
            output_filename = f"{level}_{current_problem_id}_{prompt_strategy}.json"
            output_filepath = os.path.join(specific_reasoning_output_dir, output_filename)
            
            with open(output_filepath, 'w') as f:
                json.dump(output_data, f, indent=4)
            
            print(f"\n🎉 Generation complete for L{level} P{current_problem_id}!")
            print(f"✓ Successfully generated: {len(successful_chains)} reasoning chains")
            if failed_chains:
                print(f"✗ Failed to generate: {len(failed_chains)} reasoning chains")
            print(f"📁 Results saved to: {output_filepath}")
            
            model_counts = {}
            for chain in reasoning_chains:
                model = chain.get('model', 'unknown')
                model_counts[model] = model_counts.get(model, 0) + 1
            
            print(f"\n📊 Model usage summary for L{level} P{current_problem_id}:")
            for model, count in model_counts.items():
                model_chains = [c for c in reasoning_chains if c.get('model') == model]
                successful = len([c for c in model_chains if c.get('generation_successful', False)])
                failed = len([c for c in model_chains if not c.get('generation_successful', False)])
                print(f"  {model}: {count} total ({successful} successful, {failed} failed)")
            
        except FileNotFoundError as e:
            print(f"Error processing L{level} P{current_problem_id}: {e}")
            print("Skipping to next problem.")
            continue # Continue to the next problem_id in the range
        except Exception as e:
            print(f"An unexpected error occurred while processing L{level} P{current_problem_id}: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping to next problem.")
            continue # Continue to the next problem_id in the range

if __name__ == "__main__":
    main() 