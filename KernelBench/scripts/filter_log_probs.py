import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import argparse
import modal
import json 

app = modal.App()

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"    
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
    )
    .run_commands("pip uninstall torch torchvision torchaudio")
    .run_commands(
        "pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 "
        "--index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "transformers",
        "datasets",
        "safetensors",
        "pandas",
        "huggingface-hub[hf_transfer]==0.25.2",
        "accelerate>=0.26.0",
        "scipy",
        "scikit-learn",
    )
    .run_commands("pip install flash-attn --no-build-isolation")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

TOKENIZER_NAME = "Qwen/Qwen2.5-7B"
MODEL_NAME = "Qwen/Qwen2.5-7B"


def compute_log_probability(text: str, tokenizer, model, device) -> float:
    """
    Tokenizes the input `text`, runs it through the causal LM, and sums the log
    probabilities of each token in the sequence (using teacher-forcing).
    Returns: total_log_prob (float)
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # shape: (1, seq_len)

    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.logits: (1, seq_len, vocab_size)
        logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()            # (1, seq_len-1, vocab_size)
        shift_labels = input_ids[:, 1:].contiguous()             # (1, seq_len-1)

        # Compute log-softmax over vocab dimension
        log_probs = F.log_softmax(shift_logits, dim=-1)          # (1, seq_len-1, vocab_size)

        # Gather the log-probability of the actual next token at each position
        token_log_probs = log_probs.gather(
            -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)                                             # (1, seq_len-1)

        # Sum over all tokens to get total log probability
        total_log_prob = token_log_probs.sum().item()

    return total_log_prob


def compute_conditional_log_probability(prompt: str, text: str, tokenizer, model, device) -> float:
    """
    Computes the conditional log probability P(text | prompt).
    Only calculates log probabilities for the text tokens that come after the prompt.
    
    Args:
        prompt: The conditioning context
        text: The text to compute probability for
        tokenizer, model, device: Model components
    
    Returns: conditional_log_prob (float)
    """
    # Tokenize prompt and full sequence separately
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = prompt_inputs["input_ids"].shape[1]
    
    # Tokenize the full sequence (prompt + text)
    full_text = prompt + text
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = full_inputs["input_ids"]  # shape: (1, seq_len)

    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()            # (1, seq_len-1, vocab_size)
        shift_labels = input_ids[:, 1:].contiguous()             # (1, seq_len-1)

        # Compute log-softmax over vocab dimension
        log_probs = F.log_softmax(shift_logits, dim=-1)          # (1, seq_len-1, vocab_size)

        # Gather the log-probability of the actual next token at each position
        token_log_probs = log_probs.gather(
            -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)                                             # (1, seq_len-1)

        # Only sum over the text tokens (after the prompt)
        # Note: prompt_length-1 because we shifted everything by 1 for next-token prediction
        text_token_log_probs = token_log_probs[:, prompt_length-1:]
        conditional_log_prob = text_token_log_probs.sum().item()

    return conditional_log_prob


@app.function(image=image, gpu="a100", timeout=20000)
def calculate_log_prob(text: str, prompt: str = None) -> float:
    """
    Remote-callable function that loads the Qwen model and computes log probability.
    
    Args:
        text: The text to compute probability for
        prompt: Optional conditioning prompt. If provided, computes P(text | prompt)
    
    Returns: log probability (float)
    """
    print("Loading tokenizer and model on Modal GPU...")
    
    # load tokenizer and model inside Modal
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    model.eval()  # we only need inference
    
    if prompt is not None:
        print(f"Computing conditional log probability P(text | prompt)")
        print(f"Prompt: '{prompt}'")
        print(f"Text: '{text}'")
        lp = compute_conditional_log_probability(prompt, text, tokenizer, model, device)
    else:
        print("Computing unconditional log probability P(text)")
        print(f"Text: '{text}'")
        lp = compute_log_probability(text, tokenizer, model, device)
    
    print(f"Log probability result: {lp}")
    return lp


@app.function(image=image, gpu="a100", timeout=20000)
def generate_from_prompt(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    """
    Remote‐callable function that loads the Qwen model on Modal's GPU,
    generates a continuation for the given `prompt`, and returns the generated text.

    Args:
        prompt: the input text to condition on.
        max_new_tokens: how many new tokens to sample.
        temperature: sampling temperature.
        top_k: top‐k sampling cutoff.
        top_p: nucleus (top‐p) sampling cutoff.

    Returns:
        A single string containing the prompt + generated continuation.
    """

    print("Loading tokenizer and model on Modal GPU for generation...")

    # Load tokenizer & model inside the function (on the remote GPU)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    model.eval()

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the entire sequence (prompt + continuation)
    generated_text = tokenizer.decode(
        generation_output[0], skip_special_tokens=True
    )

    print("=== Generated Text ===")
    print(generated_text)
    return generated_text


@app.local_entrypoint()
def main(text: str, prompt: str = None):
    if prompt is not None:
        print(f"Computing conditional log probability:")
        print(f"  Prompt: '{prompt}'")
        print(f"  Text: '{text}'")
    else:
        print(f"Computing unconditional log probability for: '{text}'")
    
    logp = calculate_log_prob.remote(text, prompt)
    print(f"Total log probability returned: {logp}")