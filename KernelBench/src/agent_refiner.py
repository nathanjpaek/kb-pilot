"""
Lightweight CuTe kernel refiner used by the interactive agent.

This module adapts the logic from `multiturn_optimizer_willy.py`
to operate on in-memory kernels rather than files on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from .utils import create_inference_server_from_presets, extract_all_code_blocks


def _format_eval_failure(result: Optional[Dict]) -> str:
    if not result:
        return "No evaluation result available."

    lines = []
    compiled = result.get("compiled", False)
    correctness = result.get("correctness", False)
    lines.append(f"compiled={compiled}")
    lines.append(f"correctness={correctness}")

    metadata = result.get("metadata", {})
    if metadata:
        for key, value in metadata.items():
            lines.append(f"{key}: {value}")

    if "error" in result:
        lines.append(f"error: {result['error']}")

    runtime_stats = result.get("runtime_stats")
    if runtime_stats:
        lines.append(f"runtime_stats: {runtime_stats}")

    return "\n".join(lines)


def _build_refine_prompt(
    current_code: str,
    error_text: str,
    guideline_prompt: str,
    problem_label: str,
    attempt_idx: int,
) -> str:
    return (
        "You are an expert CuTe engineer. Use the CuTe Python DSL to fix this kernel.\n"
        "Goals:\n"
        "1. Kernel must compile.\n"
        "2. Kernel must pass correctness tests.\n"
        "3. Kernel should be as performant as possible.\n\n"
        f"Guidelines:\n{guideline_prompt}\n\n"
        f"Problem label: {problem_label}\n"
        f"Attempt number: {attempt_idx}\n\n"
        "Current CuTe kernel:\n"
        f"```python\n{current_code}\n```\n\n"
        "Evaluation failure details:\n"
        f"{error_text}\n\n"
        "Output requirements:\n"
        "- Return ONLY one code block containing the full CuTe Python kernel.\n"
        "- Code must be self-contained and include the `ModelNew` class.\n"
        "- Do not include explanation outside the code block.\n"
    )


def _extract_cute_code(llm_response: str) -> Optional[str]:
    blocks = extract_all_code_blocks(llm_response)
    return blocks.get("python")


@dataclass
class CuteKernelRefinerConfig:
    max_attempts: int = 3
    server_type: str = "openai"
    model_name: Optional[str] = "o3"
    temperature: float = 1.0
    max_tokens: Optional[int] = 30000
    is_reasoning_model: bool = True
    reasoning_effort: str = "medium"


class CuteKernelRefiner:
    def __init__(self, guideline_prompt: str, config: Optional[CuteKernelRefinerConfig] = None):
        self.guideline_prompt = guideline_prompt
        self.config = config or CuteKernelRefinerConfig()
        self.query_llm = self._create_llm_client()

    def _create_llm_client(self):
        llm_kwargs = {}
        if self.config.model_name is not None:
            llm_kwargs["model_name"] = self.config.model_name
        if self.config.temperature is not None:
            llm_kwargs["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            llm_kwargs["max_tokens"] = self.config.max_tokens
        if self.config.is_reasoning_model is not None:
            llm_kwargs["is_reasoning_model"] = self.config.is_reasoning_model
        if self.config.reasoning_effort is not None:
            llm_kwargs["reasoning_effort"] = self.config.reasoning_effort

        return create_inference_server_from_presets(
            server_type=self.config.server_type,
            greedy_sample=False,
            verbose=False,
            time_generation=False,
            **llm_kwargs,
        )

    def refine(
        self,
        initial_code: str,
        evaluate_fn: Callable[[str], Dict],
        problem_label: str = "custom_kernel",
        initial_result: Optional[Dict] = None,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Attempt to fix the kernel, returning (best_code, eval_result).

        evaluate_fn should call the evaluation pipeline and return a result dict.
        """
        current_code = initial_code
        eval_result = initial_result

        for attempt in range(1, self.config.max_attempts + 1):
            if eval_result and eval_result.get("correctness"):
                return current_code, eval_result

            error_text = _format_eval_failure(eval_result)

            prompt = _build_refine_prompt(
                current_code=current_code,
                error_text=error_text,
                guideline_prompt=self.guideline_prompt,
                problem_label=problem_label,
                attempt_idx=attempt,
            )

            try:
                llm_output = self.query_llm(prompt)
            except Exception as exc:
                return None, {"error": f"LLM call failed: {exc}"}

            next_code = _extract_cute_code(llm_output)
            if not next_code:
                return None, {"error": "Refiner did not return a python code block."}

            current_code = next_code
            eval_result = evaluate_fn(current_code)

        if eval_result and eval_result.get("correctness"):
            return current_code, eval_result

        return None, eval_result

