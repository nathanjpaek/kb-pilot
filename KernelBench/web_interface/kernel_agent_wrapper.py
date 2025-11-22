"""
Wrapper for kernel agent CLI that captures all print statements
and yields them as they occur, enabling real-time streaming.
"""
import sys
import io
import queue
import threading
from typing import Iterator, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Add parent directories to path
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
SCRIPTS_DIR = PARENT_DIR / "scripts"
SRC_DIR = PARENT_DIR / "src"

for path in (CURRENT_DIR, PARENT_DIR, SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.kernel_agent_cli import KernelAgentCLI, KernelSpec as CLI_KernelSpec, DEFAULT_RETRIES

# Type alias for compatibility
KernelSpec = CLI_KernelSpec


@dataclass
class StreamMessage:
    """Message structure for streaming output"""
    type: str  # 'log', 'code', 'result', 'error', 'done'
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None


class PrintCapture:
    """Context manager that captures print statements and yields them"""
    
    def __init__(self, callback=None):
        self.callback = callback
        self.original_stdout = None
        self.original_stderr = None
        self.capture_buffer = None
        
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create custom file-like object that calls callback on write
        self.capture_buffer = io.StringIO()
        
        class StreamWrapper(io.TextIOWrapper):
            def __init__(self, stream, callback, original):
                self._stream = stream
                self._callback = callback
                self._original = original
                super().__init__(stream.buffer, encoding='utf-8', errors='replace')
            
            def write(self, text):
                if self._callback and text.strip():
                    self._callback(text)
                self._original.write(text)  # Still write to original for debugging
                return len(text)
            
            def flush(self):
                self._original.flush()
        
        sys.stdout = StreamWrapper(self.capture_buffer, self.callback, self.original_stdout)
        sys.stderr = StreamWrapper(self.capture_buffer, self.callback, self.original_stderr)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        return False


class StreamingKernelAgent:
    """Kernel agent that streams its output in real-time"""
    
    def __init__(self, spec: KernelSpec):
        self.spec = spec
        self.output_queue = queue.Queue()
        self.generation_thread = None
        self.agent = None
        self.result = None
        self.error = None
        
    def _log(self, message: str, msg_type: str = "log"):
        """Add a message to the output queue"""
        import time
        self.output_queue.put(StreamMessage(
            type=msg_type,
            content=message,
            timestamp=time.time()
        ))
    
    def _generate_kernel(self):
        """Run kernel generation in a separate thread"""
        try:
            self._log("🚀 Starting kernel generation...", "log")
            self._log(f"Language: {self.spec.language.upper()} | GPU: {self.spec.gpu}", "log")
            
            # Create agent
            self.agent = KernelAgentCLI(self.spec, DEFAULT_RETRIES)
            
            # Capture all prints
            def print_callback(text: str):
                if text.strip():
                    self._log(text.rstrip(), "log")
            
            with PrintCapture(callback=print_callback):
                # Run the agent (this will generate and evaluate)
                # We need to modify the agent's run method to return results
                self._run_agent_with_results()
                
        except Exception as e:
            import traceback
            error_msg = f"Error during generation: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg, "error")
            self.error = error_msg
        finally:
            self._log("✅ Generation complete", "done")
            self.output_queue.put(None)  # Sentinel to signal completion
    
    def _run_agent_with_results(self):
        """Modified version of agent.run() that captures results"""
        # Import the agent's internal methods
        pytorch_code = self.spec.pytorch_reference
        retries = [
            {"rag_k": self.spec.rag_k, "temperature": self.spec.temperature},
            *DEFAULT_RETRIES,
        ]
        if self.spec.max_attempts:
            retries = retries[:max(self.spec.max_attempts, 1)]
        
        best_result = None
        best_strategy = None
        
        for attempt, strategy in enumerate(retries, start=1):
            # Use fast model for first attempt if available
            from scripts.generate_and_eval_rag_modal import configure_dspy
            model_to_use = self.spec.fast_model if (attempt == 1 and self.spec.fast_model) else self.spec.model_name
            if attempt == 1 and self.spec.fast_model:
                self._log(f"⚡ Using fast model ({model_to_use}) for first attempt", "log")
            configure_dspy(model_to_use, strategy.get("temperature", self.spec.temperature))
            
            self._log(f"Attempt {attempt}: rag_k={strategy['rag_k']} temp={strategy.get('temperature', self.spec.temperature)} model={model_to_use}", "log")
            
            # Test-time scaling on first attempt
            if self.spec.test_time_scaling and attempt == 1:
                candidates_with_results = self.agent._test_time_scaling(pytorch_code, strategy)
                if candidates_with_results:
                    correct_candidates = [c for c in candidates_with_results if c["evaluation"].get("correctness", False)]
                    if correct_candidates:
                        best_candidate_result = correct_candidates[0]
                        if best_candidate_result["evaluation"].get("correctness", False):
                            self._log(f"✓ Test-time scaling: Found {len(correct_candidates)} correct candidate(s)", "log")
                            self._log("🔄 Running full evaluation on best candidate...", "log")
                            full_eval_result = self.agent._evaluate_candidate(
                                best_candidate_result["code"],
                                quiet=False,
                                fast_mode=False
                            )
                            best_candidate_result["evaluation"] = full_eval_result
                            speedup = full_eval_result.get("runtime_stats", {}).get("performance_comparison", {}).get("speedup_ratio")
                            speedup_str = f" (speedup={speedup:.2f}×)" if speedup else ""
                            self._log(f"✓ Test-time scaling: Selected best of {len(candidates_with_results)} candidates{speedup_str}", "log")
                            best_result = best_candidate_result
                            best_strategy = strategy
                            break
            
            # Single candidate generation fallback
            candidate = self.agent._attempt_generation(pytorch_code, strategy)
            if candidate is None:
                self._log("✗ Generation failed (no valid code extracted)", "log")
                continue
            
            # Stream the generated code
            self._log(f"\n{'='*60}\n📝 Generated Kernel:\n{'='*60}", "code")
            self._log(candidate, "code")
            self._log(f"{'='*60}\n", "code")
            
            eval_result = self.agent._evaluate_candidate(candidate)
            if eval_result.get("correctness", False):
                self._log("✓ Kernel passes correctness checks!", "log")
                best_result = {"code": candidate, "evaluation": eval_result}
                best_strategy = strategy
                break
            else:
                self._log("✗ Kernel failed correctness.", "log")
                refined = self.agent._maybe_refine(candidate, eval_result)
                if refined:
                    best_result = refined
                    best_strategy = strategy
                    break
        
        # Store final result
        if best_result:
            self.result = {
                "code": best_result["code"],
                "evaluation": best_result["evaluation"],
                "strategy": best_strategy or "default"
            }
            self._log(f"\n{'='*60}\n✅ SUCCESS\n{'='*60}", "result")
            
            eval_data = best_result["evaluation"]
            if eval_data.get("compiled"):
                self._log("✓ Compiled successfully", "result")
            if eval_data.get("correctness"):
                self._log("✓ Correctness verified", "result")
            
            speedup = eval_data.get("runtime_stats", {}).get("performance_comparison", {}).get("speedup_ratio")
            if speedup:
                self._log(f"⚡ Speedup: {speedup:.2f}× vs PyTorch", "result")
        else:
            self._log("\n❌ All attempts failed. Consider refining specification.", "error")
    
    def start_generation(self):
        """Start kernel generation in background thread"""
        self.generation_thread = threading.Thread(target=self._generate_kernel, daemon=True)
        self.generation_thread.start()
        return self.generation_thread
    
    def stream_output(self) -> Iterator[StreamMessage]:
        """Generator that yields messages as they become available"""
        while True:
            try:
                msg = self.output_queue.get(timeout=0.1)
                if msg is None:  # Sentinel
                    break
                yield msg
            except queue.Empty:
                continue
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get the final result after generation completes"""
        if self.generation_thread:
            self.generation_thread.join(timeout=300)  # Wait up to 5 minutes
        return self.result


def generate_kernel_streaming(spec: KernelSpec) -> Iterator[StreamMessage]:
    """
    Generate a kernel and stream all output messages.
    
    Yields StreamMessage objects with type, content, and metadata.
    """
    agent = StreamingKernelAgent(spec)
    agent.start_generation()
    
    # Stream all messages
    for msg in agent.stream_output():
        yield msg
    
    # Yield final result if available
    result = agent.get_result()
    if result:
        yield StreamMessage(
            type="result",
            content="Generation completed successfully",
            metadata=result
        )
    elif agent.error:
        yield StreamMessage(
            type="error",
            content=agent.error
        )

