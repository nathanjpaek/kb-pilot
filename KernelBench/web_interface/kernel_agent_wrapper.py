"""
Simplified wrapper for kernel generation.
Just calls the RAG system once and returns the generated code.
NO evaluation, NO test-time scaling, NO multiple attempts.
"""
import sys
import io
import queue
import threading
from typing import Iterator, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Add parent directories to path
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
SCRIPTS_DIR = PARENT_DIR / "scripts"
SRC_DIR = PARENT_DIR / "src"

for path in (CURRENT_DIR, PARENT_DIR, SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@dataclass
class KernelSpec:
    """Simplified spec for kernel generation - just what we need for RAG"""
    language: str = "cute"
    gpu: str = "H100"
    rag_k: int = 5
    model_name: str = "openai/o3"
    temperature: float = 1.0
    pytorch_reference: str = ""
    operation_description: str = ""
    paper_prompt: str = ""
    guideline_prompt: str = ""
    # Unused but kept for compatibility
    fast_model: Optional[str] = None
    ops_sequence: str = ""
    input_shapes: str = ""
    dtype: str = ""
    target_speedup: Optional[float] = None
    additional_constraints: str = ""
    problem_label: str = "custom_kernel"
    ops_list: List[str] = field(default_factory=list)
    current_level: Optional[int] = None
    current_problem_id: Optional[int] = None
    max_attempts: Optional[int] = 1
    measure_performance: bool = False
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    use_modal: bool = False
    enable_refiner: bool = False
    refiner_attempts: int = 0
    refiner_server: str = "openai"
    refiner_model: Optional[str] = None
    refiner_temperature: float = 1.0
    test_time_scaling: bool = False
    num_candidates: int = 1
    syntax_only: bool = True
    headless: bool = True


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
        
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        class StreamWrapper:
            def __init__(self, callback, original):
                self._callback = callback
                self._original = original
            
            def write(self, text):
                if self._callback and text.strip():
                    self._callback(text)
                self._original.write(text)
                return len(text)
            
            def flush(self):
                self._original.flush()
                
            def __getattr__(self, name):
                return getattr(self._original, name)
        
        sys.stdout = StreamWrapper(self.callback, self.original_stdout)
        sys.stderr = StreamWrapper(self.callback, self.original_stderr)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        return False


class StreamingKernelAgent:
    """Simplified kernel agent - just generates code, no evaluation"""
    
    def __init__(self, spec: KernelSpec):
        self.spec = spec
        self.output_queue = queue.Queue()
        self.generation_thread = None
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
        """Run kernel generation - single attempt, no evaluation"""
        try:
            self._log("🚀 Starting kernel generation...", "log")
            self._log(f"Language: {self.spec.language.upper()} | Model: {self.spec.model_name}", "log")
            
            # Capture all prints during generation
            def print_callback(text: str):
                if text.strip():
                    self._log(text.rstrip(), "log")
            
            with PrintCapture(callback=print_callback):
                # Configure DSPy
                import dspy
                from dotenv import load_dotenv
                load_dotenv()
                
                self._log(f"🤖 Configuring LLM: {self.spec.model_name}", "log")
                
                # Check if this is a reasoning model
                model_lower = self.spec.model_name.lower()
                is_reasoning_model = 'o3' in model_lower or 'o1' in model_lower
                
                if is_reasoning_model:
                    effective_temperature = 1.0
                    max_tokens = 20000
                else:
                    effective_temperature = self.spec.temperature
                    max_tokens = 16384
                
                lm = dspy.LM(self.spec.model_name, temperature=effective_temperature, max_tokens=max_tokens)
                dspy.configure(lm=lm)
                
                self._log(f"✅ LLM configured", "log")
                
                # Get paper and guideline prompts
                from scripts.cute_paperinfo_prompt import CUTE_PAPER_PROMPT
                from scripts.cute_guideline_prompt import CUTE_GUIDELINE_PROMPT
                from scripts.tilelang_paperinfo_prompt import TILELANG_PAPER_PROMPT
                from scripts.tilelang_guideline_prompt import TILELANG_GUIDELINE_PROMPT
                from scripts.tk_paperinfo_prompt import TK_PAPER_PROMPT
                from scripts.tk_guideline_prompt import TK_GUIDELINE_PROMPT
                
                GUIDELINE_BY_LANG = {
                    "cute": (CUTE_PAPER_PROMPT, CUTE_GUIDELINE_PROMPT),
                    "tilelang": (TILELANG_PAPER_PROMPT, TILELANG_GUIDELINE_PROMPT),
                    "tk": (TK_PAPER_PROMPT, TK_GUIDELINE_PROMPT),
                }
                
                paper_prompt, guideline_prompt = GUIDELINE_BY_LANG.get(
                    self.spec.language.lower(), 
                    GUIDELINE_BY_LANG["cute"]
                )
                
                # First, retrieve RAG examples to show them
                self._log(f"📚 Retrieving {self.spec.rag_k} similar examples...", "log")
                
                from src.example_selector_mafer import select_smart_examples
                
                try:
                    rag_examples = select_smart_examples(
                        problem_code=self.spec.pytorch_reference,
                        language=self.spec.language,
                        k=self.spec.rag_k,
                        current_level=self.spec.current_level,
                        current_problem_id=self.spec.current_problem_id,
                        fast_mode=False,
                    )
                    
                    # Send RAG examples to frontend
                    if rag_examples:
                        self._log(f"✅ Found {len(rag_examples)} similar examples", "log")
                        for i, example in enumerate(rag_examples, 1):
                            example_data = {
                                "index": i,
                                "problem_name": example.get("problem_name", f"Example {i}"),
                                "score": example.get("score", 0),
                                "reference_code": example.get("reference_code", ""),
                                "dsl_code": example.get("code", ""),
                            }
                            self.output_queue.put(StreamMessage(
                                type="rag_example",
                                content=f"Example {i}: {example_data['problem_name']}",
                                metadata=example_data,
                                timestamp=__import__('time').time()
                            ))
                except Exception as e:
                    self._log(f"⚠️ Could not retrieve examples: {e}", "log")
                
                # Now call RAG generation
                self._log("🤖 Generating optimized kernel...", "log")
                
                from src.prompt_constructor_rag import prompt_generate_custom_dsl_rag_enhanced
                
                generated = prompt_generate_custom_dsl_rag_enhanced(
                    ref_arch_src=self.spec.pytorch_reference,
                    language=self.spec.language,
                    paper_prompt=paper_prompt,
                    guideline_prompt=guideline_prompt,
                    problem_description=self.spec.operation_description or "Optimize this kernel",
                    k=self.spec.rag_k,
                    current_level=self.spec.current_level,
                    current_problem_id=self.spec.current_problem_id,
                    extra_ops=self.spec.ops_list if self.spec.ops_list else None,
                )
                
                self._log("✅ Generation complete!", "log")
                
                # Extract code from the response
                from src.utils import extract_first_code
                
                clean_code = extract_first_code(generated, ["python"])
                
                if clean_code:
                    self._log(f"\n{'='*60}", "log")
                    self._log("📝 Generated Kernel:", "log")
                    self._log(f"{'='*60}\n", "log")
                    
                    # Store result
                    self.result = {
                        "code": clean_code,
                        "evaluation": {"generated": True}
                    }
                    
                    # Send code to frontend
                    self._log(clean_code, "code")
                    
                    self._log(f"\n{'='*60}", "log")
                    self._log("✅ Kernel ready to copy!", "result")
                else:
                    # If extraction failed, show raw output
                    self._log("⚠️ Could not extract clean code, showing raw output:", "log")
                    self._log(generated, "code")
                    self.result = {
                        "code": generated,
                        "evaluation": {"generated": True, "raw": True}
                    }
                
        except Exception as e:
            import traceback
            error_msg = f"Error during generation: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg, "error")
            self.error = error_msg
        finally:
            self._log("", "done")
            self.output_queue.put(None)  # Sentinel to signal completion
    
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
