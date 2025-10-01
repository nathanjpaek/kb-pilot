"""
DSPy RAG Pipeline for TileLang Kernel Optimization

This implements a retrieval-augmented generation system for optimizing
PyTorch code to TileLang using DSPy's RAG framework.
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import dspy
from dspy import Example


# ============= Configuration =============


def configure_dspy(
    model: str = "openai/o4-mini",
    api_key: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 20000,
) -> dspy.LM:
    """Configure DSPy with the specified language model."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

    lm = dspy.LM(model, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    dspy.configure(lm=lm)
    return lm


# ============= Data Loading =============


def load_tilelang_examples(examples_dir: str) -> Tuple[List[Example], List[str], List[Dict]]:
    """Load TileLang examples and create a corpus for retrieval.

    Args:
        examples_dir: Directory containing example subdirectories

    Returns:
        Tuple of (examples, corpus_texts, corpus_metadata)
    """
    examples = []
    corpus_texts = []
    corpus_metadata = []

    examples_path = Path(examples_dir)
    if not examples_path.exists():
        raise ValueError(f"Examples directory not found: {examples_dir}")

    # Process each example directory
    for example_dir in sorted(examples_path.iterdir()):
        if not example_dir.is_dir():
            continue

        example_id = example_dir.name
        original_path = example_dir / "original.py"
        tilelang_path = example_dir / "tilelang.py"

        # Validate required files exist
        if not (original_path.exists() and tilelang_path.exists()):
            print(f"Warning: Skipping {example_id} - missing required files")
            continue

        # Read source files
        try:
            original_code = original_path.read_text(encoding="utf-8")
            tilelang_code = tilelang_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading files for {example_id}: {e}")
            continue

        # Create training example
        example = Example(
            original_code=original_code, tilelang_code=tilelang_code, metadata={"example_id": example_id}
        ).with_inputs("original_code")
        examples.append(example)

        # Create corpus entries for retrieval
        # Entry 1: Original code with context
        corpus_texts.append(
            f"Example: {example_id}\n" f"Type: Original PyTorch Implementation\n" f"Code:\n{original_code}"
        )
        corpus_metadata.append({"type": "original", "example_id": example_id, "dir_path": str(example_dir)})

        # Entry 2: TileLang implementation with context
        corpus_texts.append(
            f"Example: {example_id}\n" f"Type: TileLang Optimized Implementation\n" f"Code:\n{tilelang_code}"
        )
        corpus_metadata.append({"type": "tilelang", "example_id": example_id, "dir_path": str(example_dir)})

    print(f"Loaded {len(examples)} examples ({len(corpus_texts)} corpus documents)")
    return examples, corpus_texts, corpus_metadata


# ============= Retriever Setup =============


def setup_retriever(
    corpus_texts: List[str],
    corpus_metadata: List[Dict],
    embedder_model: str = "openai/text-embedding-3-small",
    embedding_dims: int = 512,
    k: int = 3,
    max_characters: int = 6000,
) -> dspy.Retrieve:
    """Set up the DSPy retriever with embeddings.

    Args:
        corpus_texts: List of text documents
        corpus_metadata: Metadata for each document
        embedder_model: Model to use for embeddings
        embedding_dims: Dimension of embeddings
        k: Number of documents to retrieve
        max_characters: Maximum characters per document

    Returns:
        Configured retriever
    """
    # Truncate corpus texts if needed
    truncated_texts = [text[:max_characters] for text in corpus_texts]

    print(f"Setting up retriever with {len(truncated_texts)} documents...")

    # Create embedder and retriever
    embedder = dspy.Embedder(embedder_model, dimensions=embedding_dims)
    retriever = dspy.retrievers.Embeddings(embedder=embedder, corpus=truncated_texts, k=k)

    # Attach metadata for later use
    retriever.corpus_metadata = corpus_metadata

    return retriever


# ============= RAG Module =============


class TileLangRAG(dspy.Module):
    """RAG module for TileLang code generation."""

    def __init__(self, retriever: dspy.Retrieve):
        super().__init__()
        self.retriever = retriever
        self.generator = dspy.ChainOfThought("context, pytorch_code -> tilelang_code")

    def forward(self, pytorch_code: str) -> dspy.Prediction:
        """Generate TileLang code from PyTorch code.

        Args:
            pytorch_code: Input PyTorch code to optimize

        Returns:
            Prediction with tilelang_code field
        """
        # Retrieve relevant examples
        retrieval_results = self.retriever(pytorch_code)

        # Build context from retrieved examples
        context = self._build_context(retrieval_results)

        # Generate TileLang code
        return self.generator(context=context, pytorch_code=pytorch_code)

    def _build_context(self, retrieval_results) -> str:
        """Build context string from retrieval results."""
        context_parts = []
        seen_examples = set()

        for i, passage in enumerate(retrieval_results.passages):
            # Get metadata if available
            if hasattr(self.retriever, "corpus_metadata") and i < len(retrieval_results.passages):
                # Find the corresponding metadata entry
                corpus_index = self._find_corpus_index(passage)
                if corpus_index is not None and corpus_index < len(self.retriever.corpus_metadata):
                    metadata = self.retriever.corpus_metadata[corpus_index]
                    example_id = metadata.get("example_id", "")

                    # Skip duplicate examples
                    if example_id in seen_examples:
                        continue
                    seen_examples.add(example_id)

                    # Try to load full example
                    full_example = self._load_full_example(metadata)
                    if full_example:
                        context_parts.append(full_example)
                        continue

            # Fallback to passage text
            context_parts.append(f"Retrieved Example {i+1}:\n{passage}")

        return "\n\n---\n\n".join(context_parts)

    def _find_corpus_index(self, passage: str) -> Optional[int]:
        """Find the index of a passage in the corpus."""
        # This is a simplified implementation - in practice, the retriever
        # should provide this mapping
        for i, corpus_text in enumerate(self.retriever.corpus):
            if passage in corpus_text or corpus_text in passage:
                return i
        return None

    def _load_full_example(self, metadata: Dict) -> Optional[str]:
        """Load full example code from metadata."""
        if "dir_path" not in metadata:
            return None

        example_dir = Path(metadata["dir_path"])
        example_id = metadata.get("example_id", "Unknown")

        try:
            original_code = (example_dir / "original.py").read_text(encoding="utf-8")
            tilelang_code = (example_dir / "tilelang.py").read_text(encoding="utf-8")

            return f"""Example: {example_id}

Original PyTorch Implementation:
```python
{original_code}
```

TileLang Optimized Implementation:
```python
{tilelang_code}
```"""
        except Exception:
            return None


# ============= Evaluation Metrics =============


def tilelang_correctness_metric(example: Example, pred: dspy.Prediction, trace=None) -> float:
    """Evaluate the correctness of generated TileLang code.

    Returns a score between 0 and 1.
    """
    if not hasattr(pred, "tilelang_code") or not pred.tilelang_code:
        return 0.0

    code = pred.tilelang_code
    score = 0.0

    # Check for essential TileLang components
    checks = [
        ("import tilelang", 0.2),  # TileLang import
        ("class ModelNew", 0.2),  # Model class definition
        ("def forward", 0.2),  # Forward method
        ("@T.prim_func", 0.2),  # TileLang decorator
        ("T.grid", 0.1),  # Grid computation
        ("T.block", 0.1),  # Block structure
    ]

    for pattern, weight in checks:
        if pattern in code:
            score += weight

    # Additional check for @T.kernel as alternative to @T.prim_func
    if "@T.kernel" in code and "@T.prim_func" not in code:
        score += 0.2

    return min(score, 1.0)


# ============= Pipeline Setup =============


class TileLangPipeline:
    """Main pipeline for TileLang code generation."""

    def __init__(
        self,
        examples_dir: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        retriever_k: int = 3,
    ):
        """Initialize the TileLang pipeline.

        Args:
            examples_dir: Directory containing training examples
            model: Language model to use (defaults to gpt-4o-mini)
            api_key: API key for the model
            retriever_k: Number of examples to retrieve
        """
        # Configure DSPy
        if model:
            configure_dspy(model=model, api_key=api_key)
        else:
            configure_dspy(api_key=api_key)

        # Load examples
        self.examples, corpus_texts, corpus_metadata = load_tilelang_examples(examples_dir)

        # Set up retriever
        self.retriever = setup_retriever(corpus_texts, corpus_metadata, k=retriever_k)

        # Create RAG module
        self.rag = TileLangRAG(self.retriever)
        self.optimized_rag = None

    def generate(self, pytorch_code: str) -> str:
        """Generate TileLang code from PyTorch code."""
        model = self.optimized_rag if self.optimized_rag else self.rag
        result = model(pytorch_code)
        return result.tilelang_code

    def optimize(
        self,
        metric=None,
        train_ratio: float = 0.2,
        max_bootstrapped_demos: int = 2,
        max_labeled_demos: int = 2,
        seed: int = 42,
    ) -> "TileLangRAG":
        """Optimize the pipeline using DSPy optimizers.

        Args:
            metric: Evaluation metric (defaults to tilelang_correctness_metric)
            train_ratio: Ratio of examples to use for training
            max_bootstrapped_demos: Maximum bootstrapped demonstrations
            max_labeled_demos: Maximum labeled demonstrations
            seed: Random seed for reproducibility

        Returns:
            Optimized RAG module
        """
        if metric is None:
            metric = tilelang_correctness_metric

        # Split examples
        random.Random(seed).shuffle(self.examples)
        train_size = max(5, int(len(self.examples) * train_ratio))
        trainset = self.examples[:train_size]
        valset = self.examples[train_size:]

        print(f"Optimizing with {len(trainset)} training and {len(valset)} validation examples")

        # Use MIPROv2 optimizer
        optimizer = dspy.MIPROv2(metric=metric, auto="medium", num_threads=4)  # Reduced for stability

        self.optimized_rag = optimizer.compile(
            self.rag,
            trainset=trainset,
            valset=valset,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            requires_permission_to_run=False,
        )

        return self.optimized_rag

    def save(self, path: str):
        """Save the optimized model."""
        if self.optimized_rag:
            self.optimized_rag.save(path)
            print(f"Saved optimized model to {path}")
        else:
            print("No optimized model to save. Run optimize() first.")

    def load(self, path: str):
        """Load an optimized model."""
        self.optimized_rag = TileLangRAG(self.retriever)
        self.optimized_rag.load(path)
        print(f"Loaded optimized model from {path}")


# ============= Main Usage Example =============


def main():
    """Example usage of the TileLang pipeline."""
    # Initialize pipeline
    pipeline = TileLangPipeline(examples_dir="./examples")

    # Load example PyTorch code
    pytorch_file = "../KernelBench/KernelBench/level1/23_Softmax.py"
    if Path(pytorch_file).exists():
        pytorch_code = Path(pytorch_file).read_text(encoding="utf-8")
    else:
        # Fallback example
        pytorch_code = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.relu(x)
"""

    # Generate TileLang code
    print("Generating TileLang optimization...")
    tilelang_code = pipeline.generate(pytorch_code)
    print("\nGenerated TileLang Code:")
    print(tilelang_code)

    # Optimize pipeline if enough examples
    if len(pipeline.examples) >= 10:
        print("\nOptimizing pipeline...")
        pipeline.optimize()
        pipeline.save("optimized_tilelang_rag.json")

        # Test optimized version
        optimized_code = pipeline.generate(pytorch_code)
        print("\nOptimized TileLang Code:")
        print(optimized_code)

    # Inspect prompts
    print("\nInspecting last prompt:")
    dspy.inspect_history(n=1)


if __name__ == "__main__":
    # Set up model at module level
    configure_dspy()
    main()
