"""
Hierarchical Summarization of CuTe DSL Documentation

This script uses multi-stage LLM summarization to compress extensive CuTe DSL 
documentation into a concise guideline prompt suitable for kernel generation.

Strategy:
1. Level 1: Summarize individual markdown files into JSON summaries
2. Level 2: Aggregate summaries by category/directory into section summaries  
3. Level 3: Synthesize all sections into final comprehensive guideline prompt
4. Level 4: Include curated kernel examples as patterns

This overcomes long context limitations by progressive compression.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
CUTE_REF_DIR = REPO_ROOT / "src" / "cute_reference"
BACKGROUND_DIR = CUTE_REF_DIR / "background_basics"
KERNEL_EXAMPLES_DIR = CUTE_REF_DIR / "kernel_examples"
CACHE_DIR = SCRIPT_DIR / ".cute_summary_cache"
CACHE_DIR.mkdir(exist_ok=True)

# OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@dataclass
class DocSummary:
    """Summary of a single documentation file"""
    file_path: str
    category: str  # e.g., "01-cute_fundamentals"
    title: str
    key_concepts: List[str]
    critical_patterns: List[str]
    code_snippets: List[str]
    constraints: List[str]
    compressed_summary: str  # 100-300 words


@dataclass
class CategorySummary:
    """Summary of a documentation category"""
    category_name: str
    category_path: str
    file_summaries: List[DocSummary]
    unified_concepts: List[str]
    essential_patterns: List[str]
    common_pitfalls: List[str]
    compressed_overview: str  # 200-400 words


@dataclass
class KernelPattern:
    """Extracted pattern from kernel examples"""
    name: str
    operations: List[str]
    architecture: str  # ampere, hopper, blackwell
    key_techniques: List[str]
    code_pattern: str  # simplified/abstracted code


def call_llm(
    messages: List[Dict[str, str]], 
    model: str = "gpt-4o",
    temperature: float = 0.3,
    response_format: Dict[str, str] = None
) -> str:
    """Call OpenAI API with caching"""
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format
            
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return None


def summarize_single_doc(file_path: Path, category: str) -> DocSummary:
    """Summarize a single markdown documentation file using LLM"""
    print(f"  📄 Summarizing {file_path.name}...")
    
    # Check cache
    cache_key = f"{category}_{file_path.stem}.json"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        print(f"    ✅ Using cached summary")
        with open(cache_path, 'r') as f:
            data = json.load(f)
            return DocSummary(**data)
    
    # Read file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Prompt for summarization
    system_prompt = """You are an expert at distilling technical documentation into structured summaries.
Extract the most critical information for someone writing GPU kernels with CuTe DSL.
Focus on practical patterns, constraints, and gotchas rather than verbose explanations."""

    user_prompt = f"""Summarize this CuTe DSL documentation file into a structured JSON format.

Documentation content:
{content}

Return ONLY a valid JSON object with this exact structure:
{{
  "title": "Brief descriptive title",
  "key_concepts": ["concept1", "concept2", ...],
  "critical_patterns": ["pattern1 with brief context", "pattern2 with brief context", ...],
  "code_snippets": ["minimal code example 1", "minimal code example 2", ...],
  "constraints": ["constraint1", "constraint2", ...],
  "compressed_summary": "A 1000-1500 word summary of the essential information from this document"
}}

Focus on actionable information. Be concise but precise."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Call LLM with JSON mode
    response = call_llm(
        messages, 
        model="gpt-4o",
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    if not response:
        return None
    
    try:
        summary_data = json.loads(response)
        doc_summary = DocSummary(
            file_path=str(file_path),
            category=category,
            title=summary_data.get("title", file_path.stem),
            key_concepts=summary_data.get("key_concepts", []),
            critical_patterns=summary_data.get("critical_patterns", []),
            code_snippets=summary_data.get("code_snippets", []),
            constraints=summary_data.get("constraints", []),
            compressed_summary=summary_data.get("compressed_summary", "")
        )
        
        # Cache result
        with open(cache_path, 'w') as f:
            json.dump(asdict(doc_summary), f, indent=2)
        
        return doc_summary
    except json.JSONDecodeError as e:
        print(f"    ⚠️  JSON parsing failed: {e}")
        return None


def aggregate_category_summaries(category_name: str, doc_summaries: List[DocSummary]) -> CategorySummary:
    """Aggregate multiple document summaries into a category summary"""
    print(f"\n  📚 Aggregating category: {category_name}")
    
    # Check cache
    cache_key = f"category_{category_name}.json"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        print(f"    ✅ Using cached category summary")
        with open(cache_path, 'r') as f:
            data = json.load(f)
            # Reconstruct DocSummary objects
            doc_summaries_data = data.pop('file_summaries', [])
            data['file_summaries'] = [DocSummary(**ds) for ds in doc_summaries_data]
            return CategorySummary(**data)
    
    # Combine all doc summaries
    combined_text = f"Category: {category_name}\n\n"
    for i, doc_sum in enumerate(doc_summaries, 1):
        combined_text += f"## Document {i}: {doc_sum.title}\n"
        combined_text += f"Concepts: {', '.join(doc_sum.key_concepts)}\n"
        combined_text += f"Summary: {doc_sum.compressed_summary}\n\n"
    
    system_prompt = """You are synthesizing multiple related documentation summaries into a unified category overview.
Identify common themes, essential patterns, and critical constraints across all documents.
Eliminate redundancy while preserving unique insights."""

    user_prompt = f"""Synthesize these related CuTe DSL documentation summaries into a unified category overview.

{combined_text}

Return ONLY a valid JSON object with this structure:
{{
  "unified_concepts": ["core concept 1", "core concept 2", ...],
  "essential_patterns": ["essential pattern 1 with context", "essential pattern 2 with context", ...],
  "common_pitfalls": ["pitfall 1", "pitfall 2", ...],
  "compressed_overview": "A 200-400 word synthesis of this category's key information"
}}

Eliminate redundancy. Focus on what a kernel developer absolutely needs to know."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = call_llm(
        messages,
        model="gpt-4o", 
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    if not response:
        return None
    
    try:
        cat_data = json.loads(response)
        category_summary = CategorySummary(
            category_name=category_name,
            category_path="",  # Will be set by caller
            file_summaries=doc_summaries,
            unified_concepts=cat_data.get("unified_concepts", []),
            essential_patterns=cat_data.get("essential_patterns", []),
            common_pitfalls=cat_data.get("common_pitfalls", []),
            compressed_overview=cat_data.get("compressed_overview", "")
        )
        
        # Cache result
        cache_data = asdict(category_summary)
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        return category_summary
    except json.JSONDecodeError as e:
        print(f"    ⚠️  JSON parsing failed: {e}")
        return None


def extract_kernel_patterns(kernel_examples_dir: Path) -> List[KernelPattern]:
    """Extract patterns from kernel examples"""
    print(f"\n🔍 Extracting kernel patterns from examples...")
    
    cache_path = CACHE_DIR / "kernel_patterns.json"
    if cache_path.exists():
        print(f"  ✅ Using cached kernel patterns")
        with open(cache_path, 'r') as f:
            data = json.load(f)
            return [KernelPattern(**kp) for kp in data]
    
    patterns = []
    
    # Traverse kernel examples and extract key patterns
    # For now, we'll do a simpler analysis - just identify file structure
    # You could extend this to actually parse and summarize code
    
    arch_dirs = {
        "ampere": kernel_examples_dir / "official_cutlass" / "ampere",
        "hopper": kernel_examples_dir / "official_cutlass" / "hopper",
        "blackwell": kernel_examples_dir / "official_cutlass" / "blackwell",
    }
    
    for arch, arch_path in arch_dirs.items():
        if not arch_path.exists():
            continue
        
        py_files = list(arch_path.rglob("*.py"))
        for py_file in py_files[:5]:  # Sample first 5 from each arch
            pattern_name = py_file.stem.replace("_", " ").title()
            
            # Simple pattern extraction - could be enhanced with LLM
            patterns.append(KernelPattern(
                name=pattern_name,
                operations=[pattern_name.split()[0]],  # Crude extraction
                architecture=arch,
                key_techniques=["tiled_copy", "mma_operations"] if "gemm" in py_file.name else ["elementwise"],
                code_pattern=f"# See {py_file.name} for reference"
            ))
    
    # Cache patterns
    with open(cache_path, 'w') as f:
        json.dump([asdict(p) for p in patterns], f, indent=2)
    
    return patterns


def synthesize_final_guideline(
    category_summaries: List[CategorySummary],
    kernel_patterns: List[KernelPattern]
) -> str:
    """Synthesize all summaries into final guideline prompt"""
    print(f"\n🎯 Synthesizing final guideline prompt...")
    
    cache_path = CACHE_DIR / "final_guideline.txt"
    if cache_path.exists():
        print(f"  ✅ Using cached final guideline")
        with open(cache_path, 'r') as f:
            return f.read()
    
    # Build comprehensive context
    context = "# CuTe DSL Documentation Summaries\n\n"
    
    for cat_summary in category_summaries:
        context += f"## {cat_summary.category_name}\n"
        context += f"{cat_summary.compressed_overview}\n\n"
        context += f"**Key Concepts:** {', '.join(cat_summary.unified_concepts)}\n\n"
        context += f"**Essential Patterns:**\n"
        for pattern in cat_summary.essential_patterns:
            context += f"- {pattern}\n"
        context += f"\n**Common Pitfalls:**\n"
        for pitfall in cat_summary.common_pitfalls:
            context += f"- {pitfall}\n"
        context += "\n---\n\n"
    
    system_prompt = """You are an expert at creating concise, actionable technical guidelines.
Your task is to synthesize comprehensive documentation into a single, highly compressed guideline 
that an LLM can use to generate correct CuTe DSL kernels.

The guideline should:
1. Be extremely information-dense (every sentence matters)
2. Focus on syntax, patterns, and constraints
3. Include critical gotchas and common mistakes
4. Be organized for quick reference
5. Use technical terminology precisely
6. Be under 3000 words but maximally informative"""

    user_prompt = f"""Synthesize these CuTe DSL documentation summaries into a single comprehensive guideline prompt.

{context}

Create a guideline that would help an LLM generate correct CuTe DSL kernels. Structure it logically:
1. Core concepts (layouts, tensors, types)
2. Kernel structure and decorators
3. Memory operations (copy atoms, MMA)
4. Thread-value layouts and tiling
5. Critical patterns and idioms
6. Common mistakes to avoid
7. Architecture-specific considerations

Be maximally information-dense. Every sentence should provide actionable guidance."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    final_guideline = call_llm(
        messages,
        model="o3",  # Use reasoning model for synthesis
        temperature=1.0
    )
    
    if final_guideline:
        # Cache result
        with open(cache_path, 'w') as f:
            f.write(final_guideline)
        return final_guideline
    
    return "ERROR: Failed to generate final guideline"


def main(dry_run: bool = False, force_regenerate: bool = False):
    """Main hierarchical summarization pipeline
    
    Args:
        dry_run: If True, only count files and estimate cost without calling LLMs
        force_regenerate: If True, clear cache and regenerate everything
    """
    print("🚀 Starting CuTe DSL Documentation Summarization Pipeline\n")
    print("=" * 80)
    
    if force_regenerate:
        print("\n🗑️  Force regenerate mode: Clearing cache...")
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)
        print("✅ Cache cleared\n")
    
    # Count files for dry run
    total_md_files = 0
    categories = []
    for category_dir in sorted(BACKGROUND_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        md_files = list(category_dir.glob("*.md"))
        total_md_files += len(md_files)
        if md_files:
            categories.append(category_dir.name)
    
    if dry_run:
        print(f"\n📊 Dry Run Summary:")
        print(f"  - Total markdown files: {total_md_files}")
        print(f"  - Total categories: {len(categories)}")
        print(f"\n💰 Estimated API Cost:")
        print(f"  - Doc summaries: {total_md_files} × $0.05 = ${total_md_files * 0.05:.2f}")
        print(f"  - Category summaries: {len(categories)} × $0.10 = ${len(categories) * 0.10:.2f}")
        print(f"  - Final synthesis: 1 × $0.50 = $0.50")
        print(f"  - Total estimated: ${total_md_files * 0.05 + len(categories) * 0.10 + 0.50:.2f}")
        print(f"\n✅ Dry run complete. Run without --dry-run to execute.")
        return None
    
    # Phase 1: Summarize individual documents
    print("\n📖 Phase 1: Summarizing individual documentation files...")
    
    all_doc_summaries_by_category = defaultdict(list)
    failed_docs = []
    
    # Process all markdown files in background_basics
    for category_dir in sorted(BACKGROUND_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        print(f"\n📁 Processing category: {category_name}")
        
        md_files = sorted(category_dir.glob("*.md"))
        for md_file in md_files:
            try:
                doc_summary = summarize_single_doc(md_file, category_name)
                if doc_summary:
                    all_doc_summaries_by_category[category_name].append(doc_summary)
                else:
                    failed_docs.append(str(md_file))
            except Exception as e:
                print(f"    ❌ Error processing {md_file.name}: {e}")
                failed_docs.append(str(md_file))
    
    total_summarized = sum(len(v) for v in all_doc_summaries_by_category.values())
    print(f"\n✅ Phase 1 complete: Summarized {total_summarized} documents")
    if failed_docs:
        print(f"⚠️  Failed to summarize {len(failed_docs)} documents:")
        for doc in failed_docs[:5]:  # Show first 5
            print(f"   - {doc}")
    
    # Phase 2: Aggregate by category
    print("\n📚 Phase 2: Aggregating summaries by category...")
    
    category_summaries = []
    failed_categories = []
    
    for category_name, doc_summaries in sorted(all_doc_summaries_by_category.items()):
        if doc_summaries:
            try:
                cat_summary = aggregate_category_summaries(category_name, doc_summaries)
                if cat_summary:
                    category_summaries.append(cat_summary)
                else:
                    failed_categories.append(category_name)
            except Exception as e:
                print(f"  ❌ Error aggregating {category_name}: {e}")
                failed_categories.append(category_name)
    
    print(f"\n✅ Phase 2 complete: Aggregated {len(category_summaries)} categories")
    if failed_categories:
        print(f"⚠️  Failed to aggregate {len(failed_categories)} categories: {failed_categories}")
    
    # Phase 3: Extract kernel patterns
    print("\n🔍 Phase 3: Extracting kernel patterns...")
    try:
        kernel_patterns = extract_kernel_patterns(KERNEL_EXAMPLES_DIR)
        print(f"✅ Phase 3 complete: Extracted {len(kernel_patterns)} kernel patterns")
    except Exception as e:
        print(f"⚠️  Pattern extraction failed: {e}")
        kernel_patterns = []
    
    # Phase 4: Synthesize final guideline
    print("\n🎯 Phase 4: Synthesizing final comprehensive guideline...")
    
    try:
        final_guideline = synthesize_final_guideline(category_summaries, kernel_patterns)
        
        if final_guideline and not final_guideline.startswith("ERROR"):
            print("\n" + "=" * 80)
            print("✅ Pipeline complete!\n")
            print(f"Final guideline length: {len(final_guideline)} characters")
            print(f"Final guideline word count: ~{len(final_guideline.split())} words")
            
            # Save to output
            output_path = CACHE_DIR / "CUTE_GUIDELINE_PROMPT_FINAL.txt"
            with open(output_path, 'w') as f:
                f.write(final_guideline)
            
            print(f"\n📝 Saved final guideline to: {output_path}")
            
            # Show preview
            print("\n📄 Preview (first 500 chars):")
            print("-" * 80)
            print(final_guideline[:500] + "...")
            print("-" * 80)
            
            return final_guideline
        else:
            print(f"\n❌ Final synthesis failed")
            return None
    except Exception as e:
        print(f"\n❌ Error in final synthesis: {e}")
        import traceback
        traceback.print_exc()
        return None


# Run pipeline and set the guideline prompt
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hierarchical summarization of CuTe DSL documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to estimate cost
  python -m scripts.cute_guideline_prompt --dry-run
  
  # Run full pipeline (uses cache)
  python -m scripts.cute_guideline_prompt
  
  # Force regenerate everything
  python -m scripts.cute_guideline_prompt --force
  
  # Use cheaper model for testing
  python -m scripts.cute_guideline_prompt --model gpt-4o-mini
        """
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files and estimate cost without calling LLMs"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear cache and regenerate everything"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to use for summarization (default: gpt-4o)"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 Running in dry-run mode (no API calls will be made)\n")
    
    if args.force:
        print("⚡ Force regenerate mode enabled\n")
    
    CUTE_GUIDELINE_PROMPT = main(
        dry_run=args.dry_run,
        force_regenerate=args.force
    )
    
    if CUTE_GUIDELINE_PROMPT:
        print("\n✅ Success! CUTE_GUIDELINE_PROMPT is ready to use.")
        print("\nTo use in your code:")
        print("  from scripts.cute_guideline_prompt import CUTE_GUIDELINE_PROMPT")
    else:
        print("\n❌ Pipeline failed. Check errors above.")
        sys.exit(1)
else:
    # When imported, try to load from cache
    cache_path = CACHE_DIR / "CUTE_GUIDELINE_PROMPT_FINAL.txt"
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            CUTE_GUIDELINE_PROMPT = f.read()
    else:
        # Fallback to placeholder
        CUTE_GUIDELINE_PROMPT = """TODO: Run this script to generate the comprehensive CuTe guideline prompt.
        
Run: python -m scripts.cute_guideline_prompt

Or use the shell script:
Run: ./scripts/run_cute_summarization.sh"""

# Export the final prompt
__all__ = ['CUTE_GUIDELINE_PROMPT']
