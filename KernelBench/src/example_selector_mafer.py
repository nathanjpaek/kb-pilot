"""
Smart RAG Example Selector + Quality Analysis + Cleanup

ALL-IN-ONE toolkit for RAG improvement:
1. SmartRAGSelector - Better example selection with multi-factor scoring
2. analyze_rag_quality() - Analyze example database quality
3. clean_rag_examples() - Remove low-quality examples

Goal: Retrieve examples that are more likely to lead to correct, fast kernels.

Usage:
    # Use smart selector in your RAG pipeline
    from src.example_selector_mafer import select_smart_examples
    
    # Analyze quality
    from src.example_selector_mafer import analyze_rag_quality
    analyze_rag_quality(language="cute", level=2)
    
    # Clean up bad examples
    from src.example_selector_mafer import clean_rag_examples
    clean_rag_examples(language="cute", level=2, dry_run=True)
"""

import os
import re
import shutil
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np


class SmartRAGSelector:
    """
    Enhanced RAG selector that considers multiple signals for better example retrieval.
    """
    
    def __init__(self, correct_dsl_dir: str, kernelbench_dir: str):
        self.correct_dsl_dir = correct_dsl_dir
        self.kernelbench_dir = kernelbench_dir
        self.example_cache = {}
        self.metadata_cache = {}
    
    def select_examples(self,
                       problem_code: str,
                       k: int = 5,
                       current_level: int = None,
                       current_problem_id: int = None) -> List[Dict]:
        """
        Select k best examples using multi-factor scoring.
        
        Returns:
            List of {
                "code": str,
                "problem_name": str,
                "score": float,
                "operations": list,
                "speedup": float,
            }
        """
        
        # Extract features from target problem
        target_features = self._extract_features(problem_code)
        
        # Get all candidate examples
        candidates = self._load_all_examples(current_level, current_problem_id)
        
        # Score each candidate
        scored = []
        for candidate in candidates:
            score = self._compute_score(target_features, candidate)
            candidate["score"] = score
            scored.append(candidate)
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k
        return scored[:k]
    
    def _extract_features(self, code: str) -> Dict:
        """Extract features from problem code for similarity matching"""
        
        features = {
            "operations": self._extract_operations(code),
            "complexity": self._estimate_complexity(code),
            "has_activation": self._has_activation(code),
            "has_gemm": "linear" in code.lower() or "matmul" in code.lower(),
            "has_residual": "+" in code and "x" in code,
            "num_operations": len(self._extract_operations(code)),
        }
        
        return features
    
    def _extract_operations(self, code: str) -> List[str]:
        """Extract PyTorch operations from code"""
        
        ops = []
        
        # Common PyTorch operations
        patterns = {
            "relu": r"relu|ReLU",
            "gelu": r"gelu|GELU",
            "sigmoid": r"sigmoid",
            "tanh": r"tanh",
            "softmax": r"softmax",
            "linear": r"nn\.Linear|F\.linear",
            "matmul": r"matmul|mm\(",
            "add": r"\s\+\s",
            "mul": r"\s\*\s",
            "sub": r"\s-\s",
            "div": r"\s/\s",
            "clamp": r"clamp|clip",
            "layernorm": r"LayerNorm|layer_norm",
            "batchnorm": r"BatchNorm|batch_norm",
        }
        
        for op, pattern in patterns.items():
            if re.search(pattern, code):
                ops.append(op)
        
        return ops
    
    def _estimate_complexity(self, code: str) -> str:
        """Estimate problem complexity: simple, medium, complex"""
        
        num_ops = len(self._extract_operations(code))
        
        if num_ops <= 2:
            return "simple"
        elif num_ops <= 4:
            return "medium"
        else:
            return "complex"
    
    def _has_activation(self, code: str) -> bool:
        """Check if problem has activation functions"""
        activations = ["relu", "gelu", "sigmoid", "tanh", "swish", "silu"]
        return any(act in code.lower() for act in activations)
    
    def _load_all_examples(self, exclude_level: int = None, exclude_id: int = None) -> List[Dict]:
        """Load all available examples with metadata"""
        
        examples = []
        
        # Scan all levels in correct_dsl_dir
        for level_dir in os.listdir(self.correct_dsl_dir):
            if not level_dir.startswith("level"):
                continue
            
            level_num = int(level_dir.replace("level", ""))
            level_path = os.path.join(self.correct_dsl_dir, level_dir)
            
            if not os.path.isdir(level_path):
                continue
            
            # Load all .py files in this level
            for filename in os.listdir(level_path):
                if not filename.endswith(".py"):
                    continue
                
                # Skip if this is the current problem
                if exclude_level == level_num:
                    problem_id = int(filename.split("_")[0])
                    if problem_id == exclude_id:
                        continue
                
                # Load file
                filepath = os.path.join(level_path, filename)
                with open(filepath, "r") as f:
                    content = f.read()
                
                # Extract metadata from docstring if available
                metadata = self._extract_metadata(content)
                
                # Get reference problem code
                problem_name = filename.replace(".py", "")
                ref_code = self._get_reference_code(level_num, problem_name)
                
                if ref_code:
                    features = self._extract_features(ref_code)
                    
                    examples.append({
                        "code": content,
                        "solution_code": content,  # raw solution; caller can clean
                        "reference_code": ref_code,
                        "problem_name": problem_name,
                        "level": level_num,
                        "operations": features["operations"],
                        "complexity": features["complexity"],
                        "speedup": metadata.get("speedup", 0.0),
                        "compiled": metadata.get("compiled", True),
                        "correct": metadata.get("correct", True),
                    })
        
        return examples
    
    def _extract_metadata(self, code: str) -> Dict:
        """Extract metadata from kernel docstring"""
        
        metadata = {}
        
        # Look for speedup_ratio in docstring
        match = re.search(r"speedup_ratio['\"]?\s*:\s*([0-9.]+)", code)
        if match:
            metadata["speedup"] = float(match.group(1))
        
        # Look for compiled/correctness
        if "compiled=True" in code:
            metadata["compiled"] = True
        if "correctness=True" in code:
            metadata["correct"] = True
        
        return metadata
    
    def _get_reference_code(self, level: int, problem_name: str) -> str:
        """Get reference PyTorch code for this problem"""
        
        # Build path to reference
        problem_id = int(problem_name.split("_")[0])
        ref_path = os.path.join(self.kernelbench_dir, f"level{level}", f"{problem_id}_*.py")
        
        # Find matching file
        import glob
        matches = glob.glob(ref_path)
        
        if matches:
            with open(matches[0], "r") as f:
                return f.read()
        
        return ""
    
    def _compute_score(self, target_features: Dict, candidate: Dict) -> float:
        """
        Multi-factor scoring for example selection.
        
        Factors:
        1. Operation overlap (40%)
        2. Complexity match (20%)
        3. Performance tier (20%)
        4. Correctness (20%)
        """
        
        score = 0.0
        
        # Factor 1: Operation overlap (40 points)
        target_ops = set(target_features["operations"])
        candidate_ops = set(candidate["operations"])
        
        if target_ops and candidate_ops:
            overlap = len(target_ops & candidate_ops) / len(target_ops | candidate_ops)
            score += overlap * 40
        
        # Factor 2: Complexity match (20 points)
        if target_features["complexity"] == candidate["complexity"]:
            score += 20
        elif abs(["simple", "medium", "complex"].index(target_features["complexity"]) - 
                 ["simple", "medium", "complex"].index(candidate["complexity"])) == 1:
            score += 10  # Adjacent complexity levels
        
        # Factor 3: Performance tier (20 points)
        speedup = candidate.get("speedup", 0.0)
        if speedup >= 0.9:
            score += 20  # Excellent performance
        elif speedup >= 0.7:
            score += 15  # Good performance
        elif speedup >= 0.5:
            score += 10  # Acceptable performance
        elif speedup > 0:
            score += 5   # Poor but working
        
        # Factor 4: Correctness (20 points)
        if candidate.get("compiled", False):
            score += 10
        if candidate.get("correct", False):
            score += 10
        
        return score


# Convenience function
def select_smart_examples(problem_code: str,
                         language: str,
                         k: int = 5,
                         current_level: int = None,
                         current_problem_id: int = None) -> List[Dict]:
    """
    Select k best RAG examples using smart scoring.
    
    Args:
        problem_code: Reference PyTorch code
        language: DSL language (cute, tilelang, etc)
        k: Number of examples to retrieve
        current_level: Current problem level (for exclusion)
        current_problem_id: Current problem ID (for exclusion)
        
    Returns:
        List of top k examples with scores
    """
    
    REPO_TOP_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    
    correct_dsl_dir = os.path.join(REPO_TOP_PATH, f"src/prompts/correct_{language}")
    kernelbench_dir = os.path.join(REPO_TOP_PATH, "KernelBench")
    
    selector = SmartRAGSelector(correct_dsl_dir, kernelbench_dir)
    
    return selector.select_examples(
        problem_code=problem_code,
        k=k,
        current_level=current_level,
        current_problem_id=current_problem_id,
    )


# =============================================================================
# QUALITY ANALYSIS TOOLS
# =============================================================================

def analyze_rag_quality(language: str = "cute", level: int = 2) -> Dict:
    """
    Analyze RAG example quality for a given level.
    
    Args:
        language: DSL language
        level: Problem level
        
    Returns:
        Dictionary with analysis results
    """
    
    REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    level_dir = os.path.join(REPO_TOP_PATH, f"src/prompts/correct_{language}/level{level}")
    
    if not os.path.exists(level_dir):
        print(f"❌ Directory not found: {level_dir}")
        return {}
    
    examples = []
    
    for filename in os.listdir(level_dir):
        if not filename.endswith(".py"):
            continue
        
        filepath = os.path.join(level_dir, filename)
        
        with open(filepath, "r") as f:
            content = f.read()
        
        # Extract metadata
        speedup = 0.0
        match = re.search(r"speedup_ratio['\"]?\s*:\s*([0-9.]+)", content)
        if match:
            speedup = float(match.group(1))
        
        has_eval = "Evaluation Result:" in content
        compiled = "compiled=True" in content
        correct = "correctness=True" in content
        is_slow = "very_slow" in filename.lower()
        
        examples.append({
            "filename": filename,
            "speedup": speedup,
            "has_eval": has_eval,
            "compiled": compiled,
            "correct": correct,
            "is_slow": is_slow,
            "filepath": filepath,
        })
    
    # Sort by speedup
    examples.sort(key=lambda x: x["speedup"], reverse=True)
    
    # Analysis
    print(f"\n{'='*80}")
    print(f"RAG EXAMPLES ANALYSIS - Level {level} ({language.upper()})")
    print(f"{'='*80}")
    print(f"Total examples: {len(examples)}")
    
    # Count by status
    with_eval = sum(1 for e in examples if e["has_eval"])
    compiled_count = sum(1 for e in examples if e["compiled"])
    correct_count = sum(1 for e in examples if e["correct"])
    slow_count = sum(1 for e in examples if e["is_slow"])
    
    print(f"\n📊 Status:")
    print(f"  With evaluation metadata: {with_eval}/{len(examples)}")
    print(f"  Compiled: {compiled_count}/{len(examples)}")
    print(f"  Correct: {correct_count}/{len(examples)}")
    print(f"  Marked slow: {slow_count}/{len(examples)}")
    
    # Performance tiers
    excellent = [e for e in examples if e["speedup"] >= 0.9]
    good = [e for e in examples if 0.7 <= e["speedup"] < 0.9]
    acceptable = [e for e in examples if 0.5 <= e["speedup"] < 0.7]
    poor = [e for e in examples if 0 < e["speedup"] < 0.5]
    unknown = [e for e in examples if e["speedup"] == 0]
    
    print(f"\n⚡ Performance Tiers:")
    print(f"  Excellent (≥0.9×): {len(excellent)}")
    print(f"  Good (0.7-0.9×): {len(good)}")
    print(f"  Acceptable (0.5-0.7×): {len(acceptable)}")
    print(f"  Poor (<0.5×): {len(poor)}")
    print(f"  Unknown: {len(unknown)}")
    
    # Top performers
    print(f"\n🏆 Top 10 Examples (by speedup):")
    for i, example in enumerate(examples[:10], 1):
        status = "✓" if example["correct"] else "✗"
        speedup_str = f"{example['speedup']:.2f}×" if example['speedup'] > 0 else "unknown"
        print(f"  {i}. {status} {example['filename']}: {speedup_str}")
    
    # Bottom performers
    if len(examples) > 10:
        print(f"\n⚠️  Bottom 10 Examples:")
        for i, example in enumerate(examples[-10:], 1):
            status = "✓" if example["correct"] else "✗"
            speedup_str = f"{example['speedup']:.2f}×" if example['speedup'] > 0 else "unknown"
            print(f"  {i}. {status} {example['filename']}: {speedup_str}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if poor:
        print(f"\n  1. ⚠️  REMOVE {len(poor)} POOR EXAMPLES (speedup <0.5×)")
        print(f"     Command: python -c \"from src.example_selector_mafer import clean_rag_examples; clean_rag_examples('{language}', {level}, execute=True)\"")
    
    if slow_count:
        print(f"\n  2. ⚠️  REMOVE {slow_count} EXAMPLES MARKED 'very_slow'")
    
    if unknown and len(unknown) > len(examples) * 0.3:
        print(f"\n  3. ℹ️  EVALUATE {len(unknown)} EXAMPLES WITH MISSING METADATA")
    
    if len(excellent) < 5:
        print(f"\n  4. 📈 NEED MORE HIGH-QUALITY EXAMPLES (only {len(excellent)} excellent)")
        print(f"     Focus on generating kernels with speedup ≥0.9×")
    
    # Statistics
    if [e for e in examples if e["speedup"] > 0]:
        speedups = [e["speedup"] for e in examples if e["speedup"] > 0]
        print(f"\n📈 STATISTICS:")
        print(f"  Mean speedup: {np.mean(speedups):.2f}×")
        print(f"  Median speedup: {np.median(speedups):.2f}×")
        print(f"  Std dev: {np.std(speedups):.2f}")
        print(f"  Success rate: {correct_count/len(examples):.1%}")
    
    return {
        "total": len(examples),
        "excellent": len(excellent),
        "good": len(good),
        "acceptable": len(acceptable),
        "poor": len(poor),
        "unknown": len(unknown),
        "examples": examples,
    }


def clean_rag_examples(language: str = "cute",
                      level: int = 2,
                      dry_run: bool = True,
                      backup: bool = True,
                      execute: bool = False) -> Dict:
    """
    Remove low-quality examples from RAG database.
    
    Removal criteria:
    - Speedup < 0.3× (very slow)
    - Marked "very_slow" in filename
    - compiled=False or correctness=False
    - Missing ModelNew class
    
    Args:
        language: DSL language
        level: Problem level
        dry_run: Preview changes without executing (default: True)
        backup: Create backup before removing (default: True)
        execute: Actually execute removal (overrides dry_run)
        
    Returns:
        Dictionary with cleanup results
    """
    
    REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    level_dir = os.path.join(REPO_TOP_PATH, f"src/prompts/correct_{language}/level{level}")
    
    if not os.path.exists(level_dir):
        print(f"❌ Directory not found: {level_dir}")
        return {}
    
    # Override dry_run if execute=True
    if execute:
        dry_run = False
    
    # Create backup
    if backup and not dry_run:
        backup_dir = os.path.join(
            REPO_TOP_PATH,
            f"src/prompts/correct_{language}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copytree(level_dir, os.path.join(backup_dir, f"level{level}"))
        print(f"📦 Backup created: {backup_dir}")
    
    print(f"\n{'='*80}")
    print(f"CLEANING RAG EXAMPLES - Level {level} ({language.upper()})")
    print(f"Mode: {'🔍 DRY RUN (preview only)' if dry_run else '🗑️  EXECUTE (will delete)'}")
    print(f"{'='*80}")
    
    to_remove = []
    to_keep = []
    removal_reasons = defaultdict(int)
    
    for filename in sorted(os.listdir(level_dir)):
        if not filename.endswith(".py"):
            continue
        
        filepath = os.path.join(level_dir, filename)
        
        with open(filepath, "r") as f:
            content = f.read()
        
        # Check removal criteria
        should_rm, reason = _should_remove_file(filepath, content)
        
        if should_rm:
            to_remove.append({
                "filename": filename,
                "filepath": filepath,
                "reason": reason,
            })
            removal_reasons[reason] += 1
        else:
            to_keep.append(filename)
    
    # Print removals
    if to_remove:
        print(f"\n{'Would remove' if dry_run else 'Removing'} {len(to_remove)} files:")
        for item in to_remove:
            print(f"  ✗ {item['filename']}")
            print(f"     Reason: {item['reason']}")
            
            if not dry_run:
                os.remove(item['filepath'])
    else:
        print(f"\n✓ No files need removal - all examples are good quality!")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Kept: {len(to_keep)}")
    print(f"Removed: {len(to_remove)}")
    
    if removal_reasons:
        print(f"\nRemoval reasons:")
        for reason, count in removal_reasons.items():
            print(f"  • {reason}: {count}")
    
    if dry_run:
        print(f"\n⚠️  This was a DRY RUN - no files were actually removed")
        print(f"💡 Run with execute=True to actually remove files")
        print(f"   Example: clean_rag_examples('{language}', {level}, execute=True)")
    else:
        print(f"\n✓ Cleanup complete!")
        print(f"   Removed: {len(to_remove)} files")
        print(f"   Remaining: {len(to_keep)} high-quality examples")
    
    return {
        "kept": len(to_keep),
        "removed": len(to_remove),
        "removal_reasons": dict(removal_reasons),
        "removed_files": [r["filename"] for r in to_remove],
    }


def _should_remove_file(filepath: str, content: str) -> Tuple[bool, str]:
    """
    Determine if example should be removed.
    
    Returns:
        (should_remove: bool, reason: str)
    """
    
    filename = os.path.basename(filepath)
    
    # Rule 1: Files explicitly marked as very_slow
    if "very_slow" in filename.lower():
        return True, "Explicitly marked as very_slow"
    
    # Rule 2: Extract speedup from docstring
    match = re.search(r"speedup_ratio['\"]?\s*:\s*([0-9.]+)", content)
    if match:
        speedup = float(match.group(1))
        if speedup < 0.3:
            return True, f"Very slow (speedup={speedup:.2f}×)"
    
    # Rule 3: Not compiled
    if "compiled=False" in content:
        return True, "Does not compile"
    
    # Rule 4: Not correct
    if "correctness=False" in content:
        return True, "Fails correctness tests"
    
    # Rule 5: Missing ModelNew class
    if "class ModelNew" not in content and "def ModelNew" not in content:
        return True, "Missing ModelNew implementation"
    
    return False, ""


# =============================================================================
# CONVENIENCE FUNCTIONS FOR SCRIPTS
# =============================================================================

def analyze_all_levels(language: str = "cute"):
    """Analyze all levels for a language"""
    
    print(f"\n{'='*80}")
    print(f"MULTI-LEVEL RAG ANALYSIS - {language.upper()}")
    print(f"{'='*80}")
    
    all_results = {}
    
    for level in [1, 2, 3]:
        result = analyze_rag_quality(language, level)
        all_results[f"level{level}"] = result
        
        if result:
            print(f"\n📊 Level {level} Summary:")
            print(f"   Total: {result['total']}")
            print(f"   Excellent: {result['excellent']}")
            print(f"   Poor: {result['poor']}")
    
    return all_results


def get_example_stats(language: str = "cute", level: int = 2) -> Dict:
    """Get quick stats about examples without printing"""
    
    REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    level_dir = os.path.join(REPO_TOP_PATH, f"src/prompts/correct_{language}/level{level}")
    
    if not os.path.exists(level_dir):
        return {"error": "Directory not found"}
    
    total = 0
    correct = 0
    speedups = []
    
    for filename in os.listdir(level_dir):
        if not filename.endswith(".py"):
            continue
        
        total += 1
        filepath = os.path.join(level_dir, filename)
        
        with open(filepath, "r") as f:
            content = f.read()
        
        if "correctness=True" in content:
            correct += 1
        
        match = re.search(r"speedup_ratio['\"]?\s*:\s*([0-9.]+)", content)
        if match:
            speedups.append(float(match.group(1)))
    
    return {
        "total": total,
        "correct": correct,
        "mean_speedup": np.mean(speedups) if speedups else 0.0,
        "median_speedup": np.median(speedups) if speedups else 0.0,
    }


# =============================================================================
# MAIN CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Example Quality Tools")
    parser.add_argument("--language", default="cute", help="DSL language")
    parser.add_argument("--level", type=int, help="Specific level")
    parser.add_argument("--all-levels", action="store_true", help="Analyze all levels")
    
    # Commands
    parser.add_argument("--analyze", action="store_true", help="Analyze quality")
    parser.add_argument("--clean", action="store_true", help="Clean low-quality examples")
    parser.add_argument("--execute", action="store_true", help="Actually remove files (use with --clean)")
    
    args = parser.parse_args()
    
    # Default: analyze
    if not args.clean:
        args.analyze = True
    
    if args.analyze:
        if args.all_levels:
            analyze_all_levels(args.language)
        elif args.level:
            analyze_rag_quality(args.language, args.level)
        else:
            print("Specify --level N or --all-levels")
    
    if args.clean:
        if args.level:
            dry_run = not args.execute
            clean_rag_examples(args.language, args.level, dry_run=dry_run)
        else:
            print("Specify --level N for cleaning")
