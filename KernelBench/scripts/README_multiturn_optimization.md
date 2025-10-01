# Multiturn Optimization for TileLang Kernels

This system provides sophisticated multiturn optimization for TileLang kernels using DSPy and advanced performance analysis. It takes existing kernels and iteratively improves them by analyzing bottlenecks and applying targeted optimizations.

## Quick Start

### 1. List slow kernels that need optimization
```bash
python run_multiturn_optimization.py --list-slow --threshold 0.8
```

### 2. Optimize a specific kernel
```bash
python run_multiturn_optimization.py --kernel src/prompts/correct_tilelang/level2/2_10.py
```

### 3. Optimize with custom parameters
```bash
python run_multiturn_optimization.py \
    --kernel level2/2_10.py \
    --target-speedup 1.8 \
    --max-iterations 10 \
    --model openai/o1-preview
```

## How It Works

### 1. **Performance Analysis**
The system analyzes kernels across multiple dimensions:
- **Memory Bandwidth**: Access patterns, coalescing, bank conflicts
- **Computational Efficiency**: Arithmetic intensity, redundant operations
- **Parallelism**: Thread utilization, workload balance, occupancy
- **Algorithmic**: Fusion opportunities, reduction patterns, control flow

### 2. **RAG-Enhanced Optimization**
- Retrieves relevant optimization examples from existing kernels
- Classifies optimization techniques (shared memory, pipelining, tensor cores, etc.)
- Provides context-aware suggestions based on performance category

### 3. **Iterative Refinement**
- Generates targeted optimizations based on bottleneck analysis
- Evaluates each iteration on Modal infrastructure
- Tracks optimization history and maintains best kernel
- Stops when target performance is reached or improvements plateau

### 4. **Performance Categories**
- **SLOW** (< 0.8x): Focus on memory bandwidth and algorithmic issues
- **MODERATE** (0.8-1.2x): Optimize parallelism and memory patterns  
- **GOOD** (> 1.2x): Fine-tune with advanced techniques

## Command Line Options

```bash
python run_multiturn_optimization.py [OPTIONS]

Required (choose one):
  --kernel, -k PATH        Path to kernel file to optimize
  --list-slow, -l          List kernels with low speedup ratios

Optimization Parameters:
  --target-speedup FLOAT   Target speedup ratio (default: 1.5)
  --max-iterations INT     Maximum optimization iterations (default: 5)
  --min-improvement FLOAT  Minimum improvement threshold (default: 0.05)

Model Configuration:
  --model MODEL           DSPy model: openai/o3, openai/o3-mini, 
                         openai/o1-preview, openai/gpt-4o (default: openai/o3)
  --temperature FLOAT     Model temperature (default: 0.7)
  --rag-k INT            Number of RAG examples (default: 3)

Evaluation:
  --gpu GPU              GPU type: H100, A100, L40S, A10G, L4, T4 (default: H100)

Output:
  --no-save              Don't save optimized kernels
  --quiet, -q            Reduce output verbosity
```

## Examples

### Find and optimize the slowest kernels
```bash
# List kernels slower than 0.6x
python run_multiturn_optimization.py --list-slow --threshold 0.6

# Optimize the slowest one
python run_multiturn_optimization.py --kernel level1/1_38.py --target-speedup 2.0
```

### Aggressive optimization with O3
```bash
python run_multiturn_optimization.py \
    --kernel level2/2_61.py \
    --model openai/o3 \
    --target-speedup 2.5 \
    --max-iterations 8 \
    --rag-k 5
```

### Quick optimization with O1
```bash
python run_multiturn_optimization.py \
    --kernel level1/1_94.py \
    --model openai/o1-preview \
    --target-speedup 1.2 \
    --max-iterations 3
```

## Optimization Strategies

The system applies different strategies based on current performance:

### Critical Performance Issues (< 0.5x speedup)
- Memory bandwidth optimization
- Aggressive shared memory usage
- Algorithmic changes for locality
- Mixed precision arithmetic

### Moderate Issues (0.5-0.8x speedup)  
- Thread block configuration tuning
- Memory coalescing patterns
- Register tiling techniques
- Workload rebalancing

### Fine-tuning (0.8-1.2x speedup)
- Advanced pipelining
- Architecture-specific optimizations
- Block size and thread count tuning
- Instruction-level optimizations

### Advanced Optimizations (> 1.2x speedup)
- Cutting-edge optimization techniques
- Sophisticated memory hierarchies
- Architecture-specific features
- Performance boundary pushing

## Output Files

### Optimized Kernels
```
src/prompts/correct_tilelang/level2/2_10_optimized_1.234x.py
```
Contains the best optimized kernel with performance metadata.

### Optimization History
```
results/multiturn_logs/optimization_history_10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh.json
```
Detailed JSON log of each optimization iteration including:
- Analysis results
- Optimization suggestions  
- Code changes
- Performance improvements
- Evaluation results

### DSPy Prompt History
```
results/multiturn_logs/dspy_history_*.txt
```
Complete DSPy interaction history for debugging and analysis.

## Integration with Existing Workflow

### After RAG Generation
```bash
# Generate initial kernel
python generate_and_eval_rag_modal.py --level 2 --problem_id 10

# Optimize if slow
python run_multiturn_optimization.py --kernel src/prompts/correct_tilelang/level2/2_10.py
```

### Batch Optimization
```bash
# Find all slow kernels
python run_multiturn_optimization.py --list-slow --threshold 0.8 > slow_kernels.txt

# Optimize each one (example with GNU parallel)
cat slow_kernels.txt | grep "📉" | awk '{print $3}' | \
  parallel python run_multiturn_optimization.py --kernel {}
```

## Performance Tips

### Model Selection
- **openai/o3**: Best optimization quality, highest cost
- **openai/o3-mini**: Good quality, moderate cost  
- **openai/o1-preview**: Fast iterations, lower cost
- **openai/gpt-4o**: Quick prototyping, lowest cost

### Parameter Tuning
- Start with default parameters for most kernels
- Increase `rag-k` for complex optimizations (5-8)
- Use higher `max-iterations` for very slow kernels (8-12)
- Lower `min-improvement` for fine-tuning (0.01-0.03)

### Resource Management
- Use A100/L40S for faster evaluation cycles during development
- Reserve H100 for final optimization runs
- Monitor Modal credits and OpenAI API costs

## Troubleshooting

### Common Issues

**"Could not find kernel file"**
- Use `--list-slow` to see available kernels
- Check file paths and ensure kernel exists
- Try relative paths like `level2/2_10.py`

**"Optimization failed correctness check"**
- The generated code has bugs
- Try different model or lower temperature
- Check optimization suggestions for validity

**"Improvement below threshold"**
- Kernel may be near optimal
- Try lower `min-improvement` threshold
- Use more aggressive optimization targets

### Debug Mode
```bash
# Enable verbose logging
python run_multiturn_optimization.py --kernel level2/2_10.py --verbose

# Check optimization history
cat results/multiturn_logs/optimization_history_*.json | jq '.[] | .bottleneck_analysis'
```

## Advanced Usage

### Custom Configuration
For complex scenarios, modify `multiturn_optimization.py` directly:
```python
config = OptimizationConfig()
config.kernel_file = "path/to/kernel.py"
config.custom_analysis_prompt = "Focus on memory coalescing..."
main(config)
```

### Integration with Research Workflows
The system can be integrated into larger research pipelines:
- Automated optimization of kernel datasets
- A/B testing of optimization strategies
- Performance regression detection
- Optimization technique analysis

## Contributing

To add new optimization strategies:
1. Extend `PerformanceAnalyzer` with new bottleneck categories
2. Add optimization patterns to `MultiturnOptimizationRAG`
3. Update `create_optimization_guidelines()` with new strategies
4. Test on representative kernel set

For questions or issues, please check the optimization logs and consider opening an issue with:
- Kernel file being optimized
- Configuration used
- Error messages or unexpected behavior
- Optimization history JSON (if available) 