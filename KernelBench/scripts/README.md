# TileLang Reasoning Data Generation

This script generates reasoning chain datasets for teaching models and engineers how to approach TileLang kernel optimization problems. It creates preference datasets by generating step-by-step reasoning chains that trace the path from a problem statement to an optimal kernel solution.

## Quick Start

```bash
python create_reasoning_data.py level=1 problem=94 prompt=diverse_strategy
```

## Command Format

```bash
python create_reasoning_data.py level=<LEVEL> problem=<PROBLEM_ID> prompt=<STRATEGY> [OPTIONS]
```

## Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `level` | KernelBench problem level | `level=1` |
| `problem` | Problem ID within the level | `problem=94` |
| `prompt` | Prompting strategy to use | `prompt=diverse_strategy` |

## Available Prompt Strategies

### 1. `diverse_strategy`
**Purpose**: Explores diverse optimization strategies by systematically evaluating multiple approaches at each decision point.

**Models Used**: `gpt-4o-mini`, `o3`, `gpt-4o`

**Example**:
```bash
python create_reasoning_data.py level=1 problem=94 prompt=diverse_strategy
```

**What it does**: Generates reasoning chains that compare different tiling shapes, memory layouts, thread mappings, and optimization strategies before making design decisions.

### 2. `good_reasoning`
**Purpose**: Generates high-quality reasoning chains using various structured approaches.

**Models Used**: `o1`, `o3`, `qwen_2.5`, `qwen_2.5_instruct`

**Templates Available**:
- General-to-specific reasoning
- Top-to-bottom analysis
- Why-focused explanations
- Syntax-focused reasoning
- Alternative consideration approaches
- Concise reasoning
- Concept-only reasoning
- Normal structured reasoning

**Example**:
```bash
python create_reasoning_data.py level=1 problem=94 prompt=good_reasoning
```

### 3. `bad_reasoning`
**Purpose**: Generates reasoning chains with intentional flaws for contrast learning.

**Models Used**: `gpt-4o-mini`, `gpt-4.1-nano`

**Templates Available**:
- Chains with syntax errors
- Chains with subtle reasoning mistakes
- Rambling, unfocused chains
- Chains missing key information
- Chains without examples

**Example**:
```bash
python create_reasoning_data.py level=1 problem=94 prompt=bad_reasoning
```

## Optional Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `num_chains` | Number of reasoning chains to generate | `8` | `num_chains=4` |
| `prompts_dir` | Directory containing kernel files | `../src/prompts/correct_tilelang` | `prompts_dir=./kernels` |
| `reasoning_dir` | Output directory for results | `../reasoning_data` | `reasoning_dir=./output` |

## Usage Examples

### Basic Usage
```bash
# Generate 8 diverse strategy chains for problem 94
python create_reasoning_data.py level=1 problem=94 prompt=diverse_strategy

# Generate 4 good reasoning chains for problem 93
python create_reasoning_data.py level=1 problem=93 prompt=good_reasoning num_chains=4

# Generate bad reasoning examples for contrast learning
python create_reasoning_data.py level=1 problem=94 prompt=bad_reasoning num_chains=6
```

### Custom Directories
```bash
# Use custom input and output directories
python create_reasoning_data.py level=2 problem=15 prompt=diverse_strategy \
    prompts_dir=./my_kernels reasoning_dir=./my_output
```

### Different Problem Levels
```bash
# Level 1 problems (basic)
python create_reasoning_data.py level=1 problem=94 prompt=diverse_strategy

# Level 2 problems (intermediate)
python create_reasoning_data.py level=2 problem=23 prompt=good_reasoning

# Level 3 problems (advanced)
python create_reasoning_data.py level=3 problem=7 prompt=diverse_strategy
```

## Output

The script generates a JSON file with the following format:

```json
{
    "level": 1,
    "problem_id": 94,
    "prompt_strategy": "diverse_strategy",
    "source_file": "../src/prompts/correct_tilelang/1_94.py",
    "num_chains_requested": 8,
    "num_chains_successful": 7,
    "num_chains_failed": 1,
    "reasoning_chains": [
        {
            "chain_id": 1,
            "model": "gpt-4o",
            "reasoning_chain": "Let me think through this MSE loss optimization...",
            "timestamp": "2025-01-20T10:30:45.123456Z",
            "generation_successful": true,
            ...
        }
    ],
    "generation_timestamp": "2025-01-20T10:30:45.123456Z"
}
```

### Output Location
Results are saved to: `../reasoning_data/reasoning_level_{LEVEL}_problem_{PROBLEM_ID}_{STRATEGY}.json`

## Prerequisites

### Environment Variables
Set the following API keys:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export TOGETHER_API_KEY="your_together_api_key_here"  # For Qwen models
```

### Dependencies
- Python packages: `torch`, `openai`, `together` (if using Qwen models)
- Access to TileLang infrastructure and kernel files

### File Structure
Ensure your project has the following structure:
```
KernelBench/
├── scripts/
│   ├── create_reasoning_data.py
│   └── reasoning_prompts/
├── src/
│   ├── prompts/correct_tilelang/
│   ├── utils.py
│   ├── tilelang_paperinfo_prompt.py
│   └── tilelang_icl_prompt.py
└── reasoning_data/  # Will be created automatically
```

## Model Support

### OpenAI Models
- `gpt-4o`: General-purpose, high-quality generations
- `gpt-4o-mini`: Faster, cost-effective option
- `o1`: Advanced reasoning model with chain-of-thought
- `o3`: Latest reasoning model (requires special access)

### Together AI Models (Qwen)
- `qwen_2.5`: General Qwen model
- `qwen_2.5_instruct`: Instruction-tuned variant

## Troubleshooting

### Common Issues

1. **Missing API Keys**:
   ```
   Error: OPENAI_API_KEY not set
   ```
   Solution: Set your API keys in environment variables

2. **Kernel File Not Found**:
   ```
   Error: No files found for level 1, problem 94
   ```
   Solution: Check that kernel files exist in the prompts directory

3. **Model Access Issues**:
   ```
   Error: Only support o1 and o3 for now
   ```
   Solution: Use supported models or update the model whitelist

4. **Template Formatting Errors**:
   ```
   Error: 'tilelang_info'
   ```
   Solution: This should be fixed in the latest version - check template compatibility

### Debug Mode
Add debug prints by looking for the "FINAL PROMPT BEING SENT" output to see exactly what's being sent to the models.

## Advanced Usage

### Batch Processing
Generate reasoning data for multiple problems:
```bash
for problem in 93 94 95; do
    python create_reasoning_data.py level=1 problem=$problem prompt=diverse_strategy
done
```

### Strategy Comparison
Compare different strategies for the same problem:
```bash
python create_reasoning_data.py level=1 problem=94 prompt=diverse_strategy num_chains=4
python create_reasoning_data.py level=1 problem=94 prompt=good_reasoning num_chains=4
python create_reasoning_data.py level=1 problem=94 prompt=bad_reasoning num_chains=4
```

## Contributing

When adding new prompt strategies:
1. Add templates to `reasoning_prompts/`
2. Update `PROMPT_STRATEGIES` configuration
3. Test with different problem types
4. Update this README with new strategy documentation 