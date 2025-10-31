---
topic: "autotuning"
difficulty: "advanced"
related_topics: ["profiling", "performance_tips", "jit_caching"]
---

# Autotuning CuTe DSL Kernels

## Overview

**Autotuning** automatically searches for optimal kernel parameters by trying different configurations and measuring performance. Instead of manually tuning, autotuning finds the best parameters for your specific hardware and problem size.

**Key concepts:**
- Parameter spaces
- Search strategies (grid, random, bayesian)
- Performance measurement
- Caching tuned parameters
- Hardware-specific tuning
- Problem-size-specific tuning

---

## What is Autotuning?

### The Problem

**Manual tuning is hard:**
```python
# Which configuration is best?
TILE_M = 64  # or 128? or 256?
TILE_N = 64  # or 128? or 256?
TILE_K = 16  # or 32? or 64?
NUM_STAGES = 2  # or 3? or 4? or 5?
BLOCK_SIZE = 256  # or 128? or 512?

# Depends on:
# - GPU architecture (Ampere vs Hopper)
# - Problem size (small vs large)
# - Data types (fp16 vs fp32)
# - Memory constraints
# - Register pressure
```

### The Solution

**Let autotuner find optimal parameters:**
```python
@cute.autotune(
    parameters={
        'TILE_M': [64, 128, 256],
        'TILE_N': [64, 128, 256],
        'TILE_K': [16, 32, 64],
        'NUM_STAGES': [2, 3, 4],
        'BLOCK_SIZE': [128, 256, 512]
    },
    metric='time'  # Minimize time
)
@cute.kernel
def autotuned_gemm(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    TILE_M: cutlass.Constexpr,
    TILE_N: cutlass.Constexpr,
    TILE_K: cutlass.Constexpr,
    NUM_STAGES: cutlass.Constexpr,
    BLOCK_SIZE: cutlass.Constexpr
):
    # Kernel implementation using parameters
    pass

# Autotuner tries all combinations and caches best
```

---

## Simple Autotuning

### Grid Search

**Try all combinations:**
```python
import torch
import itertools
from typing import Dict, List, Tuple

def grid_search_autotune(
    kernel,
    param_space: Dict[str, List],
    test_args: Tuple,
    num_iters: int = 100
) -> Dict:
    """
    Grid search autotuning
    
    Args:
        kernel: Kernel function
        param_space: Dictionary of parameter names to possible values
        test_args: Arguments to pass to kernel
        num_iters: Number of timing iterations
    
    Returns:
        Best parameters and timing
    """
    
    # Generate all combinations
    param_names = list(param_space.keys())
    param_values = list(param_space.values())
    
    best_time = float('inf')
    best_params = None
    
    # Try all combinations
    total_configs = 1
    for values in param_values:
        total_configs *= len(values)
    
    print(f"Testing {total_configs} configurations...")
    
    for i, values in enumerate(itertools.product(*param_values)):
        # Create parameter dict
        params = dict(zip(param_names, values))
        
        print(f"\n[{i+1}/{total_configs}] Testing: {params}")
        
        try:
            # Time this configuration
            time = time_configuration(kernel, params, test_args, num_iters)
            
            print(f"  Time: {time*1000:.3f} ms")
            
            # Track best
            if time < best_time:
                best_time = time
                best_params = params
                print(f"  ★ New best!")
        
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    print(f"\n=== Best Configuration ===")
    print(f"Parameters: {best_params}")
    print(f"Time: {best_time*1000:.3f} ms")
    
    return {
        'params': best_params,
        'time': best_time
    }

def time_configuration(kernel, params, args, num_iters):
    """Time a specific parameter configuration"""
    
    # Calculate grid/block from parameters
    grid, block = calculate_launch_config(params, args)
    
    # Warm up
    for _ in range(10):
        kernel.launch(grid=grid, block=block)(*args, **params)
    torch.cuda.synchronize()
    
    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        kernel.launch(grid=grid, block=block)(*args, **params)
    end.record()
    
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms / 1000 / num_iters  # Average time in seconds

# Usage
param_space = {
    'TILE_M': [64, 128, 256],
    'TILE_N': [64, 128, 256],
    'TILE_K': [16, 32, 64],
    'NUM_STAGES': [2, 3, 4]
}

best = grid_search_autotune(my_kernel, param_space, (A, B, C))
```

### Random Search

**Sample random configurations (faster than grid search):**
```python
import random

def random_search_autotune(
    kernel,
    param_space: Dict[str, List],
    test_args: Tuple,
    num_trials: int = 50,
    num_iters: int = 100
) -> Dict:
    """
    Random search autotuning
    
    More efficient than grid search for large parameter spaces.
    """
    
    best_time = float('inf')
    best_params = None
    
    print(f"Testing {num_trials} random configurations...")
    
    for trial in range(num_trials):
        # Sample random configuration
        params = {
            name: random.choice(values)
            for name, values in param_space.items()
        }
        
        print(f"\n[{trial+1}/{num_trials}] Testing: {params}")
        
        try:
            time = time_configuration(kernel, params, test_args, num_iters)
            
            print(f"  Time: {time*1000:.3f} ms")
            
            if time < best_time:
                best_time = time
                best_params = params
                print(f"  ★ New best!")
        
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    print(f"\n=== Best Configuration ===")
    print(f"Parameters: {best_params}")
    print(f"Time: {best_time*1000:.3f} ms")
    
    return {
        'params': best_params,
        'time': best_time
    }

# Usage - much faster for large spaces
param_space = {
    'TILE_M': [64, 96, 128, 192, 256],
    'TILE_N': [64, 96, 128, 192, 256],
    'TILE_K': [16, 32, 48, 64],
    'NUM_STAGES': [2, 3, 4, 5],
    'BLOCK_SIZE': [128, 256, 512]
}

# Grid search: 5*5*4*4*3 = 1200 configs
# Random search: 50 configs (much faster!)
best = random_search_autotune(my_kernel, param_space, (A, B, C), num_trials=50)
```

---

## Advanced Autotuning

### Bayesian Optimization

**Use machine learning to guide search:**
```python
try:
    from bayes_opt import BayesianOptimization
except ImportError:
    print("Install: pip install bayesian-optimization")

def bayesian_autotune(
    kernel,
    param_space: Dict[str, Tuple[int, int]],  # (min, max) for each param
    test_args: Tuple,
    num_trials: int = 30
) -> Dict:
    """
    Bayesian optimization autotuning
    
    Uses Gaussian Process to model performance and intelligently
    choose next configurations to try.
    
    Args:
        param_space: Dict of parameter names to (min, max) ranges
        num_trials: Number of configurations to try
    """
    
    def objective(**params):
        """Objective function to minimize (negative time for maximization)"""
        
        # Convert float params to int
        int_params = {k: int(v) for k, v in params.items()}
        
        try:
            time = time_configuration(kernel, int_params, test_args, num_iters=20)
            # Return negative (BayesianOptimization maximizes)
            return -time
        except Exception as e:
            # Return very bad value for invalid configs
            return -1000.0
    
    # Create optimizer
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=param_space,
        random_state=42,
        verbose=2
    )
    
    # Run optimization
    optimizer.maximize(
        init_points=10,  # Random exploration
        n_iter=num_trials  # Bayesian optimization
    )
    
    # Get best params
    best_params = {k: int(v) for k, v in optimizer.max['params'].items()}
    best_time = -optimizer.max['target']
    
    print(f"\n=== Best Configuration (Bayesian) ===")
    print(f"Parameters: {best_params}")
    print(f"Time: {best_time*1000:.3f} ms")
    
    return {
        'params': best_params,
        'time': best_time
    }

# Usage
param_space = {
    'TILE_M': (64, 256),
    'TILE_N': (64, 256),
    'TILE_K': (16, 64),
    'NUM_STAGES': (2, 5)
}

best = bayesian_autotune(my_kernel, param_space, (A, B, C), num_trials=30)
```

### Genetic Algorithm

**Evolve good configurations:**
```python
import random
from typing import List

def genetic_autotune(
    kernel,
    param_space: Dict[str, List],
    test_args: Tuple,
    population_size: int = 20,
    num_generations: int = 10
) -> Dict:
    """
    Genetic algorithm autotuning
    
    Evolves population of configurations over generations.
    """
    
    def create_individual():
        """Create random configuration"""
        return {
            name: random.choice(values)
            for name, values in param_space.items()
        }
    
    def evaluate_fitness(individual):
        """Fitness = 1/time (higher is better)"""
        try:
            time = time_configuration(kernel, individual, test_args, num_iters=10)
            return 1.0 / time
        except:
            return 0.0
    
    def crossover(parent1, parent2):
        """Create child by combining parents"""
        child = {}
        for name in param_space.keys():
            # Randomly choose from parent1 or parent2
            child[name] = random.choice([parent1[name], parent2[name]])
        return child
    
    def mutate(individual, mutation_rate=0.1):
        """Randomly change some parameters"""
        mutated = individual.copy()
        for name, values in param_space.items():
            if random.random() < mutation_rate:
                mutated[name] = random.choice(values)
        return mutated
    
    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    
    best_ever = None
    best_fitness = 0.0
    
    print(f"Starting genetic algorithm: {num_generations} generations")
    
    for gen in range(num_generations):
        print(f"\n=== Generation {gen+1} ===")
        
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness
        fitness_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Track best
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_ever = fitness_scores[0][1]
            print(f"New best! Fitness: {best_fitness:.2f}, Time: {1/best_fitness*1000:.3f} ms")
            print(f"Config: {best_ever}")
        
        # Selection: Keep top 50%
        survivors = [ind for _, ind in fitness_scores[:population_size//2]]
        
        # Create next generation
        next_generation = survivors.copy()
        
        while len(next_generation) < population_size:
            # Select parents (tournament selection)
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child)
            
            next_generation.append(child)
        
        population = next_generation
    
    print(f"\n=== Best Configuration (Genetic) ===")
    print(f"Parameters: {best_ever}")
    print(f"Time: {1/best_fitness*1000:.3f} ms")
    
    return {
        'params': best_ever,
        'time': 1/best_fitness
    }

# Usage
param_space = {
    'TILE_M': [64, 128, 256],
    'TILE_N': [64, 128, 256],
    'TILE_K': [16, 32, 64],
    'NUM_STAGES': [2, 3, 4]
}

best = genetic_autotune(my_kernel, param_space, (A, B, C))
```

---

## Caching Tuned Parameters

### Save and Load Results

**Cache results for future use:**
```python
import json
import hashlib
import torch

class AutotuneCache:
    """Cache autotuned parameters"""
    
    def __init__(self, cache_file="autotune_cache.json"):
        self.cache_file = cache_file
        self.cache = self.load_cache()
    
    def load_cache(self):
        """Load cache from disk"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get_cache_key(
        self,
        kernel_name: str,
        problem_size: Tuple,
        dtype: str,
        device: str
    ) -> str:
        """Generate unique cache key"""
        
        # Get GPU architecture
        device_props = torch.cuda.get_device_properties(device)
        arch = f"sm_{device_props.major}{device_props.minor}"
        
        # Create key from all relevant factors
        key_data = {
            'kernel': kernel_name,
            'problem_size': problem_size,
            'dtype': dtype,
            'arch': arch
        }
        
        # Hash to create short key
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str):
        """Get cached parameters"""
        return self.cache.get(key)
    
    def set(self, key: str, params: Dict, time: float):
        """Cache parameters"""
        self.cache[key] = {
            'params': params,
            'time': time
        }
        self.save_cache()

# Usage
cache = AutotuneCache()

# Check cache
cache_key = cache.get_cache_key(
    kernel_name='gemm',
    problem_size=(4096, 4096, 4096),
    dtype='float16',
    device='cuda:0'
)

cached = cache.get(cache_key)
if cached:
    print("Using cached parameters:")
    print(cached['params'])
    best_params = cached['params']
else:
    print("Running autotuning...")
    result = grid_search_autotune(kernel, param_space, args)
    
    # Cache result
    cache.set(cache_key, result['params'], result['time'])
    best_params = result['params']
```

---

## Problem-Size-Specific Tuning

### Different Sizes Need Different Parameters

**Small vs large problems:**
```python
def size_specific_autotune(
    kernel,
    param_space: Dict[str, List],
    problem_sizes: List[Tuple],
    dtype: str = 'float16'
) -> Dict[Tuple, Dict]:
    """
    Autotune for different problem sizes
    
    Returns:
        Dictionary mapping problem size to best parameters
    """
    
    results = {}
    
    for size in problem_sizes:
        M, N, K = size
        print(f"\n=== Tuning for size {M}×{N}×{K} ===")
        
        # Create test data
        A = torch.randn(M, K, device='cuda', dtype=getattr(torch, dtype))
        B = torch.randn(K, N, device='cuda', dtype=getattr(torch, dtype))
        C = torch.zeros(M, N, device='cuda', dtype=getattr(torch, dtype))
        
        # Autotune for this size
        result = grid_search_autotune(
            kernel,
            param_space,
            (cute.from_dlpack(A), cute.from_dlpack(B), cute.from_dlpack(C))
        )
        
        results[size] = result
        
        print(f"Best params for {size}: {result['params']}")
        print(f"Time: {result['time']*1000:.3f} ms")
    
    return results

# Usage
problem_sizes = [
    (512, 512, 512),      # Small
    (2048, 2048, 2048),   # Medium
    (4096, 4096, 4096),   # Large
    (8192, 8192, 8192)    # Very large
]

size_configs = size_specific_autotune(gemm_kernel, param_space, problem_sizes)

# Use appropriate config at runtime
def get_best_params_for_size(M, N, K):
    # Find closest tuned size
    for size, config in size_configs.items():
        if M <= size[0] and N <= size[1] and K <= size[2]:
            return config['params']
    # Default to largest
    return size_configs[max(size_configs.keys())]['params']
```

---

## Multi-Objective Optimization

### Optimize for Multiple Metrics

**Balance time and memory:**
```python
def pareto_autotune(
    kernel,
    param_space: Dict[str, List],
    test_args: Tuple,
    num_trials: int = 100
) -> List[Dict]:
    """
    Multi-objective autotuning
    
    Find Pareto frontier: configurations where improving one
    metric requires making another worse.
    
    Returns:
        List of Pareto-optimal configurations
    """
    
    results = []
    
    print(f"Testing {num_trials} configurations for Pareto frontier...")
    
    for trial in range(num_trials):
        # Random configuration
        params = {
            name: random.choice(values)
            for name, values in param_space.items()
        }
        
        try:
            # Measure time
            time = time_configuration(kernel, params, test_args, num_iters=20)
            
            # Estimate memory usage
            memory_usage = estimate_memory_usage(params)
            
            results.append({
                'params': params,
                'time': time,
                'memory': memory_usage
            })
            
            print(f"[{trial+1}] Time: {time*1000:.2f} ms, Memory: {memory_usage/1024:.1f} KB")
        
        except Exception as e:
            continue
    
    # Find Pareto frontier
    pareto_frontier = []
    
    for result in results:
        is_dominated = False
        
        for other in results:
            if other == result:
                continue
            
            # Check if 'other' dominates 'result'
            # (better in all objectives)
            if (other['time'] <= result['time'] and
                other['memory'] <= result['memory'] and
                (other['time'] < result['time'] or other['memory'] < result['memory'])):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_frontier.append(result)
    
    print(f"\n=== Pareto Frontier ({len(pareto_frontier)} configurations) ===")
    for i, config in enumerate(sorted(pareto_frontier, key=lambda x: x['time'])):
        print(f"{i+1}. Time: {config['time']*1000:.2f} ms, "
              f"Memory: {config['memory']/1024:.1f} KB")
        print(f"   Params: {config['params']}")
    
    return pareto_frontier

def estimate_memory_usage(params):
    """Estimate shared memory usage from parameters"""
    TILE_M = params.get('TILE_M', 128)
    TILE_N = params.get('TILE_N', 128)
    TILE_K = params.get('TILE_K', 32)
    NUM_STAGES = params.get('NUM_STAGES', 2)
    
    # Shared memory for A and B tiles
    smem_per_stage = (TILE_M * TILE_K + TILE_K * TILE_N) * 2  # Float16 = 2 bytes
    total_smem = smem_per_stage * NUM_STAGES
    
    return total_smem

# Usage
pareto_configs = pareto_autotune(gemm_kernel, param_space, (A, B, C))

# User can choose based on preference:
# - Fastest (first in list)
# - Most memory-efficient (last in list)
# - Balanced (middle of list)
```

---

## Autotuning Framework Integration

### OpenTuner Integration

**Use OpenTuner framework:**
```python
try:
    import opentuner
    from opentuner import ConfigurationManipulator
    from opentuner import IntegerParameter
    from opentuner import MeasurementInterface
    from opentuner import Result
except ImportError:
    print("Install: pip install opentuner")

class KernelTuner(MeasurementInterface):
    """OpenTuner interface for kernel autotuning"""
    
    def __init__(self, kernel, test_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel = kernel
        self.test_args = test_args
    
    def manipulator(self):
        """Define search space"""
        manipulator = ConfigurationManipulator()
        
        manipulator.add_parameter(
            IntegerParameter('TILE_M', 64, 256)
        )
        manipulator.add_parameter(
            IntegerParameter('TILE_N', 64, 256)
        )
        manipulator.add_parameter(
            IntegerParameter('TILE_K', 16, 64)
        )
        manipulator.add_parameter(
            IntegerParameter('NUM_STAGES', 2, 5)
        )
        
        return manipulator
    
    def run(self, desired_result, input, limit):
        """Run and time configuration"""
        cfg = desired_result.configuration.data
        
        try:
            time = time_configuration(
                self.kernel,
                cfg,
                self.test_args,
                num_iters=20
            )
            return Result(time=time)
        except:
            return Result(time=float('inf'))
    
    def save_final_config(self, configuration):
        """Save best configuration"""
        print(f"Best configuration: {configuration.data}")

# Usage
# tuner = KernelTuner(my_kernel, (A, B, C))
# tuner.main()
```

---

## Best Practices

### ✅ DO

**Start with coarse grid:**
```python
# Phase 1: Coarse grid (fast)
coarse_space = {
    'TILE_M': [64, 128, 256],
    'TILE_K': [16, 32, 64]
}

# Phase 2: Fine grid around best (accurate)
fine_space = {
    'TILE_M': [112, 128, 144],
    'TILE_K': [24, 32, 40]
}
```

**Cache results:**
```python
cache = AutotuneCache()
key = cache.get_cache_key(...)
if key in cache:
    use_cached_params()
else:
    autotune_and_cache()
```

**Use problem-size-specific tuning:**
```python
# Tune for representative sizes
sizes = [small, medium, large]
for size in sizes:
    autotune(size)
```

**Validate results:**
```python
# Check autotuned kernel produces correct results
result = autotuned_kernel(inputs)
reference = reference_kernel(inputs)
assert torch.allclose(result, reference)
```

### ❌ DON'T

**Don't tune on different hardware:**
```python
# ✗ BAD: Tune on A100, use on V100
# Results won't transfer!

# ✓ GOOD: Tune on target hardware
```

**Don't forget constraints:**
```python
# ✗ BAD: Invalid configurations
params = {'TILE_M': 512, 'TILE_N': 512}  # > 1024 threads!

# ✓ GOOD: Validate constraints
assert params['TILE_M'] * params['TILE_N'] <= 1024
```

**Don't overtune:**
```python
# ✗ BAD: 10,000 trials for 1% improvement
# Diminishing returns!

# ✓ GOOD: Stop when good enough
if improvement < 0.01:
    break
```

---

## Summary

**Autotuning strategies:**
- **Grid search:** Exhaustive but slow
- **Random search:** Faster, often as good
- **Bayesian optimization:** Smart, sample-efficient
- **Genetic algorithm:** Good for complex spaces

**What to tune:**
- Tile sizes (TILE_M, TILE_N, TILE_K)
- Pipeline depth (NUM_STAGES)
- Block size (threads per block)
- Vectorization width
- Unroll factors

**Best practices:**
1. Cache tuned parameters
2. Tune per problem size
3. Validate correctness
4. Start coarse, refine
5. Use target hardware

**Tools:**
- Custom grid/random search
- Bayesian optimization (bayes_opt)
- Genetic algorithms
- OpenTuner framework

**Key insight:** Autotuning can find 2-5× better configurations than manual tuning, and adapts to different hardware automatically. The time spent autotuning once pays off in faster execution forever.

---

## Next Steps

- [Profiling](./profiling.md) - Measure what to tune
- [Performance Tips](./performance_tips.md) - What parameters matter
- [Case Studies](../09_case_studies/) - Real autotuning examples

---

## Further Reading

- [OpenTuner](http://opentuner.org/)
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)
- [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)
- [CUTLASS Profiler](https://github.com/NVIDIA/cutlass/tree/main/tools/profiler)