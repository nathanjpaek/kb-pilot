# Debugging Notes for CuTe Kernels 2_76 and 2_9

## Problem Statement
Both kernels compile successfully but fail correctness tests on Modal.

## What We Know
- ✅ Both kernels compile (compiled=True)
- ❌ Both fail correctness (correctness=False) 
- ✅ No runtime errors (runtime_error='')
- ✅ Reference models use nn.Linear (matches our implementation)
- ✅ Kernel logic looks correct (GEMM + bias + optional ops + ReLU)

## Hypothesis
The issue might be with how weights are being passed or initialized. Working example 2_12.py uses direct nn.Parameter instead of nn.Linear.

## Next Steps to Try

### Option 1: Match 2_12.py pattern exactly
Change from:
```python
self.linear = nn.Linear(in_features, out_features)
```
To:
```python
self.weight = nn.Parameter(torch.empty(out_features, in_features))
self.bias = nn.Parameter(torch.empty(out_features))
self.reset_parameters()  # Using nn.init methods
```

### Option 2: Add verbose error output
Modify the evaluation to capture actual numerical differences between outputs.

### Option 3: Test with simpler values
Create a test with known inputs/weights to verify kernel math is correct.

## Questions
1. Are the weights being transferred correctly from Model to ModelNew?
2. Is there a dtype mismatch somewhere?
3. Is the evaluation harness properly initializing both models with the same seed?
4. Could there be a layout/stride issue with how we're wrapping tensors?

