# MXFP8 Triton Kernel Profiling Guide

This guide explains how to profile the MXFP8 Triton kernel implementation to establish a performance baseline on your RTX 5070 TI GPU.

## Overview

The profiling script (`profile_mxfp8_triton.py`) measures the performance of:
1. **mxfp8_group_quantize** - Quantization step (FP32/BF16 → FP8)
2. **mxfp8_block_scaled_matmul_triton** - The main matmul kernel
3. **triton_mxfp8_blockscaled_linear** - Full end-to-end function

## Requirements

- CUDA-capable GPU with SM100+ (Blackwell architecture)
- PyTorch with CUDA support
- triton_kernels package with MXFP8 support
- sglang installed

## Quick Start

### Basic Usage

```bash
# Run with default test cases (small, medium, large)
python profile_mxfp8_triton.py

# Run only small test cases
python profile_mxfp8_triton.py --test-cases small

# Run with custom matrix sizes
python profile_mxfp8_triton.py --custom 512,1024,2048

# Run all test cases
python profile_mxfp8_triton.py --test-cases all
```

### Detailed Profiling

For detailed analysis with Chrome trace visualization:

```bash
# Generate Chrome trace (view in chrome://tracing)
python profile_mxfp8_triton.py --test-cases medium --use-profiler
```

This will create `mxfp8_triton_trace.json` which can be opened in Chrome at `chrome://tracing`.

### Customization

```bash
# Change output dtype
python profile_mxfp8_triton.py --dtype float16

# Adjust number of iterations for timing
python profile_mxfp8_triton.py --iterations 200

# Combine options
python profile_mxfp8_triton.py --test-cases large --dtype bfloat16 --iterations 50
```

## Output

The script provides:

1. **Per-function timing**: Mean and standard deviation of execution time
2. **Throughput metrics**: GB/s for quantization, TFLOPS for matmul
3. **Summary table**: Comparison across all test cases
4. **Chrome trace** (if `--use-profiler` is used): Detailed timeline visualization

### Example Output

```
================================================================================
MXFP8 Triton Kernel Performance Baseline
================================================================================

GPU: NVIDIA RTX 5070 TI
CUDA Version: 12.x
PyTorch Version: 2.x

================================================================================
Test Case: M=512, N=1024, K=2048
================================================================================
Profiling mxfp8_group_quantize...
  Mean time: 0.123 ± 0.005 ms
  Throughput: 85.23 GB/s

Profiling mxfp8_block_scaled_matmul_triton...
  Mean time: 0.456 ± 0.012 ms
  Performance: 4.70 TFLOPS

Profiling triton_mxfp8_blockscaled_linear (end-to-end)...
  Mean time: 0.612 ± 0.015 ms
  Performance: 3.50 TFLOPS

================================================================================
SUMMARY
================================================================================
M        N        K        Quant (ms)  Matmul (ms)  E2E (ms)    E2E TFLOPS
--------------------------------------------------------------------------------
512      1024     2048     0.123       0.456        0.612       3.50
...
```

## Understanding the Results

### Key Metrics

- **Quantization Time**: Time to quantize input from FP32/BF16 to FP8
- **Matmul Time**: Time for the core matrix multiplication kernel
- **End-to-End Time**: Total time including quantization + matmul
- **TFLOPS**: Theoretical peak performance (2 * M * N * K / time)

### Performance Targets

For RTX 5070 TI (Blackwell architecture), typical targets:
- **Small matrices** (M<512): 2-5 TFLOPS
- **Medium matrices** (512-2048): 5-15 TFLOPS  
- **Large matrices** (M>2048): 15-30+ TFLOPS

### Bottleneck Analysis

If quantization time is significant compared to matmul:
- Consider kernel fusion opportunities
- Optimize quantization kernel separately

If matmul is slow:
- Check memory access patterns
- Tune block sizes (block_m, block_n, block_k)
- Adjust num_stages parameter
- Verify tensor layouts and alignment

## Next Steps for Optimization

1. **Analyze Chrome Trace**: Identify hotspots and stalls
2. **Profile Memory Access**: Check for uncoalesced memory access
3. **Tune Block Sizes**: Experiment with different block configurations
4. **Kernel Fusion**: Consider fusing quantization with matmul
5. **Compare with Baseline**: Use results to measure optimization impact

## Troubleshooting

### "MXFP8 requires Blackwell GPUs (SM100+)"
- Verify your GPU supports SM100
- Check: `python -c "from sglang.srt.utils import is_sm100_supported; print(is_sm100_supported())"`

### Import Errors
- Ensure sglang is properly installed
- Check triton_kernels package is available
- Verify CUDA and PyTorch versions are compatible

### Out of Memory
- Reduce test case sizes
- Use smaller `--custom` matrix dimensions
- Close other GPU applications

## Files Generated

- `mxfp8_triton_trace.json`: Chrome trace file (if `--use-profiler` is used)
  - Open in Chrome: `chrome://tracing`
  - Useful for detailed performance analysis

## Notes

- The script uses CUDA events for accurate timing
- Warmup runs are performed before measurement
- Results may vary based on GPU state and other processes
- For consistent results, run on a dedicated GPU
