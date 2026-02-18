#!/usr/bin/env python3
"""
Profile the MXFP8 Triton kernel implementation to establish a performance baseline.

This script profiles:
1. mxfp8_group_quantize - quantization step
2. mxfp8_block_scaled_matmul_triton - the main matmul kernel
3. triton_mxfp8_blockscaled_linear - the full end-to-end function

Usage:
    python profile_mxfp8_triton.py
"""

import argparse
import time
from typing import Dict, List, Tuple

import torch
import torch.profiler

from sglang.srt.layers.quantization.fp8_utils import (
    mxfp8_group_quantize,
    triton_mxfp8_blockscaled_linear,
)
from sglang.srt.layers.quantization.fp8_kernel import mxfp8_block_scaled_matmul_triton
from sglang.srt.utils import is_sm100_supported


def check_requirements():
    """Check if the system meets requirements for MXFP8."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()


def warmup(func, *args, **kwargs):
    """Warmup GPU by running the function multiple times."""
    for _ in range(10):
        func(*args, **kwargs)
    torch.cuda.synchronize()


def profile_with_cuda_events(
    func, *args, num_iterations: int = 100, **kwargs
) -> Tuple[float, float]:
    """
    Profile a function using CUDA events for accurate timing.
    
    Returns:
        (mean_time_ms, std_time_ms)
    """
    # Warmup
    warmup(func, *args, **kwargs)
    
    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_event.record()
        
        result = func(*args, **kwargs)
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms)
    
    mean_time = sum(times) / len(times)
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    std_time = variance ** 0.5
    
    return mean_time, std_time


def profile_quantization(input_tensor: torch.Tensor, num_iterations: int = 100) -> Dict:
    """Profile the mxfp8_group_quantize function."""
    print("Profiling mxfp8_group_quantize...")
    
    def quantize():
        return mxfp8_group_quantize(input_tensor)
    
    mean_time, std_time = profile_with_cuda_events(quantize, num_iterations=num_iterations)
    
    result = {
        "function": "mxfp8_group_quantize",
        "input_shape": input_tensor.shape,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "throughput_gbps": (input_tensor.numel() * 4 * 2) / (mean_time * 1e-3) / 1e9,  # FP32 input + FP8 output
    }
    
    print(f"  Mean time: {mean_time:.3f} ± {std_time:.3f} ms")
    print(f"  Throughput: {result['throughput_gbps']:.2f} GB/s")
    print()
    
    return result


def determine_block_sizes(M: int, N: int, K: int) -> Tuple[int, int, int, int]:
    """
    Determine appropriate block sizes and num_stages based on matrix dimensions.
    
    The kernel requires:
    - block_m must be a multiple of 128 (rep_m = block_m // 128)
    - block_n must be a multiple of 128 (rep_n = block_n // 128)  
    - block_k must be a multiple of 128 (rep_k = block_k // 128)
    
    Additionally, rep_k should match scale_k from the packed scale tensor:
    - scale_k = (K // 32) // 4 = K // 128
    - So ideally: block_k // 128 = K // 128, meaning block_k should be K (or a multiple)
    
    Tries to find a configuration that fits within shared memory limits.
    """
    # The kernel requires block sizes to be multiples of 128
    # For block_k, try to match K's scale_k dimension
    # scale_k = K // 128, so we want rep_k = block_k // 128 = K // 128
    # This means block_k should be K rounded up to the nearest multiple of 128
    block_k = ((K + 127) // 128) * 128
    
    # Start with conservative defaults based on matrix size
    if M < 256 and N < 512:
        # Very small matrices - use minimum block size (128)
        # May still hit shared memory limits, will fallback to smaller num_stages
        block_m, block_n = 128, 128
        num_stages = 2
    elif M < 512 and N < 1024:
        # Small-medium matrices
        block_m, block_n = 128, 128
        num_stages = 3
    elif M < 1024:
        # Medium matrices
        block_m, block_n = 128, 128
        num_stages = 3
    else:
        # Large matrices - can try more aggressive configs
        block_m, block_n = 128, 256
        num_stages = 4
    
    return (block_m, block_n, block_k, num_stages)


def profile_matmul_kernel(
    q_input: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,
    block_m: int = None,
    block_n: int = None,
    block_k: int = None,
    num_stages: int = None,
    num_iterations: int = 100,
) -> Dict:
    """Profile the mxfp8_block_scaled_matmul_triton kernel."""
    print("Profiling mxfp8_block_scaled_matmul_triton...")
    
    M, K = q_input.shape
    N = weight.shape[0]
    
    # Auto-determine block sizes if not provided
    # NOTE: Kernel requires block_m, block_n, block_k to be multiples of 128
    # (rep_m = block_m // 128, rep_n = block_n // 128, rep_k = block_k // 128)
    if block_m is None or block_n is None or block_k is None or num_stages is None:
        block_m, block_n, block_k, num_stages = determine_block_sizes(M, N, K)
        # Ensure block sizes are multiples of 128
        block_m = ((block_m + 127) // 128) * 128
        block_n = ((block_n + 127) // 128) * 128
        block_k = ((block_k + 127) // 128) * 128
        print(f"  Using block sizes: block_m={block_m}, block_n={block_n}, block_k={block_k}, num_stages={num_stages}")
    
    # Try different configurations if the first one fails
    # NOTE: All block sizes must be multiples of 128 due to kernel constraints
    # (rep_m = block_m // 128, rep_n = block_n // 128, rep_k = block_k // 128)
    # We can only reduce num_stages to reduce shared memory usage
    configs_to_try = [
        (block_m, block_n, block_k, num_stages),
        (block_m, block_n, block_k, num_stages - 1) if num_stages > 1 else None,
        (block_m, block_n, block_k, num_stages - 2) if num_stages > 2 else None,
        (block_m, block_n, block_k, 1),  # Most conservative: num_stages=1
        # Can't reduce block sizes below 128 due to kernel constraints
        # If 128x128x128 with num_stages=1 doesn't work, the kernel cannot run on this GPU
    ]
    
    configs_to_try = [c for c in configs_to_try if c is not None]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_configs = []
    for c in configs_to_try:
        if c not in seen:
            seen.add(c)
            unique_configs.append(c)
    configs_to_try = unique_configs
    
    last_error = None
    working_config = None
    initial_config = (block_m, block_n, block_k, num_stages)
    
    for bm, bn, bk, ns in configs_to_try:
        try:
            def matmul():
                return mxfp8_block_scaled_matmul_triton(
                    q_input,
                    a_scale,
                    weight,
                    b_scale,
                    output_dtype,
                    block_m=bm,
                    block_n=bn,
                    block_k=bk,
                    num_stages=ns,
                )
            
            # Test if it works
            warmup(matmul)
            torch.cuda.synchronize()
            
            # If we get here, it works - use this config
            working_config = (bm, bn, bk, ns)
            if working_config != initial_config:
                print(f"  Adjusted to: block_m={bm}, block_n={bn}, block_k={bk}, num_stages={ns}")
            break
        except Exception as e:
            last_error = e
            continue
    
    if working_config is None:
        error_msg = f"Failed to find working block size configuration.\n"
        error_msg += f"Last error: {last_error}\n"
        error_msg += f"Note: The kernel requires block_m, block_n, block_k to be multiples of 128.\n"
        error_msg += f"Minimum configuration (128x128x128, num_stages=1) still exceeds shared memory limits.\n"
        error_msg += f"This may indicate the matrix dimensions are too small or the GPU has limited shared memory.\n"
        error_msg += f"Try using larger matrix dimensions (M, N, K >= 256) or skip this test case."
        raise RuntimeError(error_msg)
    
    block_m, block_n, block_k, num_stages = working_config
    
    def matmul():
        return mxfp8_block_scaled_matmul_triton(
            q_input,
            a_scale,
            weight,
            b_scale,
            output_dtype,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_stages=num_stages,
        )
    
    mean_time, std_time = profile_with_cuda_events(matmul, num_iterations=num_iterations)
    
    M, K = q_input.shape
    N = weight.shape[0]
    # Compute FLOPS: 2 * M * N * K (for matmul)
    flops = 2 * M * N * K
    tflops = flops / (mean_time * 1e-3) / 1e12
    
    result = {
        "function": "mxfp8_block_scaled_matmul_triton",
        "input_shape": (M, K),
        "weight_shape": (N, K),
        "output_shape": (M, N),
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_stages": num_stages,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "tflops": tflops,
    }
    
    print(f"  Mean time: {mean_time:.3f} ± {std_time:.3f} ms")
    print(f"  Performance: {tflops:.2f} TFLOPS")
    print()
    
    return result


def profile_end_to_end(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
    num_iterations: int = 100,
) -> Dict:
    """Profile the full triton_mxfp8_blockscaled_linear function."""
    print("Profiling triton_mxfp8_blockscaled_linear (end-to-end)...")
    
    def e2e():
        try:
            return triton_mxfp8_blockscaled_linear(
                input_tensor,
                weight,
                weight_scale,
                input_scale=None,
                bias=None,
                output_dtype=output_dtype,
            )
        except Exception as e:
            if "shared memory" in str(e) or "OutOfResources" in str(type(e).__name__):
                print(f"  Warning: End-to-end function hit shared memory limit: {e}")
                print("  Note: This may need manual block size tuning in triton_mxfp8_blockscaled_linear")
                raise
            raise
    
    M, K = input_tensor.shape
    N = weight.shape[0]
    
    try:
        mean_time, std_time = profile_with_cuda_events(e2e, num_iterations=num_iterations)
    except Exception as e:
        error_str = str(e)
        error_type = type(e).__name__
        # Check for various error types that indicate the kernel can't run
        if any(keyword in error_str.lower() for keyword in [
            "shared memory", "outofresources", "reshape", 
            "assertion", "divisible", "multiple"
        ]) or "OutOfResources" in error_type:
            print(f"  Skipping end-to-end profiling due to constraints")
            print(f"  Error type: {error_type}")
            print(f"  Error message: {error_str[:200]}...")  # Truncate long errors
            return {
                "function": "triton_mxfp8_blockscaled_linear",
                "input_shape": (M, K),
                "weight_shape": (N, K),
                "error": error_str,
                "error_type": error_type,
                "skipped": True,
            }
        raise
    
    flops = 2 * M * N * K
    tflops = flops / (mean_time * 1e-3) / 1e12
    
    result = {
        "function": "triton_mxfp8_blockscaled_linear",
        "input_shape": (M, K),
        "weight_shape": (N, K),
        "output_shape": (M, N),
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "tflops": tflops,
    }
    
    print(f"  Mean time: {mean_time:.3f} ± {std_time:.3f} ms")
    print(f"  Performance: {tflops:.2f} TFLOPS")
    print()
    
    return result


def profile_with_torch_profiler(
    func, *args, num_iterations: int = 10, **kwargs
) -> Dict:
    """Profile using PyTorch profiler for detailed analysis."""
    print("Running PyTorch profiler (detailed analysis)...")
    
    warmup(func, *args, **kwargs)
    
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_iterations):
            func(*args, **kwargs)
            torch.cuda.synchronize()
    
    # Print key metrics
    print("\n=== PyTorch Profiler Summary ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Export to Chrome trace format
    trace_file = "mxfp8_triton_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nChrome trace exported to: {trace_file}")
    print("  Open in Chrome: chrome://tracing")
    print()
    
    return {
        "trace_file": trace_file,
        "profiler_summary": str(prof.key_averages()),
    }


def run_benchmark_suite(
    test_cases: List[Tuple[int, int, int]],  # List of (M, N, K) tuples
    dtype: torch.dtype = torch.bfloat16,
    use_profiler: bool = False,
    num_iterations: int = 100,
):
    """Run a suite of benchmarks with different matrix sizes."""
    print("=" * 80)
    print("MXFP8 Triton Kernel Performance Baseline")
    print("=" * 80)
    print()
    
    all_results = []
    
    for M, N, K in test_cases:
        print(f"\n{'='*80}")
        print(f"Test Case: M={M}, N={N}, K={K}")
        print(f"{'='*80}")
        
        # Create test tensors
        input_tensor = torch.randn(M, K, dtype=dtype, device="cuda")
        weight_fp32 = torch.randn(N, K, dtype=torch.float32, device="cuda") / 4
        
        # Quantize weight
        weight, weight_scale = mxfp8_group_quantize(weight_fp32)
        torch.cuda.synchronize()
        
        # Profile quantization
        quant_result = profile_quantization(input_tensor, num_iterations=num_iterations)
        
        # Quantize input for matmul
        q_input, input_scale = mxfp8_group_quantize(input_tensor.to(torch.float32))
        torch.cuda.synchronize()
        
        # Pack scales
        from sglang.srt.layers.quantization.fp8_utils import _pack_mxfp8_scales
        a_scale_packed = _pack_mxfp8_scales(input_scale)
        b_scale_packed = _pack_mxfp8_scales(weight_scale)
        
        # Profile matmul kernel
        try:
            matmul_result = profile_matmul_kernel(
                q_input, a_scale_packed, weight, b_scale_packed, dtype, num_iterations=num_iterations
            )
        except Exception as e:
            error_msg = str(e)
            if "reshape" in error_msg.lower() or "shared memory" in error_msg.lower():
                print(f"  Skipping matmul profiling due to: {error_msg}")
                matmul_result = {
                    "function": "mxfp8_block_scaled_matmul_triton",
                    "input_shape": q_input.shape,
                    "weight_shape": weight.shape,
                    "error": error_msg,
                    "skipped": True,
                }
            else:
                raise
        
        # Profile end-to-end
        e2e_result = profile_end_to_end(input_tensor, weight, weight_scale, dtype, num_iterations=num_iterations)
        
        # Optional: Detailed profiler
        if use_profiler:
            profile_with_torch_profiler(
                triton_mxfp8_blockscaled_linear,
                input_tensor,
                weight,
                weight_scale,
                input_scale=None,
                bias=None,
                output_dtype=dtype,
                num_iterations=10,
            )
        
        all_results.append({
            "test_case": (M, N, K),
            "quantization": quant_result,
            "matmul": matmul_result,
            "end_to_end": e2e_result,
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'M':<8} {'N':<8} {'K':<8} {'Quant (ms)':<12} {'Matmul (ms)':<12} {'E2E (ms)':<12} {'E2E TFLOPS':<12}")
    print("-" * 80)
    
    # Count successful runs
    successful_quant = sum(1 for r in all_results if not r["quantization"].get("skipped", False))
    successful_matmul = sum(1 for r in all_results if not r["matmul"].get("skipped", False))
    successful_e2e = sum(1 for r in all_results if not r["end_to_end"].get("skipped", False))
    
    for result in all_results:
        M, N, K = result["test_case"]
        quant_time = result["quantization"]["mean_time_ms"]
        
        if result["matmul"].get("skipped", False):
            matmul_time_str = "N/A"
        else:
            matmul_time = result["matmul"]["mean_time_ms"]
            matmul_time_str = f"{matmul_time:.3f}"
        
        if result["end_to_end"].get("skipped", False):
            e2e_time_str = "N/A"
            e2e_tflops_str = "N/A"
        else:
            e2e_time = result["end_to_end"]["mean_time_ms"]
            e2e_tflops = result["end_to_end"]["tflops"]
            e2e_time_str = f"{e2e_time:.3f}"
            e2e_tflops_str = f"{e2e_tflops:.2f}"
        
        print(
            f"{M:<8} {N:<8} {K:<8} {quant_time:<12.3f} {matmul_time_str:<12} "
            f"{e2e_time_str:<12} {e2e_tflops_str:<12}"
        )
    
    print("=" * 80)
    print(f"\nSuccessful runs: Quantization={successful_quant}/{len(all_results)}, "
          f"Matmul={successful_matmul}/{len(all_results)}, "
          f"End-to-End={successful_e2e}/{len(all_results)}")
    
    # Print baseline summary for successful runs
    if successful_matmul > 0:
        print("\n" + "=" * 80)
        print("BASELINE PERFORMANCE (Successful Runs Only)")
        print("=" * 80)
        matmul_results = [r for r in all_results if not r["matmul"].get("skipped", False)]
        if matmul_results:
            print(f"\nMatmul Kernel Performance:")
            for r in matmul_results:
                M, N, K = r["test_case"]
                matmul = r["matmul"]
                if "tflops" in matmul:
                    print(f"  M={M}, N={N}, K={K}: {matmul['mean_time_ms']:.3f} ms, "
                          f"{matmul['tflops']:.2f} TFLOPS")
        
        quant_results = [r for r in all_results if not r["quantization"].get("skipped", False)]
        if quant_results:
            print(f"\nQuantization Performance:")
            for r in quant_results:
                M, N, K = r["test_case"]
                quant = r["quantization"]
                print(f"  M={M}, K={K}: {quant['mean_time_ms']:.3f} ms, "
                      f"{quant['throughput_gbps']:.2f} GB/s")
    
    print("=" * 80)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Profile MXFP8 Triton kernel performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run small test cases
  python profile_mxfp8_triton.py --test-cases small

  # Run with custom matrix sizes
  python profile_mxfp8_triton.py --custom 512,1024,2048

  # Run with detailed profiler (generates Chrome trace)
  python profile_mxfp8_triton.py --test-cases medium --use-profiler

  # Run all test cases
  python profile_mxfp8_triton.py --test-cases all
        """,
    )
    parser.add_argument(
        "--test-cases",
        type=str,
        default="small,medium,large",
        help="Comma-separated list of test case sizes: small,medium,large,all",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Output dtype",
    )
    parser.add_argument(
        "--use-profiler",
        action="store_true",
        help="Use PyTorch profiler for detailed analysis (slower, generates Chrome trace)",
    )
    parser.add_argument(
        "--custom",
        type=str,
        help="Custom test case: M,N,K (e.g., 128,512,1024)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for timing (default: 100)",
    )
    
    args = parser.parse_args()
    
    # Check requirements
    check_requirements()
    
    # Map dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # Define test cases
    test_case_sets = {
        "small": [
            (128, 256, 512),
            (256, 512, 1024),
        ],
        "medium": [
            (512, 1024, 2048),
            (1024, 2048, 4096),
        ],
        "large": [
            (2048, 4096, 8192),
            (4096, 8192, 16384),
        ],
        "all": [
            (128, 256, 512),
            (256, 512, 1024),
            (512, 1024, 2048),
            (1024, 2048, 4096),
            (2048, 4096, 8192),
            (4096, 8192, 16384),
        ],
    }
    
    # Select test cases
    if args.custom:
        # Parse custom test case
        parts = args.custom.split(",")
        if len(parts) != 3:
            raise ValueError("Custom test case must be in format M,N,K")
        test_cases = [(int(parts[0]), int(parts[1]), int(parts[2]))]
    else:
        selected = args.test_cases.split(",")
        test_cases = []
        for sel in selected:
            if sel not in test_case_sets:
                raise ValueError(f"Unknown test case set: {sel}")
            test_cases.extend(test_case_sets[sel])
    
    # Remove duplicates and sort
    test_cases = sorted(list(set(test_cases)))
    
    print(f"Selected test cases: {test_cases}")
    print(f"Output dtype: {dtype}")
    print()
    
    # Run benchmarks
    results = run_benchmark_suite(
        test_cases, dtype=dtype, use_profiler=args.use_profiler, num_iterations=args.iterations
    )
    
    print("\nProfiling complete!")
    print("\nNext steps for optimization:")
    print("1. Analyze the Chrome trace (if --use-profiler was used)")
    print("2. Identify bottlenecks in the kernel")
    print("3. Optimize memory access patterns")
    print("4. Tune block sizes and num_stages")
    print("5. Consider kernel fusion opportunities")


if __name__ == "__main__":
    main()
