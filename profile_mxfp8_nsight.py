#!/usr/bin/env python3
"""
Profile MXFP8 Triton kernel with NVIDIA Nsight Compute.

This script is designed to be run with Nsight Compute for detailed hardware-level profiling.

Usage:
    # Command line profiling
    ncu --set full python profile_mxfp8_nsight.py
    
    # Or with specific metrics
    ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
        python profile_mxfp8_nsight.py
    
    # Export to file
    ncu -o mxfp8_profile python profile_mxfp8_nsight.py
    ncu --import mxfp8_profile.ncu-rep
"""

import argparse
import sys

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    mxfp8_group_quantize,
    triton_mxfp8_blockscaled_linear,
    _pack_mxfp8_scales,
)
from sglang.srt.layers.quantization.fp8_kernel import mxfp8_block_scaled_matmul_triton


def check_requirements():
    """Check if the system meets requirements."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()


def determine_block_sizes(M: int, N: int, K: int) -> tuple:
    """
    Determine appropriate block sizes and num_stages based on matrix dimensions.
    
    The kernel requires:
    - block_m must be a multiple of 128 (rep_m = block_m // 128)
    - block_n must be a multiple of 128 (rep_n = block_n // 128)  
    - block_k must be a multiple of 128 (rep_k = block_k // 128)
    
    Additionally, rep_k should match scale_k from the packed scale tensor:
    - scale_k = (K // 32) // 4 = K // 128
    - So ideally: block_k // 128 = K // 128, meaning block_k should be K (or a multiple)
    """
    # For block_k, try to match K's scale_k dimension
    block_k = ((K + 127) // 128) * 128
    
    # Start with conservative defaults based on matrix size
    if M < 256 and N < 512:
        block_m, block_n = 128, 128
        num_stages = 2
    elif M < 512 and N < 1024:
        block_m, block_n = 128, 128
        num_stages = 3
    elif M < 1024:
        block_m, block_n = 128, 128
        num_stages = 3
    else:
        block_m, block_n = 128, 256
        num_stages = 4
    
    return (block_m, block_n, block_k, num_stages)


def find_working_config(q_input, a_scale, weight, b_scale, output_dtype, M, N, K):
    """
    Find a working block size configuration by trying different num_stages.
    """
    block_m, block_n, block_k, num_stages = determine_block_sizes(M, N, K)
    
    # Try different configurations if the first one fails
    configs_to_try = [
        (block_m, block_n, block_k, num_stages),
        (block_m, block_n, block_k, num_stages - 1) if num_stages > 1 else None,
        (block_m, block_n, block_k, num_stages - 2) if num_stages > 2 else None,
        (block_m, block_n, block_k, 1),  # Most conservative: num_stages=1
    ]
    
    configs_to_try = [c for c in configs_to_try if c is not None]
    
    # Remove duplicates
    seen = set()
    unique_configs = []
    for c in configs_to_try:
        if c not in seen:
            seen.add(c)
            unique_configs.append(c)
    
    last_error = None
    for bm, bn, bk, ns in unique_configs:
        try:
            result = mxfp8_block_scaled_matmul_triton(
                q_input, a_scale, weight, b_scale, output_dtype,
                block_m=bm, block_n=bn, block_k=bk, num_stages=ns
            )
            if (bm, bn, bk, ns) != (block_m, block_n, block_k, num_stages):
                print(f"  Using adjusted config: block_m={bm}, block_n={bn}, block_k={bk}, num_stages={ns}")
            return result, (bm, bn, bk, ns)
        except Exception as e:
            last_error = e
            continue
    
    raise RuntimeError(
        f"Failed to find working block size configuration. Last error: {last_error}\n"
        f"Try using larger matrix dimensions or different test case."
    )


def profile_kernel_with_nsight(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.bfloat16,
    num_warmup: int = 5,
    num_iterations: int = 10,
    profile_quantization: bool = False,
    profile_matmul: bool = True,
    profile_e2e: bool = False,
):
    """
    Profile the kernel with setup for Nsight Compute.
    
    Args:
        M, N, K: Matrix dimensions
        dtype: Output dtype
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations to profile
        profile_quantization: Whether to profile quantization step
        profile_matmul: Whether to profile matmul kernel
        profile_e2e: Whether to profile end-to-end function
    """
    print(f"Profiling MXFP8 kernel: M={M}, N={N}, K={K}")
    print(f"Output dtype: {dtype}")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Profile iterations: {num_iterations}")
    print()
    
    # Create test tensors
    input_tensor = torch.randn(M, K, dtype=dtype, device="cuda")
    weight_fp32 = torch.randn(N, K, dtype=torch.float32, device="cuda") / 4
    
    # Quantize weight
    weight, weight_scale = mxfp8_group_quantize(weight_fp32)
    torch.cuda.synchronize()
    
    # Prepare for matmul if needed
    matmul_config = None
    if profile_matmul:
        q_input_test, input_scale_test = mxfp8_group_quantize(input_tensor.to(torch.float32))
        a_scale_packed_test = _pack_mxfp8_scales(input_scale_test)
        b_scale_packed_test = _pack_mxfp8_scales(weight_scale)
        try:
            _, matmul_config = find_working_config(
                q_input_test, a_scale_packed_test, weight, b_scale_packed_test, dtype, M, N, K
            )
            print(f"Found working matmul config: block_m={matmul_config[0]}, block_n={matmul_config[1]}, "
                  f"block_k={matmul_config[2]}, num_stages={matmul_config[3]}")
        except Exception as e:
            print(f"Warning: Could not find working matmul config: {e}")
            print("Skipping matmul profiling")
            profile_matmul = False
    
    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        if profile_quantization:
            mxfp8_group_quantize(input_tensor.to(torch.float32))
        if profile_matmul or profile_e2e:
            q_input, input_scale = mxfp8_group_quantize(input_tensor.to(torch.float32))
            if profile_matmul and matmul_config is not None:
                a_scale_packed = _pack_mxfp8_scales(input_scale)
                b_scale_packed = _pack_mxfp8_scales(weight_scale)
                bm, bn, bk, ns = matmul_config
                mxfp8_block_scaled_matmul_triton(
                    q_input, a_scale_packed, weight, b_scale_packed, dtype,
                    block_m=bm, block_n=bn, block_k=bk, num_stages=ns
                )
            if profile_e2e:
                triton_mxfp8_blockscaled_linear(
                    input_tensor, weight, weight_scale, output_dtype=dtype
                )
    torch.cuda.synchronize()
    
    print("Starting profiling (Nsight Compute will capture this)...")
    print("=" * 80)
    
    # Profile quantization
    if profile_quantization:
        print("\n[Profiling: mxfp8_group_quantize]")
        for i in range(num_iterations):
            result = mxfp8_group_quantize(input_tensor.to(torch.float32))
            torch.cuda.synchronize()
            if i == 0:
                print(f"  Iteration {i+1}/{num_iterations} completed")
    
    # Profile matmul kernel
    if profile_matmul:
        print("\n[Profiling: mxfp8_block_scaled_matmul_triton]")
        q_input, input_scale = mxfp8_group_quantize(input_tensor.to(torch.float32))
        a_scale_packed = _pack_mxfp8_scales(input_scale)
        b_scale_packed = _pack_mxfp8_scales(weight_scale)
        
        # Use config found during warmup, or find it now
        if matmul_config is None:
            print("  Finding working block size configuration...")
            try:
                test_result, matmul_config = find_working_config(
                    q_input, a_scale_packed, weight, b_scale_packed, dtype, M, N, K
                )
            except Exception as e:
                print(f"  ERROR: Could not find working configuration: {e}")
                print("  Skipping matmul profiling")
                profile_matmul = False
        
        if profile_matmul and matmul_config is not None:
            block_m, block_n, block_k, num_stages = matmul_config
            print(f"  Using: block_m={block_m}, block_n={block_n}, block_k={block_k}, num_stages={num_stages}")
            for i in range(num_iterations):
                result = mxfp8_block_scaled_matmul_triton(
                    q_input, a_scale_packed, weight, b_scale_packed, dtype,
                    block_m=block_m, block_n=block_n, block_k=block_k, num_stages=num_stages
                )
                torch.cuda.synchronize()
                if i == 0:
                    print(f"  Iteration {i+1}/{num_iterations} completed")
                    print(f"  Output shape: {result.shape}")
    
    # Profile end-to-end
    if profile_e2e:
        print("\n[Profiling: triton_mxfp8_blockscaled_linear]")
        for i in range(num_iterations):
            result = triton_mxfp8_blockscaled_linear(
                input_tensor, weight, weight_scale, output_dtype=dtype
            )
            torch.cuda.synchronize()
            if i == 0:
                print(f"  Iteration {i+1}/{num_iterations} completed")
                print(f"  Output shape: {result.shape}")
    
    print("\n" + "=" * 80)
    print("Profiling complete!")
    print("\nNsight Compute should have captured the kernel execution.")
    print("Check the output above or the Nsight Compute report for details.")


def main():
    parser = argparse.ArgumentParser(
        description="Profile MXFP8 Triton kernel with Nsight Compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full profile with Nsight Compute
  ncu --set full python profile_mxfp8_nsight.py
  
  # Profile specific metrics
  ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \\
      python profile_mxfp8_nsight.py
  
  # Export to file
  ncu -o mxfp8_profile python profile_mxfp8_nsight.py
  
  # Profile only matmul kernel
  ncu --set full python profile_mxfp8_nsight.py --profile-matmul
  
  # Custom matrix size
  ncu --set full python profile_mxfp8_nsight.py --M 512 --N 1024 --K 2048
        """,
    )
    parser.add_argument(
        "--M", type=int, default=128, help="Input matrix M dimension (default: 128)"
    )
    parser.add_argument(
        "--N", type=int, default=256, help="Weight matrix N dimension (default: 256)"
    )
    parser.add_argument(
        "--K", type=int, default=512, help="Matrix K dimension (default: 512)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Output dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="Number of profiling iterations (default: 10)",
    )
    parser.add_argument(
        "--profile-quantization",
        action="store_true",
        help="Profile quantization step",
    )
    parser.add_argument(
        "--profile-matmul",
        action="store_true",
        default=True,
        help="Profile matmul kernel (default: True)",
    )
    parser.add_argument(
        "--profile-e2e",
        action="store_true",
        help="Profile end-to-end function",
    )
    parser.add_argument(
        "--no-profile-matmul",
        action="store_true",
        help="Disable matmul profiling",
    )
    
    args = parser.parse_args()
    
    # Check requirements
    check_requirements()
    
    # Map dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # Handle --no-profile-matmul
    profile_matmul = args.profile_matmul and not args.no_profile_matmul
    
    # Ensure at least one component is profiled
    if not (args.profile_quantization or profile_matmul or args.profile_e2e):
        print("Warning: No profiling targets selected. Defaulting to matmul kernel.")
        profile_matmul = True
    
    # Run profiling
    try:
        profile_kernel_with_nsight(
            M=args.M,
            N=args.N,
            K=args.K,
            dtype=dtype,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            profile_quantization=args.profile_quantization,
            profile_matmul=profile_matmul,
            profile_e2e=args.profile_e2e,
        )
    except Exception as e:
        print(f"\nError during profiling: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
