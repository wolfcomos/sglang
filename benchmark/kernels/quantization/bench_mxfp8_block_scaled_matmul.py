import argparse
import os

import torch
import triton

from sglang.srt.layers.quantization.fp8_kernel import (
    mxfp8_block_scaled_matmul_triton,
)
from sglang.srt.layers.quantization.fp8_utils import (
    _pack_mxfp8_scales,
    mxfp8_group_quantize,
    triton_mxfp8_blockscaled_linear,
)
from sglang.srt.utils import is_sm100_supported


def _get_triton_mxfp8_upcast():
    try:
        from triton_kernels.numerics_details.mxfp import upcast_from_mxfp_torch
    except Exception as err:
        raise RuntimeError(
            "MXFP8 dequantization requires triton_kernels with MXFP8 support."
        ) from err
    return upcast_from_mxfp_torch


def _mxfp8_group_dequant(q: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    upcast_from_mxfp_torch = _get_triton_mxfp8_upcast()
    return upcast_from_mxfp_torch(q, scale_u8, torch.float32, axis=1)


def _test_accuracy_once(M, N, K, input_dtype, device):
    torch.manual_seed(0)
    input_fp32 = torch.randn((M, K), dtype=torch.float32, device=device) / 4
    input_fp16 = input_fp32.to(input_dtype)

    weight_fp32 = torch.randn((N, K), dtype=torch.float32, device=device) / 4
    weight_q, weight_scale_u8 = mxfp8_group_quantize(weight_fp32)

    with torch.inference_mode():
        q_input, input_scale_u8 = mxfp8_group_quantize(input_fp16.to(torch.float32))
        a_dq = _mxfp8_group_dequant(q_input, input_scale_u8)
        b_dq = _mxfp8_group_dequant(weight_q, weight_scale_u8)
        ref_out = torch.matmul(a_dq, b_dq.t()).to(input_dtype)

        out = triton_mxfp8_blockscaled_linear(
            input=input_fp16,
            weight=weight_q,
            weight_scale=weight_scale_u8,
        )

    rel_error = (
        torch.mean(torch.abs(out.to(torch.float32) - ref_out.to(torch.float32)))
        / torch.mean(torch.abs(ref_out.to(torch.float32)))
    )
    assert rel_error < 0.02, f"Accuracy test failed: rel_error={rel_error:.4f}"
    print(f"M: {M}, N: {N}, K: {K}, type: {input_dtype} OK")


def test_accuracy():
    if not is_sm100_supported():
        print("Skipping accuracy test: MXFP8 requires Blackwell (SM100+)")
        return

    Ms = [1, 127, 128, 129, 255, 256]
    NKs = [
        (256, 512),
        (384, 1024),
        (512, 2048),
        (768, 1024),
    ]
    input_dtypes = [torch.bfloat16]
    for M in Ms:
        for N, K in NKs:
            for input_dtype in input_dtypes:
                _test_accuracy_once(M, N, K, input_dtype, "cuda")


# Test configurations from test_block_fp8.py TestMXFP8DenseLinear
Ms = [1, 127, 128, 129, 255, 256, 512, 1024]
NKs = [
    (256, 512),
    (384, 1024),
    (512, 2048),
    (768, 1024),
]

# Filter valid configurations: K must be divisible by 128, M by 128, N by 128 or 256
def is_valid_config(M, N, K):
    if K % 128 != 0:
        return False
    if M % 128 != 0:
        return False
    if N % 256 != 0 and N % 128 != 0:
        return False
    return True


valid_configs = [
    (M, N, K) for M in Ms for N, K in NKs if is_valid_config(M, N, K)
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=valid_configs,
        x_log=False,
        line_arg="provider",
        line_vals=["mxfp8_triton"],
        line_names=["MXFP8 Triton"],
        styles=[("blue", "-")],
        ylabel="ms",
        plot_name="mxfp8-block-scaled-matmul",
        args={},
    )
)
def benchmark(M, N, K, provider):
    if not is_sm100_supported():
        return (0, 0, 0)

    device = "cuda"
    torch.manual_seed(0)
    input_fp32 = torch.randn((M, K), dtype=torch.float32, device=device) / 4
    weight_fp32 = torch.randn((N, K), dtype=torch.float32, device=device) / 4

    # Quantize input and weight
    q_input, input_scale_u8 = mxfp8_group_quantize(input_fp32)
    weight_q, weight_scale_u8 = mxfp8_group_quantize(weight_fp32)

    # Pack scales
    a_scale_packed = _pack_mxfp8_scales(input_scale_u8)
    b_scale_packed = _pack_mxfp8_scales(weight_scale_u8)

    # Determine block sizes
    block_m = 128
    block_n = 256 if N % 256 == 0 else 128
    block_k = 128

    quantiles = [0.5, 0.2, 0.8]
    if provider == "mxfp8_triton":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: mxfp8_block_scaled_matmul_triton(
                q_input,
                a_scale_packed,
                weight_q,
                b_scale_packed,
                output_dtype=torch.bfloat16,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_stages=4,
            ),
            quantiles=quantiles,
        )

    return ms, min_ms, max_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./bench_mxfp8_block_scaled_matmul_res",
        help="Path to save mxfp8 block scaled matmul benchmark results",
    )
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)

    test_accuracy()

    benchmark.run(print_data=True, show_plots=True, save_path=args.save_path)
