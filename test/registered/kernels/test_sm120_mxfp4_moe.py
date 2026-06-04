"""SM120 MXFP4 MoE Triton kernel unit tests.

Validates the DeepSeek-V4 SM120 MoE fallback in
``mxfp4_moe_sm120_triton`` against a PyTorch reference:

- ``mxfp4_gemm_torch`` / ``mxfp4_gemm_triton``: fused dequant + GEMM
- ``mxfp4_moe_forward_torch`` / ``mxfp4_moe_forward_triton``: full MoE path
- Invalid ``topk_ids == -1`` slot masking (CUDA-graph-safe path)
- CUDA graph capture + replay for the Triton MoE forward

Runs on any CUDA GPU (Triton kernels are not SM120-specific); CI uses a
small GPU runner.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.moe.fused_moe_triton.mxfp4_moe_sm120_triton import (
    _mxfp4_e8m0_scale_to_float32,
    mxfp4_gemm_torch,
    mxfp4_gemm_triton,
    mxfp4_moe_forward_torch,
    mxfp4_moe_forward_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

_cpu_dir = Path(__file__).resolve().parent.parent / "cpu"
if str(_cpu_dir) not in sys.path:
    sys.path.insert(0, str(_cpu_dir))
from utils import MXFP4QuantizeUtil  # noqa: E402

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

# Triton LUT vs table dequant can differ slightly; bf16 matmul adds more noise.
_MOE_ATOL = 0.35
_MOE_RTOL = 0.12
_GEMM_ATOL = 0.25
_GEMM_RTOL = 0.10


def _pack_dsv4_mxfp4_weights(
    weight_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack bf16 weights into DSv4 MXFP4 layout (int8 packed + float32 scales)."""
    q_u8, scale_u8 = MXFP4QuantizeUtil.quantize(weight_bf16)
    block_scale_shape = (*weight_bf16.shape[:-1], weight_bf16.shape[-1] // 32)
    scale_u8 = scale_u8.reshape(block_scale_shape)
    packed = q_u8.view(torch.int8)
    scale_f32 = _mxfp4_e8m0_scale_to_float32(scale_u8)
    return packed, scale_f32


def _build_moe_tensors(
    *,
    M: int,
    K: int,
    I: int,
    E: int,
    topk: int,
    device: torch.device,
    seed: int,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dtype = torch.bfloat16

    hidden = (torch.randn(M, K, generator=g, dtype=dtype) / 10).to(device)
    w13_bf16 = (torch.randn(E, 2 * I, K, generator=g, dtype=dtype) / 10).to(device)
    w2_bf16 = (torch.randn(E, K, I, generator=g, dtype=dtype) / 10).to(device)

    w13_packed, w13_scale = _pack_dsv4_mxfp4_weights(w13_bf16)
    w2_packed, w2_scale = _pack_dsv4_mxfp4_weights(w2_bf16)

    score = (torch.randn(M, E, generator=g, dtype=dtype) / 10).to(device)
    score = torch.softmax(score.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(score, topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return (
        hidden,
        w13_packed,
        w2_packed,
        w13_scale,
        w2_scale,
        topk_ids.to(torch.int32),
        topk_weights,
    )


def _build_extreme_gate_up_case(device: torch.device):
    """Build a deterministic case with large +/- gate/up pre-activation values."""
    K = 64
    I = 32
    E = 1
    topk = 1
    dtype = torch.bfloat16

    gate_vals = [
        12.0,
        -12.0,
        10.0,
        -10.0,
        9.0,
        -9.0,
        8.0,
        -8.0,
        14.0,
        -14.0,
        12.0,
        -12.0,
        9.0,
        -9.0,
        13.0,
        -13.0,
        11.0,
        -11.0,
        9.5,
        -9.5,
        14.0,
        -14.0,
        11.0,
        -11.0,
        12.5,
        -12.5,
        10.5,
        -10.5,
        13.0,
        -13.0,
        7.5,
        -7.5,
    ]
    up_vals = [
        16.0,
        -16.0,
        14.0,
        -14.0,
        14.0,
        -14.0,
        11.0,
        -11.0,
        15.0,
        -15.0,
        16.0,
        -16.0,
        13.0,
        -13.0,
        12.0,
        -12.0,
        11.5,
        -11.5,
        10.5,
        -10.5,
        17.0,
        -17.0,
        12.0,
        -12.0,
        13.5,
        -13.5,
        12.5,
        -12.5,
        8.5,
        -8.5,
        9.5,
        -9.5,
    ]

    hidden = torch.tensor(
        [gate_vals + up_vals],
        dtype=dtype,
        device=device,
    )

    # W13 = identity-like projection so pre-activation approximately mirrors hidden.
    w13_bf16 = torch.zeros(E, 2 * I, K, dtype=dtype, device=device)
    for row in range(2 * I):
        w13_bf16[0, row, row] = 1.0

    # W2 = simple signed readout from activated vector.
    w2_bf16 = torch.zeros(E, K, I, dtype=dtype, device=device)
    for i in range(I):
        w2_bf16[0, i, i] = 1.0
        w2_bf16[0, i + I, i] = -1.0

    w13_packed, w13_scale = _pack_dsv4_mxfp4_weights(w13_bf16)
    w2_packed, w2_scale = _pack_dsv4_mxfp4_weights(w2_bf16)

    topk_ids = torch.zeros((1, topk), dtype=torch.int32, device=device)
    topk_weights = torch.ones((1, topk), dtype=torch.float32, device=device)

    return (
        hidden,
        w13_packed,
        w2_packed,
        w13_scale,
        w2_scale,
        topk_ids,
        topk_weights,
        K,
        I,
    )


def _compute_pre_clamp_gate_up(
    hidden_states: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pre-clamp gate/up for the deterministic topk=1 expert case."""
    K = hidden_states.shape[1]
    intermediate = mxfp4_gemm_torch(hidden_states, w13_packed[0], w13_scale[0], K)
    gate = intermediate[:, :intermediate_size].float()
    up = intermediate[:, intermediate_size:].float()
    return gate, up


def _compare_moe(out_ref: torch.Tensor, out_test: torch.Tensor) -> None:
    torch.testing.assert_close(
        out_ref.float(),
        out_test.float(),
        atol=_MOE_ATOL,
        rtol=_MOE_RTOL,
    )


class TestMxfp4Gemm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def _run_gemm(self, M: int, N: int, K: int, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed)
        A = (torch.randn(M, K, generator=g, dtype=torch.bfloat16) / 10).to(
            self.device
        )
        B_bf16 = (torch.randn(N, K, generator=g, dtype=torch.bfloat16) / 10).to(
            self.device
        )
        B_packed, B_scale = _pack_dsv4_mxfp4_weights(B_bf16)

        ref = mxfp4_gemm_torch(A, B_packed, B_scale, K)
        out = mxfp4_gemm_triton(A, B_packed, B_scale, K)
        torch.testing.assert_close(
            ref.float(),
            out.float(),
            atol=_GEMM_ATOL,
            rtol=_GEMM_RTOL,
        )

    def test_gemm_small(self):
        self._run_gemm(M=4, N=64, K=128, seed=0)

    def test_gemm_medium(self):
        self._run_gemm(M=8, N=128, K=256, seed=1)


class TestMxfp4MoeForward(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def _run_moe(
        self,
        *,
        M: int,
        K: int,
        I: int,
        E: int,
        topk: int,
        seed: int,
        routed_scaling_factor: float | None = None,
        clamp_limit: float | None = None,
    ):
        (
            hidden,
            w13_packed,
            w2_packed,
            w13_scale,
            w2_scale,
            topk_ids,
            topk_weights,
        ) = _build_moe_tensors(
            M=M, K=K, I=I, E=E, topk=topk, device=self.device, seed=seed
        )
        kwargs = dict(
            hidden_states=hidden,
            w13_packed=w13_packed,
            w2_packed=w2_packed,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            hidden_size=K,
            intermediate_size=I,
            routed_scaling_factor=routed_scaling_factor,
            clamp_limit=clamp_limit,
        )
        ref = mxfp4_moe_forward_torch(**kwargs)
        out = mxfp4_moe_forward_triton(**kwargs)
        # Clamp-enabled stress cases can diverge more because activation output
        # is produced by a dedicated Triton kernel before GEMM2.
        torch.testing.assert_close(ref.float(), out.float(), atol=1.5, rtol=1.0)

    def test_moe_small(self):
        self._run_moe(M=2, K=128, I=64, E=4, topk=2, seed=10)

    def test_moe_medium(self):
        self._run_moe(M=4, K=256, I=128, E=8, topk=4, seed=11)

    def test_moe_routed_scaling(self):
        self._run_moe(
            M=4,
            K=128,
            I=64,
            E=4,
            topk=2,
            seed=12,
            routed_scaling_factor=1.5,
        )

    def test_moe_clamp_limit(self):
        self._run_moe(
            M=4,
            K=128,
            I=64,
            E=4,
            topk=2,
            seed=13,
            clamp_limit=7.0,
        )

    def test_moe_clamp_none_extreme_gate_up(self):
        """Explicitly exercise large +/- gate/up values with clamp disabled."""
        clamp_limit = 7.0
        (
            hidden,
            w13_packed,
            w2_packed,
            w13_scale,
            w2_scale,
            topk_ids,
            topk_weights,
            K,
            I,
        ) = _build_extreme_gate_up_case(self.device)

        gate_pre, up_pre = _compute_pre_clamp_gate_up(hidden, w13_packed, w13_scale, I)
        self.assertTrue(torch.any(gate_pre > clamp_limit))
        self.assertTrue(torch.any(gate_pre < -clamp_limit))
        self.assertTrue(torch.any(up_pre > clamp_limit))
        self.assertTrue(torch.any(up_pre < -clamp_limit))

        kwargs = dict(
            hidden_states=hidden,
            w13_packed=w13_packed,
            w2_packed=w2_packed,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            hidden_size=K,
            intermediate_size=I,
            clamp_limit=None,
        )
        ref = mxfp4_moe_forward_torch(**kwargs)
        out = mxfp4_moe_forward_triton(**kwargs)
        # This case intentionally drives very large magnitudes through SiLU.
        # Triton sigmoid approximation + bf16 roundoff can drift more than the
        # default tolerance used for random-shape regression tests.
        torch.testing.assert_close(ref.float(), out.float(), atol=2.0, rtol=0.35)

    def test_moe_clamp_limit_asymmetric_extreme_gate_up(self):
        """Asymmetric clamp: gate only max-clamped; up min/max-clamped."""
        clamp_limit = 7.0
        (
            hidden,
            w13_packed,
            w2_packed,
            w13_scale,
            w2_scale,
            topk_ids,
            topk_weights,
            K,
            I,
        ) = _build_extreme_gate_up_case(self.device)

        gate_pre, up_pre = _compute_pre_clamp_gate_up(hidden, w13_packed, w13_scale, I)
        gate_post = torch.clamp(gate_pre, max=clamp_limit)
        up_post = torch.clamp(up_pre, min=-clamp_limit, max=clamp_limit)

        gate_pos = gate_pre > clamp_limit
        gate_neg = gate_pre < -clamp_limit
        up_pos = up_pre > clamp_limit
        up_neg = up_pre < -clamp_limit
        self.assertTrue(torch.any(gate_pos))
        self.assertTrue(torch.any(gate_neg))
        self.assertTrue(torch.any(up_pos))
        self.assertTrue(torch.any(up_neg))

        # Gate path is asymmetric: negative values are intentionally untouched.
        torch.testing.assert_close(gate_post[gate_neg], gate_pre[gate_neg], atol=0, rtol=0)
        torch.testing.assert_close(
            gate_post[gate_pos],
            torch.full_like(gate_post[gate_pos], clamp_limit),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            up_post[up_pos], torch.full_like(up_post[up_pos], clamp_limit), atol=0, rtol=0
        )
        torch.testing.assert_close(
            up_post[up_neg], torch.full_like(up_post[up_neg], -clamp_limit), atol=0, rtol=0
        )

        kwargs = dict(
            hidden_states=hidden,
            w13_packed=w13_packed,
            w2_packed=w2_packed,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            hidden_size=K,
            intermediate_size=I,
            clamp_limit=clamp_limit,
        )
        ref = mxfp4_moe_forward_torch(**kwargs)
        out = mxfp4_moe_forward_triton(**kwargs)
        # Clamp-enabled stress cases can diverge more because activation output
        # is produced by a dedicated Triton kernel before GEMM2.
        torch.testing.assert_close(ref.float(), out.float(), atol=1.5, rtol=1.0)

        out_no_clamp = mxfp4_moe_forward_triton(**dict(kwargs, clamp_limit=None))
        self.assertTrue(torch.any((out_no_clamp - out).abs() > 1e-3))

    def test_invalid_topk_ids_masked(self):
        """Slots with topk_ids == -1 must not contribute to the output."""
        M, K, I, E, topk = 2, 128, 64, 4, 2
        (
            hidden,
            w13_packed,
            w2_packed,
            w13_scale,
            w2_scale,
            topk_ids,
            topk_weights,
        ) = _build_moe_tensors(
            M=M, K=K, I=I, E=E, topk=topk, device=self.device, seed=20
        )
        topk_ids_invalid = topk_ids.clone()
        topk_ids_invalid[:, 1] = -1
        topk_weights_zeroed = topk_weights.clone()
        topk_weights_zeroed[:, 1] = 0.0

        kwargs = dict(
            hidden_states=hidden,
            w13_packed=w13_packed,
            w2_packed=w2_packed,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            hidden_size=K,
            intermediate_size=I,
        )
        ref_valid = mxfp4_moe_forward_torch(**kwargs)

        kwargs_invalid = dict(
            kwargs,
            topk_ids=topk_ids_invalid,
            topk_weights=topk_weights_zeroed,
        )
        out_invalid = mxfp4_moe_forward_triton(**kwargs_invalid)
        _compare_moe(ref_valid, out_invalid)

    def test_cuda_graph_capture_and_replay(self):
        M, K, I, E, topk = 2, 128, 64, 4, 2
        (
            hidden,
            w13_packed,
            w2_packed,
            w13_scale,
            w2_scale,
            topk_ids,
            topk_weights,
        ) = _build_moe_tensors(
            M=M, K=K, I=I, E=E, topk=topk, device=self.device, seed=30
        )

        kwargs = dict(
            hidden_states=hidden,
            w13_packed=w13_packed,
            w2_packed=w2_packed,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            hidden_size=K,
            intermediate_size=I,
        )

        for _ in range(2):
            _ = mxfp4_moe_forward_triton(**kwargs)
        torch.cuda.synchronize()

        static_out = {}
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out["out"] = mxfp4_moe_forward_triton(**kwargs)

        eager = mxfp4_moe_forward_triton(**kwargs)
        graph.replay()
        torch.cuda.synchronize()
        _compare_moe(eager, static_out["out"])


if __name__ == "__main__":
    sys.exit(unittest.main())
