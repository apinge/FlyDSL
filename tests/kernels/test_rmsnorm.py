#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
RMSNorm Operator Test
Implementation of a Block-wise RMSNorm:
- Grid: (M, 1, 1) -> One block per row
- Block: (N, 1, 1) -> Threads handle columns
- Shared Memory: Used for reduction (sum of squares)

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
"""

import os

from tests.test_common import run_perftest
from tests.kernels.benchmark_common import (
    PerfRow,
    bench_gpu_us_torch,
    maybe_enable_aiter,
    print_perf_table,
)
import pytest
try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

import torch.nn.functional as F

DTYPE_FP32 = torch.float32
DTYPE_FP16 = torch.float16
DTYPE_BF16 = torch.bfloat16

EPS: float = 1e-5
from kernels.rmsnorm_kernel import (
    build_rmsnorm_module,
    KERNEL_NAME as RMSNORM_KERNEL_NAME,
    BLOCK_THREADS,
)

WARMUP_ITERS = 10
BENCH_ITERS = 100


def run_torch(input, weight, eps, residual=None):
    if residual is None:
        residual_out = None
        output = F.rms_norm(
            input=input, normalized_shape=(input.shape[-1],), weight=weight, eps=eps
        )
    else:
        residual_out = input + residual
        output = F.rms_norm(
            input=residual_out,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            eps=eps,
        )
    return output, residual_out


def run_test(M: int, N: int, dtype: str = "f32", with_residual: bool = False):
    mode = "RMSNorm + residual" if with_residual else "RMSNorm"
    print(f"\nTesting {mode} (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_rmsnorm_module(M, N, dtype, with_residual=with_residual)
    except Exception as e:
        print(f"[FAIL] Compile failed for (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False, None
    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    residual_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)

    if dtype == "f32":
        input_dev = input_t.contiguous()
        gamma_dev = gamma_t.contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP32)
        residual_dev = residual_t.contiguous() if with_residual else None
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 1e-4
    elif dtype == "f16":
        input_dev = input_t.to(DTYPE_FP16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_FP16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP16)
        residual_dev = residual_t.to(DTYPE_FP16).contiguous() if with_residual else None
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 1e-2
    elif dtype == "bf16":
        input_dev = input_t.to(DTYPE_BF16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_BF16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_BF16)
        residual_dev = residual_t.to(DTYPE_BF16).contiguous() if with_residual else None
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    # 原地写 Residual：benchmark 会多次 launch，每次须从初始 residual 恢复，否则会反复 input+(input+residual)。
    if with_residual:
        residual_dev_initial = residual_dev.clone()

    # PyTorch reference in fp32. With residual: golden 必须与内核一致——对 element 张量先 cast 再 extf 相加，
    # 不能用全精度 residual_t，否则与 fp16/bf16 residual buffer 上的融合不一致。
    if with_residual:
        res_ref = residual_dev.to(DTYPE_FP32)
    else:
        res_ref = None
    expected, residual_sum_ref = run_torch(input_ref, gamma_ref, EPS, residual=res_ref)
    expected = expected.to(DTYPE_FP32)
    if residual_sum_ref is not None:
        residual_sum_ref = residual_sum_ref.to(DTYPE_FP32)

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    def kernel_launch():
        if with_residual:
            residual_dev.copy_(residual_dev_initial)
            launch_fn(
                input_dev,
                residual_dev,
                gamma_dev,
                output_dev,
                M,
                stream=stream,
            )
        else:
            launch_fn(input_dev, gamma_dev, output_dev, M, stream=stream)

    # run_perftest returns (data, avg_us)
    _, avg_us = run_perftest(lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS)
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0

    # Bandwidth estimate (rough): read input + gamma [+ residual]; write output [+ residual sum].
    elem_bytes = 4 if dtype == "f32" else 2
    reads = 3 if with_residual else 2
    writes = 2 if with_residual else 1
    total_bytes = (reads + writes) * M * N * elem_bytes
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL rmsnorm gpu: {flydsl_gpu_us:.1f} us")

    # Verification: kernel output vs golden (both compared in fp32).
    output_actual_fp32 = output_dev.to(DTYPE_FP32)
    error = (output_actual_fp32 - expected).abs().max().item()
    print(f"Max absolute error (output): {error:.2e} (atol={atol})")

    if with_residual:
        residual_inplace_fp32 = residual_dev.to(DTYPE_FP32)
        err_res = (residual_inplace_fp32 - residual_sum_ref).abs().max().item()
        # bf16 写回 fused sum 时舍入与参考 fp32 累加再 cast 会有略大偏差。
        atol_res = max(atol, 3e-2) if dtype == "bf16" else atol
        print(f"Max absolute error (input+residual): {err_res:.2e} (atol={atol_res})")
        ok = error < atol and err_res < atol_res
    else:
        err_res = 0.0
        ok = error < atol

    if ok:
        print("PASSED")
    else:
        print("FAILED")
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(output_actual_fp32[0, :5])
        if with_residual:
            print("Residual sum Expected first row:")
            print(residual_sum_ref[0, :5])
            print("Residual (in-place) Actual first row:")
            print(residual_inplace_fp32[0, :5])
    return ok, flydsl_gpu_us

def test_all():
    print("="*80)
    print("Running RMSNorm Tests")
    print("="*80)

    shapes_env = os.environ.get("ROCDSL_RMSNORM_SHAPES", "").strip()
    if shapes_env:
        configs = []
        for part in shapes_env.split(";"):
            p = part.strip()
            if not p:
                continue
            m_s, n_s, dt = [x.strip() for x in p.split(",")]
            configs.append((int(m_s), int(n_s), dt))
    else:
        # Prefer N multiples of BLOCK_THREADS*VEC_WIDTH (=2048) to exercise the fast path.
        configs = [
            # (64, 256, "f32"),     # Aligned
            # (128, 1024, "f32"),   # Aligned
            # (32, 128, "f16"),     # Aligned
            # (64, 2000, "f32"),    # Unaligned (tail handling)
            # (16, 512, "bf16"),    # BF16
            # (1024, 8192, "bf16"), # BF16
            # (32768, 8192, "bf16"), # 原始版本
            (32768, 4096, "f16"),
            # (8000, 4096, "bf16"),
            # (32768, 4096, "bf16"),
            # (2, 4096, "bf16"),
            # (8000, 256, "bf16"),
            # (32768, 256, "bf16"),
            # (2, 256, "bf16"),
        ]

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []

    with_residual = os.environ.get("ROCDSL_RMSNORM_WITH_RESIDUAL", "0") == "1"

    failures = 0
    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_test(M, N, dtype, with_residual=with_residual)
        if not ok:
            failures += 1

        if do_compare:
            import torch
            aiter_us = None
            if maybe_enable_aiter():
                try:
                    from aiter.ops.triton.rmsnorm import rms_norm as aiter_rms_norm
                    x = torch.randn((M, N), device="cuda", dtype=DTYPE_BF16 if dtype == "bf16" else (DTYPE_FP16 if dtype == "f16" else DTYPE_FP32))
                    w = torch.rand((N,), device="cuda", dtype=x.dtype)

                    def run_aiter():
                        aiter_rms_norm(x, w, EPS)

                    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] AIter rmsnorm gpu: {aiter_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] AIter rmsnorm skipped: {type(e).__name__}: {e!r}")

            perf_rows.append(PerfRow(op="rmsnorm", shape=f"{M}x{N}", dtype=dtype, flydsl_gpu_us=flydsl_gpu_us, aiter_gpu_us=aiter_us))

    print("\n" + "="*80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("="*80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    # Ensure a non-zero exit code on failure for shell wrappers.
    if failures != 0:
        raise SystemExit(1)

if __name__ == "__main__":
    test_all()

