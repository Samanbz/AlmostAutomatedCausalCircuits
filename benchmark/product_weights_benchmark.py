"""Benchmark ProductWeights.forward() — materialized vs factored contraction."""

import gc
import itertools
import time
import tracemalloc

import torch

from src.symbolic.arithmetic.weights import DenseWeights, ProductWeights


def benchmark_forward(H, G, B, expand_L, expand_R, expand_U, n_warmup=3, n_runs=10):
    """Benchmark ProductWeights.forward for given dimensions."""
    w1 = DenseWeights(torch.randn(G, H, G, H, G, H))
    w2 = DenseWeights(torch.randn(G, H, G, H, G, H))
    pw = ProductWeights(w1, w2, expand_U=expand_U, expand_L=expand_L, expand_R=expand_R)

    shape = pw.shape
    _, _, G_L, L, G_R, R = shape
    left = torch.randn(B, G_L, L)
    right = torch.randn(B, G_R, R)

    # Warmup
    for _ in range(n_warmup):
        _ = pw.forward(left, right)

    # Timed runs
    gc.collect()

    tracemalloc.start()

    times = []
    for _ in range(n_runs):
        gc.collect()
        tracemalloc.clear_traces()
        t0 = time.perf_counter()
        out = pw.forward(left, right)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_time = sum(times) / len(times)
    # Compute materialized weight tensor size
    mat_shape = pw.shape
    mat_elements = 1
    for s in mat_shape:
        mat_elements *= s

    return {
        "H": H,
        "G": G,
        "B": B,
        "expand_L": expand_L,
        "expand_R": expand_R,
        "expand_U": expand_U,
        "output_shape": list(out.shape),
        "weight_shape": list(mat_shape),
        "weight_elements": mat_elements,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min(times) * 1000,
        "peak_mem_bytes": mem_peak,
        "peak_mem_MB": mem_peak / (1024 * 1024),
    }


def correctness_check(H, G, B, expand_L, expand_R, expand_U):
    """Check that forward produces correct output by comparing against log_weights materialization."""
    torch.manual_seed(42)
    w1 = DenseWeights(torch.randn(G, H, G, H, G, H))
    w2 = DenseWeights(torch.randn(G, H, G, H, G, H))
    pw = ProductWeights(w1, w2, expand_U=expand_U, expand_L=expand_L, expand_R=expand_R)

    shape = pw.shape
    _, _, G_L, L, G_R, R = shape
    left = torch.randn(B, G_L, L)
    right = torch.randn(B, G_R, R)

    out = pw.forward(left, right)

    # Reference: use DenseWeights with the fully materialized log_weights
    ref_w = DenseWeights(pw.log_weights)
    ref_out = ref_w.forward(left, right)

    max_diff = (out - ref_out).abs().max().item()
    return max_diff


def main():
    print("=" * 80)
    print("ProductWeights.forward() Benchmark")
    print("=" * 80)

    # Correctness check first
    print("\n--- Correctness Check ---")
    for expand_L, expand_R, expand_U in itertools.product([True, False], repeat=3):
        diff = correctness_check(4, 1, 8, expand_L, expand_R, expand_U)
        status = "PASS" if diff < 1e-4 else f"FAIL (diff={diff:.6f})"
        print(f"  expand_L={expand_L}, expand_R={expand_R}, expand_U={expand_U}: {status}")

    # Performance benchmarks
    print(
        "\n--- Performance Benchmark (Full Kronecker: expand_L=True, expand_R=True, expand_U=True) ---"
    )
    print(
        f"{'H':>4} {'G':>3} {'B':>4} {'weight_shape':>30} {'weight_elems':>14} "
        f"{'avg_ms':>10} {'min_ms':>10} {'peak_MB':>10}"
    )
    print("-" * 100)

    configs = [
        # (H, G, B)
        (2, 1, 32),
        (4, 1, 32),
        (4, 2, 32),
        (6, 1, 32),
        (8, 1, 32),
        (8, 2, 16),
    ]

    results = []
    for H, G, B in configs:
        try:
            r = benchmark_forward(H, G, B, expand_L=True, expand_R=True, expand_U=True)
            results.append(r)
            print(
                f"{r['H']:>4} {r['G']:>3} {r['B']:>4} {str(r['weight_shape']):>30} "
                f"{r['weight_elements']:>14,} {r['avg_time_ms']:>10.2f} "
                f"{r['min_time_ms']:>10.2f} {r['peak_mem_MB']:>10.2f}"
            )
        except Exception as e:
            print(f"{H:>4} {G:>3} {B:>4} {'FAILED':>30} -- {e}")

    # Also benchmark Hadamard case for comparison
    print(
        "\n--- Performance Benchmark (Hadamard: expand_L=False, expand_R=False, expand_U=False) ---"
    )
    print(
        f"{'H':>4} {'G':>3} {'B':>4} {'weight_shape':>30} {'weight_elems':>14} "
        f"{'avg_ms':>10} {'min_ms':>10} {'peak_MB':>10}"
    )
    print("-" * 100)

    for H, G, B in configs:
        try:
            r = benchmark_forward(H, G, B, expand_L=False, expand_R=False, expand_U=False)
            results.append(r)
            print(
                f"{r['H']:>4} {r['G']:>3} {r['B']:>4} {str(r['weight_shape']):>30} "
                f"{r['weight_elements']:>14,} {r['avg_time_ms']:>10.2f} "
                f"{r['min_time_ms']:>10.2f} {r['peak_mem_MB']:>10.2f}"
            )
        except Exception as e:
            print(f"{H:>4} {G:>3} {B:>4} {'FAILED':>30} -- {e}")

    return results


if __name__ == "__main__":
    main()
