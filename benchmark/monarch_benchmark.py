import torch
import torch.utils.benchmark as benchmark
from src.compilation.monarch import MonarchMatrix  # noqa: E402


def run_benchmark_config(config):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    b, c, k, b1 = config["b"], config["c"], config["k"], config["b1"]
    batch_size = config.get("batch_size", 128)

    out_dim = b * c
    in_dim = k * b1

    dense_matrix = torch.randn(out_dim, in_dim, device=device)
    dense_matrix_t = dense_matrix.t().contiguous()

    L = torch.randn(b, c, k, device=device)
    R = torch.randn(k, b, b1, device=device)
    monarch_matrix = MonarchMatrix(b=b, c=c, k=k, b1=b1, L=L, R=R).to(device)
    x = torch.randn(batch_size, in_dim, device=device)

    # memory calculation
    dense_mem_mb = (dense_matrix.nelement() * dense_matrix.element_size()) / (1024 * 1024)
    monarch_mem = sum(p.nelement() * p.element_size() for p in monarch_matrix.parameters())
    monarch_mem_mb = monarch_mem / (1024 * 1024)

    # benchmark using torch.utils.benchmark
    t_dense = benchmark.Timer(
        stmt="torch.matmul(x, w)" + ("; torch.mps.synchronize()" if device == "mps" else ""),
        globals={"x": x, "w": dense_matrix_t, "m": monarch_matrix, "torch": torch},
    )
    t_monarch = benchmark.Timer(
        stmt="m(x)" + ("; torch.mps.synchronize()" if device == "mps" else ""),
        globals={"x": x, "w": dense_matrix_t, "m": monarch_matrix, "torch": torch},
    )

    b_dense = t_dense.blocked_autorange(min_run_time=0.2)
    b_monarch = t_monarch.blocked_autorange(min_run_time=0.2)

    return {
        "dense_latency_ms": b_dense.mean * 1000,
        "monarch_latency_ms": b_monarch.mean * 1000,
        "dense_mem_mb": dense_mem_mb,
        "monarch_mem_mb": monarch_mem_mb,
    }


def benchmark_monarch(device="cpu"):
    print(f"Running benchmark on {device}")

    # Configurations to benchmark: (b, c, k, b1)
    configs = [
        # Square benchmarks
        (16, 16, 16, 16),  # 256 x 256
        (32, 32, 32, 32),  # 1024 x 1024
        (64, 64, 64, 64),  # 4096 x 4096
        (128, 128, 128, 128),  # 16384 x 16384
        # Rectangular benchmarks (expansion/contraction mapping to common transformer MLP dimensions)
        (32, 32, 64, 64),  # 1024 x 4096 (1 -> 4 expansion)
        (64, 64, 32, 32),  # 4096 x 1024 (4 -> 1 contraction)
        (64, 64, 128, 128),  # 4096 x 16384 (1 -> 4 expansion)
        (128, 128, 64, 64),  # 16384 x 4096 (4 -> 1 contraction)
    ]
    batch_size = 128

    results = []

    for b, c, k, b1 in configs:
        out_dim = b * c
        in_dim = k * b1

        # Dense Matrix
        dense_matrix = torch.randn(out_dim, in_dim, device=device)
        dense_matrix_t = dense_matrix.t().contiguous()

        # Monarch Matrix (Initialize with random L and R)
        L = torch.randn(b, c, k, device=device)
        R = torch.randn(k, b, b1, device=device)
        monarch_matrix = MonarchMatrix(b=b, c=c, k=k, b1=b1, L=L, R=R).to(device)

        # Input vector
        x = torch.randn(batch_size, in_dim, device=device)

        # Compute Parameter Memory
        dense_mem_mb = (dense_matrix.nelement() * dense_matrix.element_size()) / (1024 * 1024)
        monarch_mem = sum(p.nelement() * p.element_size() for p in monarch_matrix.parameters())
        monarch_mem_mb = monarch_mem / (1024 * 1024)

        mem_saving = dense_mem_mb / monarch_mem_mb if monarch_mem_mb > 0 else float("inf")

        print(f"\n{'=' * 60}")
        print(f"Size: {out_dim}x{in_dim} (b={b}, c={c}, k={k}, b1={b1}) | Batch={batch_size}")
        print(
            f"Memory -> Dense: {dense_mem_mb:.4f} MB | Monarch: {monarch_mem_mb:.4f} MB | Saving: {mem_saving:.2f}x"
        )
        print(f"{'=' * 60}")

        # Setup statements (handling MPS sync manually since PyTorch Timer doesn't auto-sync it yet)
        stmt_dense = "torch.matmul(x, w)"
        stmt_monarch = "m(x)"

        if device == "mps":
            stmt_dense += "; torch.mps.synchronize()"
            stmt_monarch += "; torch.mps.synchronize()"

        t_dense = benchmark.Timer(
            stmt=stmt_dense,
            globals={"x": x, "w": dense_matrix_t, "m": monarch_matrix, "torch": torch},
            label=f"Matmul ({out_dim}x{in_dim})",
            sub_label="Dense",
            description="Forward Pass",
        )

        t_monarch = benchmark.Timer(
            stmt=stmt_monarch,
            globals={"x": x, "w": dense_matrix_t, "m": monarch_matrix, "torch": torch},
            label=f"Matmul ({out_dim}x{in_dim})",
            sub_label="Monarch",
            description="Forward Pass",
        )

        # Perform measurements
        b_dense = t_dense.blocked_autorange(min_run_time=0.2)
        b_monarch = t_monarch.blocked_autorange(min_run_time=0.2)

        # Add to compare list for a final summary table
        results.append(b_dense)
        results.append(b_monarch)

    print("\n\n=== Final Summary Table ===")
    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    benchmark_monarch(device)
