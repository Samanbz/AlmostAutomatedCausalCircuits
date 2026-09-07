"""Benchmark the symbolic MD-circuit pipeline.

Times VTtree construction, circuit build, one EM epoch, and a grid (heatmap-like)
evaluation for several ``num_nodes`` and batch sizes.  Records wall time and peak
memory (CUDA ``max_memory_allocated`` when available, else max RSS).

Usage:
    python benchmark/symbolic_benchmark.py
    python benchmark/symbolic_benchmark.py --num_nodes 4 8 16 --device cpu
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import torch

from src.construction.circuit_builder import create_md_circuit
from src.construction.learned_vtree import construct_optimal_md_vtree
from src.symbolic.arithmetic.circuit import eval_circuit
from src.symbolic.arithmetic.nodes import GaussianDistribution
from src.symbolic.arithmetic.nodes.leaf_layer import LogLinearSplineDistribution
from src.symbolic.arithmetic.train import SymbolicEMTrainer


def _peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def build_circuit(num_nodes: int, data: torch.Tensor, prioritize: str, device: torch.device):
    """Build the same 4-variable MD circuit shape used in test_joint_fit.py."""
    var_to_id = {"Z0": 0, "Z1": 1, "X": 2, "Y": 3}
    x_id, y_id = var_to_id["X"], var_to_id["Y"]
    z_ids = [var_to_id["Z0"], var_to_id["Z1"]]
    md_sets = [{x_id} | set(z_ids)]

    data_mean = data.mean(0)
    data_std = data.std(0)

    t0 = time.perf_counter()
    vt = construct_optimal_md_vtree(
        data,
        md_sets,
        prioritize=prioritize,
        keep_together=[(x_id, y_id)],
    )
    vtree_s = time.perf_counter() - t0

    dists = {}
    for vid in range(4):
        if vid == y_id:
            dists[vid] = GaussianDistribution(
                var=vid, base_mean=float(data_mean[vid]), base_stddev=float(data_std[vid]) * 2
            )
        else:
            dists[vid] = LogLinearSplineDistribution(
                var=vid, base_mean=float(data_mean[vid]), base_stddev=float(data_std[vid]) * 2
            )

    t0 = time.perf_counter()
    ac = create_md_circuit(dists, vt, num_nodes=num_nodes, initialize_weights=True)
    ac.to(device)
    build_s = time.perf_counter() - t0
    return ac, vtree_s, build_s


def time_em_epoch(ac, data: torch.Tensor, batch_size: int, device: torch.device) -> float:
    trainer = SymbolicEMTrainer(ac, leaf_lr=0.01)
    N = data.shape[0]
    indices = torch.randperm(N, device=device)
    t0 = time.perf_counter()
    for start in range(0, N, batch_size):
        trainer.em_step(data[indices[start : start + batch_size]], step_size=0.1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - t0


def time_grid_eval(ac, n_vars: int, grid: int, batch_size: int, device: torch.device) -> float:
    """Evaluate the root on a 2D grid, mimicking the heatmap workload."""
    xs = torch.linspace(-3, 3, grid, device=device)
    ys = torch.linspace(-7, 7, grid, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.zeros(grid * grid, n_vars, device=device)
    pts[:, 2] = X.flatten()
    pts[:, 3] = Y.flatten()

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, pts.shape[0], batch_size):
            eval_circuit(ac, pts[i : i + batch_size], verbose=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--n_samples", type=int, default=8192)
    parser.add_argument("--grid", type=int, default=100)
    parser.add_argument("--prioritize", type=str, default="hardware")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--out", type=str, default="benchmark/results_baseline.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = torch.randn(args.n_samples, 4, device=device)

    results = {
        "device": str(device),
        "n_samples": args.n_samples,
        "grid": args.grid,
        "runs": [],
    }

    for num_nodes in args.num_nodes:
        ac, vtree_s, build_s = build_circuit(num_nodes, data, args.prioritize, device)
        for batch_size in args.batch_sizes:
            _reset_peak_memory(device)
            em_s = time_em_epoch(ac, data, batch_size, device)
            em_mem = _peak_memory_bytes(device)

            _reset_peak_memory(device)
            grid_s = time_grid_eval(ac, 4, args.grid, batch_size, device)
            grid_mem = _peak_memory_bytes(device)

            run = {
                "num_nodes": num_nodes,
                "batch_size": batch_size,
                "vtree_s": vtree_s,
                "build_s": build_s,
                "em_epoch_s": em_s,
                "em_peak_mem_bytes": em_mem,
                "grid_eval_s": grid_s,
                "grid_peak_mem_bytes": grid_mem,
            }
            results["runs"].append(run)
            print(
                f"N={num_nodes:3d} B={batch_size:5d} | vtree {vtree_s:.3f}s build {build_s:.3f}s "
                f"EM {em_s:.3f}s ({em_mem / 1e6:.1f}MB) grid {grid_s:.3f}s ({grid_mem / 1e6:.1f}MB)"
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
