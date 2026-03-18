import time

import numpy as np
import torch
from _bootstrap import ensure_project_root_on_path


ensure_project_root_on_path()


from src.compilation.monarch_circuit import MonarchCircuit  # noqa: E402
from src.compilation.tensorized_circuit import TensorizedCircuit  # noqa: E402
from src.construction.random_vtree import construct_random_vtree  # noqa: E402
from src.construction.rat_spn import create_rat_spn  # noqa: E402
from src.construction.xpc import (  # noqa: E402
    construct_random_data_region_graph,
    construct_spn_from_region_graph,
)


def measure_performance(model, data, name="Model", warmup=5, iters=20):
    device = data.device
    model = model.to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    def reset_mem():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def get_mem():
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated() / (1024**2)  # MB
        elif device.type == "mps":
            return torch.mps.current_allocated_memory() / (1024**2)  # MB
        return 0

    # Warmup
    for _ in range(warmup):
        _ = model(data)
    sync()

    reset_mem()

    start_event = time.perf_counter()
    for _ in range(iters):
        _ = model(data)
    sync()
    end_event = time.perf_counter()

    total_time = end_event - start_event
    avg_time_ms = (total_time / iters) * 1000.0
    mem_mb = get_mem()

    print(f"{name:45s} | Time (ms/fwd): {avg_time_ms:7.3f} | Max Mem (MB): {mem_mb:7.3f}")


def run_benchmarks():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Running benchmarks on: {device}")

    batch_size = 512
    num_features = 16

    print("\n--- Generating RAT-SPN (Dense Connections) ---")
    from src.symbolic.arithmetic.distributions import GaussianDistribution

    input_dists = {i: GaussianDistribution(i, 0.0, 1.0) for i in range(num_features)}
    rat_spn = create_rat_spn(
        num_features=num_features,
        num_classes=1,
        depth=4,
        num_repetitions=2,
        num_sums=16,
        num_inputs=16,
        input_dists=input_dists,
    )

    dummy_data = torch.randn((batch_size, num_features), device=device)

    print("Compiling Tensorized Circuit (RAT-SPN)...")
    tc_rat = TensorizedCircuit(rat_spn)

    print("Compiling Monarch Circuit (RAT-SPN, respect_sparsity=False)...")
    mc_rat = MonarchCircuit(rat_spn, respect_sparsity=False)

    measure_performance(tc_rat, dummy_data, "TensorizedCircuit (RAT-SPN)")
    measure_performance(mc_rat, dummy_data, "MonarchCircuit (RAT-SPN)")

    print("\n--- Generating XPC (Sparse Regional Connections) ---")
    data = np.random.randn(batch_size, num_features)
    scopes = list(range(num_features))
    vtree = construct_random_vtree(scopes)

    print("Constructing Data Region Graph...")
    region_graph = construct_random_data_region_graph(
        data=data, input_dists=input_dists, min_examples=10, split_arity=4, var_decomp=vtree
    )

    print("Extracting SPN from Region Graph...")
    xpc_spn = construct_spn_from_region_graph(region_graph, input_dists=input_dists)

    print("Compiling Tensorized Circuit (XPC)...")
    try:
        tc_xpc = TensorizedCircuit(xpc_spn)
        measure_tc = True
    except Exception as e:
        print(
            f"    [!] Failed to compile TensorizedCircuit for XPC (DAG contains skip connections): {e}"
        )
        measure_tc = False

    print("Compiling Monarch Circuit (XPC, respect_sparsity=True)...")
    try:
        mc_xpc = MonarchCircuit(xpc_spn, respect_sparsity=True)
        measure_mc = True
    except Exception as e:
        print(f"    [!] Failed to compile MonarchCircuit for XPC: {e}")
        measure_mc = False

    dummy_data_xpc = torch.tensor(data, dtype=torch.float32, device=device)

    if measure_tc:
        measure_performance(tc_xpc, dummy_data_xpc, "TensorizedCircuit (XPC)")
    if measure_mc:
        measure_performance(mc_xpc, dummy_data_xpc, "MonarchCircuit (XPC)")


if __name__ == "__main__":
    run_benchmarks()
