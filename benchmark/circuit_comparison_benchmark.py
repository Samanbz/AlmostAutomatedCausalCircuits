import multiprocessing

import torch
import torch.utils.benchmark as benchmark

from src.compilation.folded_circuit import CircuitFolder
from src.compilation.fused_circuit import CircuitFuser
from src.compilation.monarch_circuit import MonarchCircuit
from src.compilation.tensorized_circuit import TensorizedCircuit
from src.construction import (
    CircuitBuilder,
    RegionGraphBuilder,
    construct_optimal_md_vtree,
    generate_random_scm,
)
from src.symbolic import GaussianDistribution


def run_benchmark_timer_process(q, timer):
    """
    Module-level function to run the timer.
    Must be at the module level so it can be pickled by 'spawn'.
    """
    try:
        res = timer.blocked_autorange(min_run_time=0.2)
        q.put(res.mean * 1000)
    except Exception as e:
        q.put(e)


def generate_circuit(num_features, h):
    # Ensure reproducibility
    torch.manual_seed(42)

    scm = generate_random_scm(n_nodes=num_features, expected_degree=2.0)
    data = scm.sample(1000)
    data_tensor = torch.from_numpy(data.values.copy()).float()

    # Generate simple chain of MD sets
    md_sets = []
    current_set = set()
    for i in range(min(num_features, 4)):
        current_set.add(i)
        if len(current_set) >= 2:
            md_sets.append(set(current_set))
    if not md_sets and num_features >= 2:
        md_sets = [{0, 1}]

    md_vtree = construct_optimal_md_vtree(data_tensor, md_sets=md_sets)
    input_dists = {
        v: GaussianDistribution(var=v, mean=0.0, stddev=1.0) for v in range(num_features)
    }
    region_graph = RegionGraphBuilder(md_vtree=md_vtree, input_dists=input_dists).build()
    spn = CircuitBuilder(rg=region_graph, h=h, input_dists=input_dists).build()

    folded = CircuitFolder(spn).build()
    fused = CircuitFuser(folded).build()

    return fused


def run_circuit_benchmark():
    configs = [
        # (num_features, h, batch_size)
        (4, 128, 128),
        (8, 128, 128),
        (16, 128, 128),
        (32, 128, 128),
        (64, 128, 128),
        (128, 128, 128),
        (16, 64, 128),
        (16, 128, 128),
        (16, 256, 128),
        (16, 512, 128),
    ]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Running circuit comparison benchmark on {device}\n")

    print(
        f"{'Vars':<6} | {'h':<6} | {'Batch':<6} | {'Model':<12} | {'Mem (MB)':<10} | {'Time (ms)':<10}"
    )
    print("-" * 67)

    for num_features, h, batch_size in configs:
        try:
            fused = generate_circuit(num_features, h)

            tc = TensorizedCircuit(fused).to(device)
            mc = MonarchCircuit(fused).to(device)

            x = torch.randn(batch_size, num_features, device=device)

            # Compute memory (MB)
            tc_mem = sum(p.nelement() * p.element_size() for p in tc.parameters()) / (1024 * 1024)
            mc_mem = sum(p.nelement() * p.element_size() for p in mc.parameters()) / (1024 * 1024)

            # Time dense TensorizedCircuit
            t_tc = benchmark.Timer(
                stmt="tc(x)" + ("; torch.mps.synchronize()" if device == "mps" else ""),
                globals={"tc": tc, "x": x, "torch": torch},
            )
            try:
                q = multiprocessing.Queue()
                p = multiprocessing.Process(target=run_benchmark_timer_process, args=(q, t_tc))
                p.start()
                p.join(10)

                if p.is_alive():
                    p.terminate()
                    p.join()
                    tc_time = float("inf")
                else:
                    tc_time = q.get()
                    if isinstance(tc_time, Exception):
                        raise tc_time
            except Exception:
                b_tc = t_tc.blocked_autorange(min_run_time=0.2)
                tc_time = b_tc.mean * 1000

            # Time MonarchCircuit
            t_mc = benchmark.Timer(
                stmt="mc(x)" + ("; torch.mps.synchronize()" if device == "mps" else ""),
                globals={"mc": mc, "x": x, "torch": torch},
            )
            try:
                q = multiprocessing.Queue()
                p = multiprocessing.Process(target=run_benchmark_timer_process, args=(q, t_mc))
                p.start()
                p.join(10)

                if p.is_alive():
                    p.terminate()
                    p.join()
                    mc_time = float("inf")
                else:
                    mc_time = q.get()
                    if isinstance(mc_time, Exception):
                        raise mc_time
            except Exception:
                b_mc = t_mc.blocked_autorange(min_run_time=0.2)
                mc_time = b_mc.mean * 1000

            print(
                f"{num_features:<6} | {h:<6} | {batch_size:<6} | {'Tensorized':<12} | {tc_mem:<10.4f} | {tc_time:<10.3f}"
            )
            print(
                f"{num_features:<6} | {h:<6} | {batch_size:<6} | {'Monarch':<12} | {mc_mem:<10.4f} | {mc_time:<10.3f}"
            )
            print("-" * 67)

        except Exception as e:
            print(f"{num_features:<6} | {h:<6} | {batch_size:<6} | {'FAILED':<12} | {str(e)}")
            print("-" * 67)


if __name__ == "__main__":
    # Force multiprocessing to use 'spawn' instead of 'fork'
    multiprocessing.set_start_method("spawn", force=True)
    run_circuit_benchmark()
