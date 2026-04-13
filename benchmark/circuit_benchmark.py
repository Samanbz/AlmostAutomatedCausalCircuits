import math

import numpy as np
import torch

from benchmark.runner import measure_performance
from src.compilation.monarch_circuit import MonarchCircuit  # noqa: E402
from src.compilation.tensorized_circuit import TensorizedCircuit  # noqa: E402
from src.construction.random_vtree import construct_random_md_vtree
from src.construction.rat_spn import create_rat_spn  # noqa: E402
from src.construction.xpc import (
    construct_random_md_data_region_graph,
    construct_spn_from_region_graph,  # noqa: E402
)
from src.symbolic.arithmetic.distributions import GaussianDistribution


def run_benchmark_config(config):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    num_features = config["num_features"]
    batch_size = config.get("batch_size", 512)

    input_dists = {i: GaussianDistribution(i, 0.0, 1.0) for i in range(num_features)}
    rat_depth = int(math.ceil(math.log2(num_features)))
    node_copies = min(4, max(4, int(math.ceil(math.log2(num_features)))))

    metrics = {}

    # RAT-SPN Dense Benchmark
    if num_features < 256:
        try:
            rat_spn = create_rat_spn(
                num_features=num_features,
                num_classes=1,
                depth=rat_depth,
                num_repetitions=2,
                num_sums=node_copies,
                num_inputs=node_copies,
                input_dists=input_dists,
            )

            dummy_data = torch.randn((batch_size, num_features), device=device)

            tc_rat = TensorizedCircuit(rat_spn)
            mc_rat = MonarchCircuit(rat_spn, respect_sparsity=False)

            tc_res = measure_performance(tc_rat, dummy_data, "Tensorized (RAT)", warmup=2, iters=5)
            mc_res = measure_performance(mc_rat, dummy_data, "Monarch (RAT)", warmup=2, iters=5)

            metrics["rat_tc_latency_ms"] = tc_res["latency_ms"]
            metrics["rat_tc_mem_mb"] = tc_res["peak_mem_mb"]
            metrics["rat_mc_latency_ms"] = mc_res["latency_ms"]
            metrics["rat_mc_mem_mb"] = mc_res["peak_mem_mb"]
        except Exception as e:
            metrics["rat_error"] = str(e)

    # XPC Sparse Benchmark
    try:
        data = np.random.randn(batch_size, num_features)
        scopes = list(range(num_features))
        max_size = min(6, max(2, int(math.log2(num_features))))
        md_vtree = construct_random_md_vtree(set(scopes), max_subset_size=max_size)

        region_graph = construct_random_md_data_region_graph(
            data=data,
            input_dists=input_dists,
            min_examples=10,
            split_arity=4,
            md_var_decomp=md_vtree,
        )
        xpc_spn = construct_spn_from_region_graph(region_graph, input_dists=input_dists)
        dummy_data_xpc = torch.tensor(data, dtype=torch.float32, device=device)

        try:
            tc_xpc = TensorizedCircuit(xpc_spn)
            tc_res = measure_performance(
                tc_xpc, dummy_data_xpc, "Tensorized (XPC)", warmup=2, iters=5
            )
            metrics["xpc_tc_latency_ms"] = tc_res["latency_ms"]
            metrics["xpc_tc_mem_mb"] = tc_res["peak_mem_mb"]
        except Exception as e:
            metrics["xpc_tc_error"] = str(e)

        try:
            mc_xpc = MonarchCircuit(xpc_spn, respect_sparsity=True)
            mc_res = measure_performance(mc_xpc, dummy_data_xpc, "Monarch (XPC)", warmup=2, iters=5)
            metrics["xpc_mc_latency_ms"] = mc_res["latency_ms"]
            metrics["xpc_mc_mem_mb"] = mc_res["peak_mem_mb"]
        except Exception as e:
            metrics["xpc_mc_error"] = str(e)

    except Exception as e:
        metrics["xpc_error"] = str(e)

    return metrics
