import time

import numpy as np
import torch

from src.compilation.folded_circuit import CircuitFolder
from src.compilation.fused_circuit import CircuitFuser
from src.construction import (
    CircuitBuilder,
    RegionGraphBuilder,
    construct_optimal_md_vtree,
    generate_random_scm,
)
from src.symbolic import GaussianDistribution


def run_construction_benchmark():
    print(
        f"{'Vars':<6} | {'Init W':<6} | {'h':<4} | {'VTree (s)':<12} | {'RegionGraph (s)':<17} | {'Circuit (s)':<13} | {'Folded (s)':<12} | {'Fused (s)':<11}"
    )
    print("-" * 97)

    # We will test variable counts from 4 up to 1024
    for num_vars in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        h = 16

        try:
            torch.manual_seed(42)
            np.random.seed(42)

            # SCM and Data
            scm = generate_random_scm(n_nodes=num_vars, expected_degree=2.0)
            data = scm.sample(500)
            data_tensor = torch.from_numpy(data.values.copy()).float()

            # 1. VTree
            md_sets = []
            curr_set = set()
            for i in range(0, min(num_vars, 16), 4):
                curr_set.update({i, i + 1})
                md_sets.append(set(curr_set))

            t0 = time.time()
            md_vtree = construct_optimal_md_vtree(data_tensor, md_sets=md_sets)
            t1 = time.time()
            vtree_time = t1 - t0

            # 2. Region Graph
            input_dists = {
                v: GaussianDistribution(var=v, mean=0.0, stddev=1.0) for v in range(num_vars)
            }
            t0 = time.time()
            region_graph = RegionGraphBuilder(md_vtree=md_vtree, input_dists=input_dists).build()
            t1 = time.time()
            rg_time = t1 - t0

            for init_w in [False, True]:
                # 3. Circuit
                t0 = time.time()
                spn = CircuitBuilder(
                    rg=region_graph, h=h, input_dists=input_dists, initialize_weights=init_w
                ).build()
                t1 = time.time()
                circuit_time = t1 - t0

                # 4. Folded Circuit
                t0 = time.time()
                folded = CircuitFolder(spn).build()
                t1 = time.time()
                folded_time = t1 - t0

                # 5. Fused Circuit
                t0 = time.time()
                _ = CircuitFuser(folded).build()
                t1 = time.time()
                fused_time = t1 - t0

                print(
                    f"{num_vars:<6} | {str(init_w):<6} | {h:<4} | {vtree_time:<12.4f} | {rg_time:<17.4f} | {circuit_time:<13.4f} | {folded_time:<12.4f} | {fused_time:<11.4f}"
                )

        except Exception as e:
            print(f"{num_vars:<6} | {'N/A':<6} | {h:<4} | {'FAILED':<12} | {str(e)}")


if __name__ == "__main__":
    run_construction_benchmark()
