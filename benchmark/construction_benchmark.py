import math
import signal
import time

import numpy as np

from src.construction.random_vtree import construct_random_vtree
from src.construction.rat_spn import create_rat_spn
from src.construction.xpc import (
    construct_random_data_region_graph,
    construct_random_md_data_region_graph,
    construct_spn_from_region_graph,
)
from src.symbolic import MDVTree
from src.symbolic.arithmetic.distributions import GaussianDistribution


def get_md_schemes(num_features):
    """
    Generate three different sliding/overlapping scaling strategies.
    Bound maximum dimensional complexity to 6 to prevent O(2^K) explosion.
    """
    max_size = min(6, max(2, int(math.log2(num_features))))

    # 1. Sliding Window
    windows = []
    step = max(1, max_size // 2)
    for i in range(0, num_features, step):
        w = set(range(i, min(i + max_size, num_features)))
        if len(w) >= 2:
            windows.append(w)

    # 2. Supersets
    supersets = []
    for i in range(0, num_features, max_size):
        chunk_end = min(i + max_size, num_features)
        current_set = set()
        for j in range(i, chunk_end):
            current_set.add(j)
            if len(current_set) >= 2:
                supersets.append(set(current_set))

    # 3. Disjoint Chunks (baseline factorization logic)
    disjoint = []
    for i in range(0, num_features, max_size):
        w = set(range(i, min(i + max_size, num_features)))
        if len(w) >= 1:
            disjoint.append(w)

    return {
        "Sliding Windows": windows,
        "Incremental Supersets": supersets,
        "Disjoint Chunks": disjoint,
    }


def run_construction_benchmarks():
    feature_sizes = [8, 16, 32, 64, 128, 256, 512]
    batch_size = 1000

    print(f"{'Features':<8} | {'Model':<30} | {'Time (s)':<10}")
    print("-" * 55)

    for num_features in feature_sizes:
        input_dists = {i: GaussianDistribution(i, 0.0, 1.0) for i in range(num_features)}
        data = np.random.randn(batch_size, num_features)
        scopes = list(range(num_features))

        t0 = time.time()
        # RAT-SPN assumes log2 depth to reach 1-variable leaves
        depth = max(1, int(math.ceil(math.log2(num_features))))
        create_rat_spn(num_features, 1, depth, 2, 4, 4, input_dists)
        t1 = time.time()
        print(f"{num_features:<8} | {'RAT-SPN':<30} | {t1 - t0:.3f}")

        try:
            signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(TimeoutError))
            signal.alarm(30)
            t0 = time.time()
            max_size = min(6, max(2, int(math.log2(num_features))))
            vtree = construct_random_vtree(scopes, conj_len=max_size)
            rg = construct_random_data_region_graph(data, input_dists, 20, 2, vtree)
            construct_spn_from_region_graph(rg, input_dists)
            t1 = time.time()
            print(f"{num_features:<8} | {'Standard XPC':<30} | {t1 - t0:.3f}")
        except TimeoutError:
            print(f"{num_features:<8} | {'Standard XPC':<30} | {'TIMEOUT (>30s)':<10}")
        except MemoryError:
            print(f"{num_features:<8} | {'Standard XPC':<30} | {'OOM':<10}")
        except Exception as e:
            print(f"{num_features:<8} | {'Standard XPC':<30} | FAILED: {e}")
        finally:
            signal.alarm(0)

        schemes = get_md_schemes(num_features)
        for scheme_name, md_sets in schemes.items():
            try:
                t0 = time.time()
                vtree = construct_random_vtree(scopes)
                md_vtree = MDVTree.from_vtree(vtree, md_sets)

                rg_md = construct_random_md_data_region_graph(data, input_dists, 20, 2, md_vtree)
                construct_spn_from_region_graph(rg_md, input_dists)
                t1 = time.time()
                print(f"{num_features:<8} | {'MD-XPC (' + scheme_name + ')':<30} | {t1 - t0:.3f}")
            except Exception as e:
                print(f"{num_features:<8} | {'MD-XPC (' + scheme_name + ')':<30} | FAILED: {e}")

        print("-" * 55)


if __name__ == "__main__":
    run_construction_benchmarks()
