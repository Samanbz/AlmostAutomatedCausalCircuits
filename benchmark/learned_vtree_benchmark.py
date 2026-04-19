import itertools
import math
import signal
import time

import networkx as nx
import numpy as np
import torch


from src.construction.learned_vtree import construct_optimal_md_vtree, construct_optimal_vtree


def generate_mock_data_and_dag(num_samples, num_features, expected_degree):
    """
    Generate mock tensor data and a random directed acyclic graph.
    """
    data = torch.randn(num_samples, num_features, dtype=torch.float32)

    # Generate random DAG (Erdos-Renyi style, then directed and acyclic)
    p = expected_degree / (num_features - 1) if num_features > 1 else 0
    p = min(max(p, 0), 1)

    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_features))

    # To keep it acyclic, only add edges from lower index to higher index
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if np.random.rand() < p:
                dag.add_edge(i, j)

    return data, dag


def generate_md_sets(num_features):
    """
    Generate a simple chain of overlapping MD sets.
    """
    max_size = min(8, max(2, int(math.log2(num_features))))

    md_sets = []
    current_set = set()
    for i in range(min(num_features, max_size * 2)):
        current_set.add(i)
        if len(current_set) >= 2:
            md_sets.append(set(current_set))

    # Default fallback
    if not md_sets and num_features >= 2:
        md_sets = [{0, 1}]

    return md_sets


def run_benchmark():
    feature_sizes = [16, 64, 256, 1024]
    sample_sizes = [1000, 10000]
    expected_degrees = [2.0, 10.0]  # Sparse vs Denser

    print(f"{'Vars':<6} | {'Samples':<9} | {'Degree':<6} | {'Model':<35} | {'Time (s)':<10}")
    print("-" * 75)

    for num_features in feature_sizes:
        for num_samples in sample_sizes:
            for exp_degree in expected_degrees:
                # Cap the explicit degree to avoid non-sensical dense ranges in very small graphs
                if exp_degree > num_features:
                    continue

                data, dag = generate_mock_data_and_dag(num_samples, num_features, exp_degree)
                md_sets = generate_md_sets(num_features)

                configs = [
                    (
                        "Optimal VTree (Hardware)",
                        lambda: construct_optimal_vtree(data, dag, prioritize="hardware"),
                    ),
                    (
                        "Optimal VTree (Expressivity)",
                        lambda: construct_optimal_vtree(data, dag, prioritize="expressivity"),
                    ),
                    (
                        "Optimal MD-VTree (Hardware)",
                        lambda: construct_optimal_md_vtree(
                            data, md_sets, dag, prioritize="hardware"
                        ),
                    ),
                    (
                        "Optimal MD-VTree (Expressivity)",
                        lambda: construct_optimal_md_vtree(
                            data, md_sets, dag, prioritize="expressivity"
                        ),
                    ),
                ]

                for name, func in configs:
                    try:
                        # Implement a 60-second timeout to catch hanging unconstrained cuts on huge graphs
                        def handler(signum, frame):
                            raise TimeoutError()

                        signal.signal(signal.SIGALRM, handler)
                        signal.alarm(60)

                        t0 = time.time()
                        func()
                        t1 = time.time()

                        print(
                            f"{num_features:<6} | {num_samples:<9} | {exp_degree:<6.1f} | {name:<35} | {t1 - t0:.3f}"
                        )
                    except TimeoutError:
                        print(
                            f"{num_features:<6} | {num_samples:<9} | {exp_degree:<6.1f} | {name:<35} | {'TIMEOUT (>60s)':<10}"
                        )
                    except MemoryError:
                        print(
                            f"{num_features:<6} | {num_samples:<9} | {exp_degree:<6.1f} | {name:<35} | {'OOM':<10}"
                        )
                    except Exception as e:
                        print(
                            f"{num_features:<6} | {num_samples:<9} | {exp_degree:<6.1f} | {name:<35} | FAILED: {e}"
                        )
                    finally:
                        signal.alarm(0)

                print("-" * 75)


if __name__ == "__main__":
    run_benchmark()
