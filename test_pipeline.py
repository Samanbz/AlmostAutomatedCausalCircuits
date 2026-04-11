import networkx as nx
import torch

from src.construction import construct_optimal_md_vtree, construct_optimal_vtree
from src.construction.random_scm import generate_random_scm
from src.symbolic.io_utils import plot_dag, plot_scm


def main():
    print("1. Initializing random SCM...")
    n_nodes = 8
    scm = generate_random_scm(n_nodes=n_nodes, expected_degree=2.0)

    print("2. Sampling data from SCM...")
    df = scm.sample(n_samples=5000)
    data = torch.tensor(df.values, dtype=torch.float32)

    # Map DataFrame columns to integer indices since the VTree logic
    # expects node indices 0..N-1 matching the data tensor columns.
    # The DataFrame columns are returned in topological sort order from the SCM.
    col_to_idx = {name: i for i, name in enumerate(df.columns)}

    print("3. Building networkx DAG skeleton from SCM topology...")
    dag = nx.DiGraph()
    for name in df.columns:
        dag.add_node(col_to_idx[name])

    for u_name, targets in scm._adj.items():
        for v_name in targets.keys():
            dag.add_edge(col_to_idx[u_name], col_to_idx[v_name])

    print("4a. Learning optimal VTree (prioritizing expressivity)...")
    vtree = construct_optimal_vtree(data, dag, prioritize="expressivity")
    print(f"    Built VTree with root scope: {vtree.get_node_data(vtree.get_root()).scope}")

    print("4b. Learning optimal MD-VTree (prioritizing expressivity)...")
    # For experimental assumption: MD-sets are subsets of each other.
    # e.g., Set 1 is {0, 1}, Set 2 is {0, 1, 2}, Set 3 is {0, 1, 2, 3}
    md_sets = [{0, 1}, {0, 1, 2}, {0, 1, 2, 3}]

    md_vtree = construct_optimal_md_vtree(data, md_sets, dag, prioritize="hardware")
    print(
        f"    Built MD-VTree with root scope: {md_vtree.get_node_data(md_vtree.get_root()).scope}"
    )

    print("5. Plotting SCM and VTrees...")
    plot_scm(scm, output_path="scm_plot.svg")
    plot_dag(vtree, output_path="vtree_plot.svg")
    plot_dag(md_vtree, output_path="md_vtree_plot.svg")

    print("Done! Plots saved to scm_plot.svg, vtree_plot.svg, and md_vtree_plot.svg")


if __name__ == "__main__":
    main()
