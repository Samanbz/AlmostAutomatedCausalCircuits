import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import Normalize, PowerNorm

from src.construction.circuit_builder import create_md_circuit
from src.construction.learned_vtree import construct_optimal_md_vtree
from src.symbolic.arithmetic.circuit import eval_circuit
from src.symbolic.arithmetic.nodes import GaussianDistribution, SumLayer
from src.symbolic.arithmetic.nodes.leaf_layer import (
    GaussianLeafLayer,
    LogLinearSplineDistribution,
    MixtureLeafLayer,
    SplineLeafLayer,
)
from src.symbolic.arithmetic.query import compile_query
from src.symbolic.arithmetic.train import SymbolicEMTrainer
from src.symbolic.id_ast import make_cond, make_marg, make_p, make_prod
from src.symbolic.io_utils import plot_dag
from src.utils.visualization import plot_gaussian_leaf, plot_mixture_leaf


torch.set_printoptions(precision=2, sci_mode=False)  # for better debug output


# =============================================================================
# 1. Hyperparameters & Configuration
# =============================================================================
CONFIG = {
    "n_confounders": 2,
    "prioritize": "H",
    "md_sets": "XZ",
    "num_nodes": 8,
    "train_size": 0.8,
    "batch_size": 2**12,
    "total_iters": 50,
    "step_size": 0.1,
    "decay_rate": 0.98,
    "decay_every": 5,
    "log_interval": 5,
    "jitter": 0.0,
    "seed": 27,
    "grid_size": 500,
    "x_range": (-2, 2),
    "y_range": (-20, 20),
    "output_dir": "scratch",
    "data_dir": "data",
    "dataset": "N100000_Z2_DE1.5_OS0.5_S27",
    "plot": False,
    "n_eval_x": 3,
    "grid_size_3d_x": 100,
    "grid_size_3d_y": 100,
    "x_std_range": 2.5,
    "hist_bins": 50,
    "heatmap_eval_batch_size": 128,
    "heatmap_density_gamma": 0.4,
    "heatmap_diff_percentile": 98,
    "heatmap_min_x_count": 100,
}


# =============================================================================
# 2. Utilities
# =============================================================================


def compute_nll(circuit, data):
    """Negative log-likelihood on a held-out test set."""
    with torch.no_grad():
        log_probs = eval_circuit(circuit, data)
        return -log_probs.mean().item()


def log_circuit_stddevs(circuit, name="Circuit"):
    print(f"\n--- {name} Sum Node Weight StdDevs ---")
    for node_id in circuit.topological_sort():
        node = circuit.get_node_data(node_id)
        if isinstance(node, SumLayer) and getattr(node, "log_weights", None) is not None:
            weights_obj = node.log_weights
            lw = (
                weights_obj.log_weights.detach().cpu()
                if hasattr(weights_obj, "log_weights")
                else weights_obj.detach().cpu()
            )
            mask = lw > -25.0
            if mask.any():
                valid_weights = torch.exp(lw[mask])
                stddev = valid_weights.std().item() if valid_weights.numel() > 1 else 0.0
                print(
                    f"Node {node_id} stddev: {stddev:.4f} (from {mask.sum().item()} active weights out of {lw.numel()})"
                )
            else:
                print(f"Node {node_id} stddev: N/A (no active weights)")
    print("---------------------------------------\n")


def log_circuit_weights(circuit, name="Circuit"):
    print(f"\n--- {name} Sum Node Weights ---")
    for node_id in circuit.topological_sort():
        node = circuit.get_node_data(node_id)
        if isinstance(node, SumLayer) and getattr(node, "log_weights", None) is not None:
            print(f"Node {node_id} weights:")
            print(node.log_weights)
    print("---------------------------------------\n")


def compute_kl(p, q, dx):
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return np.sum(p * np.log(p / q)) * dx


def compute_jsd(p, q, dx):
    m = 0.5 * (p + q)
    return 0.5 * compute_kl(p, m, dx) + 0.5 * compute_kl(q, m, dx)


def check_integration(
    q_do_ac,
    obs_num_ac,
    obs_den_ac,
    x_val,
    grid_y,
    z_ids,
    x_id,
    y_id,
    n_vars,
    data_mean,
    device,
    label="Integration",
):
    """Check if interventional and observational queries integrate to 1.0 over Y."""
    pts = np.zeros((len(grid_y), n_vars))
    pts[:, x_id] = x_val
    pts[:, y_id] = grid_y
    for zid in z_ids:
        pts[:, zid] = data_mean[zid]

    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)

    with torch.no_grad():
        log_do = eval_circuit(q_do_ac, pts_t, verbose=False).squeeze().detach().cpu().numpy()
        p_do = np.exp(log_do)

        log_num = eval_circuit(obs_num_ac, pts_t, verbose=False).squeeze().detach().cpu().numpy()
        log_den = eval_circuit(obs_den_ac, pts_t, verbose=False).squeeze().detach().cpu().numpy()
        p_obs = np.exp(log_num - log_den)

        # int_do = np.trapezoid(p_do, grid_y)
        # int_obs = np.trapezoid(p_obs, grid_y)

        # print(f"[{label}] X={x_val:.2f} | P(Y|do(X)) integrates to: {int_do:.4f}")
        # if abs(int_do - 1.0) > 0.05:
        #     print(f"!!! WARNING: {label} interventional distribution does not integrate to 1 !!!")

    return p_do, p_obs


# =============================================================================
# 3. Main Experiment
# =============================================================================


def setup_and_train():
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])

    # -------------------------------------------------------------------------
    # Load paired observational / interventional datasets from disk.
    # -------------------------------------------------------------------------
    data_dir = CONFIG["data_dir"]
    obs_path = os.path.join(data_dir, f"observational_{CONFIG['dataset']}.csv")
    do_path = os.path.join(data_dir, f"interventional_{CONFIG['dataset']}.csv")

    if not os.path.exists(obs_path) or not os.path.exists(do_path):
        raise FileNotFoundError(
            f"Dataset not found. Run `python generate_synthetic_data.py --output_dir {data_dir}` first."
        )

    df_obs_full = pd.read_csv(obs_path)
    df_do_full = pd.read_csv(do_path)

    n_train = int(CONFIG["train_size"] * len(df_obs_full))
    n_test = len(df_obs_full) - n_train

    df_obs = df_obs_full.iloc[:n_train]
    df_test = df_obs_full.iloc[n_train : n_train + n_test]
    df_do = df_do_full.iloc[:n_train]

    data_np = df_obs.values
    data = torch.tensor(data_np, dtype=torch.float32)
    data_mean = data.mean(0).numpy()
    data_std = data.std(0).numpy()
    print(f"Data mean: {data_mean}")
    print(f"Data std: {data_std}")

    var_names = list(df_obs.columns)
    var_to_id = {v: i for i, v in enumerate(var_names)}
    n_vars = len(var_names)

    z_names = [f"Z{i}" for i in range(CONFIG["n_confounders"])]
    z_ids = [var_to_id[z] for z in z_names]
    x_id = var_to_id["X"]
    y_id = var_to_id["Y"]
    V_names = set(var_names)

    num_nodes = CONFIG["num_nodes"]

    md_sets = [set([x_id] + z_ids)]
    # md_sets = [set([x_id] + z_ids), set(z_ids)] if CONFIG["md_sets"] == "Z+XZ" else md_sets
    md_vtree = construct_optimal_md_vtree(
        data,
        md_sets,
        prioritize="hardware" if CONFIG["prioritize"] == "H" else "expressivity",
        keep_together=[(x_id, y_id)],
    )

    plot_dag(
        md_vtree,
        output_path=f"scratch/md_vtree_{CONFIG['n_confounders']}_{CONFIG['md_sets']}_{CONFIG['prioritize']}.svg",
    )

    dists = {}
    print(f"var_to_id: {var_to_id}")
    for _name, vid in var_to_id.items():
        if vid == y_id:
            # Y is unconstrained: keep a Gaussian mixture leaf.
            dists[vid] = GaussianDistribution(
                var=vid,
                base_mean=float(data_mean[vid]),
                base_stddev=float(data_std[vid]) * 2,
            )
        else:
            # Constrained variables (X and the Z's) use disjoint-support spline leaves.
            dists[vid] = LogLinearSplineDistribution(
                var=vid,
                base_mean=float(data_mean[vid]),
                base_stddev=float(data_std[vid]) * 2,
            )

    ac = create_md_circuit(
        dists,
        md_vtree,
        num_nodes=num_nodes,
        initialize_weights=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ac.to(device)

    base_root_id = ac.get_roots()[0]

    # -------------------------------------------------------------------------
    # Compile queries (before training)
    # -------------------------------------------------------------------------
    p_all = make_p(V_names)

    # Observational queries
    other_vars = V_names - {"X", "Y"} - set(z_names)
    joint_xyz_ast = make_marg(other_vars, p_all)  # P(X,Y,Z)
    obs_num_ast = make_marg(set(z_names), joint_xyz_ast)  # P(Y, X)
    obs_den_ast = make_marg({"Y"}, obs_num_ast)  # P(X)
    marg_y_ast = make_marg(set(z_names) | {"X"}, p_all)  # P(Y)
    marg_z_ast = make_marg({"X", "Y"}, p_all)  # P(Z)
    cond_yxz_ast = make_cond({"Y"}, {"X"} | set(z_names), joint_xyz_ast)  # P(Y | X, Z)
    do_summand = make_prod([cond_yxz_ast, marg_z_ast])  # P(Y | X, Z) * P(Z)
    do_ast = make_marg(set(z_names), do_summand)  # sum_z P(Y | X, Z) * P(Z)

    test_data = torch.tensor(df_test.values, dtype=torch.float32, device=device)
    data_tensor = data.to(device)

    print(f"Test NLL (before): {compute_nll(ac, test_data):.4f}")
    print(f"Data NLL (before): {compute_nll(ac, data_tensor):.4f}")

    has_spline_leaf = any(isinstance(d, LogLinearSplineDistribution) for d in dists.values())
    leaf_lr = 0.01 if has_spline_leaf else 0.05

    trainer = SymbolicEMTrainer(ac, leaf_lr=leaf_lr)
    trainer.train(
        data_tensor,
        n_iter=CONFIG["total_iters"],
        batch_size=CONFIG["batch_size"],
        step_size=CONFIG["step_size"],
        decay_rate=CONFIG["decay_rate"],
        decay_every=CONFIG["decay_every"],
        log_interval=CONFIG["log_interval"],
    )

    print("Compiling post-training query circuits...")
    obs_num_ast = make_marg(set(z_names), p_all)  # P(Y,X)
    obs_den_ast = make_marg({"Y"}, obs_num_ast)  # P(X)
    marg_y_ast = make_marg({"X"}, obs_num_ast)

    obs_num_ac, _ = compile_query(obs_num_ast, ac, base_root_id, var_to_id)
    obs_den_ac, _ = compile_query(obs_den_ast, ac, base_root_id, var_to_id)
    marg_y_ac, _ = compile_query(marg_y_ast, ac, base_root_id, var_to_id)
    pz_ac, _ = compile_query(marg_z_ast, ac, base_root_id, var_to_id)
    cond_yxz_ac, _ = compile_query(cond_yxz_ast, ac, base_root_id, var_to_id)
    q_do_ac, _ = compile_query(do_ast, ac, base_root_id, var_to_id)
    print("Compiled post-training query circuits!")

    print(f"Test NLL (after):  {compute_nll(ac, test_data):.4f}")

    # log_circuit_weights(ac, name="After Training")

    return {
        "device": device,
        "ac": ac,
        "data_mean": data_mean,
        "data_std": data_std,
        "var_to_id": var_to_id,
        "z_ids": z_ids,
        "x_id": x_id,
        "y_id": y_id,
        "n_vars": n_vars,
        "base_root_id": base_root_id,
        "p_all": p_all,
        "data": data,
        "df_obs": df_obs,
        "df_do": df_do,
        "z_names": z_names,
        "V_names": V_names,
        "q_do_ac": q_do_ac,
        "obs_num_ac": obs_num_ac,
        "obs_den_ac": obs_den_ac,
        "marg_y_ac": marg_y_ac,
        "cond_yxz_ac": cond_yxz_ac,
        "pz_ac": pz_ac,
    }


def plot_circuits(res, var_to_name):
    import os
    import webbrowser

    circuits = [("base", res["ac"]), ("cond_yxz", res["cond_yxz_ac"]), ("q_do", res["q_do_ac"])]

    for name, ac in circuits:
        # Folded
        with open(f"{CONFIG['output_dir']}/{name}_circuit_folded.dot", "w") as f:
            f.write(ac.to_dot(var_to_name=var_to_name))
        os.system(
            f"dot -Tsvg {CONFIG['output_dir']}/{name}_circuit_folded.dot -o {CONFIG['output_dir']}/{name}_circuit_folded.svg"
        )
        webbrowser.open(
            "file://" + os.path.abspath(f"{CONFIG['output_dir']}/{name}_circuit_folded.svg")
        )

        for prune in [False, True]:
            suffix = "_pruned" if prune else "_raw"
            with open(f"{CONFIG['output_dir']}/{name}_circuit{suffix}.dot", "w") as f:
                f.write(ac.unfold(prune_dead_nodes=prune).to_dot(var_to_name=var_to_name))
            os.system(
                f"dot -Tsvg {CONFIG['output_dir']}/{name}_circuit{suffix}.dot -o {CONFIG['output_dir']}/{name}_circuit{suffix}.svg"
            )
            webbrowser.open(
                "file://" + os.path.abspath(f"{CONFIG['output_dir']}/{name}_circuit{suffix}.svg")
            )


def _num_leaf_nodes(leaf):
    """Return the number of output nodes (plotted curves) for a leaf."""
    if isinstance(leaf, GaussianLeafLayer):
        return leaf.num_nodes
    if isinstance(leaf, SplineLeafLayer):
        return leaf.num_nodes
    if isinstance(leaf, MixtureLeafLayer):
        w = torch.exp(leaf.log_weights.log_weights).detach().cpu().numpy()
        if w.ndim == 6:
            w = w.squeeze(axis=(4, 5))
        return w.shape[1]
    return 1


def _plot_spline_leaf(leaf: SplineLeafLayer, ax=None):
    """Plot the per-child densities of a spline leaf layer."""
    n_groups = leaf.num_groups
    n_nodes = leaf.num_nodes
    if ax is None:
        _, axes = plt.subplots(n_groups, 1, figsize=(6, 2 * n_groups), sharex=True)
        if n_groups == 1:
            axes = [axes]
    else:
        axes = np.atleast_1d(ax).tolist()

    with torch.no_grad():
        b = leaf._split_points().cpu().numpy()  # [G, N-1]

    # Determine a plotting range from the finite split points, but leave plenty
    # of room on both sides so the exponential tails are visible.
    finite = b[np.isfinite(b)]
    lo = float(finite.min()) if finite.size else -3.0
    hi = float(finite.max()) if finite.size else 3.0
    pad = max(2.0, (hi - lo) * 0.5)
    device = leaf._log_heights.device
    xs = torch.linspace(lo - pad, hi + pad, 1001, device=device).unsqueeze(1)

    # ``forward`` indexes ``data[:, self.var]``, so we need at least var+1 columns.
    data = torch.zeros((xs.shape[0], leaf.var + 1), dtype=xs.dtype, device=device)
    data[:, leaf.var] = xs.squeeze(1)

    with torch.no_grad():
        log_probs = leaf.forward(data).cpu().numpy()  # [B, G, N]
    probs = np.exp(log_probs)
    xs_np = xs.squeeze().cpu().numpy()

    split_vals = b[0] if b.ndim == 2 else b
    for g, ax_g in enumerate(axes):
        for j in range(n_nodes):
            ax_g.plot(
                xs_np, probs[:, g, j], alpha=0.7, lw=1.0, label=f"child {j}" if j == 0 else None
            )
        for split in split_vals:
            if np.isfinite(split):
                ax_g.axvline(split, color="gray", linestyle=":", lw=0.8)
        ax_g.set_ylabel("density")
    axes[-1].set_xlabel("value")
    return axes


def plot_leaves(ac, var_to_name, output_dir, df_obs=None):
    """Plot every Gaussian / mixture / spline leaf distribution, like in test_query.py.

    If ``df_obs`` is provided, the empirical marginal of the corresponding
    variable is overlaid as a black step histogram on each subplot so the
    learned leaf can be compared against the ground truth in one diagram.
    A second curve (red, dashed) shows the same GT marginal scaled by the
    number of leaf nodes, which matches the per-node amplitude scale.
    """
    print("\n--- Plotting Leaves ---")
    for leaf_id in ac.get_leaves():
        leaf = ac.get_node_data(leaf_id)
        var = getattr(leaf, "var", None)
        var_name = var_to_name.get(var, f"var{var}")
        if isinstance(leaf, GaussianLeafLayer):
            axes = plot_gaussian_leaf(leaf)
        elif isinstance(leaf, MixtureLeafLayer):
            axes = plot_mixture_leaf(leaf)
        elif isinstance(leaf, SplineLeafLayer):
            axes = _plot_spline_leaf(leaf)
        else:
            continue

        if df_obs is not None and var is not None and var_name in df_obs.columns:
            gt_values = df_obs[var_name].values
            num_nodes = _num_leaf_nodes(leaf)
            counts, bin_edges = np.histogram(gt_values, bins=100, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            for ax in np.asarray(axes).reshape(-1):
                ax.plot(
                    bin_centers,
                    counts,
                    drawstyle="steps-mid",
                    color="black",
                    linewidth=2,
                    label="GT marginal",
                )
                ax.plot(
                    bin_centers,
                    counts * num_nodes,
                    drawstyle="steps-mid",
                    color="red",
                    linewidth=2,
                    linestyle="--",
                    label=f"GT marginal × {num_nodes}",
                )
                ax.legend(loc="best", fontsize="small")

        out_path = os.path.join(
            output_dir, f"leaf_N{CONFIG['num_nodes']}_{var_name}_distribution.png"
        )
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved leaf plot to {out_path}")


def run_experiment():
    res = setup_and_train()

    print("\n--- Plotting Unfolded & Folded Circuits ---")
    var_to_name = {v: k for k, v in res["var_to_id"].items()}

    if CONFIG["plot"]:
        plot_circuits(res, var_to_name)

    plot_leaves(res["ac"], var_to_name, CONFIG["output_dir"], df_obs=res["df_obs"])

    print("\n--- 5. Plotting ---")
    x_id_name = "X"
    y_id_name = "Y"
    df_obs = res["df_obs"]
    df_do = res["df_do"]

    # Pick one X value per leaf support interval instead of a uniform linspace.
    x_intervals = _get_x_leaf_support_intervals(res["ac"], res["x_id"])
    if x_intervals:
        contexts, context_labels = _choose_x_values_from_intervals(
            x_intervals, df_obs=df_obs, x_id_name=x_id_name
        )
        print(f"  Plotting at one X value per leaf support ({len(contexts)} values):")
        for lbl in context_labels:
            print(f"    {lbl}")
    else:
        # Fallback to the old uniform linspace if no leaf supports are available.
        x_mean = res["data_mean"][res["x_id"]]
        x_std = res["data_std"][res["x_id"]]
        n_x = CONFIG.get("n_eval_x", 6)
        contexts = np.linspace(
            x_mean - CONFIG["x_std_range"] * x_std,
            x_mean + CONFIG["x_std_range"] * x_std,
            n_x,
        )
        context_labels = [f"X={x:.2f}" for x in contexts]

    n_x = len(contexts)

    hist_bins = CONFIG.get("hist_bins", 100)
    y_edges = np.linspace(CONFIG["y_range"][0], CONFIG["y_range"][1], hist_bins + 1)
    grid_y = (y_edges[:-1] + y_edges[1:]) / 2.0
    dy = grid_y[1] - grid_y[0]

    # Ground truth from the paired datasets on disk.
    print("  Preparing GT from observational / interventional datasets...")
    x_data = df_obs[x_id_name].values
    x_bandwidth = max(0.1, (x_data.max() - x_data.min()) / 20.0)

    n_cols = 2 if n_x >= 4 else 1
    n_rows = (n_x + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    q_do_ac = res["q_do_ac"]
    obs_num_ac = res["obs_num_ac"]
    obs_den_ac = res["obs_den_ac"]
    z_ids = res["z_ids"]
    x_id = res["x_id"]
    y_id = res["y_id"]
    n_vars = res["n_vars"]
    data_mean = res["data_mean"]
    device = res["device"]

    print("\n--- Divergence Metrics Table ---")
    print(
        f"{'X Value':<10} | {'KL(GT_obs||C_obs)':<17} | {'KL(GT_do||C_do)':<17} | {'JSD(C_obs||C_do)':<17} | {'JSD(GT_obs||GT_do)':<18} | {'JSD(GT_obs||C_obs)':<18} | {'JSD(GT_do||C_do)'}"
    )
    print("-" * 115)

    # x_vals_list = []
    # kl_obs_list = []
    # kl_do_list = []
    # jsd_circ_list = []
    # jsd_gt_list = []
    # jsd_obs_list = []
    # jsd_do_list = []

    for row, (x_val, label) in enumerate(zip(contexts, context_labels)):
        ax = axes[row]

        # Use helper for integration checking and to get probabilities
        p_do, p_obs = check_integration(
            q_do_ac,
            obs_num_ac,
            obs_den_ac,
            x_val,
            grid_y,
            z_ids,
            x_id,
            y_id,
            n_vars,
            data_mean,
            device,
            label=f"After Training {label}",
        )

        # GT from datasets: P(Y|X) on observational, P(Y|X) on interventional.
        gt_obs = _py_given_x_kde(df_obs, x_id_name, y_id_name, x_val, grid_y, x_bandwidth)
        gt_do = _py_given_x_kde(df_do, x_id_name, y_id_name, x_val, grid_y, x_bandwidth)

        # kl_obs = compute_kl(gt_obs, p_obs, dy)
        # kl_do = compute_kl(gt_do, p_do, dy)

        # jsd_circ = compute_jsd(p_obs, p_do, dy)
        # jsd_gt = compute_jsd(gt_obs, gt_do, dy)
        # jsd_obs = compute_jsd(gt_obs, p_obs, dy)
        # jsd_do = compute_jsd(gt_do, p_do, dy)

        # print(
        #     f"{x_val:<10.2f} | {kl_obs:<17.4f} | {kl_do:<17.4f} | {jsd_circ:<17.4f} | {jsd_gt:<18.4f} | {jsd_obs:<18.4f} | {jsd_do:.4f}"
        # )

        # x_vals_list.append(x_val)
        # kl_obs_list.append(kl_obs)
        # kl_do_list.append(kl_do)
        # jsd_circ_list.append(jsd_circ)
        # jsd_gt_list.append(jsd_gt)
        # jsd_obs_list.append(jsd_obs)
        # jsd_do_list.append(jsd_do)

        # Plotting (smooth lines so the KDE ground truth looks continuous)
        ax.plot(
            grid_y,
            p_do,
            color="red",
            alpha=0.5,
            label="Learned P(Y|do(X))",
            linewidth=2,
        )
        ax.plot(
            grid_y,
            p_obs,
            color="blue",
            alpha=0.5,
            label="Learned P(Y|X)",
            linewidth=2,
        )
        ax.plot(
            grid_y,
            gt_do,
            color="red",
            alpha=0.3,
            label="GT P(Y|do(X))",
            linewidth=2,
        )
        ax.plot(
            grid_y,
            gt_obs,
            color="blue",
            alpha=0.3,
            label="GT P(Y|X)",
            linewidth=2,
        )

        # Calculate and plot means
        mean_do = np.sum(grid_y * p_do) * dy
        mean_obs = np.sum(grid_y * p_obs) * dy
        mean_gt_do = np.sum(grid_y * gt_do) * dy
        mean_gt_obs = np.sum(grid_y * gt_obs) * dy

        ax.axvline(mean_do, color="red", alpha=0.5, linewidth=1)
        ax.axvline(mean_obs, color="blue", alpha=0.5, linewidth=1)
        ax.axvline(mean_gt_do, color="red", linestyle="--", alpha=0.2, linewidth=1)
        ax.axvline(mean_gt_obs, color="blue", linestyle="--", alpha=0.2, linewidth=1)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Y")
        ax.set_ylabel("Density")
        if row == 0:
            ax.legend(loc="upper right", fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = f"{CONFIG['output_dir']}/tree_backdoor_{CONFIG['dataset']}_N{CONFIG['num_nodes']}_{CONFIG['md_sets']}_{CONFIG['prioritize']}.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot to {out_file}")

    # # Plot Divergence Metrics
    # fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    # axes2[0].plot(x_vals_list, kl_obs_list, label="KL(GT_obs||C_obs)")
    # axes2[0].plot(x_vals_list, kl_do_list, label="KL(GT_do||C_do)")
    # axes2[0].set_xlabel("X")
    # axes2[0].set_ylabel("KL Divergence")
    # axes2[0].set_title("KL Divergences")
    # axes2[0].legend()
    # axes2[0].grid(True)

    # axes2[1].plot(x_vals_list, jsd_circ_list, label="JSD(C_obs||C_do)")
    # axes2[1].plot(x_vals_list, jsd_gt_list, label="JSD(GT_obs||GT_do)")
    # axes2[1].set_xlabel("X")
    # axes2[1].set_ylabel("JSD")
    # axes2[1].set_title("Jensen-Shannon Divergences")
    # axes2[1].legend()
    # axes2[1].grid(True)

    # plt.tight_layout()
    # div_out_file = f"{CONFIG['output_dir']}/divergences_{CONFIG['dataset']}_N{CONFIG['num_nodes']}_{CONFIG['md_sets']}_{CONFIG['prioritize']}.png"
    # plt.savefig(div_out_file, dpi=150)
    # print(f"Saved divergence plot to {div_out_file}")

    plot_2d_heatmaps(res)


def _py_given_x_kde(df, x_col, y_col, x_val, grid_y, x_bandwidth=None, y_bandwidth=None):
    """P(Y | X = x_val) via a fast Gaussian conditional KDE.

    Only samples within a few X bandwidths of ``x_val`` are used so the
    computation stays O(local samples) rather than the full dataset size.
    """
    x_vals = df[x_col].values
    y_vals = df[y_col].values

    if x_bandwidth is None:
        x_bandwidth = max(0.1, (x_vals.max() - x_vals.min()) / 20.0)

    # Hard window for speed: the Gaussian kernel is negligible past 3 bandwidths.
    window = 3.0 * x_bandwidth
    mask = np.abs(x_vals - x_val) <= window
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]

    n = len(y_vals)
    if n == 0:
        return np.zeros_like(grid_y)

    # Gaussian weights in X.
    x_weights = np.exp(-0.5 * ((x_vals - x_val) / x_bandwidth) ** 2)
    weight_sum = x_weights.sum()
    if weight_sum <= 0:
        return np.zeros_like(grid_y)

    if y_bandwidth is None:
        # Silverman's rule of thumb on the selected Y samples.
        std = np.std(y_vals)
        iqr = np.subtract(*np.percentile(y_vals, [75, 25]))
        h = 0.9 * min(std, iqr / 1.34) * n ** (-0.2) if n > 1 else 1.0
        y_bandwidth = max(h, 1e-6)

    # Vectorized Gaussian KDE over grid_y.
    diffs = grid_y[:, None] - y_vals[None, :]  # [M, n]
    y_kernels = np.exp(-0.5 * (diffs / y_bandwidth) ** 2)
    y_kernels /= y_bandwidth * np.sqrt(2.0 * np.pi)

    kde = (x_weights[None, :] * y_kernels).sum(axis=1) / weight_sum
    return kde


def _py_given_x_surface(df, x_col, y_col, grid_x, y_edges, min_x_count=100, x_bandwidth=None):
    """P(Y | X) surface (rows=Y bins, cols=X bins) via conditional KDE.

    X positions with fewer than ``min_x_count`` samples inside the hard window
    are masked with NaN.
    """
    grid_y = (y_edges[:-1] + y_edges[1:]) / 2.0
    surface = np.zeros((len(grid_y), len(grid_x)))
    x_vals = df[x_col].values
    if x_bandwidth is None:
        x_bandwidth = max(0.1, (grid_x[-1] - grid_x[0]) / 20.0)

    for i, x_c in enumerate(grid_x):
        window = 3.0 * x_bandwidth
        mask = np.abs(x_vals - x_c) <= window
        if mask.sum() >= min_x_count:
            surface[:, i] = _py_given_x_kde(df, x_col, y_col, x_c, grid_y, x_bandwidth=x_bandwidth)
        else:
            surface[:, i] = np.nan
    return surface


def _get_x_split_points(ac, x_id):
    """Return sorted finite boundaries of the X leaf's unit supports.

    For spline leaves the split points are learnable, so the live learned
    boundaries (``_split_points()``) are used instead of the static initial
    ``node_supports``.
    """
    split_points = set()
    for leaf_id in ac.get_leaves():
        leaf = ac.get_node_data(leaf_id)
        if getattr(leaf, "var", None) != x_id:
            continue
        if hasattr(leaf, "_split_points"):
            with torch.no_grad():
                for b in leaf._split_points().cpu().numpy().ravel():
                    if not math.isinf(b):
                        split_points.add(float(b))
        else:
            supports = getattr(leaf, "node_supports", None)
            if supports is None:
                continue
            for supp in supports:
                interval = supp.intervals.get(x_id)
                if interval is None:
                    continue
                if not math.isinf(interval.low):
                    split_points.add(float(interval.low))
                if not math.isinf(interval.high):
                    split_points.add(float(interval.high))
    return sorted(split_points)


def _get_x_leaf_support_intervals(ac, x_id):
    """Return support intervals (low, high) for the X leaf's node supports.

    Each tuple corresponds to one leaf child / node support.  Infinite bounds
    are preserved; callers should clip to the data range when choosing a
    representative X value.
    """
    for leaf_id in ac.get_leaves():
        leaf = ac.get_node_data(leaf_id)
        if getattr(leaf, "var", None) != x_id:
            continue
        node_supports = getattr(leaf, "node_supports", None)
        if not node_supports:
            continue
        intervals = []
        for supp in node_supports:
            iv = supp.intervals.get(x_id)
            if iv is None:
                continue
            intervals.append((float(iv.low), float(iv.high)))
        return intervals
    return []


def _choose_x_values_from_intervals(intervals, df_obs=None, x_id_name="X", pad_fraction=0.05):
    """Pick one representative X value inside each leaf support interval.

    For finite intervals the midpoint is used.  For infinite tails the value is
    clipped to the observed data range (with a small padding) so the ground-
    truth histograms still contain samples.
    """
    if df_obs is not None and x_id_name in df_obs.columns:
        x_data = df_obs[x_id_name].values
        x_min, x_max = float(np.percentile(x_data, 0.5)), float(np.percentile(x_data, 99.5))
    else:
        x_min, x_max = float("-inf"), float("inf")

    finite_bounds = [
        bound for low, high in intervals for bound in (low, high) if math.isfinite(bound)
    ]
    span = max(finite_bounds) - min(finite_bounds) if finite_bounds else 1.0
    pad = pad_fraction * span

    values = []
    labels = []
    for i, (low, high) in enumerate(intervals):
        if math.isfinite(low) and math.isfinite(high):
            x_val = 0.5 * (low + high)
            label = f"X={x_val:.2f} (support {i})"
        elif math.isfinite(low):
            # Right tail [low, +inf).
            candidate = low + pad
            if x_max < float("inf"):
                candidate = min(candidate, x_max)
            x_val = max(low, candidate)
            label = f"X={x_val:.2f} (right tail {i})"
        elif math.isfinite(high):
            # Left tail (-inf, high].
            candidate = high - pad
            if x_min > float("-inf"):
                candidate = max(candidate, x_min)
            x_val = min(high, candidate)
            label = f"X={x_val:.2f} (left tail {i})"
        else:
            x_val = 0.5 * (x_min + x_max) if x_min > float("-inf") and x_max < float("inf") else 0.0
            label = f"X={x_val:.2f} (full range {i})"
        values.append(x_val)
        labels.append(label)
    return np.array(values), labels


def plot_2d_heatmaps(res):
    """Plot the full 2D conditional distributions as a 2x3 heatmap grid.

    Rows:
      - Top: P(Y|do(X))    (learned, ground truth, difference)
      - Bottom: P(Y|X)     (learned, ground truth, difference)

    Columns:
      - Left: learned circuit
      - Center: ground truth
      - Right: learned - ground truth
    """
    hist_bins = CONFIG.get("hist_bins", 100)
    y_edges = np.linspace(CONFIG["y_range"][0], CONFIG["y_range"][1], hist_bins + 1)
    grid_y = (y_edges[:-1] + y_edges[1:]) / 2.0
    y_res = len(grid_y)

    x_res = CONFIG.get("grid_size_3d_x", 30)
    # Scale the X axis to the observed data support (with a tiny quantile trim)
    # instead of the fixed config range, so the heatmap only spans valid Xs.
    x_obs = res["df_obs"]["X"].values
    x_min, x_max = float(np.percentile(x_obs, 0.5)), float(np.percentile(x_obs, 99.5))
    grid_x = np.linspace(x_min, x_max, x_res)

    X, Y = np.meshgrid(grid_x, grid_y)

    pts = np.zeros((x_res * y_res, res["n_vars"]))
    pts[:, res["x_id"]] = X.flatten()
    pts[:, res["y_id"]] = Y.flatten()
    for zid in res["z_ids"]:
        pts[:, zid] = res["data_mean"][zid]

    pts_t = torch.tensor(pts, dtype=torch.float32, device=res["device"])
    eval_batch = CONFIG["heatmap_eval_batch_size"]

    def _eval_log_probs(circuit, inputs, batch_size=eval_batch):
        log_probs = []
        for i in range(0, inputs.size(0), batch_size):
            batch = inputs[i : i + batch_size]
            with torch.no_grad():
                log_p = eval_circuit(circuit, batch, verbose=False).squeeze()
            log_probs.append(log_p.detach().cpu())
        return torch.cat(log_probs).numpy()

    log_do = _eval_log_probs(res["q_do_ac"], pts_t)
    circ_do = np.exp(log_do).reshape(y_res, x_res)

    log_num = _eval_log_probs(res["obs_num_ac"], pts_t)
    log_den = _eval_log_probs(res["obs_den_ac"], pts_t)
    circ_obs = np.exp(log_num - log_den).reshape(y_res, x_res)

    # Ground truth from the paired datasets on disk.
    df_obs = res["df_obs"]
    df_do = res["df_do"]
    min_x_count = CONFIG["heatmap_min_x_count"]
    gt_obs = _py_given_x_surface(df_obs, "X", "Y", grid_x, y_edges, min_x_count=min_x_count)
    gt_do = _py_given_x_surface(df_do, "X", "Y", grid_x, y_edges, min_x_count=min_x_count)

    # Apply the same X-bin masks to the learned circuit surfaces so the panels
    # share exactly the same valid support.
    x_mask_obs = ~np.isnan(gt_obs[0, :])
    x_mask_do = ~np.isnan(gt_do[0, :])
    circ_obs_masked = circ_obs.copy()
    circ_do_masked = circ_do.copy()
    circ_obs_masked[:, ~x_mask_obs] = np.nan
    circ_do_masked[:, ~x_mask_do] = np.nan

    diff_do = circ_do_masked - gt_do
    diff_obs = circ_obs_masked - gt_obs
    diff_pct = CONFIG["heatmap_diff_percentile"]
    p99_diff = np.nanpercentile(
        np.concatenate([np.abs(diff_do).ravel(), np.abs(diff_obs).ravel()]), diff_pct
    )
    diff_norm = Normalize(vmin=-p99_diff, vmax=p99_diff)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    gamma = CONFIG["heatmap_density_gamma"]
    # Single shared PowerNorm across all density panels (ignores NaN masks).
    density_vmin = np.nanmin([circ_do_masked, gt_do, circ_obs_masked, gt_obs])
    density_vmax = np.nanmax([circ_do_masked, gt_do, circ_obs_masked, gt_obs])
    density_norm = PowerNorm(gamma=gamma, vmin=density_vmin, vmax=density_vmax)

    extent = [grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]]
    mask_color = "lightgray"

    # Crop the X axis to the region that has enough samples in both datasets.
    valid_x = x_mask_obs & x_mask_do
    if valid_x.any():
        valid_idx = np.where(valid_x)[0]
        xlim = [float(grid_x[valid_idx[0]]), float(grid_x[valid_idx[-1]])]
        for ax in axes.flat:
            ax.set_xlim(xlim)

    # Draw vertical lines at the X-leaf split points so the heatmaps reflect the
    # regions used by the circuit.
    x_split_points = _get_x_split_points(res["ac"], res["x_id"])
    for ax in axes.flat:
        for x_split in x_split_points:
            ax.axvline(x_split, color="white", linestyle="--", linewidth=1, alpha=0.5)

    # Top row: P(Y|do(X))
    cmap_do = plt.cm.viridis.with_extremes(bad=mask_color)
    im0 = axes[0, 0].imshow(
        circ_do_masked,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap_do,
        norm=density_norm,
    )
    axes[0, 0].set_title("Learned P(Y|do(X))")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(
        gt_do, aspect="auto", origin="lower", extent=extent, cmap=cmap_do, norm=density_norm
    )
    axes[0, 1].set_title("Ground Truth P(Y|do(X))")
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Y")
    fig.colorbar(im1, ax=axes[0, 1])

    cmap_diff = plt.cm.coolwarm.with_extremes(bad=mask_color)
    im2 = axes[0, 2].imshow(
        diff_do,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap_diff,
        norm=diff_norm,
    )
    axes[0, 2].set_title("Difference (Learned - GT)")
    axes[0, 2].set_xlabel("X")
    axes[0, 2].set_ylabel("Y")
    fig.colorbar(im2, ax=axes[0, 2])

    # Bottom row: P(Y|X)
    im3 = axes[1, 0].imshow(
        circ_obs_masked,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap_do,
        norm=density_norm,
    )
    axes[1, 0].set_title("Learned P(Y|X)")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    fig.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(
        gt_obs, aspect="auto", origin="lower", extent=extent, cmap=cmap_do, norm=density_norm
    )
    axes[1, 1].set_title("Ground Truth P(Y|X)")
    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("Y")
    fig.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(
        diff_obs,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap_diff,
        norm=diff_norm,
    )
    axes[1, 2].set_title("Difference (Learned - GT)")
    axes[1, 2].set_xlabel("X")
    axes[1, 2].set_ylabel("Y")
    fig.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    out_file = f"{CONFIG['output_dir']}/heatmap_do__{CONFIG['dataset']}_N{CONFIG['num_nodes']}_{CONFIG['md_sets']}_{CONFIG['prioritize']}.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved 2D heatmap grid to {out_file}")


if __name__ == "__main__":
    run_experiment()
