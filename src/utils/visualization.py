import json
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

from src.symbolic.arithmetic.nodes.leaf_layer import (
    GaussianLeafLayer,
    MixtureLeafLayer,
    SplineLeafLayer,
)
from src.symbolic.arithmetic.weights import DenseWeights


def create_interactive_plot(ac, output_path="circuit.html", var_to_name=None):
    """
    Exports a SymbolicArithmeticCircuit to an interactive HTML file.
    The HTML includes a DAG visualization and a side panel to browse sum node weights.
    """
    nodes = []
    edges = []

    for node_id in ac.topological_sort():
        node = ac.get_node_data(node_id)
        tname = type(node).__name__

        label = str(node_id)
        color = "#ffffff"
        if "Sum" in tname:
            label = "+"
            color = "#ccffcc"
        elif "Product" in tname:
            label = "X"
            color = "#ffcc99"
        else:
            if hasattr(node, "var"):
                vname = var_to_name.get(node.var, node.var) if var_to_name else node.var
                label = f"{tname}({vname})"
            else:
                label = tname
            color = "#ffcccc"

        weights_data = None
        if "Sum" in tname and hasattr(node, "log_weights") and node.log_weights is not None:
            w_obj = node.log_weights
            wt = (
                w_obj.log_weights.detach().cpu()
                if hasattr(w_obj, "log_weights")
                else w_obj.detach().cpu()
            )
            if isinstance(wt, torch.Tensor):
                probs = torch.exp(wt).numpy().tolist()
                log_weights = wt.numpy().tolist()
                weights_data = {"shape": list(wt.shape), "probs": probs, "log_weights": log_weights}

        nodes.append(
            {
                "id": node_id,
                "label": label,
                "color": color,
                "shape": "circle",
                "font": {"size": 14},
                "widthConstraint": {"minimum": 70, "maximum": 70},
                "title": f"ID: {node_id}\nType: {tname}",
                "weights": weights_data,
            }
        )

        for child_id in ac.get_children(node_id):
            edges.append({"from": node_id, "to": child_id})

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Interactive Circuit Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{ font-family: sans-serif; margin: 0; padding: 0; display: flex; height: 100vh; }}
        #network {{ flex: 1; border-right: 1px solid lightgray; }}
        #panel {{ width: 400px; padding: 20px; overflow-y: auto; background-color: #f9f9f9; display: flex; flex-direction: column; }}
        .matrix-container {{ display: grid; gap: 1px; background-color: #ccc; margin-top: 5px; margin-bottom: 20px; border: 1px solid #ccc; width: max-content; }}
        .matrix-cell {{ width: 40px; height: 40px; }}
        .matrix-label {{ font-size: 14px; margin-top: 10px; font-weight: bold; color: #333; }}
    </style>
</head>
<body>
    <div id="network"></div>
    <div id="panel">
        <h3>Node Details</h3>
        <div id="node-info">Click a node to view details.</div>
        <div id="weights-view" style="display: none;">
            <h4>Weight Matrices</h4>
            <div id="matrix-wrapper"></div>
        </div>
    </div>

    <script type="text/javascript">
        const nodesData = {json.dumps(nodes)};
        const edgesData = {json.dumps(edges)};

        const container = document.getElementById('network');
        const data = {{
            nodes: new vis.DataSet(nodesData),
            edges: new vis.DataSet(edgesData)
        }};
        const options = {{
            layout: {{ hierarchical: {{ direction: 'UD', sortMethod: 'directed' }} }},
            physics: false
        }};
        const network = new vis.Network(container, data, options);

        let currentWeights = null;

        function formatLogWeight(val) {{
            if (val === 0) return "0.0000e+0";
            let absVal = Math.abs(val);
            let exponent = Math.floor(Math.log10(absVal)) - 1;
            let mantissa = val / Math.pow(10, exponent);
            let sign = exponent >= 0 ? "+" : "";
            return mantissa.toFixed(4) + "e" + sign + exponent;
        }}

        network.on("click", function (params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodesData.find(n => n.id === nodeId);
                document.getElementById('node-info').innerText = node.title;

                if (node.weights) {{
                    currentWeights = node.weights;
                    document.getElementById('weights-view').style.display = 'block';
                    renderMatrix();
                }} else {{
                    document.getElementById('weights-view').style.display = 'none';
                }}
            }}
        }});

        function renderMatrix() {{
            if (!currentWeights) return;
            const shape = currentWeights.shape;
            const probs = currentWeights.probs;
            const logWeights = currentWeights.log_weights;

            let h_sum, h_left, h_right;

            if (shape.length === 3) {{
                h_sum = shape[0]; h_left = shape[1]; h_right = shape[2];
            }} else if (shape.length === 2) {{
                h_sum = shape[0];
                h_left = shape[1]; h_right = 1;
            }} else {{
                document.getElementById('matrix-wrapper').innerHTML = "Unsupported shape: " + shape;
                return;
            }}

            const wrapper = document.getElementById('matrix-wrapper');
            wrapper.innerHTML = '';

            for (let p = 0; p < h_sum; p++) {{
                let pageData, pageLogData;
                if (shape.length === 3) {{
                    pageData = probs[p];
                    pageLogData = logWeights[p];
                }} else {{
                    pageData = probs[p].map(val => [val]);
                    pageLogData = logWeights[p].map(val => [val]);
                }}

                const label = document.createElement('div');
                label.className = 'matrix-label';
                label.innerText = `Unit ${{p}}`;
                wrapper.appendChild(label);

                const grid = document.createElement('div');
                grid.className = 'matrix-container';
                grid.style.gridTemplateColumns = `repeat(${{h_right}}, 40px)`;

                for (let i = 0; i < h_left; i++) {{
                    for (let j = 0; j < h_right; j++) {{
                        const cell = document.createElement('div');
                        cell.className = 'matrix-cell';
                        const val = pageData[i][j];
                        const logVal = pageLogData[i][j];

                        // Amplify small non-zero values
                        const intensity = val > 0 ? Math.pow(val, 0.3) : 0;
                        const colorVal = Math.floor(255 * (1 - Math.min(1.0, intensity)));

                        cell.style.backgroundColor = `rgb(${{colorVal}}, ${{colorVal}}, ${{colorVal}})`;
                        cell.title = `Prob: ${{val.toFixed(5)}}\\nLog weight: ${{formatLogWeight(logVal)}}`;
                        grid.appendChild(cell);
                    }}
                }}
                wrapper.appendChild(grid);
            }}

            // Render sum matrix across h_sum
            let sumMatrix = [];
            for (let i = 0; i < h_left; i++) {{
                sumMatrix[i] = [];
                for (let j = 0; j < h_right; j++) {{
                    let total = 0;
                    for (let p = 0; p < h_sum; p++) {{
                        let pageData;
                        if (shape.length === 3) {{
                            pageData = probs[p];
                        }} else {{
                            pageData = probs[p].map(val => [val]);
                        }}
                        total += pageData[i][j];
                    }}
                    sumMatrix[i][j] = total;
                }}
            }}

            const sumLabel = document.createElement('div');
            sumLabel.className = 'matrix-label';
            sumLabel.innerText = 'Sum across all units';
            sumLabel.style.color = '#aa0000';
            wrapper.appendChild(sumLabel);

            const sumGrid = document.createElement('div');
            sumGrid.className = 'matrix-container';
            sumGrid.style.gridTemplateColumns = `repeat(${{h_right}}, 40px)`;

            for (let i = 0; i < h_left; i++) {{
                for (let j = 0; j < h_right; j++) {{
                    const cell = document.createElement('div');
                    cell.className = 'matrix-cell';
                    const val = sumMatrix[i][j];

                    // Normalize by h_sum so the root (where h_sum=1) looks identical to its only unit
                    const avgVal = val / h_sum;
                    const intensity = avgVal > 0 ? Math.pow(avgVal, 0.3) : 0;
                    const colorVal = Math.floor(255 * (1 - Math.min(1.0, intensity)));

                    cell.style.backgroundColor = `rgb(${{colorVal}}, ${{colorVal}}, ${{colorVal}})`;
                    cell.title = "Sum Prob: " + val.toFixed(5);
                    sumGrid.appendChild(cell);
                }}
            }}
            wrapper.appendChild(sumGrid);
        }}
    </script>
</body>
</html>"""
    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"Interactive plot saved to {output_path}")


def plot_gaussian_leaf(leaf: GaussianLeafLayer, ax=None):
    """
    Plot the units of a GaussianLeafLayer.

    The leaf has ``num_groups`` groups and ``num_nodes`` nodes per group. One
    subplot is created per group (arranged vertically), and within each subplot
    all ``num_nodes`` Gaussians are overlaid. If a node has a finite support
    interval in ``node_supports``, the active region is drawn as a solid
    normalized curve with a light fill; the truncated tails are shown as dashed
    lines.

    Parameters
    ----------
    leaf : GaussianLeafLayer
        The leaf layer to visualize.
    ax : matplotlib.axes.Axes or array-like of Axes, optional
        If ``None``, a new figure with one axis per group is created. If a
        single ``Axes`` is provided, the leaf must have exactly one group. If an
        array-like of ``Axes`` is provided, its length must match
        ``leaf.num_groups``.

    Returns
    -------
    axes : numpy.ndarray of matplotlib.axes.Axes
        Array of axes, one per group.
    """
    if not isinstance(leaf, GaussianLeafLayer):
        raise ValueError("leaf must be a GaussianLeafLayer")

    num_groups = leaf.num_groups
    num_nodes = leaf.num_nodes

    means = leaf.means.detach().cpu().numpy()  # shape [G, N]
    stds = leaf.stddevs.detach().cpu().numpy()  # shape [G, N]
    supports = getattr(leaf, "node_supports", None)

    # Build / validate axes
    if ax is None:
        if num_groups == 1:
            _, axes = plt.subplots(figsize=(10, 6))
            axes = np.array([axes])
        else:
            _, axes = plt.subplots(num_groups, 1, figsize=(10, 3 + 3 * num_groups), sharex=True)
            axes = np.asarray(axes).reshape(-1)
    else:
        axes = np.asarray(ax).reshape(-1)
        if len(axes) == 1 and num_groups == 1:
            pass
        elif len(axes) != num_groups:
            raise ValueError(f"Expected {num_groups} axes (one per group), got {len(axes)}")

    # Global x-range covers all means +/- 4 stds and finite support bounds
    x_min = np.min(means - 4 * stds)
    x_max = np.max(means + 4 * stds)
    if supports and len(supports) == num_nodes:
        for i in range(num_nodes):
            interval = supports[i].intervals.get(leaf.var)
            if interval:
                if not math.isinf(interval.low):
                    x_min = min(x_min, interval.low - np.max(stds[:, i]))
                if not math.isinf(interval.high):
                    x_max = max(x_max, interval.high + np.max(stds[:, i]))
    x = np.linspace(x_min, x_max, 2000)

    for g in range(num_groups):
        ax_g = axes[g]
        for i in range(num_nodes):
            mu = means[g, i]
            sigma = stds[g, i]
            y_full = norm.pdf(x, mu, sigma)
            color = plt.cm.tab10(i % 10)

            interval = None
            if supports and len(supports) == num_nodes:
                interval = supports[i].intervals.get(leaf.var)

            if interval and (not math.isinf(interval.low) or not math.isinf(interval.high)):
                low = interval.low if not math.isinf(interval.low) else x_min
                high = interval.high if not math.isinf(interval.high) else x_max
                mask = (x >= low) & (x <= high)

                # Normalize the active region to match the truncated density
                # computed by the leaf.
                sqrt2 = math.sqrt(2)
                cdf_lo = (
                    0.5 * (1 + math.erf((low - mu) / (sigma * sqrt2)))
                    if not math.isinf(interval.low)
                    else 0.0
                )
                cdf_hi = (
                    0.5 * (1 + math.erf((high - mu) / (sigma * sqrt2)))
                    if not math.isinf(interval.high)
                    else 1.0
                )
                z = max(cdf_hi - cdf_lo, 1e-12)
                y_trunc = y_full / z

                # Dashed tails (unnormalized) and solid active region
                ax_g.plot(x[~mask], y_full[~mask], linestyle="--", color=color, alpha=0.5)
                ax_g.plot(
                    x[mask],
                    y_trunc[mask],
                    linestyle="-",
                    color=color,
                    linewidth=2,
                    label=f"Node {i}",
                )
                ax_g.fill_between(x[mask], 0, y_trunc[mask], color=color, alpha=0.2)
            else:
                ax_g.plot(x, y_full, linestyle="-", color=color, linewidth=2, label=f"Node {i}")
                ax_g.fill_between(x, 0, y_full, color=color, alpha=0.2)

        ax_g.set_title(f"Group {g} (Var: {leaf.var})")
        ax_g.set_xlabel("Value")
        ax_g.set_ylabel("Density")
        ax_g.legend(loc="best", fontsize="small")
        ax_g.grid(True, alpha=0.3)

    plt.tight_layout()
    return axes


def _gaussian_density_arrays(leaf: GaussianLeafLayer, x: np.ndarray) -> np.ndarray:
    """
    Compute the (possibly truncated) Gaussian densities for every group/node.

    Parameters
    ----------
    leaf : GaussianLeafLayer
        The leaf layer whose densities are computed.
    x : np.ndarray
        1-D array of evaluation points.

    Returns
    -------
    densities : np.ndarray
        Array of shape ``[num_groups, num_nodes, len(x)]``.
    """
    means = leaf.means.detach().cpu().numpy()
    stds = leaf.stddevs.detach().cpu().numpy()
    num_groups, num_nodes = means.shape
    supports = getattr(leaf, "node_supports", None)

    densities = np.zeros((num_groups, num_nodes, len(x)))
    for g in range(num_groups):
        for i in range(num_nodes):
            mu = means[g, i]
            sigma = stds[g, i]
            y_full = norm.pdf(x, mu, sigma)

            interval = None
            if supports and len(supports) == num_nodes:
                interval = supports[i].intervals.get(leaf.var)

            if interval and (not math.isinf(interval.low) or not math.isinf(interval.high)):
                low = interval.low if not math.isinf(interval.low) else x.min()
                high = interval.high if not math.isinf(interval.high) else x.max()
                mask = (x >= low) & (x <= high)

                sqrt2 = math.sqrt(2)
                cdf_lo = (
                    0.5 * (1 + math.erf((low - mu) / (sigma * sqrt2)))
                    if not math.isinf(interval.low)
                    else 0.0
                )
                cdf_hi = (
                    0.5 * (1 + math.erf((high - mu) / (sigma * sqrt2)))
                    if not math.isinf(interval.high)
                    else 1.0
                )
                z = max(cdf_hi - cdf_lo, 1e-12)
                y = np.zeros_like(y_full)
                y[mask] = y_full[mask] / z
            else:
                y = y_full

            densities[g, i] = y

    return densities


def plot_mixture_leaf(leaf: MixtureLeafLayer, ax=None, plot_components: bool = False):
    """
    Plot the output densities of a MixtureLeafLayer.

    The layer wraps a base Gaussian leaf and a set of mixture weights. One
    subplot is created per output group (arranged vertically), and within each
    subplot each output node is drawn as the weighted mixture of the base
    Gaussian components.

    Parameters
    ----------
    leaf : MixtureLeafLayer
        The mixture leaf layer to visualize.
    ax : matplotlib.axes.Axes or array-like of Axes, optional
        If ``None``, a new figure with one axis per output group is created. If
        an array-like of ``Axes`` is provided, its length must match the number
        of output groups.
    plot_components : bool, optional
        If ``True``, the weighted base components that make up each mixture are
        drawn as faint dashed lines behind the mixture density.

    Returns
    -------
    axes : numpy.ndarray of matplotlib.axes.Axes
        Array of axes, one per output group.
    """
    if not isinstance(leaf, MixtureLeafLayer):
        raise ValueError("leaf must be a MixtureLeafLayer")

    base_dist = leaf.base_dist
    if not isinstance(base_dist, GaussianLeafLayer):
        raise NotImplementedError(
            "plot_mixture_leaf currently only supports GaussianLeafLayer base distributions."
        )

    weights_obj = leaf.log_weights
    if weights_obj is None:
        raise ValueError("MixtureLeafLayer has no log_weights")
    if not isinstance(weights_obj, DenseWeights):
        raise NotImplementedError("plot_mixture_leaf currently only supports DenseWeights.")

    # Weights for a mixture leaf are [G_out, U_out, G_in, L, G_R, R].
    # The right-child dimensions belong to the (absent) second child and are
    # singletons for a leaf, so squeeze only those last two axes. This preserves
    # the output-node and input-node dimensions even when they are singletons.
    w = torch.exp(weights_obj.log_weights).detach().cpu().numpy()
    if w.ndim == 6:
        w = w.squeeze(axis=(4, 5))
    if w.ndim != 4:
        raise ValueError(
            f"Expected 4-D mixture weights after squeezing right-child dims, got shape {w.shape}"
        )

    num_groups_out, num_nodes_out, num_groups_in, num_nodes_in = w.shape

    # Global x-range from the base distribution
    means = base_dist.means.detach().cpu().numpy()
    stds = base_dist.stddevs.detach().cpu().numpy()
    x_min = np.min(means - 4 * stds)
    x_max = np.max(means + 4 * stds)
    supports = getattr(base_dist, "node_supports", None)
    if supports and len(supports) == base_dist.num_nodes:
        for i in range(base_dist.num_nodes):
            interval = supports[i].intervals.get(base_dist.var)
            if interval:
                if not math.isinf(interval.low):
                    x_min = min(x_min, interval.low - np.max(stds[:, i]))
                if not math.isinf(interval.high):
                    x_max = max(x_max, interval.high + np.max(stds[:, i]))
    x = np.linspace(x_min, x_max, 2000)

    base_densities = _gaussian_density_arrays(base_dist, x)

    # Build / validate axes
    if ax is None:
        if num_groups_out == 1:
            _, axes = plt.subplots(figsize=(10, 6))
            axes = np.array([axes])
        else:
            _, axes = plt.subplots(
                num_groups_out, 1, figsize=(10, 3 + 3 * num_groups_out), sharex=True
            )
            axes = np.asarray(axes).reshape(-1)
    else:
        axes = np.asarray(ax).reshape(-1)
        if len(axes) != num_groups_out:
            raise ValueError(f"Expected {num_groups_out} axes (one per group), got {len(axes)}")

    for g in range(num_groups_out):
        ax_g = axes[g]
        for h in range(num_nodes_out):
            mix_dens = np.zeros_like(x)
            for g_in in range(num_groups_in):
                for h_in in range(num_nodes_in):
                    weight = w[g, h, g_in, h_in]
                    mix_dens += weight * base_densities[g_in, h_in]

            color = plt.cm.tab10(h % 10)
            ax_g.plot(
                x,
                mix_dens,
                linestyle="-",
                color=color,
                linewidth=2,
                label=f"Mixture node {h}",
            )

            if plot_components:
                for g_in in range(num_groups_in):
                    for h_in in range(num_nodes_in):
                        weight = w[g, h, g_in, h_in]
                        if weight > 1e-4:
                            ax_g.plot(
                                x,
                                weight * base_densities[g_in, h_in],
                                linestyle="--",
                                color=color,
                                alpha=0.3,
                            )

        ax_g.set_title(f"Group {g} (Var: {base_dist.var})")
        ax_g.set_xlabel("Value")
        ax_g.set_ylabel("Density")
        ax_g.legend(loc="best", fontsize="small")
        ax_g.grid(True, alpha=0.3)

    plt.tight_layout()
    return axes


def plot_spline_leaf(leaf: SplineLeafLayer, ax=None):
    """
    Plot the per-child densities of a SplineLeafLayer.

    The leaf has ``num_groups`` groups and ``num_nodes`` spline nodes per
    group. One subplot is created per group (arranged vertically), and within
    each subplot all ``num_nodes`` piecewise-log-linear densities are overlaid.
    Learned split points are drawn as dotted vertical lines.

    Parameters
    ----------
    leaf : SplineLeafLayer
        The spline leaf layer to visualize.
    ax : matplotlib.axes.Axes or array-like of Axes, optional
        If ``None``, a new figure with one axis per group is created. If an
        array-like of ``Axes`` is provided, its length must match
        ``leaf.num_groups``.

    Returns
    -------
    axes : list of matplotlib.axes.Axes
        One axis per group.
    """
    if not isinstance(leaf, SplineLeafLayer):
        raise ValueError("leaf must be a SplineLeafLayer")

    n_groups = leaf.num_groups
    if ax is None:
        _, axes = plt.subplots(n_groups, 1, figsize=(6, 2 * n_groups), sharex=True)
        if n_groups == 1:
            axes = [axes]
    else:
        axes = np.atleast_1d(ax).tolist()

    with torch.no_grad():
        b = leaf._split_points().cpu().numpy()  # [G, N-1]

    # Plotting range from the finite split points, padded so the exponential
    # tails are visible on both sides.
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
        for j in range(leaf.num_nodes):
            ax_g.plot(xs_np, probs[:, g, j], alpha=0.7, lw=1.0)
        for split in split_vals:
            if np.isfinite(split):
                ax_g.axvline(split, color="gray", linestyle=":", lw=0.8)
        ax_g.set_ylabel("density")
    axes[-1].set_xlabel("value")
    return axes
