"""Per-X-panel overlay of the P(Y|x,z)P(z) mixture components of P(Y|do(X)).

Shared by ``scratch/probe_y_given_xz.py`` and the backdoor experiment runner
(gated by ``evaluation.plot_components``).  For every X-leaf support point the
plot overlays the compiled product components P(Y|x,z)P(z) integrated over each
Z-leaf support interval (solid, colored by z) against their empirical
counterparts (dashed) and the analytic truth (dotted) from the saved SCM
pickle — closed-form interval-truncated Gaussian moments.  Both are computed as
interval integrals, so summing the components over the z-partition recovers
P(Y|do(x)) on each side.  The learned do-density (black), the analytic
P(Y|do(X)) (black dashed), the empirical target from the interventional data
(gray band), and the high-accuracy empirical backdoor sum (red, fine z-grid
quadrature of kernel-estimated components) are drawn per panel.

The empirical helpers follow the probe and use only the FIRST confounder's
column / z-interval partition; the analytic helpers assume the
single-confounder linear-Gaussian form ``Y = a + b_x X + b_z Z + N(0, s_y)``
with ``Z`` Gaussian.  When the SCM pickle's Y mechanism has more than one
confounder parent (or is not of that form) the true curves are skipped with a
single logged warning and only circuit + empirical curves are drawn.
"""

import math
import os
import pickle
import time
from typing import Any, Dict, Optional

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.utils.data import resolve_dataset_paths
from src.logger import logger as g_logger
from src.symbolic.arithmetic.circuit import eval_circuit
from src.symbolic.arithmetic.query import compile_query
from src.symbolic.id_ast import make_cond, make_marg, make_p, make_prod

from .circuit_inspect import choose_x_values_from_intervals, get_x_leaf_support_intervals
from .empirical import _y_kde


logger = g_logger.getChild("component_plots")

# Estimator constants (kept in code, not config).
WINDOW = 3000  # X-window size for the empirical do slices
N_BOOT = 200  # bootstrap reps for the empirical do bands
N_GRID_Y = 200  # shared Y grid
N_QUAD = 5  # Gauss-Legendre points per z-support interval
MAX_EVAL_ROWS = 32768  # row cap per batched circuit evaluation
DENSITY_YLIM_PERCENTILE = 99.9  # shared y-cap for the components figure
# Sample-count floors: empirical estimates below these are unreliable and are
# omitted from the plots (the circuit curves are always drawn).
MIN_COMPONENT_EFF_SAMPLES = 50  # effective obs samples per (x, z-interval) component
MIN_DO_SAMPLES = 100  # interventional samples within +-h_x of the panel x
Z_GRID_POINTS = 161  # z-resolution of the high-accuracy empirical backdoor sum
MIN_SUM_WINDOW_SAMPLES = 500  # obs x-window size below which the sum is skipped


def compile_component_queries(ac, data_info: Dict[str, Any]) -> Dict[str, Any]:
    """Compile the do-circuit and the product circuit P(Y|X,Z)*P(Z).

    The product is the per-z integrand of the backdoor adjustment
    sum_z P(Y|X,z) P(z), compiled without marginalizing Z; integrating it
    over a z-support interval gives one weighted mixture component of
    P(Y|do(X)).
    """
    V_names = data_info["V_names"]
    z_names = data_info["z_names"]
    var_to_id = data_info["var_to_id"]

    p_all = make_p(V_names)
    joint_ast = make_marg(V_names - {"X", "Y"} - set(z_names), p_all)
    cond_ast = make_cond({"Y"}, {"X"} | set(z_names), joint_ast)
    pz_ast = make_marg({"X", "Y"}, p_all)
    prod_ast = make_prod([cond_ast, pz_ast])
    do_ast = make_marg(set(z_names), prod_ast)

    logger.info("Compiling query circuits...")
    q_prod_ac = compile_query(prod_ast, ac, var_to_id)
    q_do_ac = compile_query(do_ast, ac, var_to_id)
    logger.info("Query circuits compiled.")
    return {"q_prod_ac": q_prod_ac, "q_do_ac": q_do_ac}


def eval_yxz_at(q_ac, x_val, z_val, grid_y, data_info, device) -> np.ndarray:
    """Evaluate a compiled (x, z, y)-query on a Y-grid at fixed ``x_val``, ``z_val``.

    The do-circuit ignores Z, so any ``z_val`` works for do evaluations.
    """
    pts = np.zeros((len(grid_y), data_info["n_vars"]))
    pts[:, data_info["x_id"]] = x_val
    pts[:, data_info["y_id"]] = grid_y
    for zid in data_info["z_ids"]:
        pts[:, zid] = z_val
    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
    with torch.no_grad():
        logp = eval_circuit(q_ac, pts_t, verbose=False, keep_intermediates=False).squeeze()
    return np.atleast_1d(np.exp(logp.detach().cpu().numpy()))


def circuit_components_all(
    q_prod_ac, comp_x, z_intervals, z_clip, grid_y, data_info, device
) -> list:
    """Batched interval integrals of the compiled product circuit.

    Returns ``comps[i][j]`` — the component density on ``grid_y`` for
    z-interval ``i`` at panel ``j`` — via Gauss-Legendre quadrature with
    ``N_QUAD`` points.  All (x, z, y) evaluations are fused into a few large
    batched circuit calls rather than one small call per interval.
    """
    n_x, n_z, n_y = len(comp_x), len(z_intervals), len(grid_y)
    ts, ws = np.polynomial.legendre.leggauss(N_QUAD)
    zpts = np.empty(n_z * N_QUAD)
    wpts = np.zeros((n_z, N_QUAD))
    for i, (z_lo, z_hi) in enumerate(z_intervals):
        lo = max(float(z_lo), z_clip[0])
        hi = min(float(z_hi), z_clip[1])
        if hi > lo:
            zpts[i * N_QUAD : (i + 1) * N_QUAD] = 0.5 * (hi - lo) * ts + 0.5 * (hi + lo)
            wpts[i] = 0.5 * (hi - lo) * ws

    n_rows = n_x * n_z * N_QUAD * n_y
    log_dens = np.empty(n_rows, dtype=np.float32)
    x_id, y_id, z_ids = data_info["x_id"], data_info["y_id"], data_info["z_ids"]
    for start in range(0, n_rows, MAX_EVAL_ROWS):
        end = min(start + MAX_EVAL_ROWS, n_rows)
        idx = np.arange(start, end)
        iy = idx % n_y
        iq = (idx // n_y) % N_QUAD
        iz = (idx // (n_y * N_QUAD)) % n_z
        ix = idx // (n_y * N_QUAD * n_z)
        pts = np.zeros((end - start, data_info["n_vars"]))
        pts[:, x_id] = comp_x[ix]
        pts[:, y_id] = grid_y[iy]
        for zid in z_ids:
            pts[:, zid] = zpts[iz * N_QUAD + iq]
        pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
        with torch.no_grad():
            logp = eval_circuit(q_prod_ac, pts_t, verbose=False, keep_intermediates=False).squeeze()
        log_dens[start:end] = logp.detach().cpu().numpy()
    dens = np.exp(log_dens).reshape(n_x, n_z, N_QUAD, n_y)
    comps = (dens * wpts[None, :, :, None]).sum(axis=2)  # [n_x, n_z, n_y]
    return [[comps[j, i] for j in range(n_x)] for i in range(n_z)]


def gt_component_in_interval(df: pd.DataFrame, x_val, z_lo, z_hi, h_x, grid_y, z_col: str = "Z0"):
    """Empirical component integral over a z-support interval [z_lo, z_hi).

    Numerically computes ∫ P(Y|x,z) p(z) dz over the interval as
    ``P(Z in [z_lo, z_hi))`` times a kernel-weighted conditional of Y given
    X ≈ x within the interval (Gaussian kernel in X, bandwidth ``h_x``).
    Returns ``(component_density, z_mass, n_in_interval, n_eff)`` where the
    density is ``None`` when fewer than ``MIN_COMPONENT_EFF_SAMPLES``
    effective samples are available — the caller then omits the curve.
    """
    zs = df[z_col].values
    mask = (zs >= z_lo) & (zs < z_hi)
    n_in = int(mask.sum())
    mass = float(mask.mean())
    if n_in < 20:
        return None, mass, n_in, 0.0
    xs = df["X"].values[mask]
    ys = df["Y"].values[mask]
    w = np.exp(-0.5 * ((xs - x_val) / h_x) ** 2)
    w_sum = float(w.sum())
    n_eff = w_sum**2 / float((w**2).sum())
    if n_eff < MIN_COMPONENT_EFF_SAMPLES:
        return None, mass, n_in, float(n_eff)
    mu = float(np.average(ys, weights=w))
    sd = float(np.sqrt(np.average((ys - mu) ** 2, weights=w)))
    h_y = max(0.9 * sd * n_eff ** (-0.2), 1e-6)
    diffs = grid_y[:, None] - ys[None, :]
    dens = (w[None, :] * np.exp(-0.5 * (diffs / h_y) ** 2)).sum(axis=1) / (
        w_sum * h_y * np.sqrt(2.0 * np.pi)
    )
    return dens * mass, mass, n_in, float(n_eff)


def trapz_norm(density: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Normalize a density on ``grid_y`` by its trapezoid integral."""
    integral = np.trapezoid(density, grid_y)
    return density / integral if integral > 0 else density


# ---------------------------------------------------------------------------
# Analytic ground truth from the saved linear-Gaussian SCM pickle
# ---------------------------------------------------------------------------


def load_true_scm_params(dataset: str, data_dir: str = "data") -> Optional[Dict[str, float]]:
    """Extract the linear-Gaussian parameters behind the analytic truth.

    Expects Y's mechanism to be linear in X and ONE confounder Z
    (``Y = a_y + b_x X + b_z Z + N(loc, s_y)``, ``Z = N(mu_z, sd_z)``) and
    reads the coefficients from the SCM pickle next to the datasets.  Returns
    ``None`` (after logging a single warning) when the Y mechanism has more
    than one confounder parent or is not of the expected form — the true
    curves are then skipped while circuit + empirical curves are still drawn.
    """
    paths = resolve_dataset_paths(data_dir, dataset)
    path = paths["scm"]
    with open(path, "rb") as f:
        scm = pickle.load(f)
    y_mech = scm.get_node_data("Y")
    y_logic = getattr(y_mech, "logic", None)
    coefs = dict(y_logic.coefficients) if y_logic is not None else None
    if not coefs or "X" not in coefs:
        logger.warning(
            "Y mechanism in %s is not linear in X with confounder parents; "
            "skipping analytic-truth curves.",
            path,
        )
        return None
    b_x = float(coefs.pop("X"))
    if len(coefs) != 1:
        logger.warning(
            "Y mechanism in %s has %d confounder parents (expected 1); "
            "skipping analytic-truth curves.",
            path,
            len(coefs),
        )
        return None
    (z_name, b_z) = next(iter(coefs.items()))
    z_mech = scm.get_node_data(z_name)
    z_logic = getattr(z_mech, "logic", None)
    return {
        "a_y": float(y_logic.intercept) + float(y_mech.noise_dist.loc),
        "b_x": b_x,
        "b_z": float(b_z),
        "s_y": float(y_mech.noise_dist.scale),
        "z_mean": (float(z_logic.intercept) if z_logic is not None else 0.0)
        + float(z_mech.noise_dist.loc),
        "z_sd": float(z_mech.noise_dist.scale),
    }


def _norm_pdf(t: float) -> float:
    t = min(max(t, -40.0), 40.0)
    return math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)


def _norm_cdf(t: float) -> float:
    t = min(max(t, -40.0), 40.0)
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


def true_component_in_interval(x_val, z_lo, z_hi, tp, grid_y):
    """Analytic P(Y|x, Z in [z_lo, z_hi)) * P(Z in [z_lo, z_hi)) density.

    Z is Gaussian, so the interval-truncated moments are closed-form; Y|x,z
    is linear-Gaussian, hence Y|x, Z in interval is Gaussian with
    mean/variance propagated through the truncation moments.
    Returns ``(density, mass)``.
    """
    mu_z, sd_z = tp["z_mean"], tp["z_sd"]
    a = (z_lo - mu_z) / sd_z if math.isfinite(z_lo) else -40.0
    b = (z_hi - mu_z) / sd_z if math.isfinite(z_hi) else 40.0
    fa, fb = _norm_cdf(a), _norm_cdf(b)
    mass = fb - fa
    pa, pb = _norm_pdf(a), _norm_pdf(b)
    t_mean = (pa - pb) / mass
    t_var = 1.0 + (a * pa - b * pb) / mass - t_mean**2
    mu_y = tp["a_y"] + tp["b_x"] * x_val + tp["b_z"] * (mu_z + sd_z * t_mean)
    var_y = tp["s_y"] ** 2 + tp["b_z"] ** 2 * sd_z**2 * t_var
    dens = mass * np.exp(-0.5 * (grid_y - mu_y) ** 2 / var_y) / math.sqrt(2.0 * math.pi * var_y)
    return dens, mass


def true_do_density(x_val, tp, grid_y):
    """Analytic P(Y|do(X=x)): N(a_y + b_x x + b_z mu_z, s_y^2 + b_z^2 sd_z^2)."""
    mu = tp["a_y"] + tp["b_x"] * x_val + tp["b_z"] * tp["z_mean"]
    var = tp["s_y"] ** 2 + tp["b_z"] ** 2 * tp["z_sd"] ** 2
    return np.exp(-0.5 * (grid_y - mu) ** 2 / var) / math.sqrt(2.0 * math.pi * var)


def empirical_do_sum(
    df: pd.DataFrame, x_val, h_x, grid_y, z_grid, z_col: str = "Z0"
) -> Optional[np.ndarray]:
    """High-accuracy empirical P(Y|do(x)) = sum_z P(Y|x,z) P(z) from obs data.

    Fine z-grid quadrature of ``p_hat(z) * p_hat(y|x,z)``: ``p_hat(z)`` is a
    Silverman KDE of the observational z's; ``p_hat(y|x,z)`` uses Gaussian
    kernels in X (bandwidth ``h_x``) and Z (Silverman bandwidth) plus a
    weighted Y-KDE over the x-window samples (|x_i - x| <= 4 h_x).  Unlike
    the per-leaf-interval components there is no z-binning, so the sum has
    only KDE smoothing bias.  Returns the density on ``grid_y``, or ``None``
    when the x-window has too few samples.
    """
    xs = df["X"].values
    ys = df["Y"].values
    zs = df[z_col].values
    n = len(zs)

    dx = (xs - x_val) / h_x
    win = np.abs(dx) <= 4.0
    n_win = int(win.sum())
    if n_win < MIN_SUM_WINDOW_SAMPLES:
        logger.info(
            "Skipping empirical sum at x=%.2f: only %d obs samples within +-%.2f (need %d)",
            x_val,
            n_win,
            4.0 * h_x,
            MIN_SUM_WINDOW_SAMPLES,
        )
        return None
    yw, zw = ys[win], zs[win]
    wx = np.exp(-0.5 * dx[win] ** 2)

    # Silverman bandwidths for z (marginal + kernel in the conditional).
    sd_z = float(np.std(zs))
    iqr_z = float(np.subtract(*np.percentile(zs, [75, 25])))
    h_z = max(0.9 * min(sd_z, iqr_z / 1.34) * n ** (-0.2), 1e-6)

    # p_hat(z) on the z-grid, trapezoid-normalized.
    dz = (z_grid[:, None] - zs[None, :]) / h_z
    p_z = np.exp(-0.5 * dz**2).sum(axis=1) / (n * h_z * math.sqrt(2.0 * math.pi))
    p_z /= np.trapezoid(p_z, z_grid)

    # Y-KDE kernels of the window samples on grid_y.
    sd_y = float(np.std(yw))
    iqr_y = float(np.subtract(*np.percentile(yw, [75, 25])))
    h_y = max(0.9 * min(sd_y, iqr_y / 1.34) * n_win ** (-0.2), 1e-6)
    ky = np.exp(-0.5 * ((grid_y[:, None] - yw[None, :]) / h_y) ** 2)

    # Joint (x, z) weights of the window samples per z-grid point.
    zw_dz = (z_grid[:, None] - zw[None, :]) / h_z
    w = wx[None, :] * np.exp(-0.5 * zw_dz**2)
    w_sum = w.sum(axis=1)
    cond = (w @ ky.T) / np.maximum(w_sum[:, None], 1e-300)  # [n_z, n_y]

    # Backdoor sum with trapezoid weights over the z-grid.
    dz_step = float(z_grid[1] - z_grid[0])
    trap = np.full_like(z_grid, dz_step)
    trap[0] *= 0.5
    trap[-1] *= 0.5
    return (cond * (p_z * trap)[:, None]).sum(axis=0) / (h_y * math.sqrt(2.0 * math.pi))


def py_do_window(df: pd.DataFrame, x_val: float, grid_y: np.ndarray, h_x: float, seed: int = 0):
    """Empirical P(Y | X=x) slice via the nearest-WINDOW samples in X only.

    Returns ``(density, lo, hi)``, or ``None`` when fewer than
    ``MIN_DO_SAMPLES`` interventional samples fall within ``+-h_x`` of
    ``x_val`` (tails) — the window would be dominated by far-away points.
    """
    xs = df["X"].values
    ys = df["Y"].values
    n_near = int((np.abs(xs - x_val) <= h_x).sum())
    if n_near < MIN_DO_SAMPLES:
        logger.info(
            "Skipping empirical do curve at x=%.2f: only %d samples within +-%.2f (need %d)",
            x_val,
            n_near,
            h_x,
            MIN_DO_SAMPLES,
        )
        return None
    order = np.argsort(np.abs(xs - x_val), kind="stable")[:WINDOW]
    y_win = ys[order]
    density = _y_kde(y_win, grid_y)
    rng = np.random.default_rng(seed)
    boots = np.empty((N_BOOT, len(grid_y)))
    for b in range(N_BOOT):
        boots[b] = _y_kde(y_win[rng.integers(0, len(y_win), len(y_win))], grid_y)
    return density, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def plot_product_components(
    comp_x,
    comp_z,
    grid_y,
    prod_circ,
    prod_emp,
    prod_true,
    do_curves,
    do_true,
    do_emp_sum,
    emp_do,
    exp_id,
    context,
    output_dir: str,
):
    """Per-x overlay of the P(Y|x,z)P(z) mixture components vs the do-density.

    One panel per X-leaf support point (square-ish grid); each panel overlays
    the compiled product components for every Z-leaf support point.  Solid
    curves: circuit; dashed: empirical counterpart; dotted: analytic truth
    from the SCM (skipped where ``prod_true``/``do_true`` entries are None).
    Black: the learned do-density, the analytic truth (dashed), and the
    empirical target (gray band) — the z-integral the components must sum to.
    Red: the high-accuracy empirical backdoor sum (fine z-grid quadrature, no
    binning).  The shared density axis is capped at a high percentile so one
    spiky component does not squash the rest.  Saved as
    ``{exp_id}_product_components.png`` in ``output_dir``.
    """
    n_x, n_z = len(comp_x), len(comp_z)
    z_norm = plt.Normalize(vmin=float(np.min(comp_z)), vmax=float(np.max(comp_z)))
    colors = plt.cm.viridis(z_norm(comp_z))
    n_cols = math.ceil(math.sqrt(n_x))
    n_rows = math.ceil(n_x / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), sharex=True, sharey=True
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for j, x_val in enumerate(comp_x):
        ax = axes_flat[j]
        if emp_do[j] is not None:
            ax.fill_between(
                grid_y, emp_do[j][1], emp_do[j][2], color="black", alpha=0.10, linewidth=0
            )
            ax.plot(
                grid_y,
                emp_do[j][0],
                color="black",
                alpha=0.35,
                linewidth=1.5,
                label="GT P(Y|do(X))",
            )
        ax.plot(
            grid_y,
            do_curves[j],
            color="black",
            alpha=0.9,
            linewidth=1.2,
            label="Learned P(Y|do(X))",
        )
        if do_true[j] is not None:
            ax.plot(
                grid_y,
                do_true[j],
                color="black",
                alpha=0.9,
                linewidth=1.2,
                linestyle="--",
                label="True P(Y|do(X))",
            )
        if do_emp_sum[j] is not None:
            ax.plot(
                grid_y,
                do_emp_sum[j],
                color="tab:red",
                alpha=0.85,
                linewidth=1.6,
                label="Sum of empirical components",
            )
        for i in range(n_z):
            ax.plot(
                grid_y,
                prod_circ[i][j],
                color=colors[i],
                alpha=0.9,
                linewidth=1.0,
                label="component (circuit)" if i == 0 else None,
            )
            if prod_emp[i][j] is not None:
                ax.plot(
                    grid_y,
                    prod_emp[i][j],
                    color=colors[i],
                    alpha=0.45,
                    linewidth=1.0,
                    linestyle="--",
                    label="component (GT)" if i == 0 else None,
                )
            if prod_true[i][j] is not None:
                ax.plot(
                    grid_y,
                    prod_true[i][j],
                    color=colors[i],
                    alpha=0.9,
                    linewidth=1.0,
                    linestyle=":",
                    label="component (true)" if i == 0 else None,
                )
        ax.set_title(f"x={x_val:.2f}", fontsize=9)
        if j % n_cols == 0:
            ax.set_ylabel("density")
        if j >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Y")
    for k in range(n_x, len(axes_flat)):
        axes_flat[k].axis("off")
    pool = np.concatenate(
        [np.asarray(v).ravel() for row in prod_circ for v in row]
        + [np.asarray(v).ravel() for row in prod_emp for v in row if v is not None]
        + [np.asarray(v).ravel() for row in prod_true for v in row if v is not None]
        + [np.asarray(c).ravel() for c in do_curves]
        + [np.asarray(c).ravel() for c in do_true if c is not None]
        + [np.asarray(c).ravel() for c in do_emp_sum if c is not None]
        + [np.asarray(t).ravel() for band in emp_do if band is not None for t in band]
    )
    pool = pool[np.isfinite(pool)]
    if pool.size:
        axes_flat[0].set_ylim(0.0, float(np.percentile(pool, DENSITY_YLIM_PERCENTILE)))
    handles, labels = axes_flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    fig.suptitle(context + " | components P(Y|x,z)P(z) at Z/X leaf supports", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(output_dir, f"{exp_id}_product_components.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved product-components plot to %s", path)


def plot_components(
    ac,
    data_info: Dict[str, Any],
    cfg: Dict[str, Any],
    device,
    output_dir: str,
    exp_id: str,
) -> None:
    """Driver: component-decomposition plot of P(Y|do(X)) for the runner.

    Compiles the product/do query circuits, picks one panel per X-leaf support
    and one component per Z-leaf (first confounder) support interval, then
    evaluates the circuit, empirical, and (single-confounder linear-Gaussian)
    analytic-truth curves per panel and calls :func:`plot_product_components`.
    """
    df_obs = data_info["df_obs"]
    dataset = cfg["dataset"]["dataset"]
    data_dir = cfg["dataset"].get("data_dir", "data")
    model_cfg = cfg["model"]
    context = f"{dataset} | num_nodes={model_cfg['num_nodes']} "

    queries = compile_component_queries(ac, data_info)
    q_prod_ac = queries["q_prod_ac"]
    q_do_ac = queries["q_do_ac"]

    # Shared Y grid: cover both tails of the pooled observational and
    # interventional Y data; shared across all panels.
    pooled_y = np.concatenate([df_obs["Y"].values, data_info["df_do"]["Y"].values])
    y_lo, y_hi = float(pooled_y.min()), float(pooled_y.max())
    grid_y = np.linspace(y_lo, y_hi, N_GRID_Y)
    logger.info("Y grid: [%.2f, %.2f] (%d points)", y_lo, y_hi, N_GRID_Y)

    # Component grid: one point per X / Z (first confounder) leaf support.
    z_col = data_info["z_names"][0]
    x_intervals = get_x_leaf_support_intervals(ac, data_info["x_id"])
    comp_x, _ = choose_x_values_from_intervals(x_intervals, df_obs=df_obs, x_id_name="X")
    z_intervals = get_x_leaf_support_intervals(ac, data_info["z_ids"][0])
    comp_z, _ = choose_x_values_from_intervals(z_intervals, df_obs=df_obs, x_id_name=z_col)
    order_x = np.argsort(comp_x)
    comp_x, x_intervals = comp_x[order_x], [x_intervals[i] for i in order_x]
    order_z = np.argsort(comp_z)
    comp_z, z_intervals = comp_z[order_z], [z_intervals[i] for i in order_z]
    logger.info(
        "Component grid: %d X-leaf supports, %d Z-leaf supports",
        len(comp_x),
        len(comp_z),
    )

    # X kernel bandwidth per panel from the spacing of neighboring supports.
    spacing = np.diff(comp_x)
    h_xs = np.empty_like(comp_x)
    h_xs[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    h_xs[0], h_xs[-1] = spacing[0], spacing[-1]
    z_clip = (float(df_obs[z_col].values.min()), float(df_obs[z_col].values.max()))
    z_grid = np.linspace(z_clip[0], z_clip[1], Z_GRID_POINTS)

    # Circuit components: all interval integrals in a few batched evals.
    start = time.perf_counter()
    comp_prod_circ = circuit_components_all(
        q_prod_ac, comp_x, z_intervals, z_clip, grid_y, data_info, device
    )
    logger.info("Circuit components (batched) took %.1fs", time.perf_counter() - start)

    # Per panel: do curves + empirical/true components.
    tp = load_true_scm_params(dataset, data_dir=data_dir)
    if tp is not None:
        logger.info(
            "True SCM: Y = %.3f + %.2f X %+.2f Z + N(0, %.2f) | Z ~ N(%.2f, %.2f)",
            tp["a_y"],
            tp["b_x"],
            tp["b_z"],
            tp["s_y"],
            tp["z_mean"],
            tp["z_sd"],
        )
    comp_prod_emp = [[None] * len(comp_x) for _ in range(len(comp_z))]
    comp_prod_true = [[None] * len(comp_x) for _ in range(len(comp_z))]
    comp_do, comp_do_true, comp_do_emp_sum, comp_emp_do = [], [], [], []
    for j, x_val in enumerate(comp_x):
        comp_do.append(
            trapz_norm(eval_yxz_at(q_do_ac, x_val, 0.0, grid_y, data_info, device), grid_y)
        )
        comp_do_true.append(true_do_density(x_val, tp, grid_y) if tp is not None else None)
        emp_sum = empirical_do_sum(df_obs, x_val, float(h_xs[j]), grid_y, z_grid, z_col=z_col)
        comp_do_emp_sum.append(emp_sum)
        comp_emp_do.append(py_do_window(data_info["df_do"], x_val, grid_y, float(h_xs[j]), seed=j))
        for i, (z_lo, z_hi) in enumerate(z_intervals):
            d_emp, _mass, _n_in, _n_eff = gt_component_in_interval(
                df_obs, x_val, z_lo, z_hi, float(h_xs[j]), grid_y, z_col=z_col
            )
            comp_prod_emp[i][j] = d_emp
            comp_prod_true[i][j] = (
                true_component_in_interval(x_val, z_lo, z_hi, tp, grid_y)[0]
                if tp is not None
                else None
            )
        sum_circ = float(
            sum(np.trapezoid(comp_prod_circ[i][j], grid_y) for i in range(len(comp_z)))
        )
        logger.info(
            "Components x=%.2f | circuit mass=%.3f | empirical sum mass=%s",
            x_val,
            sum_circ,
            (
                f"{np.trapezoid(emp_sum, grid_y):.3f}"
                if emp_sum is not None
                else "n/a (insufficient samples)"
            ),
        )

    plot_product_components(
        comp_x,
        comp_z,
        grid_y,
        comp_prod_circ,
        comp_prod_emp,
        comp_prod_true,
        comp_do,
        comp_do_true,
        comp_do_emp_sum,
        comp_emp_do,
        exp_id,
        context,
        output_dir,
    )
