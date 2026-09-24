"""Generate a dataset for a custom skeleton that is neither backdoor nor frontdoor:

    X -> M -> Y        (mediated treatment effect)
    X <- U -> Y        (hidden confounder U; latent projection -> bidirected X <-> Y)
    Z -> X, Z -> M     (observed parent of treatment and mediator)

U is hidden: dropped from both CSVs. Same CLI mechanism/generation knobs as
``generate_synthetic_data.py`` (all variables are continuous here: linear-GMM
mechanisms with exact ground truth). Treatment = X (clamped in the
interventional CSV), outcome = Y. Also writes heatmap / slice plots per
(X, Y) exactly like ``explore_synthetic_data.py``.

Example:
    python generate_custom_skeleton.py --n_samples 100000 --seed 27
"""

import argparse
import os

from scripts.explore_synthetic_data import plot_heatmaps, plot_slices
from scripts.generate_synthetic_data import (
    CASE_GUIDE,
    add_generation_arguments,
    generate_paired_datasets,
    mechanism_kwargs_from_args,
    save_datasets,
)
from src.construction import effect_gap
from src.construction.skeleton import SCMSkeleton, VariableSpec


def build_skeleton() -> SCMSkeleton:
    variables = [
        VariableSpec("U", "continuous", None, hidden=True),
        VariableSpec("Z", "continuous", None),
        VariableSpec("X", "continuous", None),
        VariableSpec("M", "continuous", None),
        VariableSpec("Y", "continuous", None),
    ]
    edges = [("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y"), ("Z", "X"), ("Z", "M")]
    return SCMSkeleton(variables, edges)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=CASE_GUIDE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_generation_arguments(parser)
    parser.add_argument("--n_contexts", type=int, default=5, help="X contexts for 1D slices")
    parser.add_argument("--bins", type=int, default=60, help="Bins per continuous axis")
    args = parser.parse_args()

    # Structural counts for the output name only (the skeleton is fixed above).
    args.skeleton = "custom"
    args.kind = "continuous"
    args.z_kind = args.x_kind = args.y_kind = args.m_kind = args.w_kind = None
    args.n_confounders = 2  # U (hidden) + Z (observed)
    args.n_treatments = 1
    args.n_outcomes = 1
    args.n_bystanders = 0
    args.n_mediators = 1

    skeleton = build_skeleton()
    mechanism_kwargs = mechanism_kwargs_from_args(args)

    df_obs, df_do, scm = generate_paired_datasets(
        skeleton,
        treatments=("X",),
        n_samples=args.n_samples,
        seed=args.seed,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        **mechanism_kwargs,
    )
    paths = save_datasets(df_obs, df_do, scm, skeleton, args, mechanism_kwargs)

    gt = scm.ground_truth()
    # Full dataset name (structure + mechanism knobs), so different mechanism
    # configs never share a plots directory.
    full_name = os.path.basename(paths["observational"])[len("observational_") : -len(".csv")]
    plots_dir = os.path.join(args.output_dir, f"plots_{full_name}")
    os.makedirs(plots_dir, exist_ok=True)
    spec_t, spec_y = skeleton.spec_of("X"), skeleton.spec_of("Y")
    title = f"{paths['base_name']}  (X -> Y)"
    heatmap_path = os.path.join(plots_dir, "heatmap_X_Y.png")
    slices_path = os.path.join(plots_dir, "slices_X_Y.png")
    plot_heatmaps(df_obs, df_do, gt, "X", "Y", spec_t, spec_y, args.bins, title, heatmap_path)
    plot_slices(
        df_obs, df_do, gt, "X", "Y", spec_t, spec_y, args.n_contexts, args.bins, title, slices_path
    )
    print(f"Saved heatmap to {heatmap_path}")
    print(f"Saved slices to {slices_path}")

    gap = effect_gap(scm, "X", "Y")
    print(
        f"[X -> Y] effect gap: mean L1 = {gap['mean_l1']:.3f}, "
        f"max L1 = {gap['max_l1']:.3f} (total effect through M)"
    )


if __name__ == "__main__":
    main()
