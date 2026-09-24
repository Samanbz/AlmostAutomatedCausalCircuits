"""Generate a dataset for the "collider door" structure (neither backdoor, nor
frontdoor, nor napkin):

    Z ---> M ---> Y        (X's effect travels through the collider M)
            ^
    X ------┘

    U1: Z <-> Y,  U2: X <-> Y   (hidden confounders; latent projection)

T-ID identifies P(Y|do(X)) with determinism chain {{Z,X} ⊂ {Z,M,X}} (K = 3):

    P(Y|do(X)) = sum_{z,m} P(m|z,x) * sum_x' P(z,x') * P(y|z,m,x')

U1, U2 are hidden: dropped from both CSVs. Two variants:

- continuous (default): linear-GMM mechanisms (single component) with exact
  ground truth; coefficients chosen so the observational slope is strongly
  inflated by the X<->Y confounding (obs ~ 2.85 x vs do ~ 1.0 x).
- ``--mixed``: U1, U2, Z, X, M binary, Y CLG; hand-designed mechanisms
  (MIXED_PARAMS) with a sign-flip contrast — observational association
  E[Y|X=1] - E[Y|X=0] = +1.275 vs causal effect E[Y|do(X=1)] - E[Y|do(X=0)]
  = -1.125.

Treatment = X (clamped in the interventional CSV), outcome = Y. Also writes
heatmap / slice plots per (X, Y) exactly like ``explore_synthetic_data.py``.

Example:
    python generate_colliderdoor_data.py --n_samples 100000 --seed 47
    python generate_colliderdoor_data.py --mixed --n_samples 100000 --seed 47 \
        --dataset_name colliderdoor_mix_ZXMd_Yc_100K
"""

import argparse
import os

import numpy as np
import pandas as pd

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
from src.symbolic.scm.continuous import CLGMechanism
from src.symbolic.scm.discrete import BinaryMechanism
from src.symbolic.scm.graph import StructuralCausalModel
from src.symbolic.scm.mechanisms import LinearLogic


def build_skeleton() -> SCMSkeleton:
    variables = [
        VariableSpec("U1", "continuous", None, hidden=True),
        VariableSpec("U2", "continuous", None, hidden=True),
        VariableSpec("Z", "continuous", None),
        VariableSpec("X", "continuous", None),
        VariableSpec("M", "continuous", None),
        VariableSpec("Y", "continuous", None),
    ]
    edges = [
        ("U1", "Z"),
        ("U1", "Y"),
        ("U2", "X"),
        ("U2", "Y"),
        ("Z", "M"),
        ("X", "M"),
        ("M", "Y"),
    ]
    return SCMSkeleton(variables, edges)


def build_mixed_skeleton(cardinality: int = 2) -> SCMSkeleton:
    """Mixed variant: U1, U2, Z, X, M discrete; Y continuous (CLG in M, U1, U2)."""
    variables = [
        VariableSpec("U1", "discrete", cardinality, hidden=True),
        VariableSpec("U2", "discrete", cardinality, hidden=True),
        VariableSpec("Z", "discrete", cardinality),
        VariableSpec("X", "discrete", cardinality),
        VariableSpec("M", "discrete", cardinality),
        VariableSpec("Y", "continuous", None),
    ]
    edges = [
        ("U1", "Z"),
        ("U1", "Y"),
        ("U2", "X"),
        ("U2", "Y"),
        ("Z", "M"),
        ("X", "M"),
        ("M", "Y"),
    ]
    return SCMSkeleton(variables, edges)


EDGE_COEFS = {
    "U1->Z": 1.0,
    "U1->Y": 1.0,
    "U2->X": 2.5,
    "U2->Y": 2.0,
    "Z->M": 1.5,
    "X->M": 1.0,
    "M->Y": 1.0,
}

# Hand-designed mixed mechanisms (used with build_mixed_skeleton). The numbers
# are chosen so the observational association and the causal effect have
# OPPOSITE SIGNS:
#   - U2 -> X is strong (X mostly tracks its confounder) and U2 -> Y is strong,
#     so E[Y|X=1] - E[Y|X=0] = +1.275 (positive selection bias wins),
#   - while X inhibits M and M raises Y, so E[Y|do(X=1)] - E[Y|do(X=0)] = -1.125.
MIXED_PARAMS = {
    "p_u": 0.5,  # U1, U2 ~ Bernoulli(0.5)
    "z_given_u1": (0.15, 0.7),  # P(Z=1|U1) = a + b*U1
    "x_given_u2": (0.1, 0.8),  # P(X=1|U2) = a + b*U2
    "m_given_zx": (0.55, 0.3, -0.45),  # P(M=1|Z,X) = a + b*Z + c*X
    "y_mean": {"M": 2.5, "U1": 1.0, "U2": 3.0},  # CLG means, sigma = 0.8
    "y_sigma": 0.8,
}


def build_mixed_scm() -> StructuralCausalModel:
    """Hand-designed mixed colliderdoor SCM (all discrete except CLG outcome Y)."""
    p = MIXED_PARAMS
    scm = StructuralCausalModel()
    scm.add_variable("U1", BinaryMechanism(base_p=p["p_u"]), hidden=True)
    scm.add_variable("U2", BinaryMechanism(base_p=p["p_u"]), hidden=True)
    scm.add_variable(
        "Z",
        BinaryMechanism(logic=LinearLogic({"U1": p["z_given_u1"][1]}, p["z_given_u1"][0])),
        parents=["U1"],
    )
    scm.add_variable(
        "X",
        BinaryMechanism(logic=LinearLogic({"U2": p["x_given_u2"][1]}, p["x_given_u2"][0])),
        parents=["U2"],
    )
    scm.add_variable(
        "M",
        BinaryMechanism(
            logic=LinearLogic(
                {"Z": p["m_given_zx"][1], "X": p["m_given_zx"][2]}, p["m_given_zx"][0]
            )
        ),
        parents=["Z", "X"],
    )
    # Y | M, U1, U2 ~ N(mean(M, U1, U2), sigma^2); config idx = m*4 + u1*2 + u2
    # (mixed-radix with first-listed parent as most significant digit).
    intercepts = np.zeros(8)
    for m in range(2):
        for u1 in range(2):
            for u2 in range(2):
                intercepts[m * 4 + u1 * 2 + u2] = (
                    p["y_mean"]["M"] * m + p["y_mean"]["U1"] * u1 + p["y_mean"]["U2"] * u2
                )
    scm.add_variable(
        "Y",
        CLGMechanism(
            discrete_parent_cardinalities={"M": 2, "U1": 2, "U2": 2},
            continuous_parent_names=[],
            intercepts=intercepts,
            coefficients=np.zeros((8, 0)),
            stds=np.full(8, p["y_sigma"]),
        ),
        parents=["M", "U1", "U2"],
    )
    scm.validate_clg_structure()
    return scm


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=CASE_GUIDE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_generation_arguments(parser)
    parser.add_argument("--n_contexts", type=int, default=5, help="X contexts for 1D slices")
    parser.add_argument("--bins", type=int, default=60, help="Bins per continuous axis")
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Mixed variant: U1, U2, Z, X, M discrete (cardinality 2), Y continuous (CLG), "
        "with hand-designed mechanisms (MIXED_PARAMS) giving a sign-flip obs/do contrast.",
    )
    args = parser.parse_args()

    # Structural counts for the output name only (the skeleton is fixed above).
    args.skeleton = "colliderdoor"
    args.kind = "mixed" if args.mixed else "continuous"
    args.z_kind = args.x_kind = args.y_kind = args.m_kind = args.w_kind = None
    args.n_confounders = 2  # U1, U2 (hidden)
    args.n_treatments = 1
    args.n_outcomes = 1
    args.n_bystanders = 0
    args.n_mediators = 1

    skeleton = build_mixed_skeleton() if args.mixed else build_skeleton()

    if args.mixed:
        # Hand-designed mechanisms: no randomization, so no mechanism suffix.
        mechanism_kwargs = {}
        scm = build_mixed_scm()
        gap0 = effect_gap(scm, "X", "Y")
        print(
            f"Designed mixed SCM effect gap: mean L1 = {gap0['mean_l1']:.3f}, "
            f"max L1 = {gap0['max_l1']:.3f}"
        )

        np.random.seed(args.seed)
        print(f"Sampling {args.n_samples} observational points...")
        df_obs = scm.sample_dataset(args.n_samples, drop_hidden=True)

        print("Sampling interventional dataset (half do(X=0), half do(X=1))...")
        n_do = args.n_samples // 2
        df_do = pd.concat(
            [scm.intervene({"X": x}).sample_dataset(n_do, drop_hidden=True) for x in (0, 1)],
            ignore_index=True,
        )[df_obs.columns]
    else:
        mechanism_kwargs = mechanism_kwargs_from_args(args)
        mechanism_kwargs["edge_coefs"] = EDGE_COEFS
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
    # configs never share a plots directory. Plots live next to the CSVs.
    full_name = os.path.basename(paths["observational"])[len("observational_") : -len(".csv")]
    plots_dir = os.path.join(os.path.dirname(paths["observational"]), f"plots_{full_name}")
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
        f"max L1 = {gap['max_l1']:.3f} (total effect through the collider M)"
    )


if __name__ == "__main__":
    main()
