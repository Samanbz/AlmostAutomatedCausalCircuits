"""Generate paired observational/interventional datasets from the synthetic SCM pipeline.

Two-step generation (see ``src/construction/``):

1. Fix a skeleton — one of the factories (``backdoor_skeleton`` / ``frontdoor_skeleton``)
   or a hand-built ``SCMSkeleton`` (for mixed discrete/continuous or custom graphs).
2. ``randomize_mechanisms`` draws the mechanisms on top of it (regional-discrete,
   Dirichlet CPTs, linear-GMM, CLG — all with analytical ground truth).

The interventional dataset is paired row-wise with the observational one: row i is
drawn from do(T = t_i) with fresh noise, where t_i is the observational treatment
value of row i. It therefore represents P(rest | do(T)), not counterfactuals.

The pickled SCM gives exact ground truth: ``scm.ground_truth().marginal_prob(...)`` /
``.marginal_density(...)`` — no Monte Carlo ground truth needed.

CLI example:
    python generate_synthetic_data.py --skeleton backdoor --kind continuous \
        --n_confounders 4 --n_bystanders 2 --regions 5 --n_samples 50000 --seed 27
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.construction import (
    SCMSkeleton,
    backdoor_skeleton,
    frontdoor_skeleton,
    randomize_mechanisms,
)


# Per-process SCM, initialized once per worker (the SCM is rebuilt deterministically
# from the skeleton + seed instead of being pickled into each worker).
_worker_scm = None
_worker_treatments = None
_worker_seed = None


def _init_worker(skeleton, mechanism_kwargs, seed, treatments):
    """Rebuild the same SCM in every worker process."""
    global _worker_scm, _worker_treatments, _worker_seed
    _worker_scm = randomize_mechanisms(skeleton, seed, **mechanism_kwargs)
    _worker_treatments = treatments
    _worker_seed = seed


def _generate_interventional_chunk(args):
    """Generate interventional samples for one chunk of observational treatment values.

    For each observed treatment row t_i we sample one point from the intervened SCM
    do(T = t_i) with FRESH exogenous noise.  This is what makes the interventional
    dataset represent P(. | do(T)) rather than reproducing the observational values.

    A deterministic per-chunk seed keeps the result reproducible while ensuring
    different chunks do not draw identical random sequences.
    """
    chunk_idx, treatment_rows = args
    n = len(next(iter(treatment_rows.values())))
    if n == 0:
        return pd.DataFrame()

    # Each chunk gets its own reproducible random stream.
    np.random.seed(_worker_seed + chunk_idx)

    data = {}
    for node_id in _worker_scm.topological_sort():
        parent_ids = _worker_scm.get_parents(node_id)
        parent_data = {pid: data[pid] for pid in parent_ids}
        mechanism = _worker_scm.get_node_data(node_id)

        if node_id in _worker_treatments:
            # Fix the treatment to the value observed in the paired row.
            data[node_id] = treatment_rows[node_id]
        else:
            # Fresh noise -> the interventional distribution, not the counterfactual.
            data[node_id] = mechanism(n_samples=n, **parent_data)

    return pd.DataFrame(data)


def default_treatments(skeleton: SCMSkeleton) -> tuple:
    """Treatment variables by factory naming convention: 'X' or 'X_0', 'X_1', ..."""
    treatments = tuple(
        v.name for v in skeleton.variables if v.name == "X" or v.name.startswith("X_")
    )
    if not treatments:
        raise ValueError("Skeleton has no treatment variables named 'X'/'X_*'; pass treatments.")
    return treatments


def generate_paired_datasets(
    skeleton: SCMSkeleton,
    treatments=None,
    n_samples: int = 18432,
    seed: int = 27,
    n_workers: int = None,
    chunk_size: int = 256,
    **mechanism_kwargs,
):
    """Generate a paired observational / interventional dataset from a skeleton.

    Extra keyword arguments are forwarded to ``randomize_mechanisms``
    (``discrete_strategy``, ``regions``, ``dirichlet_alpha``, ``gmm_components``,
    ``coef_range``, ``sigma2_range``, ...).

    Returns ``(df_obs, df_do, scm)``; hidden variables are dropped from both frames
    but remain in the returned SCM for exact ground truth.
    """
    np.random.seed(seed)
    random.seed(seed)

    treatments = tuple(treatments) if treatments is not None else default_treatments(skeleton)
    scm = randomize_mechanisms(skeleton, seed, **mechanism_kwargs)

    print(f"Sampling {n_samples} observational points...")
    df_obs = scm.sample_dataset(n_samples, drop_hidden=True)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(
        f"Generating paired interventional dataset over treatments {treatments} "
        f"with {n_workers} workers (chunk_size={chunk_size})..."
    )

    treatment_vals = {t: df_obs[t].values for t in treatments}

    chunks = []
    for chunk_idx, start in enumerate(range(0, n_samples, chunk_size)):
        end = min(start + chunk_size, n_samples)
        chunks.append((chunk_idx, {t: v[start:end] for t, v in treatment_vals.items()}))

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(skeleton, mechanism_kwargs, seed, treatments),
    ) as executor:
        chunk_results = list(executor.map(_generate_interventional_chunk, chunks))

    df_do = pd.concat(chunk_results, ignore_index=True)

    hidden = [h for h in scm.hidden_variables if h in df_do.columns]
    df_do = df_do.drop(columns=hidden)
    df_do = df_do[df_obs.columns]  # align column order

    return df_obs, df_do, scm


def _fmt(value):
    """Short, filename-friendly float representation."""
    return f"{float(value):.3g}"


def _mechanism_suffix(mechanism_kwargs):
    """Filename suffix capturing the non-default mechanism randomization knobs."""
    defaults = {
        "discrete_strategy": "regional",
        "regions": 2,
        "dirichlet_alpha": 1.0,
        "gmm_components": (2, 3, 4),
        "coef_range": (0.5, 2.0),
        "sigma2_range": (0.25, 2.0),
        "intercept_range": (-1.0, 1.0),
    }
    prefix = {
        "discrete_strategy": "ST",
        "regions": "R",
        "dirichlet_alpha": "A",
        "gmm_components": "K",
        "coef_range": "CR",
        "sigma2_range": "SV",
        "intercept_range": "IR",
    }
    parts = []
    for key, short in prefix.items():
        if key in mechanism_kwargs and mechanism_kwargs[key] != defaults[key]:
            value = mechanism_kwargs[key]
            if isinstance(value, (tuple, list)):
                value = "-".join(_fmt(v) for v in value)
            parts.append(f"{short}{value}")
    edge_coefs = mechanism_kwargs.get("edge_coefs")
    if edge_coefs:
        parts.append(
            "EC"
            + "-".join(f"{k.replace('->', '>')}{float(v):g}" for k, v in sorted(edge_coefs.items()))
        )
    for key, short in (("direct_effect", "DE"), ("confounding_strength", "CS")):
        if mechanism_kwargs.get(key) is not None:
            parts.append(f"{short}{float(mechanism_kwargs[key]):g}")
    return "_".join(parts)


def _role_names(prefix, n, always_suffix=False):
    if n == 1 and not always_suffix:
        return [prefix]
    return [f"{prefix}_{i}" for i in range(n)]


def _custom_skeleton_from_args(args) -> SCMSkeleton:
    """Skeleton with per-role variable kinds (mixed/CLG layouts).

    Mirrors the factory edge patterns but lets each role be discrete or
    continuous. The CLG constraint (discrete nodes only have discrete parents)
    is enforced by SCMSkeleton validation — e.g. a discrete Y with a continuous
    parent X raises a clear ValueError.
    """
    from src.construction.skeleton import VariableSpec

    # Per-variable category counts, drawn in declaration order — same seed -> same
    # skeleton (including cardinalities), as required by the remote/parallel workflow.
    rng = np.random.default_rng(args.seed)

    def draw_cardinality():
        return int(rng.integers(args.min_categories, args.max_categories + 1))

    def role_kind(role, override):
        if override is not None:
            return override
        if args.kind == "mixed":
            return "discrete" if role == "Z" else "continuous"
        return args.kind

    def spec(name, role, hidden=False):
        kind = role_kind(role, getattr(args, f"{role.lower()}_kind"))
        card = draw_cardinality() if kind == "discrete" else None
        return VariableSpec(name, kind, card, hidden)

    confounder_names = _role_names("Z" if args.skeleton == "backdoor" else "U", args.n_confounders)
    x_names = _role_names("X", args.n_treatments)
    y_names = _role_names("Y", args.n_outcomes)
    w_names = _role_names("W", args.n_bystanders, always_suffix=True)

    if args.skeleton == "backdoor":
        hidden = args.hidden_confounders
        variables = [spec(z, "Z", hidden=hidden) for z in confounder_names]
        variables += [spec(x, "X") for x in x_names]
        variables += [spec(y, "Y") for y in y_names]
        variables += [spec(w, "W") for w in w_names]
        edges = [(z, t) for z in confounder_names for t in x_names + y_names]
        edges += [(x, y) for x in x_names for y in y_names]
    else:  # frontdoor — confounders are always hidden and always discrete
        m_names = _role_names("M", args.n_mediators)
        variables = [
            VariableSpec(u, "discrete", draw_cardinality(), hidden=True) for u in confounder_names
        ]
        variables += [spec(x, "X") for x in x_names]
        variables += [spec(m, "M") for m in m_names]
        variables += [spec(y, "Y") for y in y_names]
        variables += [spec(w, "W") for w in w_names]
        edges = [(x, m) for x in x_names for m in m_names]
        edges += [(m, y) for m in m_names for y in y_names]
        edges += [(u, t) for u in confounder_names for t in x_names + y_names]
    return SCMSkeleton(variables, edges)


def build_skeleton_from_args(args) -> SCMSkeleton:
    if args.min_categories < 2:
        raise ValueError(f"--min_categories must be >= 2, got {args.min_categories}.")
    if args.max_categories < args.min_categories:
        raise ValueError(
            f"--max_categories ({args.max_categories}) must be >= "
            f"--min_categories ({args.min_categories})."
        )
    role_overrides = (args.z_kind, args.x_kind, args.y_kind, args.m_kind, args.w_kind)
    if args.kind in ("discrete", "mixed") or any(role_overrides):
        # Discrete variables get individual cardinalities drawn from
        # [--min_categories, --max_categories] (seeded), so build via the
        # per-role path; it mirrors the factory edge patterns below.
        return _custom_skeleton_from_args(args)
    cardinality = None  # continuous kind: no discrete variables
    if args.skeleton == "backdoor":
        return backdoor_skeleton(
            n_confounders=args.n_confounders,
            n_treatments=args.n_treatments,
            n_outcomes=args.n_outcomes,
            n_bystanders=args.n_bystanders,
            kind=args.kind,
            cardinality=cardinality,
            hidden_confounders=args.hidden_confounders,
        )
    if args.skeleton == "frontdoor":
        return frontdoor_skeleton(
            n_confounders=args.n_confounders,
            n_treatments=args.n_treatments,
            n_mediators=args.n_mediators,
            n_outcomes=args.n_outcomes,
            n_bystanders=args.n_bystanders,
            kind=args.kind,
            cardinality=cardinality,
        )
    raise ValueError(f"Unknown skeleton '{args.skeleton}'.")


CASE_GUIDE = """\
CASES — which parameters apply where
  --kind discrete    All variables discrete, each with its own number of categories drawn
                     uniformly from --min_categories..--max_categories (seeded; equal
                     bounds -> uniform cardinality). Mechanisms are chosen by
                     --discrete_strategy: 'regional' (--regions R) or 'dirichlet'
                     (--dirichlet_alpha). The continuous knobs (--gmm_components,
                     --coef_*, --sigma2_*) are unused.
  --kind continuous  All variables continuous. Mechanisms are linear with Gaussian-mixture
                     noise: --coef_min/max, --sigma2_min/max, --gmm_components. The
                     discrete knobs (--discrete_strategy, --regions, --dirichlet_alpha)
                     are unused.
  --kind mixed       CLG (conditional linear Gaussian): confounders Z/U are discrete with
                     per-variable cardinalities from --min_categories..--max_categories,
                     treatments/mediators/outcomes/bystanders are continuous and get
                     regime-switching mechanisms (one linear-Gaussian regime per
                     discrete-parent configuration). Both the discrete and the
                     continuous parameter groups apply, each to their own variables.
                     GAP DIALS (no --confounding_strength here — it needs continuous
                     confounders): the P(Y|X) vs P(Y|do(X)) gap grows with (a) how
                     strongly X's CPT depends on Z — prefer --discrete_strategy
                     dirichlet --dirichlet_alpha 0.2..0.5 over --regions 2 (R = 1 can
                     degenerate X to a constant), (b) sharp regimes — small
                     --sigma2_min/max, (c) spread-out regime means — wide
                     --intercept_min/max.

PER-ROLE OVERRIDES (--z_kind/--x_kind/--y_kind/--m_kind/--w_kind)
  Mix kinds per role with --kind mixed plus e.g. --x_kind discrete (discrete treatment,
  continuous outcome). CLG CONSTRAINT: a discrete variable may only have discrete
  parents, so:
    * --y_kind discrete  requires X (--x_kind discrete) and the confounders to be
                         discrete as well;
    * continuous X -> discrete Y is impossible (no analytical ground truth exists there);
    * frontdoor confounders U are always discrete (and always hidden);
  violations fail with a clear 'CLG violation' error before any data is generated.

CONSTRAINTS
  * --regions R is capped per node at C^(C^|PA|), the number of distinct
    parent-configuration -> value mappings (CausalProfiler Def. D.2; C is that node's
    own cardinality). R = 1 gives a deterministic mechanism; larger R is more
    stochastic.
  * --coef_min must be > 0 (near-zero causal coefficients are excluded on purpose).
    Use --set_coef / --no_direct_effect / --no_confounding to place exact zeros.
  * Every generated mechanism family admits EXACT analytical ground truth. Do NOT Monte
    Carlo the ground truth: unpickle the SCM and call scm.ground_truth()
    (.marginal_prob(...) for discrete queries, .marginal_density(...) -> GaussianMixture
    for continuous ones).

ENGINEERING THE EFFECT GAP  (difference between P(Y|do(X)) and P(Y|X))
  Two signed dials, translated into the linear mechanisms concretely:
    --direct_effect V          every X*->Y* coefficient := V, so P(Y|do(X)) has
                               slope V in X (the causal effect).
    --confounding_strength S   confounder->X coefficients stay random; the
                               confounder->Y coefficients are solved so the
                               OBSERVATIONAL regression slope of Y on X becomes
                               direct_effect + S (per confounder, split evenly).
  So the observational slope is V + S while the interventional slope stays V:
    * --direct_effect -2 --confounding_strength 2  -> observational regression
      shows NO correlation (slopes cancel) but do(X) moves Y with slope -2
    * --direct_effect 1.5 --confounding_strength 0 -> P(Y|X) == P(Y|do(X)) in
      slope, with a real causal effect
    * negative S flips the bias direction (observational understates/reverses
      the effect)
  REQUIREMENTS: continuous linear mechanisms on the involved variables (use
  --kind continuous, or --z_kind continuous under --kind mixed); exactly one
  treatment; backdoor-style skeleton (frontdoor effects are mediated; discrete
  confounders carry bias in CPTs/CLG regimes and cannot be dialed this way).
  Everything else (noise shapes, other coefficients, intercepts) stays random —
  the seed still matters. --set_coef is applied first and is overridden by these
  dials on the treatment/confounder->outcome edges. Verify with
  explore_synthetic_data.py (effect-gap printout + slice plots).

OUTPUT (in --output_dir, suffixed by a name encoding structure + mechanism knobs)
  observational_<name>.csv    full SCM sample, hidden columns dropped
  interventional_<name>.csv   row i drawn from do(T = t_i) with fresh noise,
                              T = treatments (default X / X_0..X_k; override --treatments)
  scm_<name>.pkl              the SCM (carries exact ground truth)
  meta_<name>.json            skeleton + mechanism parameters for reproducibility

EXAMPLES
  # Continuous backdoor: linear + GMM noise, 2 confounders + 1 bystander
  %(prog)s --skeleton backdoor --kind continuous --n_confounders 2 --n_bystanders 1 \\
      --gmm_components 2 3 --n_samples 50000 --seed 27

  # Discrete backdoor: regional mechanisms, R = 5 noise regions, 2-4 categories per node
  %(prog)s --skeleton backdoor --kind discrete --min_categories 2 --max_categories 4 \\
      --regions 5 --n_samples 50000 --seed 27

  # Discrete backdoor with near-deterministic CPTs (Dirichlet), all nodes binary
  %(prog)s --skeleton backdoor --kind discrete --min_categories 2 --max_categories 2 \\
      --discrete_strategy dirichlet --dirichlet_alpha 0.3 --n_samples 50000 --seed 27

  # Mixed (CLG) backdoor: discrete Z (2-4 categories each), continuous X -> Y
  %(prog)s --skeleton backdoor --kind mixed --min_categories 2 --max_categories 4 \\
      --n_confounders 2 --n_samples 50000 --seed 27

  # Discrete treatment X, continuous outcome Y (valid CLG layout)
  %(prog)s --skeleton backdoor --kind mixed --x_kind discrete --min_categories 2 \\
      --max_categories 3 --n_confounders 1 --n_samples 50000 --seed 27

  # Mixed frontdoor with hidden discrete confounder
  %(prog)s --skeleton frontdoor --kind mixed --min_categories 2 --max_categories 3 \\
      --n_samples 50000 --seed 27

  # Semi-Markovian backdoor: confounders hidden (CSV drops them; latent projection of
  # the pickled SCM yields the bidirected X <-> Y graph)
  %(prog)s --skeleton backdoor --kind discrete --hidden_confounders \\
      --n_confounders 2 --n_samples 50000 --seed 27

  # Cancelled effect: observational regression shows no correlation
  # (-2 direct + 2 confounding) while do(X) moves Y strongly
  %(prog)s --skeleton backdoor --kind continuous --direct_effect -2 \\
      --confounding_strength 2 --n_confounders 2 --n_samples 50000 --seed 27

  # Pure causal effect, no confounding bias: P(Y|X) == P(Y|do(X)) in slope
  %(prog)s --skeleton backdoor --kind continuous --direct_effect 1.5 \\
      --confounding_strength 0 --n_confounders 2 --n_samples 50000 --seed 27

  # Reversed observational effect: bias stronger than and opposite to the effect
  %(prog)s --skeleton backdoor --kind continuous --direct_effect 1.0 \\
      --confounding_strength -2.5 --n_confounders 2 --n_samples 50000 --seed 27
"""


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    """Skeleton-structure, mechanism-randomization and generation CLI arguments."""
    # Step 1 — the structure. Fixed by these flags, never randomized.
    structure = parser.add_argument_group(
        "skeleton structure (step 1 — fixed, never randomized)",
        "Set sizes: |Z|/|U| confounders, |X| treatments, |Y| outcomes, |W| bystanders "
        "(disconnected nuisance roots), |M| mediators (frontdoor only). Singletons are "
        "named bare (X, Y, M, U); sets are suffixed (Z_0.., X_0..).",
    )
    structure.add_argument("--skeleton", choices=["backdoor", "frontdoor"], default="backdoor")
    structure.add_argument(
        "--kind",
        choices=["discrete", "continuous", "mixed"],
        default="continuous",
        help="variable kind for the whole SCM: 'discrete' (all discrete), 'continuous' "
        "(all continuous), or 'mixed' (CLG: discrete confounders, continuous "
        "treatments/mediators/outcomes). See CASES below.",
    )
    structure.add_argument(
        "--min_categories",
        type=int,
        default=2,
        help="minimum number of categories of every discrete variable (>= 2; all "
        "variables with --kind discrete; confounders with --kind mixed; any "
        "--*_kind discrete role). Each discrete variable draws its own cardinality "
        "uniformly from [--min_categories, --max_categories], seeded by --seed.",
    )
    structure.add_argument(
        "--max_categories",
        type=int,
        default=4,
        help="maximum number of categories of every discrete variable (>= "
        "--min_categories). Set equal to --min_categories to give every discrete "
        "variable the same cardinality.",
    )
    for role in ("z", "x", "y", "m", "w"):
        structure.add_argument(
            f"--{role}_kind",
            choices=["discrete", "continuous"],
            default=None,
            help=f"override the kind of the {role.upper()} role (default: follows "
            "--kind). CLG constraint: a discrete variable may only have discrete "
            "parents — e.g. --y_kind discrete requires discrete X and confounders.",
        )
    structure.add_argument(
        "--n_confounders",
        type=int,
        default=4,
        help="confounder set size |Z| (backdoor; hidden with --hidden_confounders) "
        "or |U| (frontdoor; always hidden)",
    )
    structure.add_argument("--n_treatments", type=int, default=1, help="treatment set size |X|")
    structure.add_argument("--n_outcomes", type=int, default=1, help="outcome set size |Y|")
    structure.add_argument(
        "--n_bystanders",
        type=int,
        default=0,
        help="bystander set size |W|: disconnected nuisance roots (no edges); useful "
        "for checking that methods ignore irrelevant dimensions",
    )
    structure.add_argument(
        "--n_mediators", type=int, default=1, help="mediator set size |M| (frontdoor)"
    )
    structure.add_argument(
        "--hidden_confounders",
        action="store_true",
        help="backdoor only: mark the Z_i hidden — they are dropped from the CSVs and "
        "the latent projection of the pickled SCM shows bidirected X <-> Y",
    )

    # Discrete case.
    discrete = parser.add_argument_group(
        "discrete-case mechanisms",
        "Applies to discrete variables: ALL variables with --kind discrete; the "
        "confounders with --kind mixed; any role set via --*_kind discrete. Unused with "
        "--kind continuous.",
    )
    discrete.add_argument(
        "--discrete_strategy",
        choices=["regional", "dirichlet"],
        default="regional",
        help="'regional' (default): CausalProfiler-style — uniform noise split into R "
        "regions, each a distinct parent-config -> value mapping (see --regions). "
        "'dirichlet': plain CPTs with rows ~ Dirichlet(alpha * 1).",
    )
    discrete.add_argument(
        "--regions",
        type=int,
        default=2,
        help="noise regions R per discrete node (regional strategy). R = 1 -> "
        "deterministic mechanism; larger R -> more stochastic. Capped at C^(C^|PA|), "
        "the number of distinct mappings.",
    )
    discrete.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=1.0,
        help="Dirichlet concentration per CPT row (dirichlet strategy): < 1 -> "
        "near-deterministic rows, 1 -> uniform-ish, > 1 -> flat/high-entropy rows",
    )

    # Continuous case.
    continuous = parser.add_argument_group(
        "continuous-case mechanisms",
        "Applies to continuous variables: ALL variables with --kind continuous; "
        "treatments/mediators/outcomes/bystanders with --kind mixed (then the discrete "
        "parents act as regime switches, i.e. CLG). Unused with --kind discrete.",
    )
    continuous.add_argument(
        "--gmm_components",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="number of Gaussian-mixture noise components K per continuous node, drawn "
        "uniformly from this list (K = 1 -> plain Gaussian noise)",
    )
    continuous.add_argument(
        "--coef_min",
        type=float,
        default=0.5,
        help="lower bound on |coefficient| of linear continuous mechanisms",
    )
    continuous.add_argument(
        "--coef_max",
        type=float,
        default=2.0,
        help="upper bound on |coefficient| of linear continuous mechanisms",
    )
    continuous.add_argument(
        "--sigma2_min",
        type=float,
        default=0.25,
        help="minimum noise variance (log-uniform per node / regime)",
    )
    continuous.add_argument(
        "--sigma2_max",
        type=float,
        default=2.0,
        help="maximum noise variance (log-uniform per node / regime)",
    )
    continuous.add_argument(
        "--intercept_min",
        type=float,
        default=-1.0,
        help="lower bound of the per-node / per-regime intercept range. Widening the "
        "range (e.g. -3..3) spreads the means of the CLG regimes further apart, which "
        "enlarges the P(Y|X) vs P(Y|do(X)) gap in mixed layouts with discrete "
        "confounders.",
    )
    continuous.add_argument(
        "--intercept_max",
        type=float,
        default=1.0,
        help="upper bound of the per-node / per-regime intercept range (>= --intercept_min)",
    )
    continuous.add_argument(
        "--set_coef",
        action="append",
        default=None,
        metavar="'A->B=VALUE'",
        help="low-level escape hatch: pin the linear coefficient of a single "
        "continuous-parent edge after the random draw (repeatable), e.g. "
        "--set_coef 'W_0->Y=0.3'. For the causal/confounding balance prefer "
        "--direct_effect / --confounding_strength, which are applied after this.",
    )
    continuous.add_argument(
        "--direct_effect",
        type=float,
        default=None,
        metavar="V",
        help="engineer the causal slope: pin every treatment->outcome edge (X*->Y*) "
        "coefficient to the signed value V, so P(Y|do(X)) has slope V in X. "
        "Requires continuous X and Y and a backdoor-style skeleton (frontdoor "
        "effects are mediated).",
    )
    continuous.add_argument(
        "--confounding_strength",
        type=float,
        default=None,
        metavar="S",
        help="engineer the confounding bias: solve the confounder->outcome "
        "coefficients (confounder->X stays random) so the OBSERVATIONAL regression "
        "slope of Y on X becomes direct_effect + S while P(Y|do(X)) keeps slope "
        "direct_effect. S may be negative (bias opposes the effect); S = 0 makes "
        "P(Y|X) coincide with P(Y|do(X)). Requires continuous confounders (--kind "
        "continuous, or --z_kind continuous under --kind mixed) and a single "
        "treatment.",
    )

    # Generation.
    generation = parser.add_argument_group("dataset generation")
    generation.add_argument("--n_samples", type=int, default=18432, help="samples (train+test)")
    generation.add_argument(
        "--seed",
        type=int,
        default=27,
        help="random seed: fixes the mechanism draw (step 2). Same skeleton + same "
        "seed -> identical SCM; different seed -> different mechanisms on the same DAG",
    )
    generation.add_argument("--output_dir", type=str, default="data", help="output directory")
    generation.add_argument("--n_workers", type=int, default=None, help="worker processes (CPUs-1)")
    generation.add_argument(
        "--chunk_size", type=int, default=256, help="samples per parallel chunk"
    )
    generation.add_argument(
        "--treatments",
        type=str,
        default=None,
        help="comma-separated treatment variables clamped in the interventional dataset "
        "(default: X / X_0..X_k)",
    )


def mechanism_kwargs_from_args(args) -> dict:
    if args.intercept_max < args.intercept_min:
        raise ValueError(
            f"--intercept_max ({args.intercept_max}) must be >= "
            f"--intercept_min ({args.intercept_min})."
        )
    kwargs = {
        "discrete_strategy": args.discrete_strategy,
        "regions": args.regions,
        "dirichlet_alpha": args.dirichlet_alpha,
        "gmm_components": tuple(args.gmm_components),
        "coef_range": (args.coef_min, args.coef_max),
        "sigma2_range": (args.sigma2_min, args.sigma2_max),
        "intercept_range": (args.intercept_min, args.intercept_max),
    }

    edge_coefs = {}
    for item in args.set_coef or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(f"--set_coef expects 'A->B=VALUE', got {item!r}.")
        edge_coefs[key.strip()] = float(value)
    if edge_coefs:
        kwargs["edge_coefs"] = edge_coefs
    if args.direct_effect is not None:
        kwargs["direct_effect"] = args.direct_effect
    if args.confounding_strength is not None:
        kwargs["confounding_strength"] = args.confounding_strength
    return kwargs


def _category_tag(skeleton: SCMSkeleton) -> str:
    """Filename tag for the discrete cardinalities, e.g. 'C3' or 'C2-4' (empty if none)."""
    cards = sorted({v.cardinality for v in skeleton.variables if v.kind == "discrete"})
    if not cards:
        return ""
    if len(cards) == 1:
        return f"C{cards[0]}"
    return f"C{cards[0]}-{cards[-1]}"


def save_datasets(df_obs, df_do, scm, skeleton, args, mechanism_kwargs) -> dict:
    """Write observational/interventional CSVs, the SCM pickle and metadata JSON.

    Returns a dict with ``base_name`` and the four output paths.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    suffix = _mechanism_suffix(mechanism_kwargs)
    if suffix:
        suffix = "_" + suffix
    hidden = "_H" if any(v.hidden for v in skeleton.variables) else ""

    role_overrides = (args.z_kind, args.x_kind, args.y_kind, args.m_kind, args.w_kind)
    if args.kind == "mixed" or any(role_overrides):
        # Per-role layout code, e.g. "Zd_Xd_Yc" (roles in declaration order).
        layout = "_".join(
            f"{v.name[0]}{v.kind[0]}"
            for i, v in enumerate(skeleton.variables)
            if i == 0 or v.name[0] != skeleton.variables[i - 1].name[0]
        )
        has_discrete = any(v.kind == "discrete" for v in skeleton.variables)
        kind_str = layout + (f"-{_category_tag(skeleton)}" if has_discrete else "")
    else:
        kind_str = args.kind + (f"-{_category_tag(skeleton)}" if args.kind == "discrete" else "")

    base_name = (
        f"N{args.n_samples}_{args.skeleton}-{kind_str}"
        f"_Z{args.n_confounders}_X{args.n_treatments}_Y{args.n_outcomes}_W{args.n_bystanders}"
        f"{'_M' + str(args.n_mediators) if args.skeleton == 'frontdoor' else ''}"
        f"{hidden}_S{args.seed}"
    )
    paths = {
        "base_name": base_name,
        "observational": os.path.join(args.output_dir, f"observational_{base_name}{suffix}.csv"),
        "interventional": os.path.join(args.output_dir, f"interventional_{base_name}{suffix}.csv"),
        "scm": os.path.join(args.output_dir, f"scm_{base_name}{suffix}.pkl"),
        "meta": os.path.join(args.output_dir, f"meta_{base_name}{suffix}.json"),
    }

    df_obs.to_csv(paths["observational"], index=False)
    df_do.to_csv(paths["interventional"], index=False)
    with open(paths["scm"], "wb") as f:
        pickle.dump(scm, f)
    with open(paths["meta"], "w") as f:
        json.dump(
            {
                "skeleton": {
                    "variables": [vars(v) for v in skeleton.variables],
                    "edges": skeleton.edges,
                },
                "treatments": list(default_treatments(skeleton)),
                "mechanism_kwargs": {
                    k: list(v) if isinstance(v, tuple) else v for k, v in mechanism_kwargs.items()
                },
                "n_samples": args.n_samples,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    for key in ("observational", "interventional", "scm", "meta"):
        print(f"Saved {key} to {paths[key]}")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate paired observational/interventional datasets from a "
        "fixed skeleton with randomized mechanisms (synthetic SCM pipeline).",
        epilog=CASE_GUIDE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_generation_arguments(parser)
    args = parser.parse_args()

    skeleton = build_skeleton_from_args(args)
    treatments = tuple(args.treatments.split(",")) if args.treatments else None
    mechanism_kwargs = mechanism_kwargs_from_args(args)

    df_obs, df_do, scm = generate_paired_datasets(
        skeleton,
        treatments=treatments,
        n_samples=args.n_samples,
        seed=args.seed,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        **mechanism_kwargs,
    )

    save_datasets(df_obs, df_do, scm, skeleton, args, mechanism_kwargs)


if __name__ == "__main__":
    main()
