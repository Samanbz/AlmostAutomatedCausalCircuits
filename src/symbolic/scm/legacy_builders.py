"""Legacy SCM builders kept for existing experiments, tests, and pickled artifacts.

These predate the skeleton + ``randomize_mechanisms`` pipeline in
``src/construction/``. New experiments should use that pipeline instead.
"""

from typing import Optional

import numpy as np

from .continuous import AdditiveNoiseMechanism
from .discrete import BinaryMechanism
from .graph import StructuralCausalModel
from .mechanisms import GaussianNoise, LinearLogic, LogisticLogic


def build_synthetic_continuous_scm(
    num_confounders: int = 4,
    seed: Optional[int] = None,
    *,
    direct_effect: float = 1.0,
    obs_slope: Optional[float] = None,
    do_std: Optional[float] = None,
    obs_std: Optional[float] = None,
    x_confound_strength: float = 1.0,
    x_noise_std: float = 1.0,
    z_noise_std: float = 1.0,
    y_noise_std: Optional[float] = None,
) -> StructuralCausalModel:
    """Build a tunable linear-Gaussian backdoor SCM.

    The model is:

        Z_i ~ N(0, z_noise_std^2)    i = 1..K
        X   = a * sum_i Z_i + noise_X
        Y   = b * X + c * sum_i Z_i + noise_Y

    where ``b = direct_effect`` and ``a = x_confound_strength``.  The coefficient
    ``c`` is solved so that the population regression slope of ``Y`` on ``X`` equals
    ``obs_slope``.

    Parameters and their single responsibility:

    * ``direct_effect`` (b): slope of ``P(Y | do(X))``.
    * ``obs_slope``: slope of ``P(Y | X)``.  If ``None``, defaults to
      ``direct_effect + 1.0`` (a moderate positive backdoor bias).
    * ``do_std``: target standard deviation of ``P(Y | do(X))``.
    * ``obs_std``: target standard deviation of ``P(Y | X)``.  Must satisfy
      ``obs_std <= do_std``.
    * ``x_confound_strength`` (a): strength of each ``Z_i -> X`` link.  Larger
      values let you realize larger slope gaps with less confounder-induced
      variance.
    * ``x_noise_std``: independent noise in ``X``.
    * ``z_noise_std``: spread of each exogenous ``Z_i``.
    * ``y_noise_std``: independent noise in ``Y``; ignored if ``do_std`` is set.

    Because this is a linear additive SCM with independent noises, the
    observational residual variance can never exceed the interventional variance:
    ``obs_std <= do_std``.  If you need the reverse ordering, a richer model
    (e.g. a mixture confounder) is required.
    """
    if seed is not None:
        np.random.seed(seed)

    b = float(direct_effect)
    obs_slope = b + 1.0 if obs_slope is None else float(obs_slope)
    delta = obs_slope - b
    k = int(num_confounders)
    if k < 1:
        raise ValueError("num_confounders must be at least 1")

    a = float(x_confound_strength)
    s_z = float(z_noise_std)
    s_x = float(x_noise_std)
    var_z = s_z * s_z
    min_var_x = k * a * a * var_z

    # Population identities for independent equal-coefficient confounders:
    #   Var(X) = k * a^2 * s_z^2 + s_x^2
    #   obs_std^2 = do_std^2 - delta^2 * Var(X)
    # If both stds are given, Var(X) is therefore pinned down.
    if do_std is not None and obs_std is not None:
        do_var = float(do_std) ** 2
        obs_var = float(obs_std) ** 2
        if do_var + 1e-12 < obs_var:
            raise ValueError(
                f"obs_std ({obs_std}) cannot exceed do_std ({do_std}) in a linear "
                "additive SCM with independent noises."
            )
        if abs(delta) < 1e-12:
            if abs(do_var - obs_var) > 1e-6:
                raise ValueError("obs_slope == direct_effect implies obs_std must equal do_std.")
            target_var_x = min_var_x + s_x * s_x
        else:
            target_var_x = (do_var - obs_var) / (delta * delta)
            if target_var_x < min_var_x - 1e-12:
                raise ValueError(
                    f"Requested moments are inconsistent: need Var(X) >= {min_var_x:.4f} "
                    f"but the std targets imply Var(X) = {target_var_x:.4f}. "
                    "Increase x_confound_strength/x_noise_std or reduce the slope gap."
                )
            s_x = float(np.sqrt(max(0.0, target_var_x - min_var_x)))
        var_x = target_var_x
    else:
        var_x = min_var_x + s_x * s_x

    if abs(delta) < 1e-12:
        c = 0.0
    else:
        # delta = (k * a * c * s_z^2) / Var(X)  =>  solve for c
        c = delta * var_x / (k * a * var_z)

    if do_std is not None:
        do_var = float(do_std) ** 2
        y_noise_needed = do_var - k * c * c * var_z
        if y_noise_needed < -1e-12:
            raise ValueError(
                f"Cannot achieve do_std={do_std} with the computed confounder "
                "contribution; reduce x_confound_strength or the slope gap."
            )
        s_y = float(np.sqrt(max(0.0, y_noise_needed)))
    elif y_noise_std is not None:
        s_y = float(y_noise_std)
    else:
        s_y = 1.0

    scm = StructuralCausalModel()

    z_names = [f"Z{i}" for i in range(k)]
    for z_name in z_names:
        scm.add_variable(
            name=z_name,
            mechanism=AdditiveNoiseMechanism(
                logic=None,
                noise_dist=GaussianNoise(0.0, s_z),
            ),
            parents=[],
            is_exogenous=True,
        )

    x_intercept = float(np.random.uniform(-0.5, 0.5))
    scm.add_variable(
        name="X",
        mechanism=AdditiveNoiseMechanism(
            logic=LinearLogic(dict.fromkeys(z_names, a), intercept=x_intercept),
            noise_dist=GaussianNoise(0.0, s_x),
        ),
        parents=z_names,
        is_exogenous=False,
    )

    y_intercept = float(np.random.uniform(-0.5, 0.5))
    y_logic = {"X": b, **dict.fromkeys(z_names, c)}
    scm.add_variable(
        name="Y",
        mechanism=AdditiveNoiseMechanism(
            logic=LinearLogic(y_logic, intercept=y_intercept),
            noise_dist=GaussianNoise(0.0, s_y),
        ),
        parents=["X"] + z_names,
        is_exogenous=False,
    )

    return scm


def build_synthetic_binary_scm(
    num_confounders: int = 4, confounding_strength: float = 1.0
) -> StructuralCausalModel:
    """Builds a binary SCM with a branching confounder tree.

    Args:
        num_confounders: Number of Z variables.
        confounding_strength: A knob to control the magnitude of confounding.
            0.0 means no confounding (P(Y|X) == P(Y|do(X))),
            > 0.0 increases the divergence.
    """
    scm = StructuralCausalModel()

    # Z0 is strictly exogenous
    scm.add_variable(
        name="Z0",
        mechanism=BinaryMechanism(logic=None, base_p=0.5),
        parents=[],
        is_exogenous=True,
    )

    # Building a branching tree for Z variables (Logistic logic)
    z_vars = ["Z0"]
    for i in range(1, num_confounders):
        parent_i = (i - 1) // 2
        z_var = f"Z{i}"
        z_vars.append(z_var)
        scm.add_variable(
            name=z_var,
            mechanism=BinaryMechanism(logic=LogisticLogic({f"Z{parent_i}": 2.0}, intercept=-1.0)),
            parents=[f"Z{parent_i}"],
            is_exogenous=False,
        )

    # X depends on Z strongly to create high confounding
    x_logic = {f"Z{i}": 2.0 for i in range(num_confounders)}
    scm.add_variable(
        "X",
        mechanism=BinaryMechanism(
            logic=LogisticLogic(coefficients=x_logic, intercept=-float(num_confounders))
        ),
        parents=[f"Z{i}" for i in range(num_confounders)],
        is_exogenous=False,
    )

    # Y is caused by X and Z
    # We want X to positively influence Y (+2.0)
    # We want Z to negatively influence Y (-confounding_strength) to create a backdoor path
    y_coeffs = {"X": 2.0}
    for z in z_vars:
        y_coeffs[z] = -1.0 * confounding_strength

    # Intercept compensates for Z so that the interventional distribution stays relatively stable
    scm.add_variable(
        "Y",
        mechanism=BinaryMechanism(
            logic=LogisticLogic(
                coefficients=y_coeffs, intercept=confounding_strength * (num_confounders / 2.0)
            )
        ),
        parents=["X"] + z_vars,
        is_exogenous=False,
    )

    return scm
