"""Step 1 of the synthetic SCM pipeline: user-fixed skeletons.

An :class:`SCMSkeleton` pins the DAG plus variable kinds/cardinalities. It is
*never* randomized, so experiments can force specific structures (backdoor,
frontdoor). Mechanisms are randomized on top of it by
``src/construction/random_mechanisms.py`` (step 2).

Hard constraint enforced here: a discrete variable may only have discrete
parents (the CLG restriction used by the exact ground-truth engine).
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple


@dataclass
class VariableSpec:
    """One variable of the skeleton: name, kind, cardinality, visibility."""

    name: str
    kind: Literal["discrete", "continuous"]
    cardinality: Optional[int] = None
    hidden: bool = False

    def __post_init__(self):
        if self.kind not in ("discrete", "continuous"):
            raise ValueError(f"kind must be 'discrete' or 'continuous', got {self.kind!r}.")
        if self.kind == "discrete":
            if self.cardinality is None:
                raise ValueError(f"Discrete variable '{self.name}' requires a cardinality.")
            if self.cardinality < 2:
                raise ValueError(
                    f"Discrete variable '{self.name}' needs cardinality >= 2, "
                    f"got {self.cardinality}."
                )
        elif self.cardinality is not None:
            raise ValueError(
                f"Continuous variable '{self.name}' must not set a cardinality "
                f"(got {self.cardinality})."
            )


class SCMSkeleton:
    """A validated DAG over :class:`VariableSpec` nodes.

    Validates on construction: unique names, edge endpoints exist, acyclicity
    (Kahn's algorithm), per-variable cardinality rules, and the CLG constraint
    (discrete variables may only have discrete parents).
    """

    def __init__(self, variables: Sequence[VariableSpec], edges: Sequence[Tuple[str, str]]):
        self.variables: List[VariableSpec] = list(variables)
        self.edges: List[Tuple[str, str]] = [(str(a), str(b)) for a, b in edges]
        self._specs: Dict[str, VariableSpec] = {}
        self._parents: Dict[str, List[str]] = {}
        self._validate()

    def _validate(self):
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate variable names: {duplicates}.")
        self._specs = {v.name: v for v in self.variables}

        for a, b in self.edges:
            for endpoint in (a, b):
                if endpoint not in self._specs:
                    raise ValueError(f"Edge ({a!r}, {b!r}) references unknown variable.")

        parents: Dict[str, List[str]] = {v.name: [] for v in self.variables}
        for a, b in self.edges:
            parents[b].append(a)
        self._parents = parents

        # Acyclicity via Kahn's algorithm (declaration order breaks ties,
        # so the resulting topological order is deterministic).
        in_degree = {name: len(parents[name]) for name in self._specs}
        queue = [v.name for v in self.variables if in_degree[v.name] == 0]
        children: Dict[str, List[str]] = {v.name: [] for v in self.variables}
        for a, b in self.edges:
            children[a].append(b)
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self._specs):
            raise ValueError("Skeleton contains a cycle.")
        self._topological_order = order

        # CLG constraint: discrete variables may only have discrete parents.
        for spec in self.variables:
            if spec.kind != "discrete":
                continue
            for parent in self._parents[spec.name]:
                if self._specs[parent].kind != "discrete":
                    raise ValueError(
                        f"CLG violation: discrete variable '{spec.name}' has "
                        f"non-discrete parent '{parent}'."
                    )

    def parents_of(self, name: str) -> List[str]:
        """Parents of ``name``, in edge-declaration order."""
        if name not in self._specs:
            raise KeyError(f"Unknown variable '{name}'.")
        return list(self._parents[name])

    def spec_of(self, name: str) -> VariableSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown variable '{name}'.")
        return self._specs[name]

    def topological_order(self) -> List[str]:
        """A deterministic topological order (ties broken by declaration order)."""
        return list(self._topological_order)

    @property
    def discrete_variables(self) -> List[VariableSpec]:
        return [v for v in self.variables if v.kind == "discrete"]

    @property
    def continuous_variables(self) -> List[VariableSpec]:
        return [v for v in self.variables if v.kind == "continuous"]

    @property
    def hidden_variables(self) -> List[VariableSpec]:
        return [v for v in self.variables if v.hidden]

    def __repr__(self) -> str:
        return f"SCMSkeleton(variables={len(self.variables)}, edges={len(self.edges)})"


def _set_names(prefix: str, n: int, always_suffix: bool = False) -> List[str]:
    """Names for a variable set: bare prefix for singletons, else prefix_0..prefix_{n-1}."""
    if n == 1 and not always_suffix:
        return [prefix]
    return [f"{prefix}_{i}" for i in range(n)]


def backdoor_skeleton(
    n_confounders: int = 1,
    n_treatments: int = 1,
    n_outcomes: int = 1,
    n_bystanders: int = 0,
    kind: Literal["discrete", "continuous"] = "discrete",
    cardinality: Optional[int] = 2,
    hidden_confounders: bool = False,
) -> SCMSkeleton:
    """Backdoor structure: Z -> X, Z -> Y and X -> Y for all set members.

    * Confounders ``Z_i`` (size ``n_confounders``); hidden when
      ``hidden_confounders=True``, forcing latent confounding between X and Y.
    * Treatments ``X_j`` (size ``n_treatments``), outcomes ``Y_k``
      (size ``n_outcomes``): every Z_i points at every X_j and Y_k, and every
      X_j points at every Y_k.
    * Bystanders ``W_l`` (size ``n_bystanders``): disconnected nuisance roots —
      useful for checking that methods ignore irrelevant dimensions.
    """
    if min(n_confounders, n_treatments, n_outcomes) < 1 or n_bystanders < 0:
        raise ValueError("set sizes must be >= 1 (>= 0 for bystanders).")
    z_names = _set_names("Z", n_confounders, always_suffix=True)
    x_names = _set_names("X", n_treatments)
    y_names = _set_names("Y", n_outcomes)
    w_names = _set_names("W", n_bystanders, always_suffix=True)

    variables = [VariableSpec(z, kind, cardinality, hidden=hidden_confounders) for z in z_names]
    variables += [VariableSpec(x, kind, cardinality) for x in x_names]
    variables += [VariableSpec(y, kind, cardinality) for y in y_names]
    variables += [VariableSpec(w, kind, cardinality) for w in w_names]

    edges = [(z, t) for z in z_names for t in x_names + y_names]
    edges += [(x, y) for x in x_names for y in y_names]
    return SCMSkeleton(variables, edges)


def frontdoor_skeleton(
    n_confounders: int = 1,
    n_treatments: int = 1,
    n_mediators: int = 1,
    n_outcomes: int = 1,
    n_bystanders: int = 0,
    kind: Literal["discrete", "continuous"] = "discrete",
    cardinality: Optional[int] = 2,
    confounder_cardinality: int = 2,
) -> SCMSkeleton:
    """Frontdoor structure: X -> M -> Y with hidden confounders U -> X, U -> Y.

    * Confounders ``U_i`` (size ``n_confounders``) are always hidden and always
      discrete, so the CLG constraint holds for both observed kinds.
    * Treatments ``X_j``, mediators ``M_l``, outcomes ``Y_k``: every X_j points
      at every M_l, every M_l at every Y_k, and every U_i at every X_j and Y_k.
    * Bystanders ``W_l``: disconnected nuisance roots.
    """
    if min(n_confounders, n_treatments, n_mediators, n_outcomes) < 1 or n_bystanders < 0:
        raise ValueError("set sizes must be >= 1 (>= 0 for bystanders).")
    u_names = _set_names("U", n_confounders)
    x_names = _set_names("X", n_treatments)
    m_names = _set_names("M", n_mediators)
    y_names = _set_names("Y", n_outcomes)
    w_names = _set_names("W", n_bystanders, always_suffix=True)

    variables = [VariableSpec(u, "discrete", confounder_cardinality, hidden=True) for u in u_names]
    variables += [VariableSpec(x, kind, cardinality) for x in x_names]
    variables += [VariableSpec(m, kind, cardinality) for m in m_names]
    variables += [VariableSpec(y, kind, cardinality) for y in y_names]
    variables += [VariableSpec(w, kind, cardinality) for w in w_names]

    edges = [(x, m) for x in x_names for m in m_names]
    edges += [(m, y) for m in m_names for y in y_names]
    edges += [(u, t) for u in u_names for t in x_names + y_names]
    return SCMSkeleton(variables, edges)
