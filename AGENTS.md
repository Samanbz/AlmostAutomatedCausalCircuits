# AGENTS.md — MonarchCausalCircuits

## Setup

```bash
conda activate pcs      # required before any command
pip install -e .        # editable install so `from src.xxx` resolves
```

You are an experienced AI researcher with a background in Causality, Causal AI, tractable probabilistic modelling as well as low-level GPU system engineering.

This will be a comprehensive framework for algebraic circuit construction, compilation and compositional inference.

## Commands

| Action | Command |
|--------|---------|
| All tests | `pytest` |
| Single file | `pytest tests/path/to/test.py` |
| Single test | `pytest tests/.../test_file.py::test_name` |
| Lint + fix | `ruff check --fix {path}` |
| Auto-format | `ruff format {path}` |
| Lint all | `ruff check --fix src/ tests/` |
| Format all | `ruff format src/ tests/` |

**Order after changes:** `ruff check --fix <file>` → `ruff format <file>` → `pytest <file>` (if tests exist).

## Project structure

- **`src/symbolic/`** — Core data structures (SCM, VTree, region graph), causal identification (`identification.py`), symbolic arithmetic circuit. No torch in base classes.
- **`src/symbolic/arithmetic/`** — `SymbolicArithmeticCircuit` (symbolic DAG) + EM trainer.
- **`src/construction/`** — Builders: region graph from VTree, circuit from region graph, learned VTree (PyMetis + MI), random SCM generators.
- **`src/compilation/`** — Compilation pipeline: symbolic → `FoldedCircuit` → `FusedCircuit` → `TensorizedCircuit` → `MonarchCircuit`. `QueryCompiler` converts `EstimandAST` → `QueryPlan`.
- **`src/utils/`** — `BitSet`, `Support`, `Interval`, `MonarchMatrix` factorization, pairwise mutual information, `DataSlice`.
- **`tests/`** — Mirrors `src/` layout. Fixtures in `conftest.py`.
- **`scratch/`** — Experiment scripts / POCs, not part of the package.
- **`benchmark/`** — Standalone benchmarking scripts (not pytest).

## Conventions

- **Imports:** `from src.module import Thing` (not `from monarch_causal_circuits`).
- **Ruff:** line-length=100, target-version="py38" (artifact; actual req is py312). Selected rules: E, W, F, I, B, C4, UP. E501 ignored (formatter handles it).
- **No typechecker** configured (mypy/pyright absent).
- **No CI/CD** configured.
- **No pre-commit hooks.**

## Two circuit representations

1. **`SymbolicArithmeticCircuit`** — symbolic DAG, supports property checking (smoothness, decomposability, marginal determinism) and EM training.
2. **`TensorizedCircuit` / `MonarchCircuit`** — compiled `nn.Module`, GPU-efficient forward pass. `MonarchCircuit` uses `MonarchMatrix` for factorized dense layers.

## Pipeline

```
SCM → VTree → RegionGraph → SymbolicArithmeticCircuit
  → FoldedCircuit → FusedCircuit → TensorizedCircuit / MonarchCircuit
    → QueryCompiler (EstimandAST → QueryPlan) → forward pass
```

Causal identification happens in `src/symbolic/identification.py` (do-calculus, produces `EstimandAST`), then compiled for execution.

## Remote workflow

- `scripts/pull_remote.sh` — rsync **from** DGX machine
- `scripts/push_remote.sh` — rsync **to** DGX machine

## MD-Circuit Layer Constraints
When modifying structural weights (`circuit_builder.py`), agents must respect these strict invariants to maintain marginal determinism:

Let $H$ be parent units, $H_L$ be left child units, $H_R$ be right child units.
Let $P(i) \subseteq \{1..H_L\} \times \{1..H_R\}$ be the set of product units assigned to parent unit $i$.
Let $J(i)$ and $K(i)$ be the set of left and right indices used by parent $i$.

1. **Universal Layer** (Both children unconstrained, parent unconstrained)
   - Mapping: Fully dense. $P(i) = \{1..H_L\} \times \{1..H_R\}$.
   - Constrained version: N/A.

2. **Synthesizing Layer** (Both children constrained)
   - Unconstrained: Dense. $P(i) = \{1..H_L\} \times \{1..H_R\}$.
   - Constrained: Product units cannot be shared across parent units.
     * $P(i) \cap P(i') = \emptyset \quad \forall i \neq i'$
     * *Implication*: If $H > H_L \times H_R$, at least $H - H_L H_R$ units will be permanently dead.

3. **Left-Mixing Layer** (Left constrained, Right unconstrained)
   - Unconstrained: No left index duplicated within a single parent unit (partial function $j \to k$).
   - Constrained: Left indices cannot be shared across *different* parent units.
     * $J(i) \cap J(i') = \emptyset \quad \forall i \neq i'$
     * *Implication*: Maximum of $H_L$ edges total. If $H > H_L$, dead units are mathematically unavoidable.

4. **Right-Mixing Layer** (Right constrained, Left unconstrained)
   - Symmetrical to Left-Mixing.
   - Constrained: $K(i) \cap K(i') = \emptyset \quad \forall i \neq i'$.
     * *Implication*: Maximum of $H_R$ edges total. If $H > H_R$, dead units are mathematically unavoidable.
