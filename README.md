# Almost Automated Causal Circuits

Tractable causal inference with **marginal-determinism (MD) probabilistic
circuits**. T-ID (tractable identification) turns an interventional query
`P(y | do(x))` on a causal graph into a compositional estimand (a COAST —
circuit of additive and subtractive terms), which is compiled into query
circuits that **share the parameters of a single MD circuit** trained on
observational data.

The showcase structures are the classic **backdoor** and **frontdoor**
settings, plus the **collider-door** structure — an identifiable graph that is
neither backdoor, frontdoor, nor napkin:

```
Z → M ← X      (X's effect reaches Y only through the collider M),  M → Y
Z ↔ Y (U1),  X ↔ Y (U2)          P(Y|do(X)) identified with determinisms
                                 {Z,X} ⊂ {Z,M,X}
```

## The pipeline

```
SCM → VTree → MD circuit (create_md_circuit) → EM training on P(V)
    → T-ID: ID(y, x, P(V), CausalGraph) → COAST (EstimandAST)
      → query circuits (shared parameters with the base circuit)
        → forward pass = P(y | do(x))
```

- **T-ID** (`src/symbolic/identification.py`) performs Shpitser–Pearl-style
  identification while tracking which *marginal-determinism* scopes the
  estimand requires, plus its circuit complexity `K`.
  `required_determinisms(y, x, graph)` reports the required determinism
  family for any identifiable query.
- **MD circuits** (`src/construction/circuit_builder.py`) are arithmetic
  circuits whose sum layers respect marginal-determinism constraints
  (universal / synthesizing / left-mixing / right-mixing / pseudo-mixing
  layers), so conditional and marginal queries are answered by parameter
  sharing instead of separate models. Vtrees are learned from data
  (`src/construction/learned_vtree.py`, Liang et al.-style pairwise-MI
  induction) or pinned per structure via a Newick-style `model.vtree` spec;
  the MD labeling is computed by `VTree.compute_md_labeling`.
- **Exact ground truth** (`src/symbolic/scm/ground_truth.py`) for synthetic
  SCMs: discrete variable elimination, GMM ancestral propagation, and CLG
  mixtures — no Monte-Carlo reference.

## Installation

```bash
conda create -n pcs python=3.12
conda activate pcs
pip install -e .
```

## Tests and benchmark

```bash
pytest                                  # full test suite
python -m benchmark.symbolic_benchmark --device cpu
```

---

# Reproducing the experiments

The pipeline is **train-only**; identification, evaluation, and plotting are
post-hoc steps that load saved checkpoints. Every step is idempotent — a
crashed job can simply be re-run.

## 0. Datasets

The datasets and trained checkpoints are **not** in this repository — download
them instead (or regenerate from scratch, §0b):

| Archive | Contents | Size |
|---------|----------|------|
| `data.tar.gz` | all synthetic + binary-BN datasets under `data/` | ~100 MB |
| `checkpoints_final.tar.gz` | every trained model's `_final.pt` (79 models) | ~20 MB |
| `results.tar.gz` | all run artifacts: metrics, grids, logs, paper figures | ~180 MB |

```bash
# from the Zenodo record: https://zenodo.org/records/22969724
tar -xzf data.tar.gz                 # restores data/
tar -xzf checkpoints_final.tar.gz    # restores experiments/models/**/<id>_final.pt
tar -xzf results.tar.gz              # restores experiments/results/
```

Per-epoch checkpoints (for the over-training curves) are not shipped; retrain
with §1 to regenerate them, or evaluate the shipped `_final.pt` directly with §2.

## 0b. Regenerating datasets from scratch

All synthetic datasets can be reproduced exactly from the generation commands
below (each also writes verification plots and prints the exact effect gap):

| Dataset | Command |
|---------|---------|
| `backdoor_cont_Z1_{25K,100K,400K,1200K}` | `python scripts/generate_synthetic_data.py --skeleton backdoor --kind continuous --n_confounders 1 --n_treatments 1 --n_outcomes 1 --n_bystanders 0 --gmm_components 1 2 --coef_min 1 --coef_max 4 --sigma2_min 0.2 --sigma2_max 7 --direct_effect -2 --confounding_strength 2 --n_samples <N> --seed 30 --dataset_name backdoor_cont_Z1_<X>` |
| `frontdoor_cont_Z1_100K_strongM3` | `python scripts/generate_synthetic_data.py --skeleton frontdoor --u_kind continuous --n_confounders 1 --n_mediators 1 --gmm_components 1 --coef_min 0.5 --coef_max 2 --sigma2_min 4 --sigma2_max 16 --set_coef 'X->M=-1.5' --set_coef 'M->Y=2' --set_coef 'U->X=2' --confounding_strength 3 --n_samples 100000 --seed 25 --dataset_name frontdoor_cont_Z1_100K_strongM3` |
| `frontdoor_mix_XMd_Yc_50K_gap` | `python scripts/generate_synthetic_data.py --skeleton frontdoor --kind mixed --x_kind discrete --m_kind discrete --min_categories 2 --max_categories 2 --discrete_strategy dirichlet --dirichlet_alpha 0.5 --n_confounders 1 --n_mediators 1 --n_samples 50000 --seed 20 --dataset_name frontdoor_mix_XMd_Yc_50K_gap` |
| `colliderdoor_cont_100K` | `python scripts/generate_colliderdoor_data.py --n_samples 100000 --seed 47 --dataset_name colliderdoor_cont_100K` |
| `colliderdoor_mix_ZXMd_Yc_100K` | `python scripts/generate_colliderdoor_data.py --mixed --n_samples 100000 --seed 47 --dataset_name colliderdoor_mix_ZXMd_Yc_100K` |

Each generator also writes verification plots (empirical vs. exact-GT
heatmaps and slices) and prints the exact effect gap. The binary-BN datasets
(`data/{asia,child,win95pts,andes}.pkl`) are the binarized MDNet benchmarks;
see §5.

## 1. Training

```bash
# one seed, in-process
python -m experiments.run_experiment --config configs/backdoor_cont_400K_N64.json --seed 27

# many seeds as GPU-pinned subprocesses (round-robin over --gpus)
python -m experiments.run_experiment --config configs/backdoor_cont_400K_N64.json \
    --seeds 27 28 29 --gpus 0 1 2

# useful flags
#   --yes-invalidate   non-interactive: drop stale artifacts when the
#                      checkpoint-relevant config changed (otherwise prompts)
#   --override training.total_iters=10   JSON-coerced config patches
```

Runs resume automatically from the newest checkpoint (`_final.pt` skips
training entirely). Per-seed artifacts:

```
experiments/results/<id>/seed_<S>/{config,metadata,metrics}.json + <id>_seed<S>.log
experiments/models/<id>/<id>/seed_<S>/<id>_epoch_XXXX.pt / <id>_final.pt
```

## 2. Evaluation (post-hoc, from checkpoints)

```bash
# exact-GT grid metrics for the final model: per-X MAE / KL (both directions)
# / Hellinger of P(Y|X) and P(Y|do(X));  writes grid_metrics.json + grids.npz
python -m experiments.evaluate_grids --results-dir experiments/results/backdoor_cont_400K_N64/seed_27 --plots

# NLL of every checkpoint (observational + interventional do-circuit)
python -m experiments.evaluate_checkpoints --results-dir experiments/results/backdoor_cont_400K_N64/seed_27

# per-checkpoint KL/Hellinger sweep vs. GT  ->  checkpoint_kl.csv
python -m experiments.evaluate_checkpoint_grids --results-dir experiments/results/backdoor_cont_400K_N64

# all three accept --seeds 27 28 29 --gpus 0 1 2 to fan out per-seed subprocesses
```

Configs whose circuit lacks the determinism the interventional query needs
(e.g. the unconstrained baseline) degrade gracefully: observational metrics
only, `do_*` empty.

## 3. Aggregate tables and paper figures

```bash
python -m experiments.make_results_table --results-root experiments/results   # results_table{,_by_config}.csv
python -m experiments.plot_results --results-root experiments/results         # figures into results/figures/
```

`plot_results.py` renders the KL/NLL-vs-N and over-training curves with
±1-std seed bands for the hardcoded backdoor config families (400K N-series
N8–N128, data sweep 25K–1200K, G1 and unconstrained ablations).

## 4. Per-seed visualizations

```bash
python -m experiments.plot_heatmaps --results-dir experiments/results/colliderdoor_cont_100K_N64/seed_27
python -m experiments.plot_x_support_slices --results-dir experiments/results/backdoor_cont_400K_N64/seed_27
python -m experiments.plot_leaves --results-dir experiments/results/colliderdoor_mixed_N16/seed_27 --plot-components
python -m experiments.plot_heatmap_comparison --results-root experiments/results --tag N_series
python -m experiments.plot_slice_comparison --results-root experiments/results --tag N_series --x-value 3.5
```

Heatmaps are the 2×2 learned-vs-analytical-GT `P(Y|X)` / `P(Y|do(X))`
panels; slices are 1-D Y densities at the learned X-support split points vs.
empirical histograms and exact GT; `plot_leaves` dumps every leaf's learned
density (per component with `--plot-components`). The comparison scripts
render cross-config paper figures (their config lists are module constants —
edit for non-backdoor families).

## 5. Real-world binary Bayesian networks (MDNet comparison)

```bash
# ours — same CLI as run_experiment; evaluates the backdoor query in-run
python -m experiments.run_binary_bn_experiment --config configs/binary_bn/asia.json --seeds 0 1 2 3 4

# MDNet baseline: separate checkout (causal-pc/) in its own env
conda create -n wang-pc python=3.11
conda activate wang-pc
pip install "jax==0.4.30" "jaxlib==0.4.30" "numpy<2" pandas scikit-learn infinite-sets
cd causal-pc && python wang_baseline_driver.py --runs 5   # -> wang_baseline_results_5run.json

# the comparison table (ours vs. MDNet: MAE + Bernoulli log-loss)
python -m experiments.make_binary_bn_table
python -m experiments.plot_binary_bn_comparison
```

Note: `causal-pc/` is gitignored in this repo (it carries its own git
history); `numpy<2` is required (`np.product`). MDNet's vtree shuffle is
unseeded, so only run-distributions reproduce. For bit-identical "ours"
metrics, set `experiment.device: "cpu"` in the binary-BN configs (GPU float
non-associativity can flip bits).

## 6. Standalone extras

```bash
python -m experiments.k_scaling            # T-ID complexity check: compiled-vs-base
                                           # circuit size vs. predicted x^K (colliderdoor mixed)
python -m experiments.fit_univariate_logspline --density skewed_bimodal   # leaf-fitting demo figure
python -m experiments.run_unconstrained_baseline   # regenerates + runs the md_sets=[] ablation
```

## Experiment inventory (`configs/`)

| Config | Studies |
|--------|---------|
| `backdoor_cont_400K_N{8,16,32,64,128}` | capacity (num_nodes) sweep at 400K samples |
| `backdoor_cont_{25K,100K,1200K}_N64` | data-size sweep at N=64 |
| `backdoor_cont_25K_N16{,_soft}` | small-data reference (+ softening A/B) |
| `backdoor_cont_400K_N64_G1` | ablation: single sum group (`max_sum_num_groups: 1`) |
| `backdoor_cont_400K_N64_unconstrained` | ablation: empty `md_sets` (no determinism) |
| `frontdoor_cont_100K_N64` | frontdoor, continuous, pinned vtree `(X,(Y,M))` |
| `frontdoor_mixed_N16` | frontdoor, discrete X/M + CLG Y |
| `colliderdoor_cont_100K_N64` | collider-door, continuous, pinned vtree `((Z,X),(M,Y))` |
| `colliderdoor_mixed_N16` | collider-door, discrete Z/X/M + CLG Y |
| `configs/binary_bn/{asia,child,win95pts,andes}.json` | MDNet real-world benchmarks |
| `base_config.json` | placeholder template — not runnable as-is |

The config file name **is** the experiment ID. Key schema (full details in
`experiments/utils/config.py`):

- `dataset`: `name` → `data/<name>/` (90/10 train/test split)
- `model`: `num_nodes` (capacity), `n_confounders`, `md_sets` (determinism
  family; default = treatment + all confounders), `vtree` (optional Newick
  pin, e.g. `"((Z,X),(M,Y))"`), `fairness_temperature`,
  `max_leaf_num_nodes`/`max_leaf_num_groups`/`max_sum_num_groups` caps
- `training`: `batch_size`, `total_iters`, `step_size`, `decay_rate`,
  `decay_every`, `leaf_lr`
- `checkpointing`: `checkpoint_every_epochs`, `save_initial`, `save_final`
- `evaluation`: grid/slice resolution and quantile ranges
- `identification` (optional): `treatment`, `outcome`, `determinisms` override

## Repository layout

| Path | Contents |
|------|----------|
| `src/symbolic/` | VTree, `CausalGraph`, T-ID (`identification.py`), SCM package, symbolic arithmetic circuit + EM trainer |
| `src/construction/` | MD circuit builder, learned vtrees, skeleton/mechanism generators, latent projection |
| `src/utils/` | `BitSet`, intervals, pairwise MI, node allocators |
| `scripts/` | Synthetic-data generators + exploration plots |
| `configs/` | Experiment configs (file name = experiment ID) |
| `experiments/` | Train-only pipeline + post-hoc evaluation/plotting |
| `tests/` | Test suite mirroring `src/` |
| `benchmark/` | Standalone performance benchmarks |
| `scratch/` | Experiment scripts / POCs (not part of the package) |

## Reference

Companion code for the paper on automating causal inference with
marginal-determinism probabilistic circuits (T-ID). MIT licensed.
