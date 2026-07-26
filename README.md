# DRGP: Disease-Relevant Gene Programs

Supervised Poisson Factorization via Coordinate Ascent Variational Inference (CAVI) for joint
gene-program discovery and phenotype classification from single-cell (or bulk) RNA-seq counts.

DRGP factorizes a cells/samples x genes count matrix into a small number of interpretable gene
programs while simultaneously regressing a patient/sample-level phenotype on those programs. Three
modes trade off how much of the factorization structure is supplied by a pathway database versus
learned de novo from data (see [Modes](#modes)).

## Install

```bash
pip install -e .
```

Requires Python >= 3.10. Core dependencies (numpy, scipy, pandas, anndata, scikit-learn,
matplotlib, h5py) are installed automatically. Optional extras:

```bash
pip install -e ".[gpu]"   # jax[cuda12] — GPU-accelerated CAVI
pip install -e ".[dev]"   # pytest — run the test suite
```

The CAVI implementation auto-detects a backend at import time: JAX+GPU > JAX+CPU > NumPy/SciPy.
The `gpu` extra is optional — everything runs on plain NumPy/SciPy if JAX is absent.

## Quickstart: CLI

The `drgp` console script (installed by `pip install -e .`) is a slim load -> fit -> evaluate ->
save wrapper. Run without installing via `python -m drgp.cli`; set `PYTHONPATH=.` from the repo
root if you haven't installed the package.

```bash
drgp \
    --data path/to/data.h5ad \
    --label-column t2dm \
    --aux-columns Sex \
    --patient-column sampleID \
    --mode unmasked \
    --n-factors 35 \
    --max-iter 3000 \
    --output-dir results/t2dm_unmasked \
    --verbose
```

Masked mode restricts the factorization to genes covered by a pathway database and fixes K to the
number of pathways; combined mode keeps all genes and adds free (de novo) programs alongside the
pathway-constrained ones:

```bash
drgp --data path/to/data.h5ad --label-column t2dm --mode masked \
    --pathway-file pathways.gmt --output-dir results/t2dm_masked

drgp --data path/to/data.h5ad --label-column t2dm --mode combined \
    --pathway-file pathways.gmt --n-drgps 10 --n-factors 0 \
    --output-dir results/t2dm_combined
```

### CLI flags

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--data` | yes | — | Path to an `.h5ad` file (cells/samples x genes) |
| `--label-column` | yes | — | obs column(s) holding the binary phenotype; pass multiple for multi-outcome inference |
| `--aux-columns` | no | none | obs columns used as auxiliary covariates in the regression head |
| `--patient-column` | no | none | obs column grouping cells into units; enables patient-grouped train/val/test splits |
| `--mode` | no | `unmasked` | `unmasked`, `masked`, or `combined` — see [Modes](#modes) |
| `--n-factors` | yes | — | K (latent programs). Ignored and recomputed for `masked`/`combined` once `--pathway-file` is loaded |
| `--pathway-file` | for masked/combined | none | GMT file; required for `--mode masked` or `--mode combined` |
| `--pathway-genes-ensembl` | no | off | GMT gene IDs are already Ensembl (skip symbol -> Ensembl conversion) |
| `--n-drgps` | no | 0 | combined mode: number of free (de novo) programs added alongside the pathways |
| `--max-iter` | no | 3000 | maximum CAVI iterations |
| `--tol` | no | 1e-3 | convergence tolerance for early stopping |
| `--check-freq` | no | 5 | iterations between convergence checks |
| `--v-warmup` | no | 50 | iterations before the supervised regression head (`v`) starts updating |
| `--early-stopping` | no | `elbo` | `elbo` or `heldout_ll` |
| `--seed` | no | 0 | model random seed |
| `--split-seed` | no | 0 | train/val/test split random seed |
| `--output-dir` | yes | — | directory for output artifacts (created if missing) |
| `--verbose` | no | off | print progress |

Run `drgp --help` (or `python -m drgp.cli --help`) for the authoritative, up-to-date list.

### Output artifacts

`--output-dir` receives (prefix `vi_` by default):

- `vi_model_params.npz` — fitted posterior parameters
- `vi_gene_programs.csv.gz` — gene x program loading matrix
- `vi_r_beta.csv.gz` — posterior inclusion probabilities for `beta` (spike-and-slab)
- `vi_theta_train.csv.gz` / `vi_theta_val.csv.gz` / `vi_theta_test.csv.gz` — per-cell program activations
- `vi_gamma_weights.csv.gz` / `vi_gamma_variance.csv.gz` — auxiliary-covariate regression weights (written only when `--aux-columns` is set)
- `vi_summary.json.gz` — config, ELBO trajectory, held-out metrics
- `training_curves.png` — ELBO / held-out-LL trajectory plot

## Input data contract

`--data` must be an `.h5ad` (AnnData) file:

- **`X` (or the layer DRGP reads): raw, non-negative integer counts.** DRGP is a Poisson/negative-binomial
  factorization — normalized, log-transformed, or otherwise non-integer expression values violate the
  generative model.
- **`obs`** must contain:
  - the column(s) named in `--label-column` — binary (or integer-codeable) phenotype label(s)
  - any column(s) named in `--aux-columns` — numeric or categorical covariates (categorical columns
    are one-hot/dummy-encoded automatically)
  - if `--patient-column` is given, that column must identify the patient/donor/sample each cell (or
    unit) belongs to, so splits can be grouped at the patient level (no donor leakage across
    train/val/test)
- **`var`** index is used as gene identifiers; symbol -> Ensembl conversion is applied automatically
  unless the mask genes are already Ensembl (`--pathway-genes-ensembl`).

For `--mode masked` / `--mode combined`, `--pathway-file` is a GMT file (`PATHWAY_NAME<TAB>URL<TAB>GENE1<TAB>GENE2...`),
e.g. MSigDB C2/Reactome. Gene symbols are converted to Ensembl IDs to match the expression matrix
unless `--pathway-genes-ensembl` is set.

## Modes

- **`unmasked`** — standard supervised Poisson factorization; all K factors are learned de novo,
  no pathway prior.
- **`masked`** — the gene x factor loading matrix (`beta`) is masked by a binary pathway membership
  matrix; K is fixed to the number of surviving pathways, and the expression matrix is restricted
  to pathway-covered genes only.
- **`combined`** — pathway-masked factors plus `--n-drgps` additional free factors over the full
  gene set, fit jointly; `--n-factors` is ignored (recomputed as `n_pathways + n_drgps`).

## Simulation framework

`drgp.simulation` generates labeled count data with known ground-truth gene programs, for
validating recovery and prediction. **DRGP does not generate the phenotype-free background counts
itself.** You supply those (a realistic control/healthy-only count matrix or its fitted NB
parameter surface) from an upstream simulator:

- **Single-cell**: fit [scDesign3](https://github.com/SONGDONGYUAN1994/scDesign3) (R) to your
  control/healthy cells to get a marginal- and dependence-preserving NB mean/dispersion surface.
- **Bulk**: fit [SPsimSeq](https://github.com/CenterForStatistics-UGent/SPsimSeq) (R) to your
  control/healthy samples to get a realized background count matrix.

Neither simulator is shipped with this package — install and run them yourself, then hand DRGP the
result via one of two background shapes:

```python
from drgp.simulation.background import RealizedBackground, ParametricBackground
```

| Class | Fields | Use with |
|---|---|---|
| `RealizedBackground` | `counts` (n_units, n_genes) non-negative integers; `gene_ids` (n_genes,); `unit_to_group` (n_units,) int, optional (default identity); `cell_type` (n_units,) int, optional | bulk (`thin_diff` injection, from SPsimSeq) |
| `ParametricBackground` | `mu` (n_units, n_genes) positive NB means; `dispersion` (n_genes,) NB size/dispersion; `gene_ids` (n_genes,); `unit_to_group` optional; `cell_type` optional | single-cell (`nb_resample` injection, from scDesign3) |

`gene_ids` length must match the gene axis of `counts`/`mu`; `dispersion` length must match
`mu.shape[1]`. `unit_to_group` defaults to the identity (each unit is its own group) when omitted.

Given a background, simulate with:

```python
import numpy as np
from drgp.simulation import simulate, SimSpec
from drgp.simulation.background import ParametricBackground

# Single-cell: unit_to_group maps each cell to its patient/donor; cell_type is
# required whenever ActivitySpec.responder_restricted=True (the single_cell default).
n_patients, cells_per_patient, n_genes = 40, 20, 500
n_units = n_patients * cells_per_patient
rng = np.random.default_rng(0)
unit_to_group = np.repeat(np.arange(n_patients), cells_per_patient)

background = ParametricBackground(
    mu=rng.gamma(2.0, 2.0, size=(n_units, n_genes)),
    dispersion=np.full(n_genes, 10.0),
    gene_ids=np.array([f"ENSG{i:011d}" for i in range(n_genes)]),
    unit_to_group=unit_to_group,
    cell_type=rng.integers(0, 4, size=n_units),
)

spec = SimSpec.for_modality("single_cell")  # or "bulk"
result = simulate(spec, background)

result.adata           # AnnData: counts (X), obs['y'] label, obs['liability'], truth in .uns
result.truth           # ProgramTruth: planted loadings, program membership, polarity
result.labels          # dict: per-unit and per-group liability/label arrays
```

Bulk mode (`SimSpec.for_modality("bulk")`) uses `RealizedBackground` instead (a fitted SPsimSeq
count matrix), sets `unit_to_group` to the identity by default (each unit is its own group), and
does not require `cell_type`.

`SimSpec.for_modality("bulk")` selects the `thin_diff` injection operator, which **requires R with
the `seqgendiff` package installed** (shells out to `Rscript`; point `DRGP_RSCRIPT` at a specific
`Rscript` binary if it isn't on `PATH`). `SimSpec.for_modality("single_cell")` uses `nb_resample`,
which is pure Python/NumPy and needs no R. Every knob (program count/size/polarity, carrier
propensity, activity, injection, label liability) is an explicit field on `SimSpec` — see
`drgp/simulation/spec.py` for the full parameter set and the defaults used in the published
benchmarks.

Recovery evaluation (Hungarian-matched, scale-free) lives in `drgp.simulation.evaluate`:

```python
from drgp.simulation.evaluate.align import hungarian_match
from drgp.simulation.evaluate.recovery import support_auprc
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```
