"""End-to-end smoke test for the CLI: load -> fit CAVI -> evaluate -> save.

Builds a tiny synthetic h5ad (like tests/test_loader.py's fixture) and runs the
CLI's main() in-process with --max-iter tiny. Confirms the whole chain runs
without a signature error and writes the expected output artifacts, for
unmasked, masked, and combined modes.

The masked/combined cases are the regression coverage for the riskiest part of
cli.py: load_pathways() returns a mask over only the genes that survive ITS
OWN filtering (a subset of the expression gene_list, reordered) -- the CLI has
to scatter that back onto the full expression gene axis (and, for masked mode,
restrict X/gene_list down to the covered genes) before handing it to
CAVI(pathway_mask=...). Get that wrong and CAVI._init_beta_mask raises a numpy
broadcast error (shape (n_pathway_genes, K) vs (n_data_genes, K)).
"""
import os

import numpy as np
import anndata as ad
import pandas as pd
import pytest


@pytest.fixture
def tiny_h5ad(tmp_path):
    rng = np.random.default_rng(0)
    n, p = 120, 40
    X = rng.poisson(2.0, size=(n, p)).astype(np.float32)
    obs = pd.DataFrame({
        "label": (rng.random(n) < 0.5).astype(int),
        "Sex": (rng.random(n) < 0.5).astype(int),
        "patient": [f"P{i // 6:02d}" for i in range(n)],
    }, index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"ENSG{i:011d}" for i in range(p)])
    A = ad.AnnData(X=X, obs=obs, var=var)
    path = tmp_path / "tiny.h5ad"
    A.write_h5ad(path)
    return str(path)


@pytest.fixture
def tiny_gmt(tmp_path):
    """2 pathways over KNOWN, disjoint subsets of the h5ad's gene IDs.

    Uses the same ENSG{i:011d} format the fixture's var index uses (already
    Ensembl, matching what DataLoader/load_pathways see after the CLI's
    convert_to_ensembl=True short-circuits to an identity mapping for
    ENSG-prefixed IDs). 6 genes/pathway clears load_pathways' adaptive-filter
    floor for small pathways (>=5 overlap genes).
    """
    pw1_genes = [f"ENSG{i:011d}" for i in range(0, 6)]     # genes 0-5
    pw2_genes = [f"ENSG{i:011d}" for i in range(6, 12)]    # genes 6-11
    path = tmp_path / "tiny.gmt"
    with open(path, "w") as f:
        f.write("PW1\turl\t" + "\t".join(pw1_genes) + "\n")
        f.write("PW2\turl\t" + "\t".join(pw2_genes) + "\n")
    return str(path)


def test_cli_smoke_unmasked(tiny_h5ad, tmp_path):
    from drgp.cli import main

    out_dir = str(tmp_path / "cli_smoke")
    main([
        "--data", tiny_h5ad,
        "--label-column", "label",
        "--aux-columns", "Sex",
        "--patient-column", "patient",
        "--n-factors", "5",
        "--max-iter", "5",
        "--mode", "unmasked",
        "--output-dir", out_dir,
    ])

    written = set(os.listdir(out_dir))
    assert "vi_model_params.npz" in written
    assert "vi_gene_programs.csv.gz" in written
    assert "vi_summary.json.gz" in written
    assert "vi_theta_train.csv.gz" in written


def test_cli_smoke_masked(tiny_h5ad, tiny_gmt, tmp_path):
    from drgp.cli import main

    out_dir = str(tmp_path / "cli_smoke_masked")
    main([
        "--data", tiny_h5ad,
        "--label-column", "label",
        "--n-factors", "999",  # deliberately wrong; masked mode must override to n_pathways=2
        "--max-iter", "5",
        "--mode", "masked",
        "--pathway-file", tiny_gmt,
        "--pathway-genes-ensembl",
        "--output-dir", out_dir,
    ])

    written = set(os.listdir(out_dir))
    assert "vi_model_params.npz" in written
    assert "vi_gene_programs.csv.gz" in written
    assert "vi_summary.json.gz" in written
    assert "vi_theta_train.csv.gz" in written

    # Regression check for the mask-alignment logic: masked mode restricts X
    # (and gene_list) to only the pathway-covered genes (12 of the 40, per
    # tiny_gmt), and the model has K=2 factors (overriding the bogus
    # --n-factors 999) -- so the saved gene-programs matrix must be 2 x 12,
    # not 2 x 40 and not the --n-factors value.
    import gzip
    import pandas as pd
    programs_path = os.path.join(out_dir, "vi_gene_programs.csv.gz")
    with gzip.open(programs_path, "rt") as f:
        programs_df = pd.read_csv(f, index_col=0)
    gene_cols = [c for c in programs_df.columns if c.startswith("ENSG")]
    assert len(gene_cols) == 12, f"expected 12 pathway-covered genes, got {len(gene_cols)}"
    assert programs_df.shape[0] == 2, f"expected K=2 (n_pathways), got {programs_df.shape[0]}"
    assert set(gene_cols) == {f"ENSG{i:011d}" for i in range(12)}


def test_cli_smoke_combined(tiny_h5ad, tiny_gmt, tmp_path):
    from drgp.cli import main

    out_dir = str(tmp_path / "cli_smoke_combined")
    main([
        "--data", tiny_h5ad,
        "--label-column", "label",
        "--n-factors", "999",  # ignored; combined mode computes n_pathways + n_drgps
        "--max-iter", "5",
        "--mode", "combined",
        "--pathway-file", tiny_gmt,
        "--pathway-genes-ensembl",
        "--n-drgps", "3",
        "--output-dir", out_dir,
    ])

    written = set(os.listdir(out_dir))
    assert "vi_model_params.npz" in written
    assert "vi_gene_programs.csv.gz" in written
    assert "vi_summary.json.gz" in written
    assert "vi_theta_train.csv.gz" in written

    # Combined mode keeps ALL genes (no restriction) and K = n_pathways + n_drgps = 2 + 3 = 5.
    import gzip
    import pandas as pd
    programs_path = os.path.join(out_dir, "vi_gene_programs.csv.gz")
    with gzip.open(programs_path, "rt") as f:
        programs_df = pd.read_csv(f, index_col=0)
    gene_cols = [c for c in programs_df.columns if c.startswith("ENSG")]
    assert len(gene_cols) == 40, f"combined mode must keep all genes, got {len(gene_cols)}"
    assert programs_df.shape[0] == 5, f"expected K=5 (2 pathways + 3 DRGPs), got {programs_df.shape[0]}"
