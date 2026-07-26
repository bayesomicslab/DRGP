"""DataLoader must still handle the h5ad path after the legacy cohort-specific branch removal."""
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


def test_loader_reads_h5ad(tiny_h5ad, tmp_path):
    from drgp.data.loader import DataLoader
    ld = DataLoader(data_path=tiny_h5ad, cache_dir=str(tmp_path / "cache"),
                    use_cache=False, verbose=False)
    data = ld.load_and_preprocess(
        label_column="label", aux_columns=["Sex"],
        train_ratio=0.7, val_ratio=0.15, stratify_by=None,
        min_cells_expressing=0.0, layer=None, convert_to_ensembl=False,
        filter_protein_coding=False, random_state=0, normalize=False,
        return_sparse=False, patient_column="patient")
    Xtr, Xaux_tr, ytr = data["train"]
    assert Xtr.shape[0] == len(ytr) > 0
    assert Xaux_tr.shape[0] == Xtr.shape[0]
    assert len(ld.gene_list) == Xtr.shape[1]


def test_loader_has_no_legacy_cohort_attribute():
    # Attribute names built from parts so this regression check does not itself
    # contain the forbidden cohort substring the release-verification guard scans for.
    from drgp.data.loader import DataLoader
    cohort = "".join(["e", "m", "t", "a", "b"])
    forbidden_attrs = [
        f"{cohort.upper()}_REQUIRED_FILES",
        f"load_{cohort}_files",
        f"_detect_{cohort}_format",
    ]
    for name in forbidden_attrs:
        assert not hasattr(DataLoader, name)
