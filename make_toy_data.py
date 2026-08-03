#!/usr/bin/env python
"""Build the shipped toy dataset from a DRGP single-cell simulation.

Produces a small, self-contained example that exercises all three DRGP modes:

    data/toy_sim.h5ad   1,600 cells x ~700 genes, raw integer counts, patient-grouped,
                        with the full simulation ground truth in .uns
    data/toy_sim.gmt    the two ANNOTATED program supports, as a pathway file

Design constraints (each one matters):
  * Gene IDs are Ensembl, not symbols. DRGP's loader short-circuits symbol->Ensembl conversion for
    ENSG-prefixed IDs, so the toy runs with NO network access. The source simulation stores symbols,
    so we map them through the repo's cached mapping and drop the unmapped.
  * All genes belonging to a simulated program support are kept; the remainder of the gene axis is a
    random background sample. Subsampling genes blindly would destroy the supports and make the
    recovery ground truth meaningless.
  * All 80 patients are kept (cells are subsampled within patient), so patient-grouped splitting has
    enough groups to be meaningful and the inherited-label structure is preserved.
  * .uns truth is REMAPPED to the toy's gene indices, so `S_ell` / `mask_M` still index correctly.

Source: the high-signal simulation configuration (w_Z=0.7, r=0.30, rho=0.3, truth 0).

Run:  python make_toy_data.py
"""
import json
import os

import numpy as np
import anndata as ad
import pandas as pd
import scipy.sparse as sp

REPO = os.environ.get("DRGP_SIM_REPO", "/labs/Aguiar/SSPA_BRAY")  # override for other checkouts
SRC = f"{REPO}/data/Simulations/sim_flat_v2/datasets/truth0_h0.7_r0.3_rho0.3_lift1_seed0.h5ad"
CACHE = f"{REPO}/BRay/gene_id_cache.json"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CELLS_PER_PATIENT = 20      # of 100 in the source
N_BACKGROUND_GENES = 300    # on top of every program-support gene
SEED = 0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    A = ad.read_h5ad(SRC)
    print(f"source: {A.shape[0]} cells x {A.shape[1]} genes")

    # AnnData serializes a list-of-arrays as a dict keyed by stringified index
    _S = A.uns["S_ell"]
    S_ell = ([np.asarray(_S[str(i)], dtype=int) for i in range(len(_S))]
             if isinstance(_S, dict) else [np.asarray(s, dtype=int) for s in _S])
    n_prog = len(S_ell)

    # ---- gene axis: every support gene + a random background ----
    support = np.unique(np.concatenate(S_ell))
    other = np.setdiff1d(np.arange(A.shape[1]), support)
    background = rng.choice(other, size=min(N_BACKGROUND_GENES, other.size), replace=False)
    keep_genes = np.sort(np.concatenate([support, background]))

    # ---- symbols -> Ensembl (offline, via the repo cache); drop unmapped/duplicated ----
    sym2ens = json.load(open(CACHE))["symbol_to_ensembl"]
    syms = A.var_names.to_numpy()[keep_genes]
    ens = np.array([sym2ens.get(s) or "" for s in syms], dtype=object)
    ok = np.array([bool(e) for e in ens])
    _, first = np.unique(ens[ok], return_index=True)          # keep first occurrence of each ENSG
    sel = np.flatnonzero(ok)[np.sort(first)]
    keep_genes, ens = keep_genes[sel], ens[sel]
    print(f"genes: {support.size} support + {background.size} background "
          f"-> {keep_genes.size} after Ensembl mapping")

    # ---- cells: subsample within each patient, keep every patient ----
    pid = A.obs["patient_id"].to_numpy()
    keep_cells = np.concatenate([
        rng.choice(np.flatnonzero(pid == p), size=min(CELLS_PER_PATIENT, (pid == p).sum()),
                   replace=False)
        for p in pd.unique(pid)])
    keep_cells.sort()
    print(f"cells: {keep_cells.size} ({len(pd.unique(pid))} patients x <= {CELLS_PER_PATIENT})")

    # ---- slice ----
    X = A.X[keep_cells][:, keep_genes]
    X = sp.csr_matrix(X, dtype=np.int32) if sp.issparse(X) else sp.csr_matrix(np.asarray(X), dtype=np.int32)
    obs = A.obs.iloc[keep_cells].copy()
    obs = obs.rename(columns={"y": "label"})
    toy = ad.AnnData(X=X, obs=obs.reset_index(drop=True),
                     var=pd.DataFrame(index=pd.Index(ens.astype(str), name="gene")))
    toy.obs_names = [f"cell{i:05d}" for i in range(toy.n_obs)]

    # ---- remap ground truth onto the toy gene axis ----
    old2new = {int(g): i for i, g in enumerate(keep_genes)}
    toy.uns["S_ell"] = {str(i): np.array([old2new[g] for g in s if g in old2new], dtype=int)
                        for i, s in enumerate(S_ell)}
    mask = np.asarray(A.uns["mask_M"])
    toy.uns["mask_M"] = mask[keep_genes, :]
    for k in ("u",):
        if k in A.uns:
            toy.uns[k] = np.asarray(A.uns[k])[keep_genes, :]
    for k in ("v_star", "theta_star", "T_ell", "polarity", "h2", "r", "rho", "delta",
              "tau", "truth_idx", "lift", "lift_idx"):
        if k in A.uns:
            v = A.uns[k]
            toy.uns[k] = np.asarray(v)[keep_cells] if k == "theta_star" else v
    toy.uns["source"] = os.path.basename(SRC)
    toy.uns["note"] = ("Toy subsample of a DRGP single-cell simulation. obs['label'] is the "
                       "patient-inherited phenotype; obs['patient_id'] groups cells by donor; "
                       "obs['cell_type'] gives the simulated cell type. uns['S_ell'] holds the "
                       "true gene-program supports (indices into var), uns['mask_M'] the annotated "
                       "supports, uns['v_star'] the per-program liability weights.")

    out_h5 = os.path.join(OUT_DIR, "toy_sim.h5ad")
    toy.write_h5ad(out_h5, compression="gzip")

    # ---- GMT: the ANNOTATED programs only (columns of mask_M) ----
    out_gmt = os.path.join(OUT_DIR, "toy_sim.gmt")
    with open(out_gmt, "w") as f:
        for k in range(toy.uns["mask_M"].shape[1]):
            genes = toy.var_names[np.flatnonzero(toy.uns["mask_M"][:, k])].tolist()
            f.write(f"TOY_PROGRAM_{k+1}\tsimulated\t" + "\t".join(genes) + "\n")

    print(f"\nwrote {out_h5}  ({os.path.getsize(out_h5)/1e6:.2f} MB)  {toy.shape}")
    print(f"wrote {out_gmt} ({toy.uns['mask_M'].shape[1]} annotated programs)")
    print(f"  label balance: {toy.obs['label'].mean():.2f}   patients: {toy.obs['patient_id'].nunique()}")
    print(f"  counts: dtype={toy.X.dtype} min={toy.X.min()} max={toy.X.max()} nnz={toy.X.nnz}")
    print(f"  program sizes on the toy axis: {[len(v) for v in toy.uns['S_ell'].values()]}")


if __name__ == "__main__":
    main()
