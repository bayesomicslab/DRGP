"""DRGP command-line entry point: load -> (optional pathway mask) -> fit CAVI -> evaluate -> save.

Slim wrapper around the real ``drgp`` package APIs. Keeps only the load -> fit ->
evaluate -> save flow; campaign plumbing (hyperparameter search, baselines, plotting
sweeps) lives elsewhere and is intentionally not part of this entry point.
"""
import argparse
import os

import numpy as np

from drgp.data.loader import DataLoader
from drgp.model.cavi import CAVI
from drgp.pathways import load_pathways
from drgp.results import compute_metrics, save_results, print_model_summary


def build_parser():
    p = argparse.ArgumentParser(prog="drgp", description="Disease-Relevant Gene Programs (DRGP)")
    p.add_argument("--data", required=True, help="Path to an .h5ad file (cells/samples x genes)")
    p.add_argument("--label-column", required=True, nargs="+",
                   help="obs column(s) holding the binary phenotype; multiple = multi-outcome")
    p.add_argument("--aux-columns", nargs="*", default=None,
                   help="obs columns used as auxiliary covariates in the regression head")
    p.add_argument("--patient-column", default=None,
                   help="obs column grouping cells into units; enables grouped splits")
    p.add_argument("--mode", choices=["unmasked", "masked", "combined"], default="unmasked")
    p.add_argument("--n-factors", type=int, required=True,
                   help="K (latent programs). Ignored (recomputed) for masked/combined "
                        "once --pathway-file is loaded.")
    p.add_argument("--pathway-file", default=None, help="GMT file; required for masked/combined")
    p.add_argument("--pathway-genes-ensembl", action="store_true",
                   help="GMT gene IDs are already Ensembl (skip symbol conversion)")
    p.add_argument("--n-drgps", type=int, default=0,
                   help="combined mode: number of free (de novo) programs added to the pathways")
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--check-freq", type=int, default=5)
    p.add_argument("--v-warmup", type=int, default=50)
    p.add_argument("--early-stopping", choices=["elbo", "heldout_ll"], default="elbo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ("masked", "combined") and not args.pathway_file:
        raise SystemExit(f"--mode {args.mode} requires --pathway-file")

    # label_column is a list from argparse (nargs="+"); a single-element list still
    # produces a 1-D y from DataLoader.get_matrices, multi-element -> (n, kappa).
    label_columns = args.label_column

    loader = DataLoader(data_path=args.data, use_cache=False, verbose=args.verbose)
    data = loader.load_and_preprocess(
        label_column=label_columns, aux_columns=args.aux_columns,
        train_ratio=0.7, val_ratio=0.15, stratify_by=None,
        min_cells_expressing=0.001, layer=None, convert_to_ensembl=True,
        filter_protein_coding=False, random_state=args.split_seed,
        normalize=False, return_sparse=False, patient_column=args.patient_column)

    X_train, X_aux_train, y_train = data["train"]
    X_val, X_aux_val, y_val = data["val"]
    X_test, X_aux_test, y_test = data["test"]
    gene_list = list(loader.gene_list)

    mask = None
    pw_names = None
    n_factors = args.n_factors
    n_pathway_factors = None
    if args.pathway_file:
        # load_pathways only returns a mask over the genes that survive its
        # OWN filtering (a subset of gene_list, in a different order) -- it
        # is NOT already aligned to the expression matrix's gene axis. Align
        # it here the same way quick_reference.py does: scatter the pathway
        # x pathway-gene matrix into a pathway x full-gene-list matrix.
        # use_cache=False: a published CLI must not write outside --output-dir.
        # load_pathways defaults to caching content-hashed pickles under a
        # shared per-user cache dir -- not appropriate for a
        # release entry point that may run anywhere.
        pathway_mat, pw_names, pathway_genes = load_pathways(
            gmt_path=args.pathway_file, gene_filter=gene_list,
            convert_to_ensembl=not args.pathway_genes_ensembl,
            use_cache=False)

        gene_to_expr_idx = {g: i for i, g in enumerate(gene_list)}
        gene_to_pathway_idx = {g: i for i, g in enumerate(pathway_genes)}
        common_genes = set(gene_list) & set(pathway_genes)
        if len(common_genes) < 10:
            raise ValueError(
                f"Too few common genes ({len(common_genes)}) between expression "
                f"data and pathways. Check --pathway-genes-ensembl / gene ID format.")

        n_pathways = pathway_mat.shape[0]
        n_genes_expr = len(gene_list)
        mask = np.zeros((n_pathways, n_genes_expr), dtype=np.float32)
        for gene in common_genes:
            mask[:, gene_to_expr_idx[gene]] = pathway_mat[:, gene_to_pathway_idx[gene]]
        pathways_per_gene = mask.sum(axis=0)
        print(f"pathways: {n_pathways} x {n_genes_expr} genes, "
              f"density={mask.mean()*100:.2f}%, "
              f"genes covered={int((pathways_per_gene > 0).sum())}")

        # Masked mode: restrict X to pathway-covered genes only (matches
        # quick_reference.py -- non-pathway genes only add Poisson noise
        # since their beta is forced to ~0 anyway).
        if args.mode == "masked":
            keep_idx = np.where(pathways_per_gene > 0)[0]
            X_train = X_train[:, keep_idx]
            X_val = X_val[:, keep_idx]
            X_test = X_test[:, keep_idx]
            mask = mask[:, keep_idx]
            gene_list = [gene_list[i] for i in keep_idx]
            print(f"[MASKED] restricted to {len(keep_idx)} pathway genes")
            n_factors = n_pathways
        elif args.mode == "combined":
            n_drgps = args.n_drgps
            drgp_rows = np.ones((n_drgps, mask.shape[1]), dtype=np.float32)
            mask = np.vstack([mask, drgp_rows])
            pw_names = list(pw_names) + [f"DRGP_{i+1}" for i in range(n_drgps)]
            n_pathway_factors = n_pathways
            n_factors = n_pathways + n_drgps
            print(f"[COMBINED] n_factors = {n_pathways} pathways + {n_drgps} free = {n_factors}")

    # splits dict for save_results: {'train': [...ids], 'val': [...], 'test': [...]}
    splits = {
        "train": list(data["splits"]["train"]),
        "val": list(data["splits"]["val"]),
        "test": list(data["splits"]["test"]),
    }

    model = CAVI(n_factors=n_factors, mode=args.mode, pathway_mask=mask,
                 pathway_names=pw_names, n_pathway_factors=n_pathway_factors,
                 random_state=args.seed)
    model.fit(X_train, y_train, X_aux_train=X_aux_train,
              X_val=X_val, y_val=y_val, X_aux_val=X_aux_val,
              max_iter=args.max_iter, check_freq=args.check_freq, tol=args.tol,
              v_warmup=args.v_warmup, early_stopping=args.early_stopping,
              verbose=args.verbose)

    print_model_summary(model, gene_list=gene_list)

    # compute_metrics takes arrays (y_true, y_pred, y_proba), not the model itself.
    # predict_proba() is the only prediction API CAVI exposes -> threshold at 0.5
    # for the hard-label metrics (accuracy/precision/recall/f1/confusion matrix).
    y_proba = model.predict_proba(X_test, X_aux_new=X_aux_test)
    y_proba_flat = np.asarray(y_proba).reshape(-1) if np.asarray(y_proba).ndim <= 1 \
        else np.asarray(y_proba)
    y_pred = (np.asarray(y_proba) >= 0.5).astype(int)

    if len(label_columns) == 1:
        metrics = compute_metrics(np.asarray(y_test).ravel(), y_pred.ravel(), y_proba_flat)
    else:
        # Multi-outcome: report per-outcome metrics.
        metrics = {}
        y_test_arr = np.asarray(y_test)
        for k, lname in enumerate(label_columns):
            metrics[lname] = compute_metrics(
                y_test_arr[:, k], y_pred[:, k], np.asarray(y_proba)[:, k])
    print(f"Test metrics: {metrics}")

    save_results(
        model, args.output_dir, gene_list=gene_list, splits=splits,
        mode=args.mode, label_columns=label_columns, aux_columns=args.aux_columns,
        program_names=pw_names if args.mode == "masked" else None,
        val_test_data={
            "X_val": X_val, "X_aux_val": X_aux_val,
            "X_test": X_test, "X_aux_test": X_aux_test,
        },
    )
    print(f"DONE -> {args.output_dir}")


if __name__ == "__main__":
    main()
