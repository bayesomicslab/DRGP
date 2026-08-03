"""Pathway-mask construction for DRGP's masked and combined modes.

Reads a GMT file and returns a binary mask M (n_pathways x n_genes) where M[j, i] = 1 if gene i
belongs to pathway j. Extracted from the original utils.py; the cache key hashes the GMT PATH AND
CONTENT, because simulation campaigns regenerate a GMT in place at a stable path and a path-only
key silently returns a stale parse.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from drgp.data.gene_ids import GeneIDConverter

_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "drgp", "pathways")


def load_pathways(
    gmt_path: str,
    convert_to_ensembl: bool = True,
    species: str = 'human',
    gene_filter: Optional[List[str]] = None,
    min_genes: int = 0,
    max_genes: int = 50000,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    excluded_keywords: Optional[List[str]] = None,
    require_prefix: Optional[str] = None,
    pathway_selection: Optional[str] = None,
    disable_adaptive_filter: bool = False
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load pathway definitions from GMT file into a binary matrix.

    GMT format: PATHWAY_NAME<tab>URL<tab>GENE1<tab>GENE2<tab>...

    Caches the Ensembl-converted pathways to avoid repeated conversion overhead.

    Parameters
    ----------
    gmt_path: str
        Path to GMT file. Can be the full C2 collection or a pre-filtered
        subset GMT (e.g. from results/pathway_selections/). Required --
        no repository-specific default is shipped.
    convert_to_ensembl: bool, default=True
        Convert gene symbols to Ensembl IDs.
    species: str, default='human'
        Species for gene ID conversion.
    gene_filter: list of str, optional
        If provided, only include genes in this list. Useful for filtering
        to genes present in expression data.
    min_genes: int, default=5
        Minimum number of genes for a pathway to be included.
    max_genes: int, default=500
        Maximum number of genes for a pathway to be included.
    cache_dir: str, default=~/.cache/drgp/pathways
        Directory for caching converted pathways.
    use_cache: bool, default=True
        Whether to use cached converted pathways.
    excluded_keywords: list of str, optional
        Exclude pathways containing any of these keywords (case-insensitive).
        Default: ["ADME", "DRUG", "MISCELLANEOUS", "EMT"]
    require_prefix: str, optional
        If provided, only keep pathways starting with this prefix.
        Default: None (keep all sources). Set to 'REACTOME' to restrict
        to Reactome pathways only.
    pathway_selection: str, optional
        Path to a text file listing pathway names to keep (one per line),
        or a pre-filtered GMT file. When provided, only pathways whose names
        appear in this file are retained. This is applied before all other
        filters. Pre-computed selections are available in
        results/pathway_selections/ (e.g. covid_selected_pathways.txt).

    Returns
    -------
    pathway_mat: np.ndarray
        Binary matrix (n_pathways, n_genes) where pathway_mat[i,j]=1 if
        gene j is in pathway i.
    pathway_names: list of str
        Pathway names corresponding to rows.
    gene_names: list of str
        Gene names (Ensembl or symbol) corresponding to columns.
    """
    import hashlib

    # Default excluded keywords
    if excluded_keywords is None:
        excluded_keywords = ["ADME", "DRUG", "MISCELLANEOUS", "EMT"]

    # Cache key based on GMT file (PATH + CONTENT) and conversion settings.
    # Content must be in the key: the simulation campaigns regenerate pathways.gmt IN PLACE at a
    # stable path, so a path-only key silently returns a stale parse. Observed 2026-07-20: after
    # the variable-length-program redesign, masked/combined GTEx fits reused a cached 3x100-gene
    # mask instead of the new 97/116/123-gene one, driving recovery to chance.
    _h = hashlib.md5(gmt_path.encode())
    try:
        with open(gmt_path, "rb") as _f:
            _h.update(_f.read())
    except OSError:
        pass                      # unreadable path -> fall back to path-only key
    gmt_hash = _h.hexdigest()[:8]
    cache_suffix = f"_ensembl_{species}" if convert_to_ensembl else "_symbols"
    cache_file = Path(cache_dir) / f"pathways_{gmt_hash}{cache_suffix}.pkl"

    pathways = None
    all_genes = None

    # Try loading from cache
    if use_cache and cache_file.exists():
        print(f"Loading cached pathways from {cache_file}...")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        pathways = cached['pathways']
        all_genes = cached['all_genes']
        print(f"  Loaded {len(pathways)} pathways with {len(all_genes)} genes from cache")

    # Parse GMT and convert if not cached
    if pathways is None:
        # Parse GMT file
        pathways = {}  # pathway_name -> set of genes
        all_genes = set()

        print(f"Loading pathways from {gmt_path}...")
        with open(gmt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                pathway_name = parts[0]
                # Skip URL (parts[1]), genes start at parts[2]
                genes = set(parts[2:])
                pathways[pathway_name] = genes
                all_genes.update(genes)

        print(f"  Loaded {len(pathways)} pathways with {len(all_genes)} unique genes (symbols)")

        # Convert gene symbols to Ensembl if requested
        if convert_to_ensembl:
            converter = GeneIDConverter()
            symbol_list = list(all_genes)
            symbol_to_ensembl, ensembl_list = converter.symbols_to_ensembl(
                symbol_list, species=species
            )

            # Build reverse mapping for valid conversions
            valid_genes = set()
            symbol_to_final = {}
            for sym, ens in zip(symbol_list, ensembl_list):
                if ens is not None:
                    symbol_to_final[sym] = ens
                    valid_genes.add(ens)

            # Update pathway gene sets
            pathways_converted = {}
            for pathway_name, genes in pathways.items():
                converted = {symbol_to_final[g] for g in genes if g in symbol_to_final}
                if converted:
                    pathways_converted[pathway_name] = converted

            pathways = pathways_converted
            all_genes = valid_genes
            print(f"  After Ensembl conversion: {len(all_genes)} genes with valid mappings")

        # Save to cache
        if use_cache:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({'pathways': pathways, 'all_genes': all_genes}, f)
            print(f"  Cached converted pathways to {cache_file}")

    # Apply pathway selection whitelist (pre-computed curated list)
    if pathway_selection is not None:
        selection_path = Path(pathway_selection)
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Pathway selection file not found: {pathway_selection}"
            )
        if selection_path.suffix == '.gmt':
            # Selection is itself a GMT — parse pathway names from it
            allowed = set()
            with open(selection_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if parts:
                        allowed.add(parts[0])
        else:
            # Plain text file with one pathway name per line
            with open(selection_path, 'r') as f:
                allowed = {line.strip() for line in f if line.strip()}
        n_before = len(pathways)
        pathways = {k: v for k, v in pathways.items() if k in allowed}
        print(f"  After pathway selection ({len(allowed)} allowed): "
              f"{len(pathways)}/{n_before} pathways")

    # Apply pathway filtering by prefix
    if require_prefix is not None:
        n_before = len(pathways)
        pathways = {k: v for k, v in pathways.items() if k.startswith(require_prefix)}
        print(f"  After prefix filter '{require_prefix}': {len(pathways)}/{n_before} pathways")

    # Apply pathway filtering by excluded keywords
    if excluded_keywords:
        excluded_upper = [kw.upper() for kw in excluded_keywords]
        n_before = len(pathways)
        pathways = {
            k: v for k, v in pathways.items()
            if not any(kw in k.upper() for kw in excluded_upper)
        }
        print(f"  After keyword exclusion {excluded_keywords}: {len(pathways)}/{n_before} pathways")

    # Recompute all_genes after pathway filtering
    all_genes = set()
    for genes in pathways.values():
        all_genes.update(genes)

    # Apply gene filter if provided
    if gene_filter is not None:
        gene_filter_set = set(gene_filter)
        all_genes = all_genes & gene_filter_set

        # Store original pathway sizes before gene filtering (for adaptive thresholding)
        original_pathway_sizes = {name: len(genes) for name, genes in pathways.items()}

        # Update pathways to only include filtered genes
        pathways_filtered = {}
        for pathway_name, genes in pathways.items():
            filtered = genes & gene_filter_set
            if filtered:
                pathways_filtered[pathway_name] = filtered
        pathways = pathways_filtered
        print(f"  After gene filter: {len(all_genes)} genes, {len(pathways)} pathways")

        if disable_adaptive_filter:
            print(f"  Adaptive overlap-size filter DISABLED — keeping all "
                  f"{len(pathways)} pathways regardless of data overlap")
        else:
            # Adaptive filtering: different thresholds based on original pathway size
            # - Small pathways (<500 genes): keep if at least 2 genes in dataset
            # - Large pathways (>=500 genes): keep if at least half of genes in dataset
            SMALL_PATHWAY_THRESHOLD = 100
            MIN_GENES_SMALL = 5

            pathways_adaptive = {}
            n_dropped_small = 0
            n_dropped_large = 0
            for name, genes in pathways.items():
                orig_size = original_pathway_sizes.get(name, len(genes))
                n_overlap = len(genes)

                if orig_size < SMALL_PATHWAY_THRESHOLD:
                    # Small pathway: require at least 2 genes
                    if n_overlap >= MIN_GENES_SMALL:
                        pathways_adaptive[name] = genes
                    else:
                        n_dropped_small += 1
                else:
                    # Large pathway: require at least half of genes in dataset
                    required = orig_size
                    if n_overlap >= required:
                        pathways_adaptive[name] = genes
                    else:
                        n_dropped_large += 1

            print(f"  After adaptive filter (small<{SMALL_PATHWAY_THRESHOLD}: ≥{MIN_GENES_SMALL}; "
                  f"large: full support): {len(pathways_adaptive)} pathways "
                  f"(dropped {n_dropped_small} small, {n_dropped_large} large)")
            pathways = pathways_adaptive

    # Filter pathways by size (after gene filter overlap)
    pathways_sized = {
        name: genes for name, genes in pathways.items()
        if min_genes <= len(genes) <= max_genes
    }
    print(f"  After size filter [{min_genes}, {max_genes}]: {len(pathways_sized)} pathways")
    pathways = pathways_sized

    if len(pathways) == 0:
        raise ValueError(
            f"Zero pathways remain after filtering (min_genes={min_genes}, "
            f"max_genes={max_genes}). Likely cause: gene ID mismatch between "
            f"expression data (gene_filter) and pathway GMT file. Check whether "
            f"convert_to_ensembl should be True or False."
        )

    # Collect only genes that appear in remaining pathways
    genes_in_pathways = set()
    for genes in pathways.values():
        genes_in_pathways.update(genes)
    all_genes = genes_in_pathways

    # Create ordered lists
    gene_names = sorted(all_genes)
    pathway_names = sorted(pathways.keys())

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Build binary matrix
    n_pathways = len(pathway_names)
    n_genes = len(gene_names)
    pathway_mat = np.zeros((n_pathways, n_genes), dtype=np.int8)

    for i, pathway_name in enumerate(pathway_names):
        for gene in pathways[pathway_name]:
            if gene in gene_to_idx:
                pathway_mat[i, gene_to_idx[gene]] = 1

    # Summary stats
    genes_per_pathway = pathway_mat.sum(axis=1)
    pathways_per_gene = pathway_mat.sum(axis=0)
    print(f"\nPathway matrix: {n_pathways} pathways x {n_genes} genes")
    print(f"  Genes/pathway: min={genes_per_pathway.min()}, max={genes_per_pathway.max()}, "
          f"mean={genes_per_pathway.mean():.1f}")
    print(f"  Pathways/gene: min={pathways_per_gene.min()}, max={pathways_per_gene.max()}, "
          f"mean={pathways_per_gene.mean():.1f}")
    print(f"  Matrix density: {pathway_mat.mean()*100:.2f}%")

    return pathway_mat, pathway_names, gene_names
