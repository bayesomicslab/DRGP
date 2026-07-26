"""Background (phenotype-free) expression supplied by the user.

DRGP does not generate backgrounds. Obtain one by fitting a marginal- and dependence-preserving
simulator to your control/healthy samples -- scDesign3 for single-cell, SPsimSeq for bulk -- then
hand the result here. Two shapes are accepted.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RealizedBackground:
    """A realized count matrix (e.g. from SPsimSeq)."""
    counts: np.ndarray                  # (n_units, n_genes) non-negative integers
    gene_ids: np.ndarray                # (n_genes,)
    unit_to_group: Optional[np.ndarray] = None   # (n_units,) int; None -> identity
    cell_type: Optional[np.ndarray] = None       # (n_units,) int; single-cell only
    kind: str = "realized"

    def __post_init__(self):
        self.counts = np.asarray(self.counts)
        if self.counts.ndim != 2:
            raise ValueError("counts must be 2-D (n_units, n_genes)")
        if len(self.gene_ids) != self.counts.shape[1]:
            raise ValueError("gene_ids length must equal counts.shape[1]")
        if self.unit_to_group is None:
            self.unit_to_group = np.arange(self.counts.shape[0])

    @property
    def n_units(self):
        return self.counts.shape[0]

    @property
    def n_genes(self):
        return self.counts.shape[1]


@dataclass
class ParametricBackground:
    """A per-unit-per-gene NB parameter surface (e.g. from scDesign3)."""
    mu: np.ndarray                      # (n_units, n_genes) positive means
    dispersion: np.ndarray              # (n_genes,) NB size/dispersion
    gene_ids: np.ndarray
    unit_to_group: Optional[np.ndarray] = None
    cell_type: Optional[np.ndarray] = None
    kind: str = "parametric"

    def __post_init__(self):
        self.mu = np.asarray(self.mu)
        if self.mu.ndim != 2:
            raise ValueError("mu must be 2-D (n_units, n_genes)")
        if len(self.dispersion) != self.mu.shape[1]:
            raise ValueError("dispersion length must equal mu.shape[1]")
        if len(self.gene_ids) != self.mu.shape[1]:
            raise ValueError("gene_ids length must equal mu.shape[1]")
        if self.unit_to_group is None:
            self.unit_to_group = np.arange(self.mu.shape[0])

    @property
    def n_units(self):
        return self.mu.shape[0]

    @property
    def n_genes(self):
        return self.mu.shape[1]
