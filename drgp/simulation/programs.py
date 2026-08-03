"""Gene-program ground truth: supports, signed loadings, polarity, and the pathway mask.

Supports are disjoint variable-length blocks carved out of a single permutation of the gene index;
|loading| ~ Unif[lo, hi], with the sign set by each program's polarity. The single-cell and bulk
configurations use the same algorithm and differ only in size range and loading scale.

Polarity is independent of the disease weight upsilon: a program can be down-regulated yet
risk-increasing. Non-negative factor models cannot represent a down-regulated program as a single
factor, which is why recovery is reported per polarity block.
"""
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np


@dataclass
class ProgramTruth:
    members: List[np.ndarray]
    loadings: np.ndarray          # (n_genes, L), signed
    polarity: List[str]
    program_type: List[str]
    upsilon: np.ndarray           # (L,)
    annotated_idx: np.ndarray
    mask: np.ndarray              # (n_genes, n_annotated) uint8


def _program_layout(spec) -> List[str]:
    return (["denovo_disease"] * spec.n_denovo_disease
            + ["pathway_disease"] * spec.n_pathway_disease
            + ["denovo_nuisance"] * spec.n_denovo_nuisance
            + ["pathway_nuisance"] * spec.n_pathway_nuisance)


def _signs(polarity: str, n: int, mixed_down_frac: float, rng) -> np.ndarray:
    if polarity == "up":
        return np.ones(n, dtype=np.float64)
    if polarity == "down":
        return -np.ones(n, dtype=np.float64)
    if polarity == "mixed":
        return np.where(rng.random(n) < mixed_down_frac, -1.0, 1.0)
    raise ValueError(f"unknown polarity {polarity!r}")


def draw_programs(spec, n_genes: int, rng) -> ProgramTruth:
    ptype = _program_layout(spec)
    L = len(ptype)
    polarity = list(spec.polarity)
    if len(polarity) != L:
        raise ValueError(f"polarity has {len(polarity)} entries but there are {L} programs")

    sizes = rng.integers(spec.size_lo, spec.size_hi + 1, size=L)
    if int(sizes.sum()) > n_genes:
        raise ValueError(f"sum(sizes)={int(sizes.sum())} exceeds n_genes={n_genes}; "
                         "supports must be disjoint")
    pool = rng.permutation(n_genes)
    offs = np.concatenate([[0], np.cumsum(sizes)])
    members = [np.sort(pool[offs[i]:offs[i + 1]]) for i in range(L)]

    loadings = np.zeros((n_genes, L), dtype=np.float32)
    for i in range(L):
        m = members[i]
        mag = spec.effect_scale * rng.uniform(spec.loading_lo, spec.loading_hi, size=m.size)
        loadings[m, i] = (_signs(polarity[i], m.size, spec.mixed_down_frac, rng) * mag).astype(np.float32)

    disease = np.array([t.endswith("disease") for t in ptype])
    upsilon = np.zeros(L, dtype=np.float64)
    mags = list(spec.upsilon_magnitudes)
    disease_idx = np.flatnonzero(disease)
    for j, li in enumerate(disease_idx):
        base = mags[j % len(mags)]
        upsilon[li] = base * (1.0 if j % 2 == 0 else -1.0)

    annotated = np.array([t.startswith("pathway") for t in ptype])
    annotated_idx = np.flatnonzero(annotated)
    mask = np.zeros((n_genes, annotated_idx.size), dtype=np.uint8)
    for j, li in enumerate(annotated_idx):
        mask[members[li], j] = 1

    return ProgramTruth(members=members, loadings=loadings, polarity=polarity,
                        program_type=ptype, upsilon=upsilon,
                        annotated_idx=annotated_idx, mask=mask)
