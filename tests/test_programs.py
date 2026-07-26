"""Program drawing must match the source implementations bit-for-bit."""
import numpy as np
import pytest
from drgp.simulation.programs import draw_programs
from drgp.simulation.spec import ProgramSpec


def test_polarity_signs_are_respected():
    spec = ProgramSpec(polarity=("up", "down", "mixed"), n_denovo_disease=1,
                       n_pathway_disease=1, n_denovo_nuisance=0, n_pathway_nuisance=1,
                       size_lo=20, size_hi=30, mixed_down_frac=0.5)
    t = draw_programs(spec, n_genes=500, rng=np.random.default_rng(0))
    assert len(t.members) == 3
    up = t.loadings[t.members[0], 0]
    dn = t.loadings[t.members[1], 1]
    mx = t.loadings[t.members[2], 2]
    assert (up > 0).all(), "up-only program must have all-positive loadings"
    assert (dn < 0).all(), "down-only program must have all-negative loadings"
    assert 0.2 < (mx < 0).mean() < 0.8, "mixed program should be ~half negative"


def test_supports_are_disjoint_and_variable_length():
    spec = ProgramSpec(polarity=("up",) * 5, n_denovo_disease=2, n_pathway_disease=2,
                       n_denovo_nuisance=0, n_pathway_nuisance=1, size_lo=50, size_hi=150)
    t = draw_programs(spec, n_genes=5000, rng=np.random.default_rng(1))
    sizes = [len(m) for m in t.members]
    assert min(sizes) >= 50 and max(sizes) <= 150
    assert len(set(sizes)) > 1, "sizes should vary"
    allg = np.concatenate(t.members)
    assert len(allg) == len(set(allg.tolist())), "supports must be disjoint"


def test_reproducible():
    spec = ProgramSpec()
    a = draw_programs(spec, 2000, np.random.default_rng(7))
    b = draw_programs(spec, 2000, np.random.default_rng(7))
    assert np.array_equal(a.loadings, b.loadings)
