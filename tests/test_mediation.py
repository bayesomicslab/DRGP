"""Genetic mediation: the calibration must ROUTE genetic variance, not add it.

The whole value of the mediation knob is that total genetic influence is held fixed while the
share flowing through expression changes. A test that only checked "activity changed" would pass
for a broken calibration, so these assert the variance identity itself.
"""
import numpy as np
import pytest

from drgp.simulation.mediation import mediate_activity, direct_w_s


def _setup(n=400, L=6, seed=0):
    rng = np.random.default_rng(seed)
    is_disease = np.array([True, True, True, False, False, False])
    upsilon = np.array([1.5, -1.5, 1.5, 0.0, 0.0, 0.0])
    carriers = (rng.random((n, L)) < 0.4).astype(float)
    aux = rng.standard_normal(n)
    return carriers, upsilon, is_disease, aux


def test_zero_fraction_is_identity():
    """m=0 must reproduce the unmediated design bit-for-bit, not merely approximately."""
    c, ups, isd, aux = _setup()
    out, info = mediate_activity(c, ups, isd, aux, 0.0, w_z=0.3, w_s=0.1)
    assert info["kappa"] == 0.0
    assert info["direct_share"] == pytest.approx(0.1)
    np.testing.assert_array_equal(np.asarray(out, dtype=np.float64), c)


@pytest.mark.parametrize("m", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("w_s", [0.05, 0.1, 0.2])
def test_total_genetic_share_is_conserved(m, w_s):
    """mediated + direct == w_s for every routing fraction: m moves variance, never creates it."""
    c, ups, isd, aux = _setup()
    _, info = mediate_activity(c, ups, isd, aux, m, w_z=0.3, w_s=w_s)
    assert info["mediated_share"] == pytest.approx(m * w_s, abs=1e-9)
    assert info["direct_share"] == pytest.approx((1 - m) * w_s, abs=1e-12)
    assert info["mediated_share"] + info["direct_share"] == pytest.approx(w_s, abs=1e-9)


def test_empirical_share_matches_the_closed_form():
    """The realised program score must carry the genetic share the algebra claims."""
    c, ups, isd, aux = _setup(n=20000)
    _, info = mediate_activity(c, ups, isd, aux, 0.5, w_z=0.3, w_s=0.1)
    # finite-sample: c _|_ s~ only in expectation, so allow a small tolerance
    assert info["empirical_share"] == pytest.approx(info["mediated_share"], rel=0.05)


def test_infeasible_when_program_channel_too_small():
    """w_z <= m*w_s has no solution; it must raise rather than return a silent nan kappa."""
    c, ups, isd, aux = _setup()
    with pytest.raises(ValueError, match="mediation infeasible"):
        mediate_activity(c, ups, isd, aux, 1.0, w_z=0.05, w_s=0.1)


def test_single_cell_aggregation_refused():
    """cell_mean standardizes before averaging, so the identity does not hold -- must refuse."""
    c, ups, isd, aux = _setup()
    with pytest.raises(NotImplementedError, match="identity"):
        mediate_activity(c, ups, isd, aux, 0.5, w_z=0.3, w_s=0.1, aggregate="cell_mean")


def test_push_follows_program_sign():
    """psi = sign(upsilon) on disease programs, 0 elsewhere: genetics pushes programs the way
    those programs already push the phenotype, and leaves nuisance programs alone."""
    c, ups, isd, aux = _setup()
    _, info = mediate_activity(c, ups, isd, aux, 0.5, w_z=0.3, w_s=0.1)
    np.testing.assert_array_equal(info["psi"], np.sign(ups) * isd)
    assert info["g"] == pytest.approx(np.abs(ups[isd]).sum())


def test_direct_w_s_helper():
    assert direct_w_s(0.1, 0.0) == pytest.approx(0.1)
    assert direct_w_s(0.1, 1.0) == pytest.approx(0.0)
    assert direct_w_s(0.1, 0.4) == pytest.approx(0.06)
