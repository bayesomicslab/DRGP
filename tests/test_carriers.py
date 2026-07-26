"""Carrier drawing must reproduce BOTH source implementations bit-for-bit."""
import numpy as np
from drgp.simulation.carriers import draw_propensity_carriers
from drgp.simulation.spec import PropensitySpec


def _reference_matrix_order(n_units, is_disease, s, rng):
    """Verbatim transcription of Simulations/dataset.py:patient_composition carrier block."""
    L = len(is_disease)
    p_g = np.clip(rng.beta(s.alpha, s.beta, size=n_units), 1e-6, 1.0 - 1e-6)
    logit_p = np.log(p_g / (1.0 - p_g))
    u_jit = rng.normal(0.0, s.sigma_u, size=(n_units, L))
    p_tilde = 1.0 / (1.0 + np.exp(-(logit_p[:, None] + u_jit)))
    prob = np.clip(np.where(np.asarray(is_disease)[None, :],
                            s.q0 + (s.q1 - s.q0) * s.lam * p_tilde, s.nuisance_rate), 0.0, 1.0)
    K = (rng.random((n_units, L)) < prob).astype(np.int8)
    return K, p_g.astype(np.float32)


def _reference_column_order(n_units, is_disease, s, rng):
    """Verbatim transcription of bulk_gtex_sim/make_truth.py carrier block."""
    L = len(is_disease)
    p_g = np.clip(rng.beta(s.alpha, s.beta, size=n_units), 1e-6, 1.0 - 1e-6)
    logit_p = np.log(p_g / (1.0 - p_g))
    u_jit = rng.normal(0.0, s.sigma_u, size=(n_units, L))
    p_tilde = 1.0 / (1.0 + np.exp(-(logit_p[:, None] + u_jit)))
    K = np.zeros((n_units, L), dtype=np.int8)
    for k in range(L):
        if is_disease[k]:
            prob = np.clip(s.q0 + (s.q1 - s.q0) * s.lam * p_tilde[:, k], 0.0, 1.0)
        else:
            prob = np.full(n_units, s.nuisance_rate)
        K[:, k] = (rng.random(n_units) < prob)
    return K, p_g.astype(np.float32)


def test_matches_matrix_order_reference():
    s = PropensitySpec()
    is_disease = [False, True, True, True, True]
    got, pg = draw_propensity_carriers(80, is_disease, s, np.random.default_rng(42), draw_order="matrix")
    want, pgw = _reference_matrix_order(80, is_disease, s, np.random.default_rng(42))
    assert np.array_equal(got, want)
    assert np.array_equal(pg, pgw)


def test_matches_column_order_reference():
    s = PropensitySpec()
    is_disease = [True] * 6 + [False, False]
    got, pg = draw_propensity_carriers(2000, is_disease, s, np.random.default_rng(7), draw_order="column")
    want, pgw = _reference_column_order(2000, is_disease, s, np.random.default_rng(7))
    assert np.array_equal(got, want)
    assert np.array_equal(pg, pgw)


def test_orders_actually_differ():
    """Guards the reason draw_order exists."""
    s = PropensitySpec()
    is_disease = [True] * 4
    a, _ = draw_propensity_carriers(500, is_disease, s, np.random.default_rng(3), draw_order="matrix")
    b, _ = draw_propensity_carriers(500, is_disease, s, np.random.default_rng(3), draw_order="column")
    assert not np.array_equal(a, b)


def test_marginal_carrier_rate_near_half():
    s = PropensitySpec()
    is_disease = [False] + [True] * 4
    K, _ = draw_propensity_carriers(20000, is_disease, s, np.random.default_rng(0), draw_order="matrix")
    assert np.all(np.abs(K.mean(0) - 0.5) < 0.02)
