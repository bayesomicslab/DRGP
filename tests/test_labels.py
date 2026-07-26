import numpy as np
from drgp.simulation.labels import liability_label
from drgp.simulation.spec import LabelSpec


def test_identity_aggregation_matches_bulk_reference():
    """Reproduces bulk_gtex_sim/make_labels.py:liability_label_bulk."""
    rng_ref = np.random.default_rng(5)
    n, K = 300, 4
    theta = (np.random.default_rng(1).random((n, K)) < 0.5).astype(float)
    ups = np.array([1.5, -1.5, 0.0, 1.5])
    prs = np.random.default_rng(2).standard_normal(n)
    w_z, w_s, prev = 0.3, 0.1, 0.26

    z = theta @ ups
    z_std = (z - z.mean()) / z.std()
    s_std = (prs - prs.mean()) / prs.std()
    eps = rng_ref.standard_normal(n)
    w_e = max(1.0 - w_z - w_s, 0.0)
    ell = np.sqrt(w_z) * z_std + np.sqrt(w_s) * s_std + np.sqrt(w_e) * eps
    tau = float(np.quantile(ell, 1.0 - prev))
    y_want = (ell > tau).astype(int)

    got = liability_label(theta, ups, LabelSpec(w_z=w_z, w_s=w_s, prevalence=prev,
                                                aggregate="identity"),
                          np.random.default_rng(5), aux_score=prs)
    assert np.array_equal(got["y"], y_want)
    assert np.isclose(got["tau"], tau)


def test_prevalence_is_hit():
    n = 4000
    theta = (np.random.default_rng(0).random((n, 3)) < 0.5).astype(float)
    ups = np.array([2.0, -1.0, 0.5])
    out = liability_label(theta, ups, LabelSpec(w_z=0.4, w_s=0.0, prevalence=0.26,
                                                aggregate="identity"),
                          np.random.default_rng(0))
    assert abs(out["y"].mean() - 0.26) < 0.02


def test_cell_mean_aggregation_produces_group_labels():
    n_cells, L, n_groups = 200, 3, 10
    group = np.repeat(np.arange(n_groups), n_cells // n_groups)
    act = np.random.default_rng(0).random((n_cells, L)).astype(np.float32)
    ups = np.array([2.0, 0.0, -2.0])
    out = liability_label(act, ups, LabelSpec(w_z=0.5, w_s=0.0, prevalence=0.5,
                                              aggregate="cell_mean"),
                          np.random.default_rng(0), unit_to_group=group)
    assert out["y"].shape == (n_groups,)
    assert out["y_unit"].shape == (n_cells,)
    assert np.array_equal(out["y_unit"], out["y"][group])
