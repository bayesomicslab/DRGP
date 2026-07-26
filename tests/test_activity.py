import numpy as np
from drgp.simulation.activity import draw_activity
from drgp.simulation.spec import ActivitySpec


def test_bulk_activity_is_the_carrier_matrix():
    """rho=1, constant magnitude, theta_base=0, no responder restriction -> identity."""
    rng = np.random.default_rng(0)
    K = (rng.random((50, 4)) < 0.5).astype(np.int8)
    spec = ActivitySpec(rho=1.0, magnitude="constant", theta_base=0.0, responder_restricted=False)
    A = draw_activity(K, np.arange(50), spec, np.random.default_rng(1))
    assert np.array_equal(A, K.astype(np.float32))


def test_single_cell_activity_respects_responders_and_rho():
    rng = np.random.default_rng(0)
    n_cells, L, n_groups = 400, 3, 4
    group = np.repeat(np.arange(n_groups), n_cells // n_groups)
    cell_type = rng.integers(0, 3, size=n_cells)
    Kg = np.ones((n_groups, L), dtype=np.int8)          # every group carries every program
    responder = [np.array([0]), np.array([1]), np.array([2])]
    spec = ActivitySpec(rho=0.5, magnitude="gamma", theta_base=0.01, responder_restricted=True)
    A = draw_activity(Kg, group, spec, np.random.default_rng(2),
                      cell_type=cell_type, responder_types=responder)
    assert A.shape == (n_cells, L)
    for l in range(L):
        active = A[:, l] > spec.theta_base
        assert active.sum() > 0
        assert set(np.unique(cell_type[active]).tolist()) <= set(responder[l].tolist()), \
            "only responder cell types may be active"
    assert np.isclose(A[A <= spec.theta_base].min(), spec.theta_base)
