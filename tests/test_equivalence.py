"""EQUIVALENCE GATE: merged modules must reproduce the source implementations bit-for-bit.

Imports the original package read-only from /labs/Aguiar/SSPA_BRAY/BRay. Skips if unavailable.
"""
import numpy as np
import pytest

SRC = "/labs/Aguiar/SSPA_BRAY/BRay"


@pytest.fixture(autouse=True)
def _src_on_path():
    import sys
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    yield


def _source_available():
    try:
        import VariationalInference.Simulations.config  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _source_available(), reason="source tree not importable")
def test_carriers_match_single_cell_source():
    """drgp carriers (matrix order) == Simulations/dataset.py propensity block."""
    from VariationalInference.Simulations import config as scfg
    from drgp.simulation.carriers import draw_propensity_carriers
    from drgp.simulation.spec import PropensitySpec

    spec = PropensitySpec(alpha=scfg.PROPENSITY_ALPHA, beta=scfg.PROPENSITY_BETA,
                          sigma_u=scfg.PROPENSITY_SIGMA_U, q0=scfg.CARRIER_Q0,
                          q1=scfg.CARRIER_Q1, lam=scfg.CARRIER_LAMBDA,
                          nuisance_rate=scfg.NUISANCE_RATE)
    L = scfg.L_COLS
    is_disease = [False] + [True] * (L - 1)      # decoy at index 0
    G = scfg.N_PATIENTS

    got, pg = draw_propensity_carriers(G, is_disease, spec, np.random.default_rng(123),
                                       draw_order="matrix")

    r = np.random.default_rng(123)
    p_g = np.clip(r.beta(spec.alpha, spec.beta, size=G), 1e-6, 1 - 1e-6)
    logit_p = np.log(p_g / (1 - p_g))
    u = r.normal(0.0, spec.sigma_u, size=(G, L))
    pt = 1.0 / (1.0 + np.exp(-(logit_p[:, None] + u)))
    isd = np.array(is_disease)
    prob = np.clip(np.where(isd[None, :], spec.q0 + (spec.q1 - spec.q0) * spec.lam * pt,
                            spec.nuisance_rate), 0.0, 1.0)
    want = (r.random((G, L)) < prob).astype(np.int8)

    assert np.array_equal(got, want), "single-cell carrier draw diverged from source"
    assert np.allclose(pg, p_g.astype(np.float32))


@pytest.mark.skipif(not _source_available(), reason="source tree not importable")
def test_labels_match_bulk_source():
    """drgp liability_label(identity) == bulk_gtex_sim/make_labels.liability_label_bulk."""
    from VariationalInference.bulk_gtex_sim.make_labels import liability_label_bulk
    from drgp.simulation.labels import liability_label
    from drgp.simulation.spec import LabelSpec

    n, K = 500, 6
    theta = (np.random.default_rng(4).random((n, K)) < 0.5).astype(float)
    ups = np.array([1.5, -1.5, 1.5, -1.5, 0.0, 0.0])
    prs = np.random.default_rng(5).standard_normal(n)

    want = liability_label_bulk(theta, ups, prs, w_z=0.3, w_s=0.1, prevalence=0.26,
                                rng=np.random.default_rng(99))
    got = liability_label(theta, ups,
                          LabelSpec(w_z=0.3, w_s=0.1, prevalence=0.26, aggregate="identity"),
                          np.random.default_rng(99), aux_score=prs)

    assert np.array_equal(np.asarray(got["y"]).astype(int), np.asarray(want["y"]).astype(int))
    assert np.isclose(got["tau"], want["tau"])
    assert np.allclose(got["ell"], want["ell"])


def test_end_to_end_single_cell_runs():
    """Full pipeline on a parametric background, no R and no protected data."""
    from drgp.simulation import simulate, SimSpec
    from drgp.simulation.background import ParametricBackground

    n_cells, p, n_groups = 400, 300, 8
    rng = np.random.default_rng(0)
    bg = ParametricBackground(
        mu=rng.gamma(2.0, 2.0, size=(n_cells, p)).astype(np.float32),
        dispersion=np.full(p, 2.0, dtype=np.float32),
        gene_ids=np.array([f"G{i}" for i in range(p)]),
        unit_to_group=np.repeat(np.arange(n_groups), n_cells // n_groups),
        cell_type=rng.integers(0, 4, size=n_cells))
    spec = SimSpec.for_modality("single_cell")
    spec.programs.size_lo, spec.programs.size_hi = 20, 30
    res = simulate(spec, bg)
    assert res.adata.shape == (n_cells, p)
    assert res.labels["y"].shape == (n_groups,)
    assert set(np.unique(res.adata.obs["y"]).tolist()) <= {0, 1}
    assert res.carriers.shape[0] == n_groups


def test_orchestrator_passes_theta_base_to_inject():
    """ORCHESTRATOR-LEVEL theta_base contract guard.

    Single-cell activity is centered by theta_base inside inject()
    (A_dev = activity - theta_base). draw_activity floors every non-active cell at theta_base,
    so after centering a NON-ACTIVE cell of program l contributes A_dev=0 and its carried genes
    stay at the UNPERTURBED background mean mu. If simulate() forgets to forward theta_base, every
    such cell instead gets A_dev=theta_base>0 and its carried genes are uniformly inflated by
    exp(delta * theta_base * loading) -- a bias on carrier genes even where no program is active.

    We drive the sim with a large uniform background mean and a large theta_base so the two paths
    are unmistakably separated, then verify: on a carried (up) gene, cells that are non-active for
    that program sit at the background mean (correct forwarding) rather than the inflated mean
    (dropped theta_base). This exercises the wiring in __init__.simulate, not inject() directly.
    """
    from drgp.simulation import simulate, SimSpec
    from drgp.simulation.background import ParametricBackground

    n_cells, p, n_groups = 600, 200, 6
    mu0 = 40.0
    theta_base = 0.5
    delta = 1.0
    rng = np.random.default_rng(7)
    bg = ParametricBackground(
        mu=np.full((n_cells, p), mu0, dtype=np.float32),
        dispersion=np.full(p, 50.0, dtype=np.float32),   # low overdispersion -> tight means
        gene_ids=np.array([f"G{i}" for i in range(p)]),
        unit_to_group=np.repeat(np.arange(n_groups), n_cells // n_groups),
        cell_type=rng.integers(0, 4, size=n_cells))
    spec = SimSpec.for_modality("single_cell")
    spec.programs.size_lo, spec.programs.size_hi = 15, 25
    spec.activity.theta_base = theta_base
    spec.injection.delta = delta

    res = simulate(spec, bg)
    counts = res.adata.X.toarray()
    activity = res.activity           # (n_cells, L), floor == theta_base for non-active cells
    load = res.truth.loadings         # (p, L)

    # Find an up program (some gene with strictly positive loading) that actually has active cells.
    L = load.shape[1]
    chosen = None
    for l in range(L):
        up_genes = np.flatnonzero(load[:, l] > 0)
        active = np.flatnonzero(activity[:, l] > theta_base + 1e-6)
        inactive = np.flatnonzero(np.isclose(activity[:, l], theta_base))
        if up_genes.size and active.size >= 20 and inactive.size >= 20:
            chosen = (l, up_genes, active, inactive)
            break
    assert chosen is not None, "no up program with both active and inactive cells was drawn"
    l, up_genes, active, inactive = chosen

    load_l = float(load[up_genes, l].mean())
    inactive_mean = counts[np.ix_(inactive, up_genes)].mean()
    active_mean = counts[np.ix_(active, up_genes)].mean()

    # Correct forwarding: inactive cells -> background mean mu0. Dropped theta_base would inflate
    # them to mu0 * exp(delta * theta_base * load_l). The two hypotheses are well separated here.
    inflated = mu0 * np.exp(delta * theta_base * load_l)
    assert abs(inactive_mean - mu0) < abs(inactive_mean - inflated), (
        f"inactive-cell mean {inactive_mean:.2f} on carried genes is closer to the "
        f"theta_base-inflated value {inflated:.2f} than to the background mean {mu0:.2f}: "
        "simulate() dropped theta_base when calling inject()")
    # And the up program must still raise active cells above inactive ones (signal is present).
    assert active_mean > inactive_mean, "up program failed to raise active cells above inactive"
