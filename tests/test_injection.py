"""nb_resample must center activity by theta_base before applying the loading perturbation.

Regression test for a bug where _nb_resample used raw activity (which already carries the
theta_base floor baked in by draw_activity) instead of the centered deviation. Left uncentered,
every carrier gene -- including genes with zero loading -- picks up a uniform bias from
theta_base itself, corrupting the injected signal. The correct form is
`A_dev = theta_star - THETA_BASE`.
"""
import numpy as np
from drgp.simulation.background import ParametricBackground
from drgp.simulation.injection import inject
from drgp.simulation.spec import InjectionSpec


def test_nb_resample_centers_activity_by_theta_base():
    rng = np.random.default_rng(0)
    n, p, L = 40, 30, 2
    theta_base = 0.5
    bg = ParametricBackground(mu=np.full((n, p), 5.0), dispersion=np.full(p, 2.0),
                              gene_ids=np.array([f"G{i}" for i in range(p)]))
    load = np.zeros((p, L), np.float32)
    load[:5, 0] = 1.0        # up program
    load[5:10, 1] = -1.0     # down program
    # non-carried gene columns 10:15 are left at loading 0 -- control block.

    # Activity floor is theta_base everywhere (as draw_activity would produce); carriers of
    # program l get theta_base + 1.0 on cells 0:20, non-carriers stay at the floor.
    act = np.full((n, L), theta_base, dtype=np.float32)
    act[:20, 0] = theta_base + 1.0
    act[:20, 1] = theta_base + 1.0

    X = inject(bg, load, act, InjectionSpec(operator="nb_resample", delta=1.0), rng,
               theta_base=theta_base)
    assert X.shape == (n, p)

    up_car, up_non = X[:20, :5].mean(), X[20:, :5].mean()
    dn_car, dn_non = X[:20, 5:10].mean(), X[20:, 5:10].mean()
    ctrl_car, ctrl_non = X[:20, 10:15].mean(), X[20:, 10:15].mean()

    assert up_car > up_non, f"up genes should be higher in carriers: {up_car} vs {up_non}"
    assert dn_car < dn_non, f"down genes should be lower in carriers: {dn_car} vs {dn_non}"
    # Control: a gene with zero loading must show no carrier-vs-non-carrier bias. This is exactly
    # what broke when A_dev was raw (uncentered) activity -- every carrier row would pick up a
    # uniform theta_base-driven shift even on genes untouched by any program.
    assert np.isclose(ctrl_car, ctrl_non, rtol=0.25), (
        f"non-carried gene should show no uniform bias from theta_base: "
        f"carrier {ctrl_car} vs non-carrier {ctrl_non}")


def test_nb_resample_zero_theta_base_matches_default():
    """theta_base=0.0 (the default) should reduce to using raw activity unchanged."""
    rng = np.random.default_rng(1)
    n, p, L = 20, 10, 1
    bg = ParametricBackground(mu=np.full((n, p), 5.0), dispersion=np.full(p, 2.0),
                              gene_ids=np.array([f"G{i}" for i in range(p)]))
    load = np.zeros((p, L), np.float32)
    load[:3, 0] = 1.0
    act = np.zeros((n, L), np.float32)
    act[:10, 0] = 1.0

    spec = InjectionSpec(operator="nb_resample", delta=1.0)
    X_default = inject(bg, load, act, spec, np.random.default_rng(5))
    X_explicit = inject(bg, load, act, spec, np.random.default_rng(5), theta_base=0.0)
    assert np.array_equal(X_default, X_explicit)
