"""Probit liability-threshold phenotype label.

    Z_g   = sum_l upsilon_l * a~_gl              (upsilon-weighted program score)
    ell_g = sqrt(w_z) Z~_g + sqrt(w_s) s~_g + sqrt(1 - w_z - w_s) eps_g
    D_g   = 1[ell_g > tau_pi]

Because every liability component is unit-variance, w_z is exactly the program share and w_s
exactly the auxiliary (genetic) share of liability variance. tau_pi is the (1 - prevalence)
empirical quantile, fixing prevalence exactly.

By default the auxiliary score enters ONLY the label, never expression, which is what isolates
genetic attribution from program recovery. That separation is a modelling choice, not a fact:
set SimSpec.mediation.fraction > 0 to route part of the auxiliary variance through program
activity instead (simulation/mediation.py). When mediation is on, the w_s reaching this function
is the DIRECT share (1-m)*w_s -- the mediated part is already inside the z term via the
perturbed activity, so passing the nominal w_s here would double-count it.

Aggregation differs by modality and is NOT unified, because the choice changes the labels:
  'identity'  (bulk)        standardize the per-unit score directly
  'cell_mean' (single-cell) standardize activity across cells, then average within group
"""
import numpy as np
from scipy.special import ndtr


def liability_label(activity, upsilon, spec, rng, unit_to_group=None, aux_score=None):
    activity = np.asarray(activity, dtype=np.float64)
    upsilon = np.asarray(upsilon, dtype=np.float64)
    if spec.w_z < 0 or spec.w_s < 0 or (spec.w_z + spec.w_s) > 1.0 + 1e-9:
        raise ValueError(f"invalid variance shares: w_z={spec.w_z}, w_s={spec.w_s}")

    if spec.aggregate == "identity":
        z = activity @ upsilon
        group_of_unit = None
        n_groups = activity.shape[0]
    elif spec.aggregate == "cell_mean":
        if unit_to_group is None:
            raise ValueError("aggregate='cell_mean' requires unit_to_group")
        group_of_unit = np.asarray(unit_to_group)
        n_groups = int(group_of_unit.max()) + 1
        sd = activity.std(axis=0, ddof=0)
        sd[sd < 1e-8] = 1.0
        a_std = (activity - activity.mean(axis=0)) / sd
        counts = np.maximum(np.bincount(group_of_unit, minlength=n_groups), 1).astype(np.float64)
        m = np.zeros((n_groups, activity.shape[1]))
        np.add.at(m, group_of_unit, a_std)
        m /= counts[:, None]
        z = m @ upsilon
    else:
        raise ValueError(f"unknown aggregate {spec.aggregate!r}")

    zsd = z.std()
    if zsd < 1e-12:
        raise ValueError("liability score has zero variance - check upsilon / carriers")
    z_std = (z - z.mean()) / zsd

    if aux_score is None:
        s_std = np.zeros(n_groups)
    else:
        s = np.asarray(aux_score, dtype=np.float64)
        ssd = s.std()
        s_std = (s - s.mean()) / (ssd if ssd > 1e-12 else 1.0)

    eps = rng.standard_normal(n_groups)
    w_e = max(1.0 - spec.w_z - spec.w_s, 0.0)
    ell = np.sqrt(spec.w_z) * z_std + np.sqrt(spec.w_s) * s_std + np.sqrt(w_e) * eps
    tau = float(np.quantile(ell, 1.0 - spec.prevalence))
    y = (ell > tau).astype(np.int8)

    struct = np.sqrt(spec.w_z) * z_std + np.sqrt(spec.w_s) * s_std
    pi_true = (ndtr((struct - tau) / np.sqrt(w_e)) if w_e > 1e-12
               else (struct > tau).astype(float))

    y_unit = y if group_of_unit is None else y[group_of_unit]
    return dict(y=y, y_unit=y_unit, ell=ell, tau=tau, z_std=z_std,
                pi_true=pi_true, aux_std=s_std)
