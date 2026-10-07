"""Genetic mediation: route part of the auxiliary variance through program activity.

In the unmediated design the auxiliary (genetic) score enters only the label, so program
recovery and genetic attribution are cleanly separated by construction. That separation is an
assumption, not a fact about biology: a variant that acts by changing a transcriptional program
would show up in BOTH expression and phenotype. This module adds that path.

MODEL
-----
With binary carrier activity c_il, program weights upsilon_l and standardized auxiliary score
s~_i, activity becomes

    a_il = c_il + kappa * psi_l * s~_i

psi_l is a SIMULATION symbol (deliberately distinct from the model latents theta/beta/upsilon/
gamma) set to sign(upsilon_l) on disease-relevant programs and 0 elsewhere, so the genetic push
moves each program in the direction that program already pushes the phenotype.

WHY kappa IS SOLVED FOR RATHER THAN SET
---------------------------------------
Adding a genetic term to activity also adds genetic variance to the program channel. If kappa
were simply sqrt(m), the TOTAL genetic influence would grow with m and any "AUC changes with m"
reading would be confounded with signal strength. Instead kappa is calibrated so the mediated
share is exactly m*w_s. Writing

    u = c @ upsilon          (binary-carrier program score, variance sigma_u^2)
    g = psi . upsilon = sum over disease programs of |upsilon_l|   (> 0)
    z = u + kappa*g*s~       (perturbed score; c independent of s~, so no cross term)

the genetic fraction of the program channel is w_z * (kappa g)^2 / (sigma_u^2 + (kappa g)^2).
Setting that equal to m*w_s and solving:

    kappa = sqrt( m * w_s * sigma_u^2 / (w_z - m*w_s) ) / g

The label then carries only the remaining DIRECT share, (1 - m)*w_s (see `direct_w_s`). Mediated
plus direct is w_s for every m, so m is a pure ROUTING knob: the same total genetic effect, moved
between the expression channel and the label channel. m=0 gives kappa=0, i.e. the original design
exactly.

Feasibility requires w_z > m*w_s -- the program channel cannot carry a genetic share larger than
the whole program channel.

SCOPE
-----
Bulk only (`aggregate='identity'`, one unit per subject), which is what the published mediated
benchmark used. The single-cell aggregation standardizes activity across cells before averaging
within a subject, so the variance identity above does not carry over unchanged; rather than apply
a calibration that silently fails to preserve the total genetic share, `mediate_activity` refuses.
"""
import numpy as np


def direct_w_s(w_s, fraction):
    """Residual DIRECT genetic share for the label once `fraction` is routed through programs."""
    return float((1.0 - float(fraction)) * float(w_s))


def mediate_activity(activity, upsilon, is_disease, aux_score, fraction, w_z, w_s,
                     aggregate="identity", tol=1e-6):
    """Return (mediated_activity, info).

    activity   : (n_units, L) binary carrier activity, one unit per subject
    upsilon    : (L,) program weights on the liability
    is_disease : (L,) bool, which programs are disease-relevant
    aux_score  : (n_units,) auxiliary score; standardized internally
    fraction   : m, share of w_s routed through the programs. 0 returns activity unchanged.
    w_z, w_s   : liability variance shares the calibration is solved against

    info carries kappa, psi, the achieved mediated share and the residual direct share, so a
    caller can record what was actually applied instead of restating the intent.
    """
    activity = np.asarray(activity, dtype=np.float64)
    upsilon = np.asarray(upsilon, dtype=np.float64)
    m = float(fraction)
    L = activity.shape[1]
    if m <= 0.0:
        return activity.astype(np.float32), dict(
            fraction=0.0, kappa=0.0, psi=np.zeros(L), g=0.0, sigma_u=float("nan"),
            mediated_share=0.0, direct_share=float(w_s), aux_std=None)

    if aggregate != "identity":
        raise NotImplementedError(
            "mediation is calibrated for the bulk/identity aggregation (one unit per subject). "
            "Under 'cell_mean' the activity is standardized across cells before averaging, so "
            "the kappa above would not preserve the total genetic share. Refusing rather than "
            "applying a calibration whose variance identity does not hold.")
    if w_z <= m * w_s:
        raise ValueError(
            f"mediation infeasible: need w_z > m*w_s, got w_z={w_z}, m*w_s={m * w_s:.4f}. "
            f"The program channel cannot carry a genetic share larger than itself.")
    if aux_score is None:
        raise ValueError("mediation requires an auxiliary score to route")

    s = np.asarray(aux_score, dtype=np.float64).ravel()
    if s.size != activity.shape[0]:
        raise ValueError(f"aux_score has {s.size} entries but activity has "
                         f"{activity.shape[0]} units")
    sd = s.std()
    s_std = (s - s.mean()) / (sd if sd > 1e-12 else 1.0)

    psi = np.zeros(L)
    psi[np.asarray(is_disease, dtype=bool)] = np.sign(upsilon[np.asarray(is_disease, dtype=bool)])
    g = float(psi @ upsilon)                       # sum of |upsilon_l| over disease programs
    sigma_u = float((activity @ upsilon).std())
    if g <= 1e-12 or sigma_u <= 1e-12:
        raise ValueError(f"cannot calibrate mediation: g={g:.3g}, sigma_u={sigma_u:.3g}")

    kappa = float(np.sqrt(m * w_s * sigma_u ** 2 / (w_z - m * w_s)) / g)
    mediated = activity + kappa * psi[None, :] * s_std[:, None]

    # verify the achieved share rather than trusting the algebra
    med_share = w_z * (kappa * g) ** 2 / (sigma_u ** 2 + (kappa * g) ** 2)
    if abs(med_share - m * w_s) > tol:
        raise AssertionError(f"mediated share {med_share} != target {m * w_s}")

    return mediated.astype(np.float32), dict(
        fraction=m, kappa=kappa, psi=psi, g=g, sigma_u=sigma_u,
        mediated_share=float(med_share), direct_share=direct_w_s(w_s, m),
        empirical_share=float(w_z * (kappa * g) ** 2 / float((mediated @ upsilon).var())),
        aux_std=s_std)
