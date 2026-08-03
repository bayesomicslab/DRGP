"""Beta-propensity carrier assignment and single-cell composition.

    p_g      ~ Beta(alpha, beta)                                subject propensity
    p~_gl    = sigmoid(logit(p_g) + u_gl),  u ~ N(0, sigma_u^2)  per-program jitter
    c_gl     ~ Bernoulli(q0 + (q1-q0) * lam * p~_gl)   disease carrier
    c_gl     ~ Bernoulli(nuisance_rate)                nuisance carrier

The per-program jitter sigma_u is load-bearing: without it the disease carrier vectors collapse to
a shared indicator, the programs become collinear, and recovery deflates.

DRAW ORDER MATTERS. The single-cell source draws the Bernoulli as one (n, L) matrix; the bulk
source draws L separate (n,) columns. Those consume the RNG stream differently and yield different
carriers from the same seed, so `draw_order` is explicit and each modality keeps its original.
"""
import numpy as np


def draw_propensity_carriers(n_units, is_disease, spec, rng, draw_order="matrix"):
    """Draw the (n_units, L) binary carrier matrix and the latent propensity p_g.

    is_disease : length-L booleans; True = disease-relevant (propensity-driven),
                 False = nuisance (drawn at spec.nuisance_rate, independent of p_g).
    draw_order : 'matrix' (single-cell) or 'column' (bulk). See module docstring.
    """
    is_disease = np.asarray(is_disease, dtype=bool)
    L = is_disease.size

    p_g = np.clip(rng.beta(spec.alpha, spec.beta, size=n_units), 1e-6, 1.0 - 1e-6)
    logit_p = np.log(p_g / (1.0 - p_g))
    u_jit = rng.normal(0.0, spec.sigma_u, size=(n_units, L))
    p_tilde = 1.0 / (1.0 + np.exp(-(logit_p[:, None] + u_jit)))

    if draw_order == "matrix":
        prob = np.clip(np.where(is_disease[None, :],
                                spec.q0 + (spec.q1 - spec.q0) * spec.lam * p_tilde,
                                spec.nuisance_rate), 0.0, 1.0)
        K = (rng.random((n_units, L)) < prob).astype(np.int8)
    elif draw_order == "column":
        K = np.zeros((n_units, L), dtype=np.int8)
        for k in range(L):
            if is_disease[k]:
                prob_k = np.clip(spec.q0 + (spec.q1 - spec.q0) * spec.lam * p_tilde[:, k], 0.0, 1.0)
            else:
                prob_k = np.full(n_units, spec.nuisance_rate)
            K[:, k] = (rng.random(n_units) < prob_k)
    else:
        raise ValueError(f"draw_order must be 'matrix' or 'column', got {draw_order!r}")

    return K, p_g.astype(np.float32)


def draw_composition(cell_types, n_groups, dirichlet_a0, rng):
    """Assign cells to groups (patients) with per-group Dirichlet cell-type composition.

    Composition is drawn independently of disease status, making cell-type proportion a
    label-uncorrelated nuisance axis. Single-cell only; bulk uses identity grouping.
    """
    cell_types = np.asarray(cell_types)
    n_cells = cell_types.size
    n_types = int(cell_types.max()) + 1
    pi_global = np.bincount(cell_types, minlength=n_types) / n_cells
    pi_g = rng.dirichlet(dirichlet_a0 * pi_global, size=n_groups)

    per_group = n_cells // n_groups
    quota = np.floor(pi_g * per_group).astype(int)
    deficit = per_group - quota.sum(axis=1)
    order = rng.permutation(n_groups)
    for g in order:
        if deficit[g]:
            resid = pi_g[g] * per_group - quota[g]
            for _ in range(int(deficit[g])):
                t = int(np.argmax(resid))
                quota[g, t] += 1
                resid[t] -= 1

    group = np.full(n_cells, -1, dtype=np.int32)
    by_type = {t: rng.permutation(np.flatnonzero(cell_types == t)).tolist() for t in range(n_types)}
    for g in order:
        for t in range(n_types):
            k = int(quota[g, t])
            if k <= 0:
                continue
            take = min(k, len(by_type[t]))
            sel = by_type[t][:take]
            del by_type[t][:take]
            group[sel] = g
    orphans = np.flatnonzero(group < 0)
    for c in orphans:
        counts = np.bincount(group[group >= 0], minlength=n_groups)
        group[c] = int(np.argmin(counts))
    return group
