"""Program activity: how carriage becomes per-unit program strength.

Single-cell: a cell is active for program l only if its group carries l, the cell is
a responder type for l, and the cell falls in a random rho-fraction of that group's eligible cells.
Magnitude is Gamma(alpha_b, lambda_b)/mean on top of a theta_base floor.

Bulk is the degenerate case: responder_restricted=False, rho=1, constant magnitude, theta_base=0,
which reduces activity to the binary subject carrier itself.
"""
import numpy as np


def draw_activity(carriers, unit_to_group, spec, rng, cell_type=None, responder_types=None):
    """Return (n_units, L) float32 activity.

    carriers: (n_groups, L) if single-cell (indexed by unit_to_group), or (n_units, L) for bulk.
    """
    carriers = np.asarray(carriers)
    unit_to_group = np.asarray(unit_to_group)
    n_units = unit_to_group.size
    L = carriers.shape[1]

    if not spec.responder_restricted and spec.rho >= 1.0 and spec.magnitude == "constant" \
            and spec.theta_base == 0.0:
        return carriers[unit_to_group].astype(np.float32)

    A = np.full((n_units, L), spec.theta_base, dtype=np.float32)
    bbar = spec.alpha_b / spec.lambda_b
    for l in range(L):
        eligible = carriers[unit_to_group, l] == 1
        if spec.responder_restricted:
            if cell_type is None or responder_types is None:
                raise ValueError("responder_restricted=True requires cell_type and responder_types")
            eligible &= np.isin(cell_type, responder_types[l])
        idx = np.flatnonzero(eligible)
        if idx.size == 0:
            continue
        if spec.rho >= 1.0:
            sel = idx
        else:
            parts = []
            for g in np.unique(unit_to_group[idx]):
                cells_g = idx[unit_to_group[idx] == g]
                k = int(round(spec.rho * len(cells_g)))
                if k > 0:
                    parts.append(rng.choice(cells_g, size=k, replace=False))
            if not parts:
                continue
            sel = np.concatenate(parts)
        if spec.magnitude == "gamma":
            b = rng.gamma(spec.alpha_b, 1.0 / spec.lambda_b, size=sel.size)
            A[sel, l] = spec.theta_base + (b / bbar).astype(np.float32)
        else:
            A[sel, l] = spec.theta_base + 1.0
    return A
