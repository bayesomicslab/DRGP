"""Unified DRGP simulation framework.

    from drgp.simulation import simulate, SimSpec
    from drgp.simulation.background import RealizedBackground
    result = simulate(SimSpec.for_modality('bulk'), background)

The modality keyword selects three things and nothing else: how activity is derived from carriers,
which injection operator runs, and how the liability score is aggregated. Programs, carriers,
labels, emission and evaluation are shared.
"""
from dataclasses import dataclass
import numpy as np

from .spec import (SimSpec, ProgramSpec, PropensitySpec, ActivitySpec,
                   InjectionSpec, LabelSpec)
from .background import RealizedBackground, ParametricBackground
from .programs import draw_programs, ProgramTruth
from .carriers import draw_propensity_carriers, draw_composition
from .activity import draw_activity
from .injection import inject
from .labels import liability_label
from .emit import to_anndata, save_truth

__all__ = ["simulate", "SimSpec", "ProgramSpec", "PropensitySpec", "ActivitySpec",
           "InjectionSpec", "LabelSpec", "RealizedBackground", "ParametricBackground",
           "SimResult", "save_truth"]


@dataclass
class SimResult:
    adata: object
    truth: ProgramTruth
    labels: dict
    activity: np.ndarray
    carriers: np.ndarray


def simulate(spec: SimSpec, background, aux_score=None) -> SimResult:
    rng = np.random.default_rng(spec.seed)
    n_genes = background.n_genes

    truth = draw_programs(spec.programs, n_genes, rng)
    is_disease = [t.endswith("disease") for t in truth.program_type]

    group = np.asarray(background.unit_to_group)
    n_groups = int(group.max()) + 1
    draw_order = "matrix" if spec.modality == "single_cell" else "column"
    carriers, _p_g = draw_propensity_carriers(n_groups, is_disease, spec.carriers, rng,
                                              draw_order=draw_order)

    responder = None
    if spec.activity.responder_restricted:
        n_types = (int(np.max(background.cell_type)) + 1) if background.cell_type is not None else 1
        responder = []
        for _ in range(len(is_disease)):
            k = int(rng.integers(spec.programs.responder_size_lo,
                                 spec.programs.responder_size_hi + 1))
            responder.append(np.sort(rng.choice(n_types, size=min(k, n_types), replace=False)))

    activity = draw_activity(carriers, group, spec.activity, rng,
                             cell_type=background.cell_type, responder_types=responder)

    # theta_base is load-bearing: inject() centers single-cell activity by it (A_dev = activity -
    # theta_base) so the theta_base floor does not become a uniform bias on every carrier gene.
    # Omitting it silently reintroduces the Task-11 centering bug. See injection._nb_resample.
    counts = inject(background, truth.loadings, activity, spec.injection, rng,
                    theta_base=spec.activity.theta_base)

    labels = liability_label(activity, truth.upsilon, spec.label, rng,
                             unit_to_group=group if spec.label.aggregate == "cell_mean" else None,
                             aux_score=aux_score)

    adata = to_anndata(counts, labels, background, truth, spec, aux=aux_score)
    return SimResult(adata=adata, truth=truth, labels=labels,
                     activity=activity, carriers=carriers)
