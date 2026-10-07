"""Unified DRGP simulation framework.

    from drgp.simulation import simulate, SimSpec
    from drgp.simulation.background import RealizedBackground
    result = simulate(SimSpec.for_modality('bulk'), background)

The modality keyword selects three things and nothing else: how activity is derived from carriers,
which injection operator runs, and how the liability score is aggregated. Programs, carriers,
labels, emission and evaluation are shared.
"""
from dataclasses import dataclass, replace
import numpy as np

from .spec import (SimSpec, ProgramSpec, PropensitySpec, ActivitySpec,
                   InjectionSpec, LabelSpec, MediationSpec)
from .background import RealizedBackground, ParametricBackground
from .programs import draw_programs, ProgramTruth
from .carriers import draw_propensity_carriers, draw_composition
from .activity import draw_activity
from .injection import inject
from .labels import liability_label
from .mediation import mediate_activity, direct_w_s
from .emit import to_anndata, save_truth

__all__ = ["simulate", "SimSpec", "ProgramSpec", "PropensitySpec", "ActivitySpec",
           "InjectionSpec", "LabelSpec", "MediationSpec", "RealizedBackground",
           "ParametricBackground", "SimResult", "save_truth", "mediate_activity"]


@dataclass
class SimResult:
    adata: object
    truth: ProgramTruth
    labels: dict
    activity: np.ndarray
    carriers: np.ndarray
    # Pre-mediation (binary carrier) activity, and what the mediation actually applied.
    # activity is what drove BOTH expression and the label; carrier_activity is the recovery
    # target, so the two must stay distinguishable whenever mediation is on.
    carrier_activity: np.ndarray = None
    mediation: dict = None


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

    carrier_activity = draw_activity(carriers, group, spec.activity, rng,
                                     cell_type=background.cell_type,
                                     responder_types=responder)

    # Mediation perturbs activity BEFORE both injection and labelling, because the whole point
    # is that the same genetic push appears in expression and in phenotype. The pre-mediation
    # carrier activity is kept as the recovery target.
    activity, med = mediate_activity(
        carrier_activity, truth.upsilon, is_disease, aux_score,
        spec.mediation.fraction, spec.label.w_z, spec.label.w_s,
        aggregate=spec.label.aggregate)

    # theta_base is load-bearing: inject() centers single-cell activity by it (A_dev = activity -
    # theta_base) so the theta_base floor does not become a uniform bias on every carrier gene.
    # See injection._nb_resample.
    counts = inject(background, truth.loadings, activity, spec.injection, rng,
                    theta_base=spec.activity.theta_base)

    # Only the DIRECT share reaches the label: the mediated part is already inside the program
    # score via the perturbed activity, so passing the full w_s would double-count it.
    label_spec = spec.label
    if med["fraction"] > 0:
        label_spec = replace(spec.label, w_s=direct_w_s(spec.label.w_s, med["fraction"]))
    labels = liability_label(activity, truth.upsilon, label_spec, rng,
                             unit_to_group=group if spec.label.aggregate == "cell_mean" else None,
                             aux_score=aux_score)
    labels["mediation"] = med["fraction"]
    labels["w_s_nominal"] = float(spec.label.w_s)

    adata = to_anndata(counts, labels, background, truth, spec, aux=aux_score)
    return SimResult(adata=adata, truth=truth, labels=labels,
                     activity=activity, carriers=carriers,
                     carrier_activity=carrier_activity, mediation=med)
