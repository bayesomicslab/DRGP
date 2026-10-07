"""Specification objects for the unified DRGP simulation framework.

Every knob is explicit here rather than living in a module-global config, so a simulation is a
pure function of (spec, background, seed). Defaults per modality are exactly the values used for
the published benchmarks -- see SimSpec.for_modality().
"""
from dataclasses import dataclass, field, asdict
from typing import Literal, Sequence
import json

Modality = Literal["single_cell", "bulk"]


@dataclass
class ProgramSpec:
    """Gene programs: how many, how big, how strong, and their up/down polarity."""
    n_denovo_disease: int = 2
    n_pathway_disease: int = 2
    n_denovo_nuisance: int = 0
    n_pathway_nuisance: int = 1
    size_lo: int = 50
    size_hi: int = 100
    loading_lo: float = 0.5
    loading_hi: float = 1.5
    effect_scale: float = 1.0          # multiplies |loading|; bulk uses log2FC units
    polarity: Sequence[str] = ("up", "up", "mixed", "down", "mixed")
    mixed_down_frac: float = 0.5
    upsilon_magnitudes: Sequence[float] = (2.0, 0.7, 2.0, 0.7)
    responder_size_lo: int = 2         # single-cell only
    responder_size_hi: int = 3         # single-cell only


@dataclass
class PropensitySpec:
    """Beta-propensity carrier chain."""
    alpha: float = 2.0
    beta: float = 2.0
    sigma_u: float = 0.5
    q0: float = 0.2
    q1: float = 0.8
    lam: float = 1.0
    nuisance_rate: float = 0.5


@dataclass
class ActivitySpec:
    """How carriage becomes per-unit program activity."""
    rho: float = 0.3                   # fraction of eligible responder cells; bulk uses 1.0
    magnitude: Literal["gamma", "constant"] = "gamma"
    alpha_b: float = 2.0
    lambda_b: float = 2.0
    theta_base: float = 0.01
    responder_restricted: bool = True   # bulk: False


@dataclass
class InjectionSpec:
    """How activity x loadings becomes perturbed counts."""
    operator: Literal["nb_resample", "thin_diff"] = "nb_resample"
    delta: float = 1.0                 # global effect scalar on centered activity
    equalize_library: bool = False     # bulk thin_lib pass
    lib_frac: float = 0.90


@dataclass
class LabelSpec:
    """Probit liability-threshold label."""
    w_z: float = 0.3                   # program (transcriptomic) variance share
    w_s: float = 0.0                   # auxiliary (genetic) variance share
    prevalence: float = 0.5
    aggregate: Literal["cell_mean", "identity"] = "cell_mean"


@dataclass
class MediationSpec:
    """How much of the auxiliary (genetic) variance acts THROUGH the programs.

    fraction = 0 is the unmediated design: the auxiliary score touches only the label.
    fraction = m routes m*w_s of the genetic variance into program activity and leaves
    (1-m)*w_s acting directly on the liability, holding the TOTAL genetic share at w_s so m
    is a routing knob rather than a signal-strength knob. See simulation/mediation.py.
    Requires w_z > fraction*w_s, and the bulk ('identity') aggregation.
    """
    fraction: float = 0.0


@dataclass
class SimSpec:
    modality: Modality = "single_cell"
    programs: ProgramSpec = field(default_factory=ProgramSpec)
    carriers: PropensitySpec = field(default_factory=PropensitySpec)
    activity: ActivitySpec = field(default_factory=ActivitySpec)
    injection: InjectionSpec = field(default_factory=InjectionSpec)
    label: LabelSpec = field(default_factory=LabelSpec)
    mediation: MediationSpec = field(default_factory=MediationSpec)
    seed: int = 0

    @classmethod
    def for_modality(cls, modality: Modality, **overrides) -> "SimSpec":
        """Defaults matching the published benchmarks for each arm."""
        if modality == "single_cell":
            spec = cls(modality="single_cell")
        elif modality == "bulk":
            spec = cls(
                modality="bulk",
                programs=ProgramSpec(
                    n_denovo_disease=4, n_pathway_disease=2,
                    n_denovo_nuisance=1, n_pathway_nuisance=1,
                    size_lo=50, size_hi=150, effect_scale=3.0,
                    polarity=("up", "down", "mixed", "mixed", "up", "mixed", "up", "up"),
                    upsilon_magnitudes=(1.5,)),
                activity=ActivitySpec(rho=1.0, magnitude="constant", theta_base=0.0,
                                      responder_restricted=False),
                injection=InjectionSpec(operator="thin_diff", equalize_library=True),
                label=LabelSpec(w_z=0.3, w_s=0.1, prevalence=0.26, aggregate="identity"))
        else:
            raise ValueError(f"unknown modality {modality!r}")
        for k, v in overrides.items():
            setattr(spec, k, v)
        return spec

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, default=list)
