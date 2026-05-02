"""Pydantic v2 config schema for the article-track CLI.

The YAML configs under ``configs/`` are loaded into these models. Validation
is strict: unknown keys raise.

Kept separate from :mod:`doe_xgb.config` (which contains the legacy
hyperparameter catalog dataclasses) to avoid breaking the dissertation-baseline
import paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .objectives import (
    ObjectiveDirection,
    ObjectiveNormalization,
    ObjectiveRole,
    ObjectiveSpec,
    ObjectiveTransform,
    ObjectivesConfig,
)


_StrictModel: Dict[str, Any] = {"extra": "forbid", "frozen": True}


class ExperimentBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    name: str
    output_root: Path = Path("experiments")
    n_replicas: int = Field(default=1, ge=1)
    seed_base: int = 42
    deterministic: bool = False


class DatasetBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    path: Path
    target: str = "y"
    target_map: Optional[Dict[str, int]] = None
    expected_sha256: Optional[str] = None


class ModelBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    estimator: str = "xgboost.XGBClassifier"
    fixed_kwargs: Dict[str, Any] = Field(default_factory=dict)


class FactorBoundSpec(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    name: str
    lo: float
    hi: float
    type: Literal["float", "int"] = "float"

    @model_validator(mode="after")
    def _check_bounds(self) -> "FactorBoundSpec":
        if self.lo >= self.hi:
            raise ValueError(f"factor {self.name!r}: lo must be < hi")
        return self


class DesignBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    kind: Literal[
        "external_csv",
        "ccd_face_centered",
        "ccd_circumscribed",
        "ccd_inscribed",
        "box_behnken",
        "full_factorial",
        "fractional_factorial",
        "lhs",
        "sobol",
        "d_optimal",
        "simplex_lattice",
    ]
    external_path: Optional[Path] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    factors: List[FactorBoundSpec] = Field(default_factory=list)
    n_center: int = 0
    fractional_resolution: Optional[int] = None
    lhs_criterion: Literal["maximin", "correlation", "centermaximin"] = "maximin"
    d_optimal_model: Optional[str] = None
    simplex_q: Optional[int] = None
    simplex_m: Optional[int] = None
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _consistency(self) -> "DesignBlock":
        if self.kind == "external_csv" and self.external_path is None:
            raise ValueError("design.kind=external_csv requires external_path")
        if self.kind == "simplex_lattice" and (self.simplex_q is None or self.simplex_m is None):
            raise ValueError("design.kind=simplex_lattice requires simplex_q and simplex_m")
        return self


class CVBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    n_splits: int = 5
    shuffle: bool = True


class EvaluationBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    cv: CVBlock = CVBlock()
    raw_metrics: List[str] = Field(
        default_factory=lambda: [
            "Accuracy_Mean",
            "Precision_Mean",
            "Recall_Mean",
            "Specificity_Mean",
            "Time_MeanFold",
        ]
    )
    per_fold_metrics: bool = True


class FactorAutoCriteriaModel(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    eigen_threshold: float = 1.0
    cumvar_threshold: float = 0.85
    min_loading_abs: float = 0.4


class FactorConstructModel(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    name: str
    members: List[str]


class FactorModelBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    mode: Literal["auto", "fixed", "manual", "none"] = "auto"
    n_factors: Optional[int] = None
    constructs: Optional[List[FactorConstructModel]] = None
    rotation: Literal["varimax", "none"] = "varimax"
    standardize: bool = True
    sign_orientation: Literal["construct_avg", "first_loading_positive", "explicit"] = "construct_avg"
    auto: FactorAutoCriteriaModel = FactorAutoCriteriaModel()

    @model_validator(mode="after")
    def _consistency(self) -> "FactorModelBlock":
        if self.mode == "fixed" and self.n_factors is None:
            raise ValueError("factor_model.mode=fixed requires n_factors")
        if self.mode == "manual" and not self.constructs:
            raise ValueError("factor_model.mode=manual requires constructs")
        return self


class ObjectiveSpecModel(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    name: str
    source: str
    direction: ObjectiveDirection
    transform: ObjectiveTransform = ObjectiveTransform.RAW
    target: Optional[float] = None
    weight: Optional[float] = None
    role: ObjectiveRole = ObjectiveRole.PRIMARY_NBI
    group: Optional[str] = None
    bounds: Optional[Tuple[float, float]] = None
    normalization: ObjectiveNormalization = ObjectiveNormalization.PAYOFF

    @field_validator("target", mode="before")
    @classmethod
    def _coerce_target(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().lower() == "individual_optimum":
            return None
        return v

    @model_validator(mode="after")
    def _direction_target_consistency(self) -> "ObjectiveSpecModel":
        if self.direction is ObjectiveDirection.TARGET and self.target is None:
            raise ValueError(
                f"objective {self.name!r}: direction=target requires a target value "
                "(use the sentinel 'individual_optimum' for runtime-resolved targets)."
            )
        if self.transform is ObjectiveTransform.FMSE and self.target is None:
            raise ValueError(
                f"objective {self.name!r}: transform=fmse requires a target value."
            )
        return self

    def to_spec(self, raw_target: Any = None) -> ObjectiveSpec:
        # Pydantic strips the "individual_optimum" sentinel; re-attach if needed.
        target = self.target
        if (
            target is None
            and self.transform is ObjectiveTransform.FMSE
            and isinstance(raw_target, str)
            and raw_target.strip().lower() == "individual_optimum"
        ):
            # Sentinel value: NaN means "compute at runtime from anchors".
            target = float("nan")
        return ObjectiveSpec(
            name=self.name,
            source=self.source,
            direction=self.direction,
            transform=self.transform,
            target=target,
            weight=self.weight,
            role=self.role,
            group=self.group,
            bounds=self.bounds,
            normalization=self.normalization,
        )


class ObjectivesBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    convention: Literal["minimize", "maximize"] = "minimize"
    specs: List[ObjectiveSpecModel]

    def to_objectives_config(self) -> ObjectivesConfig:
        return ObjectivesConfig(
            specs=tuple(s.to_spec() for s in self.specs),
            convention=self.convention,
        )


class BackwardEliminationBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    alpha: float = 0.05
    enforce_hierarchy: bool = True


class SurrogateBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    family: Literal[
        "process_quadratic",
        "mixture_scheffe",
        "custom_polynomial",
        "black_box",
    ] = "process_quadratic"
    order: str = "quadratic"
    coding: Literal["coded", "uncoded"] = "coded"
    backward_elimination: Optional[BackwardEliminationBlock] = BackwardEliminationBlock()


class WeightsBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    method: Literal["simplex_lattice", "simplex_centroid", "custom"] = "simplex_lattice"
    q: int = Field(default=2, ge=2)
    m: int = Field(default=50, ge=1)
    custom: Optional[List[List[float]]] = None


class FeasibilityBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    spherical_radius_squared: Optional[Any] = None  # float or "auto_2k_quarter"


class NBIBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    weights: WeightsBlock = WeightsBlock()
    feasibility: FeasibilityBlock = FeasibilityBlock()
    solver: Literal["slsqp", "trust_constr"] = "slsqp"
    n_starts: int = 10
    anchor_source: Literal["surrogate", "observed"] = "surrogate"
    quasi_normal: Literal["minus_phi_ones", "gram_schmidt"] = "minus_phi_ones"


class SelectionBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    rule: Literal[
        "max_quality",
        "max_accuracy",
        "distance_to_utopia",
        "knee",
        "utility",
        "lexicographic",
        "topsis",
    ] = "distance_to_utopia"
    tie_breaker: Optional[str] = None
    weights: Optional[List[float]] = None


class PostOptTriggerBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    min_normalized_obj_range: float = 0.05
    min_avg_pairwise_dist: float = 0.05
    min_unique_nondominated: int = 5
    max_weight_concentration_top1: float = 0.7
    min_curvature_score: float = 0.05
    min_spread: float = 0.5


class InnerDesignBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    kind: Literal["simplex_lattice", "simplex_centroid"] = "simplex_lattice"
    q: int = 3
    m: int = 10


class InnerObjectiveBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    name: str
    direction: ObjectiveDirection


class EllipticalConstraintBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    center: Literal["centroid", "explicit"] = "centroid"
    explicit_center: Optional[List[float]] = None
    radii: List[float]


class MBPABlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    inner_design: InnerDesignBlock = InnerDesignBlock()
    inner_objectives: List[InnerObjectiveBlock] = Field(
        default_factory=lambda: [
            InnerObjectiveBlock(name="GD", direction=ObjectiveDirection.MINIMIZE),
            InnerObjectiveBlock(name="S", direction=ObjectiveDirection.MAXIMIZE),
        ]
    )
    elliptical_constraint: EllipticalConstraintBlock = EllipticalConstraintBlock(radii=[0.30, 0.30, 0.30])


class PostOptimizationBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    enabled: Literal["always", "never", "conditional"] = "conditional"
    trigger: PostOptTriggerBlock = PostOptTriggerBlock()
    mbpa: MBPABlock = MBPABlock()


class BenchmarksBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    enabled: List[str] = Field(default_factory=lambda: ["coarse_grid", "random", "bayes", "hyperopt_tpe"])
    budget: Any = "fairness"
    selection_rule: Literal["max_accuracy", "max_quality", "distance_to_utopia"] = "max_accuracy"


class ReportingBlock(BaseModel):
    model_config = ConfigDict(**_StrictModel)
    paper_tables: bool = True
    multiobjective_metrics: List[str] = Field(
        default_factory=lambda: ["igd", "spread", "spacing_entropy", "hypervolume", "mahalanobis"]
    )


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    experiment: ExperimentBlock
    dataset: DatasetBlock
    model: ModelBlock = ModelBlock()
    design: DesignBlock
    evaluation: EvaluationBlock = EvaluationBlock()
    factor_model: FactorModelBlock = FactorModelBlock()
    objectives: ObjectivesBlock
    surrogate: SurrogateBlock = SurrogateBlock()
    nbi: NBIBlock = NBIBlock()
    selection: SelectionBlock = SelectionBlock()
    post_optimization: PostOptimizationBlock = PostOptimizationBlock()
    benchmarks: BenchmarksBlock = BenchmarksBlock()
    reporting: ReportingBlock = ReportingBlock()

    @model_validator(mode="after")
    def _at_least_two_primary(self) -> "ExperimentConfig":
        primaries = [s for s in self.objectives.specs if s.role == ObjectiveRole.PRIMARY_NBI]
        if len(primaries) < 2:
            raise ValueError(
                "objectives.specs must contain at least two role=primary_nbi entries; "
                f"found {len(primaries)}."
            )
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config and validate against :class:`ExperimentConfig`."""
    import yaml

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ExperimentConfig.model_validate(data)


__all__ = [
    "ExperimentConfig",
    "load_config",
]
