"""Operator-driven data collection for real-world policy experiments.

An *experiment* is a set of rollouts run against one policy setting, where each
rollout stages the scene at a known condition — where the object sits, which way
it points, which variant it is — so that a success rate can be attributed to that
condition rather than averaged away.

The pieces, roughly in dependency order:

===================  =====================================================
:mod:`grid`          A rectangular grid on a plane in world coordinates
:mod:`layout`        Scattering several named objects across a region
:mod:`factors`       The axes an experiment varies, and how to show them
:mod:`plan`          The ordered condition list, frozen at creation
:mod:`config`        The YAML that defines all of the above
:mod:`revise`        What editing that config means for results already on disk
:mod:`outcomes`      Verdicts, the failure taxonomy, the record schema
:mod:`store`         The on-disk layout and resume logic
:mod:`overlay`       Drawing the staging target onto a camera frame
:mod:`session`       Running rollouts against a policy loaded once
:mod:`report`        Aggregating records into a summary and a markdown report
:mod:`viz`           The grid maps and breakdown plots
:mod:`gui`           The PyQt6 operator window (imported on demand)
===================  =====================================================

``gui`` is deliberately *not* imported here: everything above it is headless and
must stay importable on a machine with no display, which is where reports get
regenerated and tests get run.
"""

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.factors import (
    AppearanceFactor,
    Factor,
    FactorLevel,
    GridPositionFactor,
    ObjectLayoutFactor,
    OrientationFactor,
    TargetObjectFactor,
    factor_from_dict,
    register_factor,
)
from policy_abstraction.experiments.grid import TableGrid, cell_label, parse_cell_label
from policy_abstraction.experiments.layout import (
    SceneObject,
    sample_layouts,
    sample_positions_uv,
)
from policy_abstraction.experiments.outcomes import (
    FailureMode,
    Outcome,
    RolloutRecord,
    StopReason,
)
from policy_abstraction.experiments.plan import PlannedRollout, build_plan
from policy_abstraction.experiments.revise import (
    ExperimentConflict,
    ExperimentReviser,
    Revision,
    RevisionAction,
)
from policy_abstraction.experiments.store import (
    ExperimentStatus,
    ExperimentStore,
    experiment_root,
)

__all__ = [
    "AppearanceFactor",
    "ExperimentConfig",
    "ExperimentConflict",
    "ExperimentReviser",
    "ExperimentStatus",
    "ExperimentStore",
    "Factor",
    "FactorLevel",
    "FailureMode",
    "GridPositionFactor",
    "ObjectLayoutFactor",
    "OrientationFactor",
    "Outcome",
    "PlannedRollout",
    "Revision",
    "RevisionAction",
    "RolloutRecord",
    "SceneObject",
    "StopReason",
    "TableGrid",
    "TargetObjectFactor",
    "build_plan",
    "cell_label",
    "experiment_root",
    "factor_from_dict",
    "parse_cell_label",
    "register_factor",
    "sample_layouts",
    "sample_positions_uv",
]
