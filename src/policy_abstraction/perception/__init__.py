"""Object detection, segmentation, and affordance computation."""

from policy_abstraction.perception.base import BasePerceptionPipeline
from policy_abstraction.perception.cotracker_correspondence import (
    InteractionPointTracker,
    PixelCorrespondence,
    TemporalDescriptors,
    TrackedPointFeatures,
)
from policy_abstraction.perception.cotracker_tracker import (
    IncrementalObjectTracker,
    OfflineObjectTracker,
    OnlineObjectTracker,
    TrackedPoints,
    mask_to_queries,
)
from policy_abstraction.perception.dino_projector import DinoPCAProjector
from policy_abstraction.perception.gt_segmentation import GtSegmentation
from policy_abstraction.perception.sd_dino_extractor import (
    SDDinoFeatureExtractor,
    SemanticCorrespondenceFinder,
)

__all__ = [
    "BasePerceptionPipeline",
    "GtSegmentation",
    "DinoPCAProjector",
    "SDDinoFeatureExtractor",
    "SemanticCorrespondenceFinder",
    "OfflineObjectTracker",
    "OnlineObjectTracker",
    "IncrementalObjectTracker",
    "TrackedPoints",
    "mask_to_queries",
    "InteractionPointTracker",
    "TrackedPointFeatures",
    "PixelCorrespondence",
    "TemporalDescriptors",
]
