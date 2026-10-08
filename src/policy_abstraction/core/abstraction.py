"""
Rollout abstraction pipeline.

Given a retrieved demonstration, orchestrates:
  1. Perception   – locate and segment task-relevant objects (GT or learned)
  2. Affordances  – compute 3-D keypoints on each object (DINOv3 clusters)
  3. Correspondence – query a VLM to map demo entities → current scene entities
  4. Feature extraction – compute distance features over the demo trajectory
  5. Trend computation – derive the temporal trends (reference signals) to follow

Primary classes / functions (to be implemented):
  - AbstractionPipeline
"""


class AbstractionPipeline:
    """Orchestrates perception → affordance → correspondence → feature extraction."""

    def __init__(self, perception, affordance_computer, vlm_selector, feature):
        self.perception = perception
        self.affordance_computer = affordance_computer
        self.vlm_selector = vlm_selector
        self.feature = feature

    def abstract(self, demo_obs: list, task: str) -> dict:
        """Abstract a demonstration rollout into feature trends.

        Parameters
        ----------
        demo_obs:
            List of observation dicts from the demonstration.
        task:
            Task description string.

        Returns
        -------
        dict
            Keys: ``"pairs"``, ``"feature_trends"`` (T,), ``"timesteps"`` (T,).
        """
        raise NotImplementedError("AbstractionPipeline.abstract() not yet implemented.")
