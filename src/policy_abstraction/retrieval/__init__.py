"""Demo retrieval via DINOv2 visual + CLIP text embeddings."""

from policy_abstraction.retrieval.encoder import DemoEncoder
from policy_abstraction.retrieval.index import DatasetIndex
from policy_abstraction.retrieval.retrieval import EmbeddingRetriever
from policy_abstraction.retrieval.visualization import (
    load_first_frame,
    render_retrieval_figure,
)

__all__ = [
    "DemoEncoder",
    "DatasetIndex",
    "EmbeddingRetriever",
    "render_retrieval_figure",
    "load_first_frame",
]
