"""Mask IoU helpers for overlap filtering and drift detection."""

from __future__ import annotations

import numpy as np


def pairwise_mask_iou(masks: np.ndarray) -> np.ndarray:
    """Compute the NxN pairwise IoU matrix for binary masks shaped (N, H, W)."""
    if masks.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    flat = masks.reshape(masks.shape[0], -1).astype(np.float32)
    inter = flat @ flat.T
    area = flat.sum(axis=1)
    union = area[:, None] + area[None, :] - inter
    union = np.maximum(union, 1e-8)
    return (inter / union).astype(np.float32)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(inter / union) if union else 0.0


def nms(
    masks: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return the indices to keep after greedy NMS on mask IoU.

    Masks are processed in descending score order; a mask is dropped if its
    IoU with any already-kept mask exceeds ``iou_threshold``.
    """
    if masks.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    kept: list[int] = []
    # Flatten once for fast pairwise IoU with already-kept masks.
    flat = masks.reshape(masks.shape[0], -1).astype(np.float32)
    areas = flat.sum(axis=1)
    for idx in order:
        keep = True
        for k in kept:
            inter = float(flat[idx] @ flat[k])
            union = float(areas[idx] + areas[k] - inter)
            if union <= 0:
                continue
            if inter / union > iou_threshold:
                keep = False
                break
        if keep:
            kept.append(int(idx))
    return np.asarray(kept, dtype=np.int64)


def subtract_overlaps(
    masks: np.ndarray,
    scores: np.ndarray,
    min_area: int = 1,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Greedy intersection-subtraction in descending score order.

    Each lower-scoring mask has the union of all already-kept higher-scoring
    masks subtracted from it; if the remainder falls below ``min_area``, the
    mask is dropped.

    Returns ``(kept_order, new_masks)`` where ``kept_order`` is a list of
    original indices in the order they were kept (descending score), and
    ``new_masks[i]`` is the updated mask for original index ``i``.
    """
    N = masks.shape[0]
    if N == 0:
        return [], {}
    H, W = masks.shape[1], masks.shape[2]
    order = np.argsort(-scores)
    union_flat = np.zeros(H * W, dtype=bool)
    flat = masks.reshape(N, -1).astype(bool)
    kept_order: list[int] = []
    new_masks: dict[int, np.ndarray] = {}
    for idx in order:
        trimmed = flat[idx] & ~union_flat
        if int(trimmed.sum()) < min_area:
            continue
        i = int(idx)
        kept_order.append(i)
        new_masks[i] = trimmed.reshape(H, W).copy()
        union_flat |= trimmed
    return kept_order, new_masks
