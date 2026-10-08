"""Constraint-satisfaction status panel renderer.

Shared between the Rerun visualizer (logged as ``constraints/panel``) and the
video composer (concatenated next to the alignment plot in the saved replay
video). Renders one row per constraint, grouped by sub-trajectory, with a
green/red status bar indicating satisfaction.
"""

from __future__ import annotations

import cv2
import numpy as np


def render_constraint_panel(
    constraint_costs: dict[str, float],
    constraints: list,
    row_h: int = 28,
    section_h: int = 22,
    width: int = 420,
    padding: int = 6,
) -> np.ndarray | None:
    """Render all constraints as a single status-panel image, grouped by sub-trajectory.

    Returns a ``[H, W, 3]`` uint8 RGB image, or ``None`` if there is nothing
    to render.
    """
    if not constraint_costs:
        return None

    subtraj_by_key: dict[str, int | None] = {}
    for c in constraints:
        prefix = "sg" if c.kind == "sub_goal" else "pc"
        subtraj_by_key[f"{prefix}/{c.label}"] = c.subtraj_idx

    groups: dict[int | None, list[str]] = {}
    for key in constraint_costs:
        st_idx = subtraj_by_key.get(key)
        groups.setdefault(st_idx, []).append(key)

    sorted_st_idxs = sorted(
        groups.keys(),
        key=lambda x: (x is None, x if x is not None else 0),
    )
    for st_idx in sorted_st_idxs:
        groups[st_idx].sort(key=lambda k: (not k.startswith("sg/"), k))

    n_rows = sum(len(v) for v in groups.values())
    n_sections = len(groups)
    header_h = row_h
    bar_w = 12
    total_h = header_h + n_sections * section_h + n_rows * row_h + padding
    img = np.full((total_h, width, 3), 30, dtype=np.uint8)

    cv2.putText(
        img,
        "Constraints",
        (padding, header_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.line(img, (0, header_h), (width, header_h), (70, 70, 70), 1)

    y = header_h
    row_i = 0
    for st_idx in sorted_st_idxs:
        keys = groups[st_idx]

        section_label = f"ST{st_idx}" if st_idx is not None else "Global"
        cv2.rectangle(img, (0, y), (width, y + section_h), (50, 50, 60), -1)
        cv2.putText(
            img,
            section_label,
            (padding, y + section_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 180, 220),
            1,
            cv2.LINE_AA,
        )
        y += section_h

        for key in keys:
            cost = constraint_costs[key]
            y0, y1 = y, y + row_h - 1
            satisfied = cost <= 0.0

            strip_color = (50, 200, 80) if satisfied else (210, 50, 50)
            cv2.rectangle(img, (0, y0 + 1), (bar_w, y1 - 1), strip_color, -1)

            bg = (40, 40, 40) if row_i % 2 == 0 else (48, 48, 48)
            cv2.rectangle(img, (bar_w + 1, y0 + 1), (width - 1, y1 - 1), bg, -1)

            is_sg = key.startswith("sg/")
            badge_txt = "SG" if is_sg else "PC"
            badge_color = (140, 80, 210) if is_sg else (210, 140, 30)
            bx = bar_w + padding
            by = y0 + row_h - 8
            cv2.putText(
                img,
                badge_txt,
                (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                badge_color,
                1,
                cv2.LINE_AA,
            )

            label = key.split("/", 1)[1] if "/" in key else key
            cv2.putText(
                img,
                label,
                (bx + 24, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            status = "OK" if satisfied else "VIOL"
            cost_str = f"{status} {cost:+.3f}"
            cost_color = (80, 220, 100) if satisfied else (240, 90, 90)
            (tw, _), _ = cv2.getTextSize(cost_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(
                img,
                cost_str,
                (width - padding - tw, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                cost_color,
                1,
                cv2.LINE_AA,
            )

            y += row_h
            row_i += 1

    return img
