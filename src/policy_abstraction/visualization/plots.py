"""Utilities for visualizing action distributions and creating evaluation videos."""

import numpy as np


def get_action_chunk_dist(chunks, idx=None):
    """
    Create a ridge plot (joy plot) visualization of action chunk distributions.

    Args:
        chunks: Array of shape [chunk_idx, timestep, action_component]
        idx: Optional index of the selected chunk to highlight

    Returns:
        numpy array: RGB image of the distribution plot
    """
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    num_chunks, num_timesteps, num_actions = chunks.shape

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color map for different action components
    colors = cm.tab10(np.linspace(0, 1, num_actions))  # type: ignore[attr-defined]

    # Determine global x-range across all actions
    all_values = chunks.flatten()
    x_min, x_max = all_values.min(), all_values.max()
    x_padding = (x_max - x_min) * 0.15 if x_max > x_min else 0.5
    x_range = np.linspace(x_min - x_padding, x_max + x_padding, 300)

    # For each timestep (depth), plot distributions for each action component
    for t in range(num_timesteps):
        y_offset = t * 0.7  # Vertical offset for stacking (time depth)

        for action_idx in range(num_actions):
            # Get action values across all chunks for this timestep and action component
            action_values = chunks[:, t, action_idx]

            # Create histogram density
            if num_chunks > 1:
                hist, bin_edges = np.histogram(
                    action_values, bins=min(25, num_chunks * 2), density=True
                )
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Interpolate to create smooth curve
                density = np.interp(x_range, bin_centers, hist, left=0, right=0)

                # Apply simple gaussian smoothing using convolution
                kernel_size = 7
                kernel = np.exp(
                    -0.5 * (np.arange(kernel_size) - kernel_size // 2) ** 2 / 2.0
                )
                kernel = kernel / kernel.sum()
                density = np.convolve(density, kernel, mode="same")
            else:
                # Single chunk: create a simple gaussian peak
                density = np.exp(-0.5 * ((x_range - action_values[0]) / 0.05) ** 2)

            # Normalize for visibility
            if density.max() > 0:
                density = density / density.max() * 0.9

            # Plot filled curve (ridge plot style)
            ax.fill_between(
                x_range,
                y_offset,
                y_offset + density,
                color=colors[action_idx],
                alpha=0.65,
                label=f"Dim {action_idx}" if t == 0 else "",
            )
            ax.plot(
                x_range,
                y_offset + density,
                color=colors[action_idx],
                linewidth=1.8,
                alpha=0.9,
            )

            # add a mark to the action selected if idx is provided
            if idx is not None and idx < num_chunks:
                # Mark the selected action for the given chunk at this timestep and action component
                if t < chunks.shape[1]:
                    selected_value = chunks[idx, t, action_idx]
                    # Draw a vertical line at the selected value
                    ax.plot(
                        [selected_value, selected_value],
                        [y_offset, y_offset + 0.9],
                        color=colors[action_idx],
                        linestyle="--",
                        linewidth=2.5,
                    )

    # Styling
    ax.set_xlabel("Action Value", fontsize=25, fontweight="bold")
    ax.set_ylabel("Timestep (Depth)", fontsize=25, fontweight="bold")
    ax.set_title(
        "Action Component Distributions over Time",
        fontsize=25,
        fontweight="bold",
        pad=15,
    )

    if num_actions <= 12:
        ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=25, framealpha=0.9
        )

    ax.grid(True, alpha=0.25, axis="x", linestyle="--")

    # Set y-ticks to show timesteps
    ax.set_yticks([t * 1.2 for t in range(num_timesteps)])
    ax.set_yticklabels([f"t={t}" for t in range(num_timesteps)], fontsize=10)

    plt.tight_layout()

    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    buf = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
    img_array = np.asarray(buf)
    # Convert RGBA to RGB
    img_array = img_array[:, :, :3]
    plt.close(fig)

    return img_array


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
    return np.asarray(buf)[:, :, :3].copy()


def plot_consistency_over_iterations(
    aggregates: list[float], selected: int | None = None
) -> np.ndarray:
    """Line plot of aggregate consistency score across iterations.

    NaN / inf entries are dropped from the line and shown as red 'x' markers
    on the baseline so the user can still see which iterations failed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5))
    xs = list(range(len(aggregates)))
    finite_mask = [np.isfinite(v) for v in aggregates]
    finite_x = [x for x, m in zip(xs, finite_mask, strict=True) if m]
    finite_y = [v for v, m in zip(aggregates, finite_mask, strict=True) if m]
    if finite_x:
        ax.plot(finite_x, finite_y, "-o", color="#2266cc", label="aggregate score")
    bad_x = [x for x, m in zip(xs, finite_mask, strict=True) if not m]
    if bad_x:
        # Plot at the bottom of the chart so they remain visible.
        baseline = min(finite_y) if finite_y else 0.0
        ax.plot(
            bad_x,
            [baseline] * len(bad_x),
            "x",
            color="#cc3333",
            markersize=10,
            label="non-finite (failed)",
        )
    if selected is not None and 0 <= selected < len(aggregates):
        ax.axvline(
            selected,
            linestyle="--",
            color="#22aa44",
            label=f"selected (iter {selected})",
        )
    ax.set_xlabel("iteration")
    ax.set_ylabel("aggregate consistency score (lower is better)")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    if finite_x or bad_x or selected is not None:
        ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    img = _fig_to_rgb(fig)
    plt.close(fig)
    return img


def plot_segment_consistency(per_segment: list[dict]) -> np.ndarray:
    """Bar chart of per-segment consistency scores for one iteration.

    Each ``per_segment`` entry is a dict with at least ``subtraj_idx``,
    ``seg_score``, ``n_segmented``, ``n_attempted``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not per_segment:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "no segments scored", ha="center", va="center")
        ax.axis("off")
        img = _fig_to_rgb(fig)
        plt.close(fig)
        return img

    labels = [f"ST{s['subtraj_idx']}" for s in per_segment]
    scores = [
        s["seg_score"]
        if s["seg_score"] is not None and np.isfinite(s["seg_score"])
        else 0.0
        for s in per_segment
    ]
    failed = [
        s["seg_score"] is None or not np.isfinite(s["seg_score"]) for s in per_segment
    ]
    success_rates = [
        (s.get("n_segmented", 0) / max(s.get("n_attempted", 1), 1)) for s in per_segment
    ]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#cc3333" if f else "#3377cc" for f in failed]
    bars = ax.bar(labels, scores, color=colors)
    for bar, sr, fail in zip(bars, success_rates, failed, strict=True):
        h = bar.get_height()
        txt = "FAIL" if fail else f"sr={sr:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            txt,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("seg_score (lower is better)")
    ax.set_title("Per-segment consistency")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    img = _fig_to_rgb(fig)
    plt.close(fig)
    return img


def plot_demo_segmentation_strips(
    demo_breakdown: list[dict], n_subtrajs: int
) -> np.ndarray:
    """Horizontal bars per demo, each cell coloured by subtraj index.

    Failed segments are drawn in grey. The reference (top-1) is the first row.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    n_demos = len(demo_breakdown)
    if n_demos == 0:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "no demos", ha="center", va="center")
        ax.axis("off")
        img = _fig_to_rgb(fig)
        plt.close(fig)
        return img

    fig, ax = plt.subplots(figsize=(8, max(2, 0.4 * n_demos + 1)))
    palette = cm.tab10(np.linspace(0, 1, max(n_subtrajs, 1)))  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]

    max_t = 0
    for d in demo_breakdown:
        for s in d.get("segments", []):
            max_t = max(max_t, int(s.get("t_end", 0)))
    max_t = max(max_t, 1)

    for row, demo in enumerate(demo_breakdown):
        for s in demo.get("segments", []):
            t0 = int(s["t_start"])
            t1 = int(s["t_end"])
            if t1 <= t0:
                continue
            color = (0.6, 0.6, 0.6) if s.get("fallback") else palette[s["subtraj_idx"]]
            ax.barh(
                y=n_demos - row - 1,
                width=t1 - t0,
                left=t0,
                height=0.7,
                color=color,
                edgecolor="white",
            )
    ax.set_yticks(range(n_demos))
    ax.set_yticklabels(
        [
            (demo_breakdown[n_demos - 1 - i].get("demo_key", f"demo {i}") or "?")[:32]
            for i in range(n_demos)
        ],
        fontsize=9,
    )
    ax.set_xlim(0, max_t)
    ax.set_xlabel("timestep")
    ax.set_title("Per-demo segmentation (grey = fallback)")
    plt.tight_layout()
    img = _fig_to_rgb(fig)
    plt.close(fig)
    return img
