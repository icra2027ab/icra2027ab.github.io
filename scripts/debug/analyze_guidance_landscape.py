"""Plot the guidance line-search loss landscape from captured npz dumps.

Capture is produced by running an eval with ``PA_GUIDANCE_CAPTURE=<dir>`` set
(see ``policy_steering.guidance_fns_torch``). Each ``infer{NNNN}_step{NN}.npz``
holds, for one denoising step at one trajectory moment, a dense sweep of the
guidance loss along the line-search ray plus the points the line search probed.

This script renders, per selected (moment, denoise-step):
  1. Total loss vs alpha, with the line-search probes overlaid, the chosen
     alpha, the L(0) "do-no-harm" baseline, the bracket limits, and the true
     on-ray global optimum -> directly shows whether the search excluded a
     better alpha.
  2. Per-pair loss vs alpha along the SHARED direction (each pair's own optimum
     vs the single shared alpha) -> the multi-alpha-per-pair what-if.
  3. Per-pair loss vs alpha along each pair's OWN gradient direction.

Usage:
    python scripts/debug/analyze_guidance_landscape.py \\
        --capture-dir tmp/guidance_capture/swap \\
        --moments start,middle,end --denoise-steps all
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402

_FNAME_RE = re.compile(r"infer(\d+)_step(\d+)\.npz")


@dataclasses.dataclass
class Args:
    capture_dir: str
    """Directory of infer*_step*.npz dumps (the PA_GUIDANCE_CAPTURE dir)."""
    out_dir: str = "tmp/guidance_landscape"
    """Where to write PNG figures and summary.json."""
    moments: str = "start,middle,end"
    """Which inference indices: keywords start/middle/end and/or explicit ints."""
    denoise_steps: str = "all"
    """"all" or a comma list of denoise-step indices (e.g. "0,4,9")."""
    log_y: bool = True
    """Log-scale the loss axis."""


def _pair_labels(d: dict) -> list[str]:
    """Synthesize readable per-pair labels from the stored type arrays."""
    dt = d.get("dist_types")
    st = d.get("source_type")
    ha = d.get("has_axis")
    hp = d.get("has_plane")
    if dt is None:
        return []
    n = len(np.atleast_1d(dt))
    src = {0: "ee", 1: "finger_l", 2: "finger_r", 3: "ori", 4: "grip"}
    out = []
    for i in range(n):
        dti = int(np.atleast_1d(dt)[i])
        sti = int(np.atleast_1d(st)[i]) if st is not None else 0
        if dti == 1:
            kind = "surface"
        elif dti == 2:
            kind = "orient"
        elif dti == 3:
            kind = "gripper"
        elif ha is not None and int(np.atleast_1d(ha)[i]):
            kind = "axis"
        elif hp is not None and int(np.atleast_1d(hp)[i]):
            kind = "plane"
        else:
            kind = "centroid"
        out.append(f"{i}:{src.get(sti, sti)}->{kind}")
    return out


def _load(capture_dir: pathlib.Path) -> dict[tuple[int, int], dict]:
    caps: dict[tuple[int, int], dict] = {}
    for p in sorted(capture_dir.glob("infer*_step*.npz")):
        m = _FNAME_RE.match(p.name)
        if not m:
            continue
        d = dict(np.load(p, allow_pickle=True))
        caps[(int(m.group(1)), int(m.group(2)))] = d
    return caps


def _select_moments(moments: str, inf_ids: list[int]) -> list[int]:
    uniq = sorted(set(inf_ids))
    chosen: list[int] = []
    for tok in moments.split(","):
        tok = tok.strip()
        if tok == "start":
            chosen.append(uniq[0])
        elif tok == "end":
            chosen.append(uniq[-1])
        elif tok == "middle":
            chosen.append(uniq[len(uniq) // 2])
        elif tok.isdigit() and int(tok) in uniq:
            chosen.append(int(tok))
    return sorted(set(chosen))


def _plot_one(d: dict, log_y: bool):
    alphas = d["alphas"]
    total = d["total_loss_sweep"]
    per_pair = d["per_pair_loss_sweep"]  # (A, N)
    per_pair_own = d.get("per_pair_own_sweep")  # (N, A)
    labels = _pair_labels(d) or [str(i) for i in range(per_pair.shape[1])]
    chosen = float(d["chosen_alpha"])
    loss0 = float(d["loss_at_zero"])
    max_a = float(d["max_alpha"])
    bracket = float(d["bracket_max"])
    g_idx = int(np.argmin(total))
    g_alpha = float(alphas[g_idx])

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    setp = (lambda ax: ax.set_yscale("log")) if log_y else (lambda ax: None)

    # Panel 1: total landscape
    ax = axes[0]
    ax.plot(alphas, total, "-", color="#1f4e9c", lw=1.6, label="L(alpha) total")
    if "probe_alphas" in d and len(d["probe_alphas"]):
        ax.scatter(
            d["probe_alphas"],
            d["probe_losses"],
            c=np.arange(len(d["probe_alphas"])),
            cmap="autumn",
            s=22,
            zorder=5,
            label="line-search probes",
        )
    ax.axhline(loss0, ls="--", color="gray", lw=1, label=f"L(0)={loss0:.3g}")
    ax.axvline(chosen, color="green", lw=1.5, label=f"chosen={chosen:.3g}")
    ax.axvline(g_alpha, color="red", ls=":", lw=1.5, label=f"on-ray min={g_alpha:.3g}")
    for xv, lab in ((max_a, "max_alpha"), (bracket, "32x")):
        ax.axvline(xv, color="black", ls=":", lw=0.7, alpha=0.5)
        ax.text(xv, ax.get_ylim()[1], lab, fontsize=7, rotation=90, va="top")
    ax.scatter([g_alpha], [total[g_idx]], marker="*", s=160, color="red", zorder=6)
    setp(ax)
    ax.set_xlabel("alpha")
    ax.set_ylabel("total loss")
    ax.set_title(f"total: chosen={chosen:.3g} vs on-ray-min={g_alpha:.3g}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    # Panel 2: per-pair along SHARED direction
    ax = axes[1]
    cmap = plt.cm.tab10
    for n in range(per_pair.shape[1]):
        c = cmap(n % 10)
        ax.plot(alphas, per_pair[:, n], color=c, lw=1.2, label=labels[n])
        an = float(alphas[int(np.argmin(per_pair[:, n]))])
        ax.axvline(an, color=c, ls=":", lw=0.8, alpha=0.7)
    ax.axvline(chosen, color="green", lw=1.5, label="shared chosen")
    setp(ax)
    ax.set_xlabel("alpha")
    ax.set_ylabel("per-pair loss")
    ax.set_title("per-pair along SHARED dir (dotted = each pair's optimum)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="best")

    # Panel 3: per-pair along OWN direction
    ax = axes[2]
    if per_pair_own is not None:
        for n in range(per_pair_own.shape[0]):
            c = cmap(n % 10)
            ax.plot(alphas, per_pair_own[n], color=c, lw=1.2, label=labels[n])
            an = float(alphas[int(np.argmin(per_pair_own[n]))])
            ax.axvline(an, color=c, ls=":", lw=0.8, alpha=0.7)
        setp(ax)
        ax.set_title("per-pair along OWN gradient dir")
    else:
        ax.set_title("per-pair own dir (not captured)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("per-pair loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="best")

    fig.suptitle(
        f"infer={int(d['inference_idx'])} step={int(d['denoise_step'])} "
        f"do_no_harm={bool(d['do_no_harm_fired'])}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, g_alpha, chosen, loss0, labels, per_pair, alphas


def main(args: Args) -> None:
    cap_dir = pathlib.Path(args.capture_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = _load(cap_dir)
    if not caps:
        raise SystemExit(f"No infer*_step*.npz found in {cap_dir}")

    inf_ids = [k[0] for k in caps]
    sel_moments = _select_moments(args.moments, inf_ids)
    summary: dict[str, dict] = {}

    for (inf, step), d in sorted(caps.items()):
        if inf not in sel_moments:
            continue
        if args.denoise_steps != "all":
            wanted = {
                int(s) for s in args.denoise_steps.split(",") if s.strip().isdigit()
            }
            if step not in wanted:
                continue
        fig, g_alpha, chosen, loss0, labels, per_pair, alphas = _plot_one(d, args.log_y)
        fname = out_dir / f"infer{inf:04d}_step{step:02d}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

        total = d["total_loss_sweep"]
        per_pair_opt = [
            float(alphas[int(np.argmin(per_pair[:, n]))])
            for n in range(per_pair.shape[1])
        ]
        loss_at_chosen = float(np.interp(chosen, alphas, total))
        summary[f"infer{inf:04d}_step{step:02d}"] = {
            "chosen_alpha": chosen,
            "on_ray_optimal_alpha": g_alpha,
            "loss_at_chosen": loss_at_chosen,
            "loss_at_optimal": float(np.min(total)),
            "loss_left_on_table": loss_at_chosen - float(np.min(total)),
            "do_no_harm_fired": bool(d["do_no_harm_fired"]),
            "chosen_outside_bracket": bool(g_alpha > float(d["bracket_max"])),
            "per_pair_labels": labels,
            "per_pair_optimal_alphas_shared_dir": per_pair_opt,
            "per_pair_alpha_spread": (max(per_pair_opt) - min(per_pair_opt))
            if per_pair_opt
            else 0.0,
        }
        print(f"  wrote {fname.name}  chosen={chosen:.3g} on-ray-min={g_alpha:.3g}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'summary.json'} ({len(summary)} captures)")


if __name__ == "__main__":
    main(tyro.cli(Args))
