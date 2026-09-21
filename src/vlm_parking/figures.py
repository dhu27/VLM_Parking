"""Phase 6 figures and table formatting, shared by scripts/analyze.py and notebooks/03_results.ipynb so the
two can't drift apart.

Colors are the dataviz method's reference palette, validated: the first slots pass every colorblind and
normal-vision floor against each other. Blue is condition A (the sign image) in every figure here, so a
reader who learns "blue is the image" is never misled; line style repeats condition for print and
colorblind readers. Text is always in ink, never in a series color.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from vlm_parking.analysis import CONDITIONS, MODELS

NAMES = {"qwen3-vl-8b": "Qwen3-VL-8B", "internvl3_5-8b": "InternVL3.5-8B", "minicpm-v-4_5": "MiniCPM-V-4.5",
         "qwen3-vl-8b-thinking": "Qwen3-VL-8B-Thinking"}
COND_NAMES = {"A": "A · sign image", "B": "B · text transcription"}
STRATA = ["1", "2", "3+"]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"A": "#2a78d6", "B": "#eb6834"}
DASH = {"A": "-", "B": (0, (4, 2))}


def ci(est: float, lo: float, hi: float) -> str:
    """A share with its interval; a blanked (suppressed) stratum shows as a dash."""
    return "—" if pd.isna(est) else f"{100 * est:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def pts(est: float, lo: float, hi: float) -> str:
    """A signed percentage-point difference with its interval, using a true minus sign."""
    s = f"{100 * est:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return s.replace("-", "−")


def stratum_table(t: pd.DataFrame, by: str) -> pd.DataFrame:
    """An accuracy_table, wide: one row per stratum, A and B side by side per model, with n signs."""
    t = t.assign(cell=[ci(e, lo, hi) for e, lo, hi in zip(t.est, t.lo, t.hi)])
    w = t.pivot_table(index=by, columns=["model", "condition"], values="cell", aggfunc="first", observed=True)
    n = t.drop_duplicates(by).set_index(by).n_signs
    out = pd.DataFrame({by: w.index.astype(str), "signs": n.reindex(w.index).values})
    for m in MODELS:
        for c in CONDITIONS:
            out[f"{NAMES[m]} {c}"] = w[(m, c)].values
    return out


def _style() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 9, "svg.fonttype": "none", "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
                         "ytick.labelcolor": INK2})


def _panel(ax, model: str, n_signs: pd.Series, first: bool) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_title(NAMES[model], loc="left", color=INK, fontsize=10, fontweight="bold", pad=18)
    ax.set_xticks(range(3), [f"{s}\n{n_signs[s]} signs" for s in STRATA])
    ax.set_xlim(-0.45, 2.45)
    ax.tick_params(length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(first)


def _line(ax, x, t: pd.DataFrame, cond: str, z: int) -> None:
    ax.errorbar(x, t.est, yerr=[t.est - t.lo, t.hi - t.est], fmt="none", ecolor=COLOR[cond], elinewidth=1.1,
                capsize=0, zorder=z)
    ax.plot(x, t.est, color=COLOR[cond], linestyle=DASH[cond], linewidth=1.5, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, solid_capstyle="round", dash_capstyle="round",
            zorder=z + 1)


def _frame(fig: Figure, title: str, note: str) -> None:
    fig.text(0.01, 0.985, title, color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.01, 0.01, note, color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))


def accuracy_by_panels(by_panels: pd.DataFrame, gaps: pd.DataFrame, majority: float) -> Figure:
    """The main figure (plan §8): accuracy against panel count, condition A solid and B dashed.

    Small multiples, one panel per model: six lines with intervals on one axis converge and hide each other,
    and the vertical gap between A and B inside each panel is the result.
    """
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9), sharey=True, facecolor=SURFACE)
    n_signs = by_panels.drop_duplicates("panels").set_index("panels").n_signs
    for ax, model in zip(axes, MODELS):
        _panel(ax, model, n_signs, ax is axes[0])
        ax.axhline(majority, color=MUTED, linewidth=0.8, zorder=1)
        for k, cond in enumerate(["B", "A"]):  # B underneath: it's the reference the gap is read against
            t = by_panels[(by_panels.model == model) & (by_panels.condition == cond)].set_index("panels").reindex(STRATA)
            _line(ax, np.arange(3) + (0.07 if cond == "B" else -0.07), t, cond, 2 + 2 * k)  # nudged apart
        g = gaps.set_index("model").loc[model]
        ax.text(0, 1.02, f"A − B  {pts(g.gap, g.lo, g.hi)}", transform=ax.transAxes, color=INK2, fontsize=8.5)

    axes[0].set_ylim(0.5, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("Verdict accuracy", color=INK2)
    axes[0].text(2.45, majority - 0.012, f"majority class {100 * majority:.1f}%", color=MUTED, fontsize=7.5,
                 ha="right", va="top")
    axes[1].set_xlabel("Panels on the sign", color=INK2)
    handles = [Line2D([], [], color=COLOR[c], linestyle=DASH[c], linewidth=1.5, marker="o", markersize=6,
                      markeredgecolor=SURFACE, markeredgewidth=1.5) for c in ("A", "B")]
    fig.legend(handles, [COND_NAMES["A"], COND_NAMES["B"]], loc="upper right", ncol=2, frameon=False,
               labelcolor=INK, fontsize=8.5, bbox_to_anchor=(0.99, 1.0))
    _frame(fig, "Accuracy by number of panels on the sign",
           "Error bars: 95% interval, resampling signs. Gray line: always answering “illegal”.")
    return fig


def perception_loss_by_panels(decomp: pd.DataFrame) -> Figure:
    """Companion to the main figure: P(wrong on the image | right on the text) against panel count.

    Same layout as accuracy_by_panels so the two read side by side. One series per panel, in the condition-A
    blue, since this is a rate of failing on the image. A single series needs no legend; the title names it.
    """
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True, facecolor=SURFACE)
    n_signs = decomp.drop_duplicates("panels").set_index("panels").n_signs
    for ax, model in zip(axes, MODELS):
        _panel(ax, model, n_signs, ax is axes[0])
        t = decomp[decomp.model == model].set_index("panels").reindex(STRATA)
        t = t.rename(columns={"p_loss": "est", "p_loss_lo": "lo", "p_loss_hi": "hi"})
        _line(ax, np.arange(3), t, "A", 2)
        overall = (t.lost_on_image.sum() / (t.lost_on_image.sum() + t.both_right.sum()))
        ax.text(0, 1.02, f"overall {100 * overall:.1f}% of text-correct queries lost", transform=ax.transAxes,
                color=INK2, fontsize=8.5)
    axes[0].set_ylim(0, 0.4)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("Wrong on the image", color=INK2)
    axes[1].set_xlabel("Panels on the sign", color=INK2)
    _frame(fig, "Lost on the image: queries answered right from the text, then wrong from the image",
           "Share of each model's text-correct queries it gets wrong from the sign image. "
           "Error bars: 95% interval, resampling signs.")
    return fig
