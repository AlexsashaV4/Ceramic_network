"""Consistent, quiet matplotlib figures.

Colours follow a validated categorical palette in a fixed order (blue,
orange, aqua, yellow for CW, EFG, MI, LFG); where all four share a scatter,
marker shapes carry identity as well. Heatmaps use one blue ramp (magnitude)
or blue <-> red with a grey midpoint (signed values).
"""
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .data import BOKS, BOK_NAMES

COLORS = {"CW": "#2a78d6", "EFG": "#eb6834", "MI": "#1baf7a", "LFG": "#eda100",
          "FBW": "#eb6834"}
MARKERS = {"CW": "o", "EFG": "s", "MI": "^", "LFG": "D"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4a6a5", "#e34948", "#9c2020"])
FIG_DIR = Path(__file__).resolve().parents[2] / "figures"


def setup():
    mpl.rcParams.update({
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"], "figure.dpi": 100, "savefig.dpi": 120,
        "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 10,
        "text.color": INK["primary"], "axes.labelcolor": INK["secondary"],
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": INK["primary"],
        "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": INK["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"], "ytick.labelcolor": INK["secondary"],
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.labelcolor": INK["secondary"],
    })


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")


def label(b):
    return f"{b} · {BOK_NAMES.get(b, b)}"


def heatmap(ax, values, rows, cols, cmap=SEQ, fmt="{:.2f}", vmin=None, vmax=None, center=None,
            annotate=True, cbar_label=None):
    """Annotated heatmap; annotation ink switches to white on dark cells."""
    values = np.asarray(values, float)
    norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax) if center is not None else None
    im = ax.imshow(values, cmap=cmap, aspect="auto", norm=norm,
                   vmin=None if norm else vmin, vmax=None if norm else vmax)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                rgb = im.cmap(im.norm(values[i, j]))[:3]
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if lum < 0.5 else INK["primary"])
    if cbar_label:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar_label, color=INK["secondary"])
        cb.outline.set_visible(False)
    return im


def confusion(ax, cm, classes=BOKS, title=None):
    heatmap(ax, cm, classes, classes, vmin=0, vmax=1, fmt="{:.0%}")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if title:
        ax.set_title(title)


def confusion_panel(results, keys, classes=BOKS, width=3.5):
    """Row-normalised confusion matrices of several (model, protocol) results, one style."""
    fig, axes = plt.subplots(1, len(keys), figsize=(width * len(keys), 3.5), squeeze=False)
    table = results.table()
    for ax, (model, protocol) in zip(axes[0], keys):
        acc = table.loc[(model, protocol), "accuracy"]
        confusion(ax, results.confusion(model, protocol), classes)
        ax.set_title(f"{model}\n{protocol} · accuracy {acc:.2f}", fontsize=9.5)
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    return fig


def short(triplet):
    """'(A, B, C)' -> 'A › B › C' for axis labels."""
    return triplet.strip("()").replace(", ", " › ")


def distribution_comparison(empirical, model, vocab, cultures, model_label, top=12, js=None):
    """For each culture: its most frequent triplets, empirical share vs the model's share."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
    for ax, i in zip(axes.flat, range(len(cultures))):
        c, e, m = cultures[i], empirical[i], model[i]
        idx = np.argsort(-e)[:top]
        y = np.arange(len(idx))
        ax.barh(y - 0.2, e[idx], height=0.38, color=INK["muted"], label="empirical")
        ax.barh(y + 0.2, m[idx], height=0.38, color=COLORS[c], label=model_label)
        ax.set_yticks(y, [short(vocab[j]) for j in idx], fontsize=7.5)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{label(c)}" + (f"  ·  JS = {js[i]:.3f} bits" if js is not None else ""), fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def hbars(ax, labels, values, colors, xerr=None, fmt="{:.2f}", ref=None, ref_label=None):
    """Horizontal bars with the value at the tip; optional reference line."""
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.6, color=colors, xerr=xerr,
            error_kw={"ecolor": INK["secondary"], "elinewidth": 1, "capsize": 0})
    for yi, v, e in zip(y, values, xerr if xerr is not None else [0] * len(y)):
        ax.text(v + (e or 0) + 0.005 * max(values), yi, fmt.format(v), va="center",
                color=INK["secondary"], fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    if ref is not None:
        ax.axvline(ref, color=INK["muted"], lw=1)
        if ref_label:
            ax.text(ref, len(labels) - 0.4, f" {ref_label}", color=INK["muted"], fontsize=8, va="bottom")


def technique_layout(chains, lanes=(1.0, -0.2, 0.6, -1.0, 0.2, -0.6)):
    """Techniques ordered left to right by their mean relative position in the chains,
    evenly spaced, and spread over a few horizontal lanes so labels do not collide."""
    pos = {}
    for s in chains:
        for i, x in enumerate(s):
            pos.setdefault(x, []).append(i / (len(s) - 1))
    order = sorted(pos, key=lambda k: np.mean(pos[k]))
    xs = np.linspace(0, 1, len(order))
    return {k: (xs[i], lanes[i % len(lanes)]) for i, k in enumerate(order)}


def transition_graph(ax, chains, layout, color, min_share=0.02, title=None):
    """Technique network of one body of knowledge: edge width = share of chains using it."""
    from collections import Counter

    counts = Counter(e for s in chains for e in zip(s[:-1], s[1:]))
    n = len(chains)
    ax.grid(False)
    ax.set_axis_off()
    for (a, b), c in counts.items():
        share = c / n
        if share < min_share:
            continue
        (x0, y0), (x1, y1) = layout[a], layout[b]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.5 + 4 * share,
                                    alpha=0.35 + 0.6 * share, shrinkA=6, shrinkB=6,
                                    connectionstyle="arc3,rad=0.12", mutation_scale=8))
    used = {x for s in chains for x in s}
    for k, (x, y) in layout.items():
        on = k in used
        ax.scatter(x, y, s=28, color=color if on else INK["grid"], zorder=3,
                   edgecolor=INK["surface"], linewidth=1.5)
        ax.text(x, y + (0.1 if y >= 0 else -0.1), textwrap.fill(k, 13), ha="center",
                va="bottom" if y >= 0 else "top", fontsize=7, linespacing=1.0,
                color=INK["secondary"] if on else INK["muted"])
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-1.45, 1.45)
    if title:
        ax.set_title(title)
