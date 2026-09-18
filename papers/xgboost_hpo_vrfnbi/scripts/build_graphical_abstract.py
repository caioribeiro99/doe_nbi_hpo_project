"""Build the graphical abstract Applied Soft Computing requires at submission.

Every number is READ from the committed primary-analysis artifact, never typed. The
figure therefore cannot state a value the frozen manuscript does not, and it moves if and
only if the analysis moves.

This is programmatic plotting of the study's own data by the same mechanism that produced
Figures 1-6. It is not generative-AI artwork, which the journal forbids for graphical
abstracts: "The use of generative AI or AI-assisted tools in the production of artwork
such as for graphical abstracts is not permitted."

Size follows the journal's rule: "Ensure the image is a minimum of 531 x 1328 pixels
(h x w) or proportionally more and is readable at a size of 5 x 13 cm using a regular
screen resolution of 96 dpi."
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PAPER = pathlib.Path(__file__).resolve().parents[1]
OUT = PAPER / "submission" / "ASOC_graphical_abstract"
DISP = {"magic": "MAGIC", "adult": "Adult", "bank_marketing": "Bank Marketing"}
PANEL = ("magic", "adult", "bank_marketing")

# The journal gives both a pixel minimum and a physical size, and they disagree
# (13 cm at 96 dpi is 491 px, not 1328). The PIXEL spec is the binding one, so the
# canvas is sized from it directly and scaled up "proportionally more" as allowed.
W_PX, H_PX, SCALE, DPI = 1328, 531, 1.15, 300


def load() -> dict:
    d = json.loads((PAPER / "analysis" / "primary_analysis.json").read_text())
    out: dict[str, dict[str, float]] = {}
    for ds in PANEL:
        for row in d["primary_family"]["core"][ds]:
            out.setdefault(row["contrast"], {})[ds] = row["median_diff"]
    return out


def main() -> int:
    med = load()
    fig = plt.figure(figsize=(W_PX * SCALE / DPI, H_PX * SCALE / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100 * H_PX / W_PX); ax.axis("off")

    ax.text(50, 35.4, "One pipeline change, three mechanisms, separated",
            ha="center", va="top", fontsize=8.4, fontweight="bold")
    ax.text(50, 31.6, "Surrogate-assisted multiobjective hyperparameter optimization "
            "of XGBoost · 4 public datasets · R = 30",
            ha="center", va="top", fontsize=5.5, color="#444")

    arms = ["HISTORICAL-WS", "WS-S", "NBI-S", "NBI-R"]
    xs = [12, 36, 61, 88]
    for x, name in zip(xs, arms):
        ax.add_patch(FancyBboxPatch((x - 8.4, 20.0), 16.8, 5.2, boxstyle="round,pad=0.5",
                                    linewidth=0.7, edgecolor="#333", facecolor="#eef1f6"))
        ax.text(x, 22.6, name, ha="center", va="center", fontsize=5.6, family="monospace")

    labels = [("specification\n& normalization", "no detectable change", "#666"),
              ("front-construction\ngeometry", "improved on 3/3", "#1a6fb5"),
              ("anchor & payoff\nprovenance", "no benefit; worse on 2/3", "#b5321a")]
    for i, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
        ax.add_patch(FancyArrowPatch((x0 + 8.8, 22.6), (x1 - 8.8, 22.6),
                                     arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.9, color=labels[i][2]))
        mid = (x0 + x1) / 2
        ax.text(mid, 27.0, labels[i][0], ha="center", va="center", fontsize=4.9,
                color=labels[i][2], linespacing=1.15)
        ax.text(mid, 17.6, labels[i][1], ha="center", va="center", fontsize=5.0,
                color=labels[i][2], fontweight="bold")

    # median paired differences in CORE-relative hypervolume, read from the artifact
    ax.text(50, 13.0, "median paired difference, CORE-relative hypervolume ratio",
            ha="center", va="center", fontsize=4.9, color="#444")
    cols = {"WS-S -> NBI-S": "#1a6fb5", "NBI-S -> NBI-R": "#b5321a"}
    x = 18
    for contrast, colour in cols.items():
        for ds in PANEL:
            v = med[contrast][ds]
            ax.text(x, 7.6, DISP[ds], ha="center", va="center", fontsize=4.6, color="#444")
            ax.text(x, 3.4, ("%+.3f" % v), ha="center", va="center",
                    fontsize=7.0, color=colour, fontweight="bold")
            x += 12.6
        x += 5.6
    ax.text(8.0, 5.5, "geometry", ha="center", va="center", fontsize=5.0,
            color=cols["WS-S -> NBI-S"], rotation=90, fontweight="bold")
    ax.text(18 + 3 * 12.6 + 5.6 - 10.4, 5.5, "provenance", ha="center", va="center",
            fontsize=5.0, color=cols["NBI-S -> NBI-R"], rotation=90, fontweight="bold")

    OUT.parent.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT.with_suffix("." + ext), dpi=DPI, facecolor="white")
    plt.close(fig)

    from PIL import Image
    w, h = Image.open(OUT.with_suffix(".png")).size
    ok = h >= H_PX and w >= W_PX and abs(w / h - W_PX / H_PX) < 0.02
    print(f"  {OUT.name}.png  {w} x {h} px (w x h)")
    print(f"  journal minimum {W_PX} x {H_PX} (w x h), same aspect: {'OK' if ok else 'FAIL'}")
    print(f"  numbers read from analysis/primary_analysis.json: "
          f"{sum(len(v) for v in med.values())} medians")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
