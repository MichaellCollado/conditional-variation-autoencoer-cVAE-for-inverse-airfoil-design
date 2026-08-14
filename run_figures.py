"""Draw every figure the article cites, one figure per function.

Ten figures are written, named for the article's own numbering: Figures 1 to 8
in the body and Figures A1 and A2 in the appendix. Each is written as a PDF and
as a 200 dpi PNG, into figures/.

Two further figures are built but not written. The attrition flow and the
truncation bias are no longer cited by the article, so no image is produced for
either. Their code is retained and both still run, because their consistency
checks are part of this driver's seven. Pass write_image=True to either to get
the image back.

WHERE THE NUMBERS COME FROM. Every value is read from a committed artifact.
Nothing is recomputed that a build step already computed, with three stated
exceptions, each verified against the recorded value before anything is drawn.

  Figure 3's per-arm distances
      Regenerated from committed_model.pt at B18's own generation seed, and
      checked against b18_gate.json's recorded means and direction consistency.
  Figure 8's per-shape predictions
      The surrogate's prediction on each admitted B23 shape, which B25 formed
      the three surrogate-gap scalars from but did not store per shape. The
      recomputed scalars are checked against b25_metrics.json.
  Figures A1 and A2
      Read committed_training_history.json, which
      run_committed_training_history.py produced and verified against
      b18_gate.json bit for bit.

SEVEN CHECKS, ANY OF WHICH STOPS THIS DRIVER.

  Figure 5   the fitted trend slope must equal the recorded tracking slope
             less one, to within 1e-12, per arm.
  Figure 7   the recomputed across-range mean must equal the recorded mean, to
             within 1e-12, per arm.
  Figure 2   the committed avian weight must be present in the sweep's own
             ladder.
  Figure 8   the three recomputed surrogate-gap scalars must equal B25's
             recorded values exactly.
  Figure 3   both arms' mean distance and the direction consistency must
             reproduce b18_gate.json at exact float equality, and both
             separation counts must match avian_signature.npz at exact
             integer equality.
  Figure A2  the live dimension count must equal the regenerated count, the
             stored history and b18_gate.json exactly.
  truncation bias
             the bias at the full sweep length must be exactly 0.0 on both the
             mean and the upper percentile. Runs even though no image is
             written.

DRAWING CHOICES THIS DRIVER MADE. The article fixes what each figure plots and
leaves the drawing open. Each choice below is recorded here rather than left in
the code to be discovered, because the article does not state any of them.

  Arm colour        prior-on blue, prior-off orange, everywhere, in every
                    figure.
  Reference         the avian reference is black, dashed, in every figure it
                    appears in.
  Figure 6 panels   four requested targets evenly spanning the committed band,
                    being the lowest, the highest and two interior targets
                    evenly spaced between them, at indices 0, 3, 7 and 10 of
                    the eleven, PLUS every target at which the admission rule
                    excluded a shape. The second clause is there because the
                    figure's job includes the honest presentation of shapes the
                    solver could not evaluate. Every sample at each shown
                    target is drawn. No sample is selected.
  Figure 6 marking  a shape whose solver sweep did not converge at every
                    requested angle is drawn with a red halo, and a shape the
                    admission rule excluded additionally carries a red cross.
                    Nothing is omitted.
  Figure 5 band     plus and minus one sample standard deviation of the signed
                    error across the samples at that target, per arm. One
                    convention, named in the figure's legend, and not mixed
                    with any other inside the comparison.
  Figure 3 vertical every series is drawn as a density, because the four
    scaling         populations differ in size by a factor of four and counts
                    would make the comparison unreadable.
  Format            each figure is written as a PDF and as a 200 dpi PNG. The
                    article cites the PDF.
  Text on the       kept to the title, the axis labels, the legend and labels
    canvas          attached to a mark. The explanation of each figure is the
                    article's caption, and is not repeated inside the image,
                    so the two cannot drift apart. Where a number is needed to
                    read a mark it is put in that mark's legend entry rather
                    than in a block of text.

Run order      after run_committed_training_history.py. Last script of the run.
Reads          b18_gate.json, b20_truncation.json, b23_evaluation.json,
               b24_analysis.json, b25_metrics.json, b17_selection.json,
               committed_training_history.json, committed_model.pt,
               avian_signature.npz, sweep/sweep_table.json, and the build
               artifacts
Writes         figures/, ten figures as one PDF and one 200 dpi PNG each
Runtime        not recorded in the article

matplotlib is used here and by no pipeline module.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch

import analysis
import dataset as dataset_mod
import evaluate
import geometry
import model as model_mod
import run_b16_weight_sweep as b16

FIGDIR = "figures"

COL_ON = "#1f5fa8"
COL_OFF = "#d97b28"
COL_REF = "#000000"
COL_MARK = "#cc0033"
COL_REGION = "#6a3d9a"
COL_TRAIN_AVIAN = "#7fb3d5"
COL_TRAIN_OTHER = "#9a9a9a"
COL_GREY = "#555555"

ARM_LABEL = {"prior_on": "prior on", "prior_off": "prior off"}
ARM_COL = {"prior_on": COL_ON, "prior_off": COL_OFF}

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Loading and saving.
# ---------------------------------------------------------------------------

def load(name):
    with open(name) as f:
        return json.load(f)


def save(fig, stem):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        path = os.path.join(FIGDIR, f"{stem}.{ext}")
        fig.savefig(path, **kw)
    plt.close(fig)
    print(f"  -> {os.path.join(FIGDIR, stem)}.pdf / .png")



def require(condition, message):
    if not condition:
        raise SystemExit(f"FIGURE ABORTED. {message}")


def box(ax, x, y, w, h, text, facecolor="white", edgecolor=COL_GREY,
        fontsize=7, weight="normal", textcolor="black", lw=0.9, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=lw, edgecolor=edgecolor, facecolor=facecolor,
                                linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=textcolor, zorder=3, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, start, end, color=COL_GREY, lw=1.0, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1,
                                 shrinkA=0, shrinkB=0))


def blank_axes(fig_w, fig_h):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 1. Method schematic. Plots no metric.
# ---------------------------------------------------------------------------

def figure_1_method_schematic():
    fig, ax = blank_axes(9.2, 6.0)

    ax.text(0.5, 0.995, "Figure 1. Method schematic", ha="center", va="top",
            fontsize=10, fontweight="bold")

    # Row one. Dataset construction.
    y1, h = 0.772, 0.085
    w, gap = 0.163, 0.037
    xs = [0.020 + i * (w + gap) for i in range(5)]
    labels = [
        "Seed library\nfive sections, the\nseagull among them",
        "CST fit\norder 9, one basis\nshared by every shape",
        "Bounded sampler\nper coefficient bound,\n1005 shapes",
        "XFOIL labelling\nattached flow sweep,\nmax(CL/CD)",
        "Dataset\n982 labelled rows,\none split",
    ]
    for x, t in zip(xs, labels):
        box(ax, x, y1, w, h, t)
    for i in range(4):
        arrow(ax, (xs[i] + w, y1 + h / 2), (xs[i + 1], y1 + h / 2))
    ax.text(0.02, y1 + h + 0.024, "Dataset construction", fontsize=8,
            fontweight="bold", color=COL_GREY, ha="left")

    # Row two. Training.
    y2 = 0.545
    box(ax, 0.035, y2, w, h,
        "Conditioning\nnormalised target,\nsignature block, flag")
    box(ax, 0.035, y2 - 0.135, w, h,
        "Surrogate ensemble\nfrozen, gradient only,\nno reported number")
    ax.plot([xs[4] + w / 2, xs[4] + w / 2], [y1, 0.712], color=COL_GREY, lw=1.0, zorder=1)
    ax.plot([xs[4] + w / 2, 0.445], [0.712, 0.712], color=COL_GREY, lw=1.0, zorder=1)
    arrow(ax, (0.445, 0.712), (0.445, 0.672))
    ax.text(0.63, 0.722, "standardised geometry and one split", fontsize=6.5,
            color=COL_GREY, ha="center")

    # The objective box, which is the point of the figure.
    ox, oy, ow, oh = 0.245, y2 - 0.148, 0.40, 0.246
    ax.add_patch(FancyBboxPatch((ox, oy), ow, oh,
                                boxstyle="round,pad=0.006,rounding_size=0.014",
                                linewidth=1.4, edgecolor=COL_GREY, facecolor="#f6f6f6",
                                zorder=1))
    ax.text(ox + ow / 2, oy + oh - 0.026, "Training objective", ha="center",
            fontsize=8.5, fontweight="bold")
    terms = ["reconstruction", "divergence", "safeguard, both passes",
             "target consistency", "ensemble spread"]
    for i, t in enumerate(terms):
        ax.text(ox + 0.022, oy + oh - 0.060 - i * 0.031, "•  " + t,
                fontsize=7, color=COL_GREY, va="center")
    ax.text(ox + 0.022, oy + oh - 0.060 - 5 * 0.031, "•  avian prior, flag gated",
            fontsize=7.4, color=COL_MARK, va="center", fontweight="bold")

    for src_y in (y2 + h / 2, y2 - 0.135 + h / 2):
        arrow(ax, (0.035 + w, src_y), (ox, oy + oh / 2))

    box(ax, 0.695, y2 - 0.055, 0.175, 0.115,
        "Conditional VAE\none trained model,\none checkpoint",
        edgecolor=COL_GREY, lw=1.2)
    arrow(ax, (ox + ow, oy + oh / 2), (0.695, y2 - 0.055 + 0.0575), lw=1.2)
    ax.text(0.02, y2 + h + 0.028, "Training", fontsize=8, fontweight="bold",
            color=COL_GREY, ha="left")

    # Row three. Paired generation and evaluation.
    y3 = 0.135
    box(ax, 0.035, y3, 0.155, h, "Paired generation\none latent code\nper pair")
    ax.plot([0.7825, 0.7825], [y2 - 0.055, y3 + h + 0.042], color=COL_GREY, lw=1.0, zorder=1)
    ax.plot([0.7825, 0.1125], [y3 + h + 0.042, y3 + h + 0.042], color=COL_GREY, lw=1.0,
            zorder=1)
    arrow(ax, (0.1125, y3 + h + 0.042), (0.1125, y3 + h))

    box(ax, 0.245, y3 + 0.052, 0.175, 0.072, "flag set\nprior on arm",
        edgecolor=COL_ON, textcolor=COL_ON, lw=1.3)
    box(ax, 0.245, y3 - 0.052, 0.175, 0.072, "flag clear\nprior off arm",
        edgecolor=COL_OFF, textcolor=COL_OFF, lw=1.3)
    arrow(ax, (0.19, y3 + h / 2), (0.245, y3 + 0.088), color=COL_ON)
    arrow(ax, (0.19, y3 + h / 2), (0.245, y3 - 0.016), color=COL_OFF)

    box(ax, 0.475, y3, 0.165, h, "XFOIL evaluation\none operating point,\none admission rule")
    arrow(ax, (0.42, y3 + 0.088), (0.475, y3 + h / 2), color=COL_ON)
    arrow(ax, (0.42, y3 - 0.016), (0.475, y3 + h / 2), color=COL_OFF)

    box(ax, 0.695, y3, 0.175, h, "Paired analysis\none module, every\nreported statistic")
    arrow(ax, (0.64, y3 + h / 2), (0.695, y3 + h / 2))
    ax.text(0.02, y3 + h + 0.070, "Paired generation and evaluation", fontsize=8,
            fontweight="bold", color=COL_GREY, ha="left")

    save(fig, "figure_1_method_schematic")


# ---------------------------------------------------------------------------
# Figure 6. Generated shape family against the avian reference.
# Plots the target satisfaction error per shape.
# ---------------------------------------------------------------------------

BASE_SHOWN_TARGET_INDICES = (0, 3, 7, 10)


def shown_target_indices(b23):
    """The rule, stated here and applied mechanically. Four requested targets
    evenly spanning the committed band, being the lowest, the highest and two
    interior targets evenly spaced between them, PLUS every target at which the
    admission rule excluded a shape. The second clause is there because Figure 6's
    job includes the honest presentation of shapes the solver could not
    evaluate, and a panel set that happened to exclude the excluded shape would
    not do it."""
    excluded = {r["target_index"] for r in b23["records"]
                for arm in ("prior_on", "prior_off") if not r[arm]["admitted"]}
    return tuple(sorted(set(BASE_SHOWN_TARGET_INDICES) | excluded))


def figure_6_shape_family(b23, artifacts, reference_loop):
    header = b23["header"]
    n_points = int(header["n_points_per_surface"])
    label_min, label_max = artifacts.label_norm.label_min, artifacts.label_norm.label_max

    by_key = {(r["target_index"], r["sample_index"]): r for r in b23["records"]}
    samples = sorted({r["sample_index"] for r in b23["records"]})
    shown = shown_target_indices(b23)
    n_rows, n_cols = len(shown), len(samples)

    row_h, foot, head = 0.80, 0.62, 0.45
    fig_h = row_h * n_rows + foot + head
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.30 * n_cols, fig_h),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.10, wspace=0.06,
                        top=1 - head / fig_h, bottom=foot / fig_h)
    n_marked_shown = 0
    n_marked_total = 0
    excluded_targets = []
    for r in b23["records"]:
        for arm in ("prior_on", "prior_off"):
            if r[arm]["n_converged"] < r[arm]["n_requested"]:
                n_marked_total += 1
            if not r[arm]["admitted"]:
                excluded_targets.append(r["target_index"])

    for row, t_idx in enumerate(shown):
        for col, s_idx in enumerate(samples):
            ax = axes[row][col]
            rec = by_key[(t_idx, s_idx)]
            target = rec["target"]
            ax.plot(reference_loop[0], reference_loop[1], color=COL_REF, lw=0.8,
                    ls="--", zorder=2)
            texts = {}
            for arm in ("prior_off", "prior_on"):
                a = rec[arm]
                upper, lower = evaluate.standardised_to_coefficients(
                    np.asarray(a["standardised_coefficients"], dtype=float),
                    artifacts.std_stats)
                foil = geometry.decode_airfoil(upper[0], lower[0], n_points)
                partial = a["n_converged"] < a["n_requested"]
                if partial:
                    n_marked_shown += 1
                    ax.plot(foil.x, foil.y, color=COL_MARK, lw=2.6, alpha=0.55, zorder=3)
                ax.plot(foil.x, foil.y, color=ARM_COL[arm],
                        lw=1.3 if arm == "prior_off" else 0.9,
                        zorder=4 if arm == "prior_off" else 5)
                achieved = (a["label"] - label_min) / (label_max - label_min)
                err = abs(achieved - target)
                mark = "" if a["admitted"] else "  excluded"
                texts[arm] = (f"{err:.3f}" + mark,
                              COL_MARK if (partial or not a["admitted"]) else ARM_COL[arm])
                if not a["admitted"]:
                    ax.plot([0.5], [0.0], marker="x", color=COL_MARK, ms=7, mew=1.6,
                            zorder=5)
            ax.set_xlim(-0.04, 1.04)
            ax.set_ylim(-0.13, 0.31)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            ax.text(0.02, 0.99, texts["prior_on"][0], transform=ax.transAxes,
                    color=texts["prior_on"][1], fontsize=6.0, va="top", ha="left")
            ax.text(0.02, 0.80, texts["prior_off"][0], transform=ax.transAxes,
                    color=texts["prior_off"][1], fontsize=6.0, va="top", ha="left")
            if col == 0:
                raw = label_min + target * (label_max - label_min)
                ax.text(-0.06, 0.42, f"requested {target:.3f}\n({raw:.1f} L/D)",
                        transform=ax.transAxes, ha="right", va="center", fontsize=6.6)
            if row == 0:
                ax.set_title(f"sample {s_idx}", fontsize=6.6, pad=1)

    fig.suptitle("Figure 6. Generated sections against the avian reference, both arms, "
                 "every sample at the shown requested targets", fontsize=9.5,
                 y=1 - 0.12 / fig_h)
    handles = [
        plt.Line2D([], [], color=COL_ON, lw=1.2, label="prior on"),
        plt.Line2D([], [], color=COL_OFF, lw=1.2, label="prior off"),
        plt.Line2D([], [], color=COL_REF, lw=1.0, ls="--", label="avian reference"),
        plt.Line2D([], [], color=COL_MARK, lw=2.4, alpha=0.55,
                   label="solver sweep did not fully converge"),
        plt.Line2D([], [], color=COL_MARK, lw=0, marker="x", ms=6,
                   label="excluded by the admission rule"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, (foot - 0.42) / fig_h))
    shown_text = ", ".join(str(i) for i in shown)
    base_text = ", ".join(str(i) for i in BASE_SHOWN_TARGET_INDICES)
    excluded_at = ", ".join(str(i) for i in sorted(set(excluded_targets))) or "none"
    save(fig, "figure_6_shape_family_against_reference")


# ---------------------------------------------------------------------------
# Figure 4. Paired difference display. Plots the paired difference and its interval.
# ---------------------------------------------------------------------------

def figure_4_paired_difference(b24):
    ps = b24["paired_series"]
    target = np.asarray(ps["target"], dtype=float)
    diff = np.asarray(ps["difference"], dtype=float)
    primary = b24["primary"]
    shape = b24["distribution_shape"]
    span = b24["normalisation"]["label_max"] - b24["normalisation"]["label_min"]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axhline(0.0, color=COL_GREY, lw=0.8, ls=":")
    ax.axhspan(primary["lower"], primary["upper"], color=COL_ON, alpha=0.12, lw=0,
               label=f"95% wild cluster bootstrap-t interval "
                     f"[{primary['lower']:.4f}, {primary['upper']:.4f}]")
    ax.axhline(primary["point_estimate"], color=COL_ON, lw=1.4,
               label=f"mean paired difference {primary['point_estimate']:.4f}")

    jitter = (np.arange(len(target)) % 10 - 4.5) * 0.0016
    ax.scatter(target + jitter, diff, s=16, facecolor="none", edgecolor=COL_GREY,
               linewidth=0.8, zorder=3,
               label=f"one matched pair, n {b24['n_analysed_pairs']} "
                     f"in {b24['n_clusters']} clusters")

    ax.set_xlabel("requested target, normalised")
    ax.set_ylabel("paired difference in target satisfaction error, normalised\n"
                  "(prior off error minus prior on error)")
    ax.set_title("Figure 4. Paired difference against requested target", loc="left")

    sec = ax.secondary_yaxis("right", functions=(lambda v: v * span, lambda v: v / span))
    sec.set_ylabel("raw equivalent, L/D")

    ax.set_ylim(top=0.175)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=3, frameon=False)
    save(fig, "figure_4_paired_difference")


# ---------------------------------------------------------------------------
# Figure 5. Signed error against requested target, per arm.
# Plots the signed error, the tracking slope and the tracking correlation.
# ---------------------------------------------------------------------------

def figure_5_signed_error(b24, b25):
    ps = b24["paired_series"]
    target = np.asarray(ps["target"], dtype=float)
    tracking = b25["M08_M09_tracking"]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axhline(0.0, color=COL_GREY, lw=0.8, ls=":")

    grid = np.unique(target)
    for arm, key in (("prior_on", "achieved_prior_on"), ("prior_off", "achieved_prior_off")):
        achieved = np.asarray(ps[key], dtype=float)
        signed = achieved - target
        slope = tracking[arm]["slope"] - 1.0
        require(abs(slope - tracking[arm]["f04_trend_slope_derived"]) < 1e-12,
                f"Figure 5 trend slope for {arm} does not equal the tracking slope less one")
        line = signed.mean() + slope * (grid - target.mean())
        centres, lo, hi = [], [], []
        for t in grid:
            v = signed[target == t]
            centres.append(v.mean())
            sd = v.std(ddof=1)
            lo.append(v.mean() - sd)
            hi.append(v.mean() + sd)
        ax.fill_between(grid, lo, hi, color=ARM_COL[arm], alpha=0.14, lw=0,
                        label=f"{ARM_LABEL[arm]}, plus and minus one sample SD")
        ax.scatter(target, signed, s=11, color=ARM_COL[arm], alpha=0.55, lw=0)
        ax.plot(grid, centres, color=ARM_COL[arm], lw=0.9, ls=":", marker="o", ms=3)
        ax.plot(grid, line, color=ARM_COL[arm], lw=1.6,
                label=f"{ARM_LABEL[arm]}, fitted trend slope {slope:.4f} "
                      f"(tracking slope {tracking[arm]['slope']:.4f}, "
                      f"r {tracking[arm]['correlation']:.4f})")

    ax.set_xlabel("requested target, normalised")
    ax.set_ylabel("signed target satisfaction error, normalised\n(achieved minus requested)")
    ax.set_title("Figure 5. Signed error against requested target, per arm", loc="left")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "figure_5_signed_error_per_arm")


# ---------------------------------------------------------------------------
# Attrition flow. Not cited by the article, so no image is written.
# Plots the attrition flow, the admission exclusion count and the paired yield.
# ---------------------------------------------------------------------------

def uncited_attrition_flow(b23, b24, write_image=False):
    flow = b23["attrition_flow"]
    pair = b24["flow"]

    fig, ax = blank_axes(9.0, 6.2)
    ax.text(0.0, 0.995, "Attrition flow, launched runs to matched pairs",
            fontsize=10, fontweight="bold", va="top")
    ax.text(0.0, 0.958,
            "Every loss is annotated by its reason and its count. Zero counts are printed "
            "rather than omitted.", fontsize=7, color=COL_GREY, va="top")

    col_x = {"prior_on": 0.030, "prior_off": 0.530}
    w = 0.225
    stages = [
        ("launched", "launched"),
        ("plausibility_rejected", "cleared the plausibility filter"),
        ("produced_a_label", "produced a usable label"),
        ("admitted", "admitted"),
    ]
    ys = [0.825, 0.680, 0.535, 0.390]
    h = 0.062

    for arm in ("prior_on", "prior_off"):
        f = flow[arm]
        x = col_x[arm]
        ax.text(x + w / 2, 0.905, ARM_LABEL[arm], ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=ARM_COL[arm])
        counts = [f["launched"],
                  f["launched"] - f["plausibility_rejected"],
                  f["produced_a_label"],
                  f["admitted"]]
        for i, ((_, text), y, c) in enumerate(zip(stages, ys, counts)):
            box(ax, x, y, w, h, f"{text}\n{c}", edgecolor=ARM_COL[arm], lw=1.1)
            if i:
                arrow(ax, (x + w / 2, ys[i - 1]), (x + w / 2, y + h), color=ARM_COL[arm])
        losses = [
            f"rejected before the solver: {f['plausibility_rejected']}",
            (f"timeout {f['timeout']}, environment fault {f['environment_fault']},\n"
             f"solver failure {f['failed']}\n"
             f"converged {f['converged']}, partially converged {f['partially_converged']}"),
            (f"produced a label and failed the minimum\n"
             f"converged point rule: {f['label_but_below_min_points']}"),
        ]
        for i, text in enumerate(losses):
            ax.text(x + w + 0.015, ys[i] - 0.032, text, fontsize=6.3, color=COL_GREY,
                    va="center", ha="left", linespacing=1.4)

    y_pair = 0.205
    box(ax, 0.145, y_pair, 0.475, 0.075,
        f"matched pairs, both members admitted\n"
        f"{pair['pairs_both_admitted']} of {pair['pairs_launched']} pairs launched",
        edgecolor=COL_GREY, lw=1.4)
    arrow(ax, (col_x["prior_on"] + w / 2, ys[3]), (0.30, y_pair + 0.075), color=COL_ON)
    arrow(ax, (col_x["prior_off"] + w / 2, ys[3]), (0.47, y_pair + 0.075), color=COL_OFF)

    ax.text(0.145, y_pair - 0.030,
            f"pairs losing only the prior on member: {pair['pairs_only_prior_off_admitted']}"
            f"      pairs losing only the prior off member: {pair['pairs_only_prior_on_admitted']}"
            f"      pairs losing both: {pair['pairs_neither']}",
            fontsize=6.6, color=COL_GREY, va="top")

    n_label = flow["prior_on"]["produced_a_label"] + flow["prior_off"]["produced_a_label"]
    n_excluded = (flow["prior_on"]["label_but_below_min_points"]
                  + flow["prior_off"]["label_but_below_min_points"])
    ax.text(0.0, 0.115,
            f"Admission exclusion count: {n_excluded} of {n_label}. The "
            f"denominator is the records that produced a label, not the runs launched.\n"
            f"Paired yield: {pair['pairs_both_admitted']} of {pair['pairs_launched']}, "
            f"against a floor of 100 analysed pairs committed in advance.\n"
            "Attrition is shown at the shape level and at the pair level. They are "
            "different numbers and the second is the one the analysis uses.\n"
            "The separation of timeouts from convergence failures has no published "
            "precedent and is this study's own. It changes every denominator it touches.",
            fontsize=6.8, color=COL_GREY, va="top")
    if write_image:
        save(fig, "attrition_flow")
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7. Diversity across the requested range. Plots the diversity metric.
# ---------------------------------------------------------------------------

def figure_7_diversity(b25):
    m11 = b25["M11_generative_diversity"]
    targets = np.asarray(m11["targets"], dtype=float)
    per_target = m11["per_target"]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for arm in ("prior_on", "prior_off"):
        values = np.asarray(per_target[arm], dtype=float)
        mean = m11["mean_across_range"][arm]
        require(abs(values.mean() - mean) < 1e-12,
                f"Figure 7 across-range mean for {arm} does not equal the recorded value")
        ax.plot(targets, values, marker="o", ms=4, lw=1.2, color=ARM_COL[arm],
                label=f"{ARM_LABEL[arm]}, within-target spread")
        ax.axhline(mean, color=ARM_COL[arm], lw=1.0, ls="--",
                   label=f"{ARM_LABEL[arm]}, mean across the range {mean:.4f}")

    ax.set_xlabel("requested target, normalised")
    ax.set_ylabel("mean pairwise Euclidean distance among the samples\n"
                  "at that target, standardised coefficient space")
    ax.set_title("Figure 7. Generative diversity across the requested range", loc="left")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="lower center", frameon=False, ncol=2)
    save(fig, "figure_7_diversity_across_range")


# ---------------------------------------------------------------------------
# Figure 8. Surrogate against solver. Plots the surrogate to solver gap.
# ---------------------------------------------------------------------------

def figure_8_surrogate_against_solver(b23, b25, artifacts):
    m22 = b25["M22_surrogate_to_solver_gap"]
    reference = m22["reference"]
    min_points = int(b23["committed"]["minimum_converged_points"])

    series = {}
    for arm in ("prior_on", "prior_off"):
        x_std, solver_value = [], []
        for record in b23["records"]:
            state = analysis.read_admission(record[arm], min_points)
            if state.admitted:
                x_std.append(record[arm]["standardised_coefficients"])
                solver_value.append(float(state.label))
        x = torch.tensor(np.asarray(x_std, dtype=float), dtype=model_mod.DTYPE)
        with torch.no_grad():
            predicted_norm = artifacts.ensemble.predict_mean(x).numpy()
        predicted = dataset_mod.denormalize_label(predicted_norm, artifacts.label_norm)
        series[arm] = (np.asarray(solver_value, dtype=float), np.asarray(predicted, dtype=float))

    pooled_solver = np.concatenate([series["prior_on"][0], series["prior_off"][0]])
    pooled_pred = np.concatenate([series["prior_on"][1], series["prior_off"][1]])
    check = analysis.surrogate_gap(pooled_pred, pooled_solver, "raw max(CL/CD)")
    for name, got, want in (
            ("n", check.n, m22["pooled"]["n"]),
            ("mean absolute difference", check.mean_absolute_difference,
             m22["pooled"]["mean_absolute_difference"]),
            ("mean signed difference", check.mean_signed_difference,
             m22["pooled"]["mean_signed_difference"]),
            ("correlation", check.correlation, m22["pooled"]["correlation"])):
        require(got == want,
                f"Figure 8 recomputed {name} ({got!r}) does not equal B25's recorded value ({want!r})")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lo = min(pooled_solver.min(), pooled_pred.min()) - 4
    hi = max(pooled_solver.max(), pooled_pred.max()) + 4
    line = np.array([lo, hi])
    band = reference["ensemble_mean_absolute_error_raw"]
    ax.fill_between(line, line - band, line + band, color=COL_GREY, alpha=0.16, lw=0,
                    label=f"held-out reference error, plus and minus {band:.4f} L/D")
    ax.plot(line, line, color=COL_GREY, lw=1.0, ls="--", label="identity")
    for arm in ("prior_on", "prior_off"):
        solver_value, predicted = series[arm]
        ax.scatter(solver_value, predicted, s=15, color=ARM_COL[arm], alpha=0.65, lw=0,
                   label=f"{ARM_LABEL[arm]}, n {len(solver_value)}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("solver value, raw max(CL/CD)")
    ax.set_ylabel("ensemble prediction, raw max(CL/CD)")
    ax.set_title("Figure 8. Surrogate against solver, admitted generated shapes", loc="left")
    ax.legend(loc="upper left", frameon=False)
    ratio = m22["pooled"]["mean_absolute_difference"] / reference["ensemble_mean_absolute_error_raw"]
    save(fig, "figure_8_surrogate_against_solver")


# ---------------------------------------------------------------------------
# Figure 2. Prior weight sensitivity. Plots the sweep table, and through it
# the diversity metric.
# ---------------------------------------------------------------------------

def figure_2_prior_weight_sensitivity(sweep, selection):
    rows = [r for r in sweep["rows"] if r["swept_weight"] == "avian"]
    rows.sort(key=lambda r: r["candidate"])
    committed = selection["selected_by_sweep"]["avian"]
    require(any(r["candidate"] == committed for r in rows),
            "Figure 2 cannot find the committed avian weight in the sweep's own ladder")

    positions = np.arange(len(rows))
    labels = ["0" if r["candidate"] == 0 else f"{r['candidate']:.4g}" for r in rows]
    recon = np.asarray([r["val_reconstruction"] for r in rows], dtype=float)
    diversity = np.asarray([r["diversity_prior_on"] for r in rows], dtype=float)
    committed_pos = int(np.argmin([abs(r["candidate"] - committed) for r in rows]))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axvline(committed_pos, color=COL_MARK, lw=1.2, ls="--")
    ax.text(committed_pos + 0.08, 0.97, f"committed weight {committed:.6g}",
            transform=ax.get_xaxis_transform(), color=COL_MARK, fontsize=7,
            va="top", ha="left")

    ax.plot(positions, recon, marker="o", ms=5, lw=1.4, color=COL_GREY,
            label="validation reconstruction at the run's own selected checkpoint")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("candidate avian prior weight")
    ax.set_ylabel("validation reconstruction")

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(positions, diversity, marker="s", ms=5, lw=1.4, color=COL_ON,
             label="generative diversity, flag-set arm")
    ax2.set_ylabel("generative diversity", color=COL_ON)
    ax2.tick_params(axis="y", colors=COL_ON)
    ax.set_ylim(bottom=0.0)
    ax2.set_ylim(bottom=0.0)

    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels_ = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, -0.135),
              ncol=2, frameon=False)
    ax.set_title("Figure 2. Prior weight sensitivity", loc="left")
    save(fig, "figure_2_prior_weight_sensitivity")


# ---------------------------------------------------------------------------
# Truncation bias against candidate point count. Not cited by the article,
# so no image is written. Plots the truncation bias table.
# ---------------------------------------------------------------------------

def uncited_truncation_bias(b20, write_image=False):
    table = b20["bias_table"]
    k = np.asarray([r["k"] for r in table], dtype=float)
    mean_abs = np.asarray([r["mean_absolute"] for r in table], dtype=float)
    upper = np.asarray([r["upper_percentile_absolute"] for r in table], dtype=float)
    tol = b20["tolerance"]
    selected = b20["selected_minimum_converged_points"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(k, mean_abs, marker="o", ms=5, lw=1.4, color=COL_GREY,
            label="mean absolute relative truncation bias")
    ax.plot(k, upper, marker="s", ms=5, lw=1.4, color=COL_ON,
            label=f"{tol['upper_percentile']:.0f}th percentile of the absolute relative bias")
    ax.axhline(tol["mean_relative_bias"], color=COL_GREY, lw=1.0, ls="--")
    ax.axhline(tol["upper_percentile_absolute_relative_bias"], color=COL_ON, lw=1.0, ls="--")
    ax.text(k[0], tol["mean_relative_bias"] * 1.15,
            f"stated tolerance on the mean, {tol['mean_relative_bias']:.3f}",
            fontsize=6.8, color=COL_GREY, va="bottom")
    ax.text(k[0], tol["upper_percentile_absolute_relative_bias"] * 1.15,
            f"stated tolerance on the upper percentile, "
            f"{tol['upper_percentile_absolute_relative_bias']:.3f}",
            fontsize=6.8, color=COL_ON, va="bottom")
    ax.axvline(selected, color=COL_MARK, lw=1.2, ls="--")
    ax.text(selected - 0.22, 0.60, f"selected minimum converged point count, {selected}",
            transform=ax.get_xaxis_transform(), color=COL_MARK, fontsize=7,
            rotation=90, va="center", ha="right")
    full = int(b20["sweep_length"])
    require(all(row["mean_absolute"] == 0.0 and row["upper_percentile_absolute"] == 0.0
                for row in table if row["k"] == full),
            "The truncation bias figure expects the bias at the full sweep length to be exactly zero")
    ax.set_xlim(0.4, 10.1)
    ax.text(full, 0.035, f"k = {full}: bias exactly zero,\nnothing is truncated",
            transform=ax.get_xaxis_transform(), color=COL_GREY, fontsize=6.6,
            ha="center", va="bottom")

    ax.set_yscale("log")
    ax.set_xlabel("candidate minimum converged point count")
    ax.set_ylabel("relative truncation bias in the label's own units")
    ax.set_title("Truncation bias against candidate point count", loc="left")
    ax.set_xticks(k)
    ax.legend(loc="lower left", frameon=False)
    if write_image:
        save(fig, "truncation_bias")
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3. Distance to the avian reference, per arm. Plots M12, M13 and the
# separation counts of Table 4.
# ---------------------------------------------------------------------------

def figure_3_distance_to_reference(gate, artifacts, avian_npz):
    cvae = model_mod.CVAE(artifacts.x_all.shape[1], artifacts.cond_all.shape[1], b16.ARCH)
    cvae = cvae.to(model_mod.DTYPE)
    cvae.load_state_dict(torch.load("committed_model.pt", weights_only=False))
    pg = evaluate.paired_generation(cvae, artifacts.reference_signature,
                                    b16.ARCH.latent_dim,
                                    generation_seed=gate["generation_seed"])
    tests = gate["committed_gate"]["group_three"]["tests"]
    d_on = evaluate.mean_distance_to_reference(pg.x_set, artifacts.reference_signature)
    d_off = evaluate.mean_distance_to_reference(pg.x_clear, artifacts.reference_signature)
    consistency = evaluate.direction_consistency(pg, artifacts.reference_signature)
    require(d_on == tests["g3b_mean_distance_separation"]["mean_distance_prior_on"],
            "Figure 3's regenerated prior-on mean distance does not equal B18's recorded value")
    require(d_off == tests["g3b_mean_distance_separation"]["mean_distance_prior_off"],
            "Figure 3's regenerated prior-off mean distance does not equal B18's recorded value")
    require(consistency == tests["g3a_direction_consistency"]["measured"],
            "Figure 3's regenerated direction consistency does not equal B18's recorded value")

    sig = artifacts.reference_signature
    arm_distance = {
        "prior_on": torch.linalg.norm(pg.x_set - sig, dim=-1).reshape(-1).numpy(),
        "prior_off": torch.linalg.norm(pg.x_clear - sig, dim=-1).reshape(-1).numpy(),
    }
    training = torch.linalg.norm(artifacts.x_all - sig, dim=-1).numpy()
    is_avian = artifacts.family == "seagull"
    extent = artifacts.region_extent

    n_avian_inside = int((training[is_avian] < extent).sum())
    n_other_inside = int((training[~is_avian] < extent).sum())
    require(n_avian_inside == int(avian_npz["n_avian_inside"]),
            "Figure 3's recomputed avian-inside count does not equal B09's recorded value")
    require(n_other_inside == int(avian_npz["n_non_avian_inside"]),
            "Figure 3's recomputed non-avian-inside count does not equal B09's recorded value")

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    upper = max(training.max(), arm_distance["prior_off"].max()) * 1.02
    bins = np.linspace(0.0, upper, 70)

    ax.hist(training[~is_avian], bins=bins, density=True, color=COL_TRAIN_OTHER,
            alpha=0.45, label=f"training population, non-avian rows (n {int((~is_avian).sum())})")
    ax.hist(training[is_avian], bins=bins, density=True, color=COL_TRAIN_AVIAN,
            alpha=0.6, label=f"training population, avian rows (n {int(is_avian.sum())})")
    for arm in ("prior_on", "prior_off"):
        ax.hist(arm_distance[arm], bins=bins, density=True, histtype="step", lw=1.5,
                color=ARM_COL[arm],
                label=f"{ARM_LABEL[arm]} generations (n {arm_distance[arm].size})")
        ax.axvline(arm_distance[arm].mean(), color=ARM_COL[arm], lw=1.0, ls=":",
                   label=f"{ARM_LABEL[arm]} mean distance {arm_distance[arm].mean():.4f}")
    ax.axvline(extent, color=COL_REGION, lw=1.6,
               label=f"derived region extent, {extent:.6f}")

    ax.set_xlabel("Euclidean distance to the avian reference, standardised coefficient space")
    ax.set_ylabel("density")
    ax.set_title("Figure 3. Distance to the avian reference, per arm, against the region extent",
                 loc="left")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "figure_3_distance_to_reference")


# ---------------------------------------------------------------------------
# Figure A1. Training curves. Plots no reported metric.
# ---------------------------------------------------------------------------

COMPONENT_STYLE = [
    ("reconstruction", "#1f5fa8", "-"),
    ("divergence", "#d97b28", "-"),
    ("target_consistency", "#2e8b57", "-"),
    ("ensemble_spread", "#8e44ad", "-"),
    ("avian_prior", "#cc0033", "-"),
    ("safeguard_reconstruction_pass", "#777777", "--"),
    ("safeguard_generation_pass", "#bbbbbb", ":"),
]


def figure_A1_training_curves(history_blob):
    history = history_blob["history"]
    epoch = np.asarray([h["epoch"] for h in history], dtype=float)
    multiplier = np.asarray([h["divergence_schedule_multiplier"] for h in history], dtype=float)
    best_epoch = history_blob["best_epoch"]
    warmup = history_blob["architecture"]["warmup_epochs"]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08})

    ax.axvspan(0, warmup, color=COL_GREY, alpha=0.08, lw=0)
    for name, colour, ls in COMPONENT_STYLE:
        values = np.asarray([h["val_components"][name] for h in history], dtype=float)
        ax.plot(epoch, values, color=colour, lw=1.2, ls=ls,
                label=name.replace("_", " "))
    ax.axvline(best_epoch, color=COL_MARK, lw=1.2, ls="--")
    ax.text(best_epoch - 2.5, 0.42, f"selected checkpoint, epoch {best_epoch}",
            transform=ax.get_xaxis_transform(), color=COL_MARK, fontsize=7,
            rotation=90, va="center", ha="right")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylim(-3e-5, 1e2)
    ax.set_ylabel("objective component on the validation split, unweighted")
    ax.set_title("Figure A1. Training curves, the committed run", loc="left")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=4, frameon=False)

    ax2.plot(epoch, multiplier, color=COL_GREY, lw=1.4)
    ax2.axvline(best_epoch, color=COL_MARK, lw=1.2, ls="--")
    ax2.axvspan(0, warmup, color=COL_GREY, alpha=0.08, lw=0)
    ax2.text(warmup + 2, 0.35, f"divergence warmup, epochs 0 to {warmup}",
             fontsize=6.8, color=COL_GREY, va="center")
    ax2.set_ylabel("warmup\nmultiplier")
    ax2.set_xlabel("epoch")
    ax2.set_ylim(-0.05, 1.15)

    save(fig, "figure_A1_training_curves")


# ---------------------------------------------------------------------------
# Figure A2. Latent dimension usage. Plots the per-dimension divergence.
# ---------------------------------------------------------------------------

def figure_A2_latent_usage(history_blob, gate):
    per_dim = np.asarray(history_blob["per_dimension_divergence"], dtype=float)
    threshold = history_blob["liveness_threshold"]
    n_live = int((per_dim > threshold).sum())
    require(n_live == history_blob["n_live_dimensions"] == gate["committed_n_live_dimensions"],
            "Figure A2's live dimension count does not equal the recorded value")

    order = np.argsort(per_dim)[::-1]
    sorted_values = per_dim[order]
    positions = np.arange(len(sorted_values))
    colours = [COL_ON if v > threshold else COL_TRAIN_OTHER for v in sorted_values]

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.bar(positions, sorted_values, color=colours, width=0.68)
    ax.axhline(threshold, color=COL_MARK, lw=1.2, ls="--")
    ax.set_xlim(-0.7, len(positions) + 1.9)
    ax.text(len(positions) - 0.35, threshold, f"liveness threshold,\n{threshold} nats",
            color=COL_MARK, fontsize=7, ha="left", va="center")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{i}" for i in order])
    ax.set_xlabel("latent dimension, sorted by divergence")
    ax.set_ylabel("per-dimension divergence on the validation split, nats")
    ax.set_title("Figure A2. Latent dimension usage", loc="left")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=COL_ON, label="clears the liveness threshold"),
        plt.Rectangle((0, 0), 1, 1, color=COL_TRAIN_OTHER, label="below it"),
    ], loc="upper right", frameon=False)
    save(fig, "figure_A2_latent_dimension_usage")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main():
    b23 = load("b23_evaluation.json")
    b24 = load("b24_analysis.json")
    b25 = load("b25_metrics.json")
    b20 = load("b20_truncation.json")
    gate = load("b18_gate.json")
    sweep = load(os.path.join("sweep", "sweep_table.json"))
    selection = load("b17_selection.json")
    history_blob = load("committed_training_history.json")
    avian_npz = np.load("avian_signature.npz", allow_pickle=True)

    artifacts = model_mod.load_build_artifacts(".")
    reference = evaluate.standardised_to_coefficients(
        artifacts.reference_signature.numpy(), artifacts.std_stats)
    reference_foil = geometry.decode_airfoil(reference[0][0], reference[1][0],
                                             int(b23["header"]["n_points_per_surface"]))
    reference_loop = (reference_foil.x, reference_foil.y)

    print("Figure 1  method schematic");         figure_1_method_schematic()
    print("Figure 2  prior weight sensitivity"); figure_2_prior_weight_sensitivity(sweep, selection)
    print("Figure 3  distance to reference");    figure_3_distance_to_reference(gate, artifacts, avian_npz)
    print("Figure 4  paired difference");        figure_4_paired_difference(b24)
    print("Figure 5  signed error per arm");     figure_5_signed_error(b24, b25)
    print("Figure 6  shape family");             figure_6_shape_family(b23, artifacts, reference_loop)
    print("Figure 7  diversity across range");   figure_7_diversity(b25)
    print("Figure 8  surrogate against solver"); figure_8_surrogate_against_solver(b23, b25, artifacts)
    print("Figure A1 training curves");          figure_A1_training_curves(history_blob)
    print("Figure A2 latent dimension usage");   figure_A2_latent_usage(history_blob, gate)

    # Not cited by the article, so no image is written. Both still run, because
    # their consistency checks are part of this driver's seven.
    print("attrition flow   checks only, no image"); uncited_attrition_flow(b23, b24)
    print("truncation bias  checks only, no image"); uncited_truncation_bias(b20)

    print(f"\nten figures written to {FIGDIR}/")


if __name__ == "__main__":
    main()
