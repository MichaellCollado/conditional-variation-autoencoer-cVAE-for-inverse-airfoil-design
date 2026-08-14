"""The population, the labels, the conditioning array and the split.

Holds the seed derivation rule, the bounded perturbation sampler and its width
selection, the population build, the label normalisation, the flag draw and
its balance tests, the conditioning assembly, and the stratified split.

Draws are constructed as a seed coefficient plus or minus a width, so they
cannot land outside the interval by construction. The plausibility filter in
geometry.py is applied afterwards, as a separate second-stage rejection on top
of an already bounded draw.

rng_for carries the seed derivation rule. Every random stream anywhere in the
build derives from one base seed by that one offset rule.

assemble_conditioning writes the layout the article gives in Table 3: column 0
the normalised target, columns 1 to 20 the standardised avian signature where
the flag is set and a zero block where it is clear, and the final column the
flag itself, 1.0 when set. It takes the flag draw as an argument and opens no
file.

ks_two_sample is the two-sample Kolmogorov-Smirnov statistic, written out by
hand because no scipy is installed and none should be.

Called by model.py, run_b16_weight_sweep.py, run_b24_analysis.py,
run_b25_metrics.py, run_figures.py and check_b24_studentised_secondaries.py.

Public API
    rng_for
    bounded_perturbation, one_draw, Draw, select_perturbation_width,
        WidthCandidateResult, build_population, Population, PopulationRow
    fit_seeds, SeedFit, nearest_seed_family,
        shape_thickness_and_camber_magnitude, seed_population_camber_range
    derive_label_normalization, normalize_label, denormalize_label,
        LabelNormalization
    draw_flag, test_flag_independence, IndependenceResult, cohens_d,
        ks_two_sample, decile_coverage, DecileCoverage
    assemble_conditioning, ConditioningArtifact
    split_dataset, Split
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import geometry


# ---------------------------------------------------------------------------
# Base seed and the offset rule. Set here at first need -- see module
# docstring.
# ---------------------------------------------------------------------------

BASE_SEED = 20260806  # the calendar date this was set, an
                       # arbitrary but stated and reproducible choice.


def rng_for(build_step_number: int, substream: int = 0) -> np.random.Generator:
    """One documented offset rule for every RNG stream in this build."""
    return np.random.default_rng(BASE_SEED + 1000 * build_step_number + substream)


# ---------------------------------------------------------------------------
# Bounded perturbation. The bound itself: draws are constructed inside
# [-width, +width] by the construction of the draw, not by rejecting draws
# that land outside it after the fact.
# ---------------------------------------------------------------------------

def bounded_perturbation(rng: np.random.Generator, coefficients: np.ndarray,
                          width: float) -> np.ndarray:
    """seed_coefficient +/- width, per coefficient, independently. Every
    entry of the result satisfies abs(result_i - coefficients_i) <= width
    by construction; there is nothing downstream of this call that could
    make that untrue."""
    if width < 0:
        raise ValueError("width must be >= 0")
    delta = rng.uniform(-width, width, size=coefficients.shape)
    return coefficients + delta


# ---------------------------------------------------------------------------
# Measuring a shape's thickness and camber magnitude, on the same interior
# grid B03's plausibility filter already uses -- so "spans a stated
# camber and thickness range" is measured consistently with what the
# filter itself enforces, not on some other grid that could disagree with it.
# ---------------------------------------------------------------------------

def shape_thickness_and_camber_magnitude(upper: geometry.Surface, lower: geometry.Surface,
                                          bounds: geometry.PlausibilityBounds
                                          ) -> Tuple[float, float]:
    grid = geometry._interior_grid(bounds.n_grid_points, bounds.edge_margin)
    thickness, camber, _, _ = geometry.measure_thickness_and_camber(upper, lower, grid)
    return float(np.max(thickness)), float(np.max(np.abs(camber)))


def seed_population_camber_range(seeds: Sequence[geometry.SeedGeometry],
                                  bounds: geometry.PlausibilityBounds) -> Tuple[float, float]:
    """The five seeds' own camber-magnitude range, measured directly from
    their raw coordinates (no CST round trip), the same way B03 measured
    the seed population's own thickness range. This is the target range
    the sampled population's camber must span; it is not invented, it is
    read off the seed files."""
    magnitudes = []
    for seed in seeds:
        _, cam = shape_thickness_and_camber_magnitude(seed.upper, seed.lower, bounds)
        magnitudes.append(cam)
    return min(magnitudes), max(magnitudes)


# ---------------------------------------------------------------------------
# Per-seed CST fit, at the committed order.
# ---------------------------------------------------------------------------

@dataclass
class SeedFit:
    seed: geometry.SeedGeometry
    family: str
    order: int
    upper_coefficients: np.ndarray
    lower_coefficients: np.ndarray


def fit_seeds(seeds: Sequence[geometry.SeedGeometry], order: int) -> List[SeedFit]:
    fits = []
    for seed in seeds:
        fit_u = geometry.fit_surface(seed.upper, order)
        fit_l = geometry.fit_surface(seed.lower, order)
        # Family label is the seed FILE's stem, not the descriptive name on
        # line 1 of the .dat file -- these differ for at least the seagull
        # seed (B05's partial-run script records the same finding).
        family = Path(seed.path).stem
        fits.append(SeedFit(seed=seed, family=family, order=order,
                             upper_coefficients=fit_u.coefficients,
                             lower_coefficients=fit_l.coefficients))
    return fits


# ---------------------------------------------------------------------------
# One perturb-decode-filter draw.
# ---------------------------------------------------------------------------

@dataclass
class Draw:
    accepted: bool
    reason: str
    upper_coefficients: np.ndarray
    lower_coefficients: np.ndarray
    max_thickness: float = float("nan")
    camber_magnitude: float = float("nan")


def one_draw(seed_fit: SeedFit, width: float, bounds: geometry.PlausibilityBounds,
             rng: np.random.Generator, n_points_per_surface: int = 160) -> Draw:
    upper_c = bounded_perturbation(rng, seed_fit.upper_coefficients, width)
    lower_c = bounded_perturbation(rng, seed_fit.lower_coefficients, width)
    decoded = geometry.decode_airfoil(upper_c, lower_c,
                                       n_points_per_surface=n_points_per_surface)
    verdict = geometry.plausibility_filter(decoded.upper, decoded.lower, bounds)
    if not verdict.accepted:
        return Draw(accepted=False, reason=verdict.reason,
                    upper_coefficients=upper_c, lower_coefficients=lower_c)
    max_t, cam = shape_thickness_and_camber_magnitude(decoded.upper, decoded.lower, bounds)
    return Draw(accepted=True, reason="ok",
                upper_coefficients=upper_c, lower_coefficients=lower_c,
                max_thickness=max_t, camber_magnitude=cam)


# ---------------------------------------------------------------------------
# the committed specification's selection rule: smallest width whose accepted, pooled population
# spans a stated camber-and-thickness range.
# ---------------------------------------------------------------------------

@dataclass
class WidthCandidateResult:
    width: float
    n_trials: int
    n_accepted: int
    achieved_thickness_range: Tuple[float, float]
    achieved_camber_range: Tuple[float, float]
    covers_target: bool


def select_perturbation_width(
    seed_fits: Sequence[SeedFit],
    bounds: geometry.PlausibilityBounds,
    candidate_widths: Sequence[float],
    target_thickness_range: Tuple[float, float],
    target_camber_range: Tuple[float, float],
    trials_per_seed: int,
    coverage_tolerance: float,
    rng: np.random.Generator,
) -> Tuple[float, List[WidthCandidateResult]]:
    """Coverage rule, stated before running: at a candidate width, pool
    every accepted draw across all five seeds. The width covers the target
    if the pooled achieved thickness range reaches within
    coverage_tolerance * (target span) of both target edges, and likewise
    for camber magnitude. candidate_widths must be given in ascending
    order; the first one that covers both target ranges is chosen."""
    t_lo, t_hi = target_thickness_range
    c_lo, c_hi = target_camber_range
    t_span = t_hi - t_lo
    c_span = c_hi - c_lo

    results = []
    for width in candidate_widths:
        thicknesses = []
        cambers = []
        n_trials = 0
        n_accepted = 0
        for seed_fit in seed_fits:
            for _ in range(trials_per_seed):
                draw = one_draw(seed_fit, width, bounds, rng)
                n_trials += 1
                if draw.accepted:
                    n_accepted += 1
                    thicknesses.append(draw.max_thickness)
                    cambers.append(draw.camber_magnitude)

        if thicknesses:
            achieved_t = (min(thicknesses), max(thicknesses))
            achieved_c = (min(cambers), max(cambers))
            covers = (
                achieved_t[0] <= t_lo + coverage_tolerance * t_span
                and achieved_t[1] >= t_hi - coverage_tolerance * t_span
                and achieved_c[0] <= c_lo + coverage_tolerance * c_span
                and achieved_c[1] >= c_hi - coverage_tolerance * c_span
            )
        else:
            achieved_t = (float("nan"), float("nan"))
            achieved_c = (float("nan"), float("nan"))
            covers = False

        results.append(WidthCandidateResult(
            width=width, n_trials=n_trials, n_accepted=n_accepted,
            achieved_thickness_range=achieved_t, achieved_camber_range=achieved_c,
            covers_target=covers,
        ))
        if covers:
            return width, results

    raise ValueError(
        "No candidate width in the searched range produced an accepted population "
        "spanning the target camber-and-thickness range. Extend candidate_widths."
    )


# ---------------------------------------------------------------------------
# Building the committed population, at the chosen width and the stated
# per-seed count.
# ---------------------------------------------------------------------------

@dataclass
class PopulationRow:
    family: str
    is_seed: bool
    upper_coefficients: np.ndarray
    lower_coefficients: np.ndarray


@dataclass
class Population:
    rows: List[PopulationRow]
    accepted_count: Dict[str, int]
    rejected_count: Dict[str, int]
    width: float
    per_seed_count: int


def build_population(seed_fits: Sequence[SeedFit], width: float, per_seed_count: int,
                      bounds: geometry.PlausibilityBounds, rng: np.random.Generator,
                      max_trials_per_seed: int) -> Population:
    rows: List[PopulationRow] = []
    accepted_count: Dict[str, int] = {}
    rejected_count: Dict[str, int] = {}

    for seed_fit in seed_fits:
        family = seed_fit.family
        rows.append(PopulationRow(
            family=family, is_seed=True,
            upper_coefficients=seed_fit.upper_coefficients.copy(),
            lower_coefficients=seed_fit.lower_coefficients.copy(),
        ))
        accepted_count.setdefault(family, 0)
        rejected_count.setdefault(family, 0)

        n_accepted_this_seed = 0
        n_trials_this_seed = 0
        while n_accepted_this_seed < per_seed_count:
            if n_trials_this_seed >= max_trials_per_seed:
                raise RuntimeError(
                    f"family {family!r}: reached max_trials_per_seed="
                    f"{max_trials_per_seed} with only {n_accepted_this_seed} of "
                    f"{per_seed_count} accepted. The committed width may be too "
                    f"tight, or max_trials_per_seed too low; not looping forever."
                )
            draw = one_draw(seed_fit, width, bounds, rng)
            n_trials_this_seed += 1
            if draw.accepted:
                rows.append(PopulationRow(
                    family=family, is_seed=False,
                    upper_coefficients=draw.upper_coefficients,
                    lower_coefficients=draw.lower_coefficients,
                ))
                accepted_count[family] += 1
                n_accepted_this_seed += 1
            else:
                rejected_count[family] += 1

    return Population(rows=rows, accepted_count=accepted_count,
                       rejected_count=rejected_count, width=width,
                       per_seed_count=per_seed_count)


def nearest_seed_family(row: PopulationRow, seed_fits: Sequence[SeedFit]) -> str:
    """Nearest seed by Euclidean distance in coefficient space (upper and
    lower concatenated). Used only by the falsification check, to compare
    against the family label the sampler actually recorded."""
    row_vec = np.concatenate([row.upper_coefficients, row.lower_coefficients])
    best_family, best_dist = None, float("inf")
    for seed_fit in seed_fits:
        seed_vec = np.concatenate([seed_fit.upper_coefficients, seed_fit.lower_coefficients])
        dist = float(np.linalg.norm(row_vec - seed_vec))
        if dist < best_dist:
            best_dist = dist
            best_family = seed_fit.family
    return best_family


# ---------------------------------------------------------------------------
# B12. Fix the split. One split, used by every model trained in
# this build (the surrogate at B13, the generative model at B15). Not
# assigned to a named file by the target file structure table; kept here
# because it operates directly on the labelled dataset B07 produced and
# because B08 (also in this file) is the first consumer of its output.
#
# Open parameter set here: the validation fraction (the build plan: stated value).
# No selection rule is named for it anywhere in the reviewed plan text --
# unlike the perturbation width (B06) or the timeout (B05), it does not
# carry a "smallest/largest value such that..." procedure. Treated as a
# stated value the same way B04's solver operating point was: a disclosed,
# justified figure, not an optimised one. 0.20 is used: a conventional,
# round fraction with no tighter empirical basis stated anywhere in the
# corpus, disclosed as such rather than presented as derived.
# ---------------------------------------------------------------------------

@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    validation_fraction: float
    per_family_validation_count: Dict[str, int]
    per_family_total_count: Dict[str, int]


def split_dataset(family: np.ndarray, validation_fraction: float,
                   rng: np.random.Generator) -> Split:
    """Shuffle within each family under one recorded seed; take the stated
    validation fraction from each family (stratified by family, not a
    single pooled shuffle, so a family cannot be reduced to zero rows in
    validation by chance the way an unstratified split could).

    `family` is a 1-D array of family labels, one per row of the dataset
    (row i of `family` corresponds to row i of dataset.npz's arrays).
    Returns two disjoint, sorted index arrays into that same row axis whose
    union is every row exactly once.
    """
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be in (0, 1)")

    train_idx: List[np.ndarray] = []
    val_idx: List[np.ndarray] = []
    per_family_validation_count: Dict[str, int] = {}
    per_family_total_count: Dict[str, int] = {}

    for fam in np.unique(family):
        idx = np.where(family == fam)[0]
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * validation_fraction))
        val_idx.append(shuffled[:n_val])
        train_idx.append(shuffled[n_val:])
        per_family_validation_count[str(fam)] = int(n_val)
        per_family_total_count[str(fam)] = int(len(shuffled))

    return Split(
        train_idx=np.sort(np.concatenate(train_idx)),
        val_idx=np.sort(np.concatenate(val_idx)),
        validation_fraction=validation_fraction,
        per_family_validation_count=per_family_validation_count,
        per_family_total_count=per_family_total_count,
    )


# ---------------------------------------------------------------------------
# B08. Label normalisation. Kept in this file, not geometry.py: this is a
# property of the label/target column, not of the shared shape
# representation geometry.py owns (geometry.py's own "standardisation"
# entry in the file structure table is for the CST coefficient columns,
# used by the dataset build, the prior's distance computation, and the
# evaluation path alike -- the label only ever appears here and in
# model.py's target-consistency term, so it does not need that sharing).
#
# Open parameter set here (jointly with the geometry standardisation stats
# below): the row set the label min/max are computed over (the build plan: the committed specification,
# author's choice, disclosed). See derive_label_normalization's caller in
# the B08 driver for the actual row-set choice and its justification;
# nothing here decides the row set itself.
# ---------------------------------------------------------------------------

@dataclass
class LabelNormalization:
    label_min: float
    label_max: float
    row_set_description: str


def derive_label_normalization(labels: np.ndarray, row_set_description: str
                                ) -> LabelNormalization:
    return LabelNormalization(
        label_min=float(labels.min()),
        label_max=float(labels.max()),
        row_set_description=row_set_description,
    )


def normalize_label(labels, norm: LabelNormalization):
    span = norm.label_max - norm.label_min
    if span == 0:
        raise ValueError("label_max equals label_min; normalisation is undefined")
    return (labels - norm.label_min) / span


def denormalize_label(normalized, norm: LabelNormalization):
    span = norm.label_max - norm.label_min
    return normalized * span + norm.label_min


# ---------------------------------------------------------------------------
# B11. Test the flag assignment against the label. No the build plan
# row, no specification entry -- the build plan states plainly that this step
# exists on the locked design's authority alone: the prior-on/off ablation is
# clean only if the flag gating the avian term is independent of the
# label. If it were not, the ablation would be measuring that correlation,
# not the prior term's effect. Placed here rather than in prior.py because
# it is a property of the conditioning array's flag column, which this
# file assembles at B10; prior.py owns the avian reference and the penalty
# term, not the flag itself.
#
# This step is not gated: run it and record the result whichever way it
# falls. It is not a pass/fail gate
# the way B18's mechanism gate is.
#
# The flag-clear fraction is drawn here, independently of both label and
# geometry (the rng stream below never reads either), and this draw is
# what B10 consumes -- B10 does not redraw it.
# ---------------------------------------------------------------------------

def draw_flag(rng: np.random.Generator, n_rows: int, flag_clear_fraction: float
              ) -> np.ndarray:
    """Boolean array, True where the flag is CLEAR. Independent Bernoulli
    draw per row at the given fraction; the draw reads nothing about the
    row's label or geometry, so independence holds by construction of the
    draw -- the tests below check that this construction was not
    undermined somewhere else (e.g. a row order that secretly correlates
    with the label)."""
    if not (0.0 < flag_clear_fraction < 1.0):
        raise ValueError("flag_clear_fraction must be in (0, 1)")
    return rng.uniform(size=n_rows) < flag_clear_fraction


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference (a - b), pooled sample standard
    deviation (ddof=1). The effect-size half of B11's two-part comparison."""
    n_a, n_b = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic, hand-implemented: the
    locked stack is numpy and torch, and no scipy is installed in this
    environment, so this does not reach for scipy.stats.ks_2samp. Returns
    (D, the asymptotic two-sided critical value at alpha=0.05, using
    Kolmogorov's standard c(0.05)=1.35810 formula), so the statistic can be
    read against a stated threshold without a p-value routine. The
    distribution-test half of B11's two-part comparison."""
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    all_vals = np.concatenate([a_sorted, b_sorted])
    all_vals.sort()
    cdf_a = np.searchsorted(a_sorted, all_vals, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, all_vals, side="right") / len(b_sorted)
    d_stat = float(np.max(np.abs(cdf_a - cdf_b)))
    n_a, n_b = len(a_sorted), len(b_sorted)
    critical_at_05 = 1.35810 * np.sqrt((n_a + n_b) / (n_a * n_b))
    return d_stat, float(critical_at_05)


@dataclass
class DecileCoverage:
    n_well_populated_deciles: int
    n_well_populated_deciles_missing_flag_clear: int
    per_decile_counts: List[Tuple[int, int]]  # (total_in_decile, flag_clear_in_decile)


def decile_coverage(labels: np.ndarray, flag_clear: np.ndarray,
                     min_decile_population: int) -> DecileCoverage:
    """Deciles of the label distribution (quantile-based edges, so each
    decile holds close to n/10 rows by construction). A decile counts as
    'well populated' if it holds at least min_decile_population rows
    total. For each well-populated decile, record whether at least one
    flag-clear row falls in it -- this is the third of B11's three checks,
    the one the build plan's B10 entry cites by name ('flag-clear rows appear in
    every well populated target decile')."""
    edges = np.quantile(labels, np.linspace(0.0, 1.0, 11))
    edges = edges.copy()
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.digitize(labels, edges[1:-1], right=True)

    per_decile = []
    n_well = 0
    n_missing = 0
    for d in range(10):
        in_decile = bin_idx == d
        total = int(in_decile.sum())
        clear_in_decile = int((in_decile & flag_clear).sum())
        per_decile.append((total, clear_in_decile))
        if total >= min_decile_population:
            n_well += 1
            if clear_in_decile == 0:
                n_missing += 1
    return DecileCoverage(n_well_populated_deciles=n_well,
                           n_well_populated_deciles_missing_flag_clear=n_missing,
                           per_decile_counts=per_decile)


@dataclass
class IndependenceResult:
    effect_size_cohens_d: float
    ks_statistic: float
    ks_critical_at_05: float
    ks_exceeds_critical: bool
    coverage: DecileCoverage


def test_flag_independence(labels: np.ndarray, flag_clear: np.ndarray,
                            min_decile_population: int) -> IndependenceResult:
    """The three results B11's logic names, computed together: effect
    size, distribution test, decile coverage. Oriented flag-set minus
    flag-clear on the effect size and the KS statistic; the sign has no
    privileged direction here since, unlike B24's paired difference, there
    is no "favours the prior" orientation for a pre-training independence
    check."""
    clear_labels = labels[flag_clear]
    set_labels = labels[~flag_clear]
    d = cohens_d(set_labels, clear_labels)
    ks_stat, ks_crit = ks_two_sample(set_labels, clear_labels)
    coverage = decile_coverage(labels, flag_clear, min_decile_population)
    return IndependenceResult(
        effect_size_cohens_d=d, ks_statistic=ks_stat, ks_critical_at_05=ks_crit,
        ks_exceeds_critical=ks_stat > ks_crit, coverage=coverage,
    )


# ---------------------------------------------------------------------------
# B10. Assemble the conditioning representation. The null representation
# is a zero block plus a separate indicator flag column. No sentinel constant, no gating
# layer -- both were retired (a sentinel value could be confused with a
# real standardised coordinate; a gating layer is exactly the composite
# machinery the committed specification would require justifying, which the zero-block choice
# avoids needing at all). The flag itself is B11's committed draw, read
# here, not redrawn.
# ---------------------------------------------------------------------------

@dataclass
class ConditioningArtifact:
    array: np.ndarray  # (n_rows, 1 + n_signature_columns + 1)
    n_signature_columns: int
    realized_flag_clear_fraction: float


def assemble_conditioning(normalized_labels: np.ndarray, flag_clear: np.ndarray,
                           standardized_signature: np.ndarray) -> ConditioningArtifact:
    """For every row: column 0 is the normalised target (the row's own
    normalised label). Columns 1..1+n_sig are the standardised avian
    signature where the flag is set, and zero where the flag is clear.
    The final column is the flag itself: 1.0 where set (signature block
    present), 0.0 where clear (zero block present) -- the same sense B14's
    'flag gated avian term' reads, so a downstream reader does not have to
    track an inverted convention across files."""
    n_rows = normalized_labels.shape[0]
    n_sig = standardized_signature.shape[0]
    if flag_clear.shape[0] != n_rows:
        raise ValueError("flag_clear must have one entry per row")

    array = np.zeros((n_rows, 1 + n_sig + 1), dtype=float)
    array[:, 0] = normalized_labels
    flag_set = ~flag_clear
    array[flag_set, 1:1 + n_sig] = standardized_signature
    array[:, 1 + n_sig] = flag_set.astype(float)

    return ConditioningArtifact(
        array=array, n_signature_columns=n_sig,
        realized_flag_clear_fraction=float(flag_clear.mean()),
    )
