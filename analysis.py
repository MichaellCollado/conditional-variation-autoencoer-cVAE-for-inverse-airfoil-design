"""Every reported statistic and every reported count. Nothing else computes one.

Holds the truncation analysis that sets the admission rule, the admission rule
itself, the target satisfaction error and the paired differences, the primary
outcome and its cluster-resampled interval, every secondary, the declared
sensitivity check, and the distribution and Monte Carlo diagnostics.

Nothing in this file decides a threshold. The truncation tolerance, the primary
statistic, the interval level, the resample count, the cluster unit and the
bootstrap method are all read from params.py, all committed before the
evaluation run, and none of them is defined here. So the minimum converged
point count cannot be chosen in this file and justified afterwards.

Target satisfaction error is computed in normalised units throughout. The raw
ratio figure is a derived accompaniment and never the reported quantity.

interval_reading is where an interval is read. A lower bound at or below zero
with an upper bound at or above zero reads as no detected difference.

Called by run_b20_truncation_analysis.py, run_b21_paired_yield.py,
run_b23_paired_evaluation.py, run_b24_analysis.py, run_b25_metrics.py,
run_figures.py and check_b24_studentised_secondaries.py.

Public API
    label_over_first_k, truncation_bias_table, TruncationBiasRow,
        select_minimum_point_count, peak_angle_distribution
    physical_plausibility, PhysicalPlausibility, is_admitted, read_admission,
        ArmAdmission
    normalised_error, PairedData
    cluster_robust_se, wild_cluster_bootstrap_t, WildClusterResult,
        pairs_cluster_bootstrap, PairsClusterResult
    mean_statistic, median_statistic, trimmed_mean, trimmed_mean_statistic,
        win_fraction, win_fraction_statistic, slope_difference_statistic
    ols_slope, pearson_correlation
    order_statistic_quantiles, interpolated_quantiles,
        quantiles_with_convention
    distribution_shape, DistributionShape, proportion_difference,
        ProportionDifference, monte_carlo_error, MonteCarloError
    surrogate_gap, SurrogateGap
    interval_reading
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# B20. Truncation bias against candidate minimum point count. Metric M27.
# ---------------------------------------------------------------------------

def label_over_first_k(polar: np.ndarray, k: int) -> float:
    """max(CL/CD) over the first k converged points of one sweep.

    'First k' is by ascending angle of attack, which is the order the sweep
    was requested in and the order XFOIL's accumulated polar dump carries. A
    partially converged sweep loses whichever angles failed, so its surviving
    points stay in that order with gaps, and its first k are its first k
    surviving points. That is what a truncated sweep actually is in this
    pipeline, and it is what the admission rule counts.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    head = polar[:k]
    return float(np.max(head[:, 1] / head[:, 2]))


@dataclass
class TruncationBiasRow:
    """One candidate minimum point count's bias, across every fully converged
    sweep. All four bias figures are relative to that sweep's own full-sweep
    label, per B20_TRUNCATION_TOLERANCE['quantity']."""
    k: int
    n_sweeps: int
    mean_signed: float          # <= 0 by construction; truncating a max can only lower it
    mean_absolute: float
    upper_percentile_absolute: float
    worst_absolute: float
    n_sweeps_unaffected: int    # sweeps whose peak already lies inside the first k


def truncation_bias_table(polars: Sequence[np.ndarray], candidate_counts: Sequence[int],
                           upper_percentile: float) -> List[TruncationBiasRow]:
    """M27. For each candidate count, compare the maximum over the first k
    points against the maximum over the whole sweep, across every supplied
    sweep.

    The reference is the whole sweep's own maximum, computed once per sweep
    outside the candidate loop. Computing it inside, from the truncated set,
    is the wrong-but-wired failure this step's falsification check exists to
    catch: it would give near-zero bias at every candidate and make any
    threshold look defensible.
    """
    full = np.array([float(np.max(p[:, 1] / p[:, 2])) for p in polars])
    rows: List[TruncationBiasRow] = []
    for k in candidate_counts:
        truncated = np.array([label_over_first_k(p, k) for p in polars])
        relative = (truncated - full) / full
        absolute = np.abs(relative)
        rows.append(TruncationBiasRow(
            k=int(k), n_sweeps=len(polars),
            mean_signed=float(relative.mean()),
            mean_absolute=float(absolute.mean()),
            upper_percentile_absolute=float(np.percentile(absolute, upper_percentile)),
            worst_absolute=float(absolute.max()),
            n_sweeps_unaffected=int((absolute == 0.0).sum()),
        ))
    return rows


def select_minimum_point_count(rows: Sequence[TruncationBiasRow],
                                mean_tolerance: float,
                                upper_percentile_tolerance: float) -> Optional[int]:
    """The smallest candidate count whose mean absolute relative bias and
    upper-percentile absolute relative bias both sit inside the stated
    tolerances. Returns None if no candidate does, which the caller reports
    as the finding rather than relaxing anything.

    This is the whole of the selection. It reads the tolerance it is handed
    and the table it is handed, and it has no other input.
    """
    for row in sorted(rows, key=lambda r: r.k):
        if (row.mean_absolute <= mean_tolerance
                and row.upper_percentile_absolute <= upper_percentile_tolerance):
            return row.k
    return None


def peak_angle_distribution(polars: Sequence[np.ndarray]) -> np.ndarray:
    """The angle of attack at which each sweep attains its own max(CL/CD).
    Recorded alongside the bias table per B20's own logic text, because it is
    what the bias table is a consequence of: a rule admitting k points is
    admitting the sweeps whose peak lies inside the first k."""
    return np.array([float(p[int(np.argmax(p[:, 1] / p[:, 2])), 0]) for p in polars])


# ---------------------------------------------------------------------------
# the committed specification. Physical plausibility of purportedly converged solver output.
#
# Distinct from the committed specification's PRE-solver geometric filter, which B03 builds and which
# rejects a shape before the solver ever sees it. This one runs AFTER
# convergence and asks whether a point XFOIL reported as converged is
# physically meaningful.
#
# The criterion is stated in params.PARAMS['plausibility_filter_criterion'],
# together with the difference from the closest source that the committed specification requires.
# In short: this build tests what XFOIL's accumulated polar dump exposes,
# which is drag and transition location, and the closest source tests solver
# internals the dump does not carry.
# ---------------------------------------------------------------------------

@dataclass
class PhysicalPlausibility:
    n_points: int
    n_implausible: int
    keep: np.ndarray            # boolean mask over the polar's rows
    reasons: List[str]          # one per implausible point, in row order


def physical_plausibility(polar: Optional[np.ndarray]) -> PhysicalPlausibility:
    """Per converged point, require drag strictly positive and both
    transition locations inside [0, 1] chord.

    Drag at or below zero from a viscous solve is not a small drag, it is a
    non-physical result, and it would produce an unbounded or negative
    lift-to-drag ratio that the label's maximum would then select. Both
    failures land in the label rather than beside it, which is why the test
    is on the points and not on the finished label.

    A transition location outside the chord is XFOIL reporting a boundary
    layer state off the surface it was solving on. Exactly 0.0 and exactly
    1.0 are both admitted: 1.0 is how the dump records no transition, with
    laminar flow to the trailing edge, which is an ordinary result at these
    Reynolds numbers and not a fault.
    """
    if polar is None or len(polar) == 0:
        return PhysicalPlausibility(0, 0, np.zeros(0, dtype=bool), [])

    cd = polar[:, 2]
    top_xtr, bot_xtr = polar[:, 5], polar[:, 6]
    bad_cd = cd <= 0.0
    bad_xtr = (top_xtr < 0.0) | (top_xtr > 1.0) | (bot_xtr < 0.0) | (bot_xtr > 1.0)
    keep = ~(bad_cd | bad_xtr)

    reasons = []
    for i in np.flatnonzero(~keep):
        why = []
        if bad_cd[i]:
            why.append(f"CD={cd[i]:.6g} <= 0")
        if bad_xtr[i]:
            why.append(f"transition outside [0,1] (top {top_xtr[i]:.4f}, "
                       f"bot {bot_xtr[i]:.4f})")
        reasons.append(f"alpha={polar[i, 0]:.3f}: " + "; ".join(why))

    return PhysicalPlausibility(
        n_points=len(polar), n_implausible=int((~keep).sum()),
        keep=keep, reasons=reasons,
    )


# ---------------------------------------------------------------------------
# The admission rule. Applied to a record, at B21, B23 and B24 alike.
# ---------------------------------------------------------------------------

def is_admitted(label: Optional[float], n_converged: int,
                 minimum_converged_points: int) -> bool:
    """B24's admission rule in its scalar form: a record is admitted only if
    it has a label AND cleared the committed minimum converged point count.
    Both are read explicitly and neither defaults.

    evaluate.admitted() is the same rule applied to a ShapeRecord, and calls
    the same comparison. One rule, two callers, no second definition.
    """
    if label is None:
        return False
    return int(n_converged) >= int(minimum_converged_points)


# ===========================================================================
# B24. The analysis. 
#
# Everything below produces a reported number. Nothing below chooses one.
# The primary statistic, the interval level, the resample count, the cluster
# unit and the bootstrap method were all committed before B23 ran and are
# read from params.PARAMS by the driver, not restated here.
# ===========================================================================


# ---------------------------------------------------------------------------
# Admission and pairing.
# ---------------------------------------------------------------------------

@dataclass
class ArmAdmission:
    """One arm of one record, as the admission rule sees it."""
    admitted: bool
    field_present: bool
    label: Optional[float]
    n_usable: int
    recomputed: bool                # the rule recomputed from label and n_usable
    agrees: Optional[bool]          # stored field against the recomputation


def read_admission(arm: dict, minimum_converged_points: int) -> ArmAdmission:
    """Read the admission field EXPLICITLY, with no default.

    The pre-registration: 'The admission field is read explicitly, with no default
    value. A record missing the field is excluded and counted, not silently
    admitted.' A dict.get with a default is exactly the silent admission that
    sentence forbids, so the key's absence is a distinct, counted outcome and
    not a False that reads like an ordinary rejection.

    The rule is also recomputed from the stored label and usable point count,
    and the agreement is recorded. That is a cross-check on B23's stored
    field and not a second admission rule. When the two disagree the STORED
    field governs, because it is the field the pre-registration names, and the
    disagreement is reported.
    """
    present = "admitted" in arm
    stored = bool(arm["admitted"]) if present else False
    label = arm.get("label", None)
    n_usable = int(arm.get("n_usable", 0))
    recomputed = is_admitted(label, n_usable, minimum_converged_points)
    return ArmAdmission(
        admitted=stored, field_present=present, label=label, n_usable=n_usable,
        recomputed=recomputed, agrees=(stored == recomputed) if present else None,
    )


@dataclass
class PairedData:
    """The analysed pairs, and nothing else. One row per pair in which BOTH
    members were admitted, which is the complete-case pairing the pre-registration
    commits. A pair with one surviving member contributes nothing here and its
    surviving member appears in no array below.

    Every error is in NORMALISED units. raw_span is carried so the
    driver can state the accompanying raw figure without the normalisation
    being recomputed anywhere else.
    """
    target: np.ndarray              # requested target, normalised
    target_index: np.ndarray        # cluster id
    sample_index: np.ndarray
    achieved_on: np.ndarray         # achieved efficiency, normalised
    achieved_off: np.ndarray
    error_on: np.ndarray            # |achieved - requested|, normalised
    error_off: np.ndarray
    difference: np.ndarray          # error_off - error_on; positive favours the prior
    raw_span: float

    def __len__(self) -> int:
        return len(self.difference)

    @property
    def n_clusters(self) -> int:
        return len(np.unique(self.target_index))

    def cluster_members(self) -> List[np.ndarray]:
        """Row positions belonging to each cluster, in cluster order. This is
        the unit the bootstrap resamples. A cluster is a requested target and
        it is drawn whole."""
        return [np.flatnonzero(self.target_index == g)
                for g in np.unique(self.target_index)]


def normalised_error(achieved_raw: float, requested_normalised: float,
                     label_min: float, label_max: float) -> Tuple[float, float]:
    """Target satisfaction error in normalised units, with the achieved value
    normalised by the same B08 artifact the model was trained against.

    Returns (achieved_normalised, absolute_error_normalised).

    The requested target is already normalised: the committed band is stated
    in normalised units and the conditioning column the model reads is the
    normalised target. The achieved value arrives from the solver in raw
    max(CL/CD), so it is the one that has to be carried across. Comparing a
    raw achieved value against a normalised request, or reporting the
    difference in raw ratio units, is what the committed specification rules out.
    """
    span = label_max - label_min
    if span == 0:
        raise ValueError("label_max equals label_min; normalisation is undefined")
    achieved = (achieved_raw - label_min) / span
    return achieved, abs(achieved - requested_normalised)


# ---------------------------------------------------------------------------
# The cluster-robust standard error, and the wild cluster bootstrap-t.
#
# Cluster unit: the requested target (params slot
# cluster_unit_bootstrap_method). Method: the wild cluster bootstrap-t with
# Rademacher weights, with the unrefined percentile cluster bootstrap reported
# alongside it. Both were committed during the build, before evaluation, per the committed specification.
# ---------------------------------------------------------------------------

def cluster_robust_se(values: np.ndarray, cluster_id: np.ndarray) -> float:
    """The cluster-robust standard error of the MEAN of `values`, clustering
    on `cluster_id`.

    This is the CRVE for a regression on a constant. With X a column of ones,
    V = (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1 collapses to
    sum_g (sum_{i in g} u_i)^2 / n^2, with u the residuals about the mean.

    The finite-sample correction is the usual c = [G/(G-1)] * [(n-1)/(n-K)].
    Here K = 1, so the second factor is exactly one and c = G/(G-1). It is
    written out rather than simplified away, because a reader checking this
    against the source is checking for both factors.
    """
    values = np.asarray(values, dtype=float)
    cluster_id = np.asarray(cluster_id)
    n = len(values)
    residual = values - values.mean()
    groups = np.unique(cluster_id)
    g_count = len(groups)
    sums = np.array([residual[cluster_id == g].sum() for g in groups])
    correction = (g_count / (g_count - 1)) * ((n - 1) / (n - 1))
    variance = correction * float((sums ** 2).sum()) / (n ** 2)
    return float(np.sqrt(variance))


def order_statistic_quantiles(sorted_values: np.ndarray, level: float
                              ) -> Tuple[float, float]:
    """The two-sided interval endpoints as EXACT order statistics.

    The committed resample count is 9999 precisely so that (B + 1) * alpha/2
    is an integer at the committed 95 percent level, which makes the
    percentile indices exact and means no interpolation convention has to be
    chosen or disclosed. That commitment is enforced here rather than assumed:
    a count and level that do not give an integer index raise, instead of
    silently falling back to an interpolating quantile.
    """
    b = len(sorted_values)
    alpha = 1.0 - level
    lower_rank = (b + 1) * alpha / 2.0
    if abs(lower_rank - round(lower_rank)) > 1e-9:
        raise ValueError(
            "(B+1)*alpha/2 = %r is not an integer at B=%d, level=%r; the "
            "committed exact-index convention does not apply"
            % (lower_rank, b, level))
    lower_rank = int(round(lower_rank))
    upper_rank = b + 1 - lower_rank
    return float(sorted_values[lower_rank - 1]), float(sorted_values[upper_rank - 1])


def interpolated_quantiles(sorted_values: np.ndarray, level: float
                           ) -> Tuple[float, float]:
    """Linearly interpolated quantiles, for the one case where the exact-index
    convention cannot apply: the exhaustive enumeration of all 2^G Rademacher
    weight vectors, whose count is fixed by the cluster count and is not free
    to be chosen so that (B + 1) * alpha/2 lands on an integer.

    Also used when a bootstrap has had to discard a replicate, since the
    surviving count is then whatever it is and the exact index no longer
    lands. Which convention was used is recorded on the result and reported,
    rather than being decided silently.
    """
    alpha = 1.0 - level
    return (float(np.quantile(sorted_values, alpha / 2.0, method="linear")),
            float(np.quantile(sorted_values, 1.0 - alpha / 2.0, method="linear")))


def quantiles_with_convention(sorted_values: np.ndarray, level: float
                              ) -> Tuple[float, float, str]:
    """The exact order-statistic endpoints where the count allows it, and the
    interpolated ones where it does not, with the convention named either way.

    The committed resample count was chosen so the exact index lands. A
    discarded replicate moves the count off it, and the honest response is to
    say which convention was used rather than to quietly interpolate or to
    quietly pad the count back up.
    """
    try:
        lo, hi = order_statistic_quantiles(sorted_values, level)
        return lo, hi, "exact order statistic, (B+1)*alpha/2 an integer"
    except ValueError:
        lo, hi = interpolated_quantiles(sorted_values, level)
        return lo, hi, (f"linearly interpolated: {len(sorted_values)} usable "
                        f"replicates does not admit an exact index")


@dataclass
class WildClusterResult:
    """The refined interval on the mean, and the pieces it is built from."""
    point_estimate: float
    cluster_robust_se: float
    t_statistic: float
    lower: float
    upper: float
    t_quantile_lower: float
    t_quantile_upper: float
    n_resamples: int
    n_degenerate: int               # replicates whose bootstrap SE was zero
    p_value_null_imposed: Optional[float]
    n_clusters: int
    distinct_weight_vectors: int
    exhaustive: bool
    quantile_convention: str


def _wild_t_statistics(sums: np.ndarray, n_g: np.ndarray, n: int,
                       weights: np.ndarray, correction: float
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised bootstrap t statistics for a wild cluster bootstrap on the
    mean. `sums` holds the per-cluster sum of whatever residual is being
    resampled and `weights` is (B, G) of +/-1.

    Returns (numerators, bootstrap standard errors). The numerator is the
    bootstrap mean's departure from the value the residuals were formed
    about. The caller decides what that value was, which is the only
    difference between the restricted and the unrestricted form.
    """
    numerator = weights @ sums / n
    star_sums = weights * sums[None, :] - n_g[None, :] * numerator[:, None]
    variance = correction * (star_sums ** 2).sum(axis=1) / (n ** 2)
    return numerator, np.sqrt(variance)


def _rademacher(rng, n_resamples: int, n_clusters: int) -> np.ndarray:
    return rng.integers(0, 2, size=(n_resamples, n_clusters)).astype(float) * 2.0 - 1.0


def _all_rademacher(n_clusters: int) -> np.ndarray:
    """Every distinct Rademacher weight vector. Only called when the caller
    has decided the cluster count makes exhaustion cheap."""
    grid = np.arange(2 ** n_clusters)
    bits = ((grid[:, None] >> np.arange(n_clusters)[None, :]) & 1).astype(float)
    return bits * 2.0 - 1.0


def wild_cluster_bootstrap_t(values: np.ndarray, cluster_id: np.ndarray,
                             rng, n_resamples: int, level: float,
                             null_value: float = 0.0,
                             exhaustive: bool = False) -> WildClusterResult:
    """The committed estimator for the primary outcome's interval.

    Whole clusters are the resampling unit and they are the ONLY resampling
    unit. One Rademacher weight is drawn per cluster and multiplies every
    residual in that cluster together. No pair is drawn, resampled or
    reweighted independently of its cluster anywhere in this function. That is
    the property B24's falsification check exists to test, and it is a
    property of this construction rather than a claim about it.

    The interval is the percentile-t built from UNRESTRICTED residuals, which
    is the bootstrap-t interval Cameron, Gelbach and Miller recommend for a
    confidence interval. The p-value beside it is the RESTRICTED form, with
    the null imposed on the residuals, which is the form they recommend for a
    test. The two differ only in what the residuals are formed about, and both
    are the same wild cluster bootstrap with the same weights.

    One property of Rademacher weights is worth stating rather than
    discovering. t*(-w) = -t*(w) exactly, because flipping every weight flips
    the numerator and leaves the quadratic standard error untouched. The
    bootstrap t distribution is therefore exactly symmetric, and the interval
    is symmetric about the point estimate in standard-error units. That is a
    property of the weight distribution and not of this data.
    """
    values = np.asarray(values, dtype=float)
    cluster_id = np.asarray(cluster_id)
    n = len(values)
    groups = np.unique(cluster_id)
    g_count = len(groups)
    n_g = np.array([float((cluster_id == g).sum()) for g in groups])
    correction = (g_count / (g_count - 1)) * ((n - 1) / (n - 1))

    point = float(values.mean())
    se = cluster_robust_se(values, cluster_id)
    t_stat = (point - null_value) / se if se > 0 else float("nan")

    weights = _all_rademacher(g_count) if exhaustive else _rademacher(
        rng, n_resamples, g_count)

    # Unrestricted: residuals about the point estimate. This gives the interval.
    residual_u = values - point
    sums_u = np.array([residual_u[cluster_id == g].sum() for g in groups])
    num_u, se_u = _wild_t_statistics(sums_u, n_g, n, weights, correction)
    good = se_u > 0
    t_sorted = np.sort(num_u[good] / se_u[good])
    if exhaustive:
        q_lo, q_hi = interpolated_quantiles(t_sorted, level)
        convention = "linearly interpolated: the enumeration's size is not free"
    else:
        q_lo, q_hi, convention = quantiles_with_convention(t_sorted, level)

    # Restricted: residuals about the null. This gives the p-value.
    residual_r = values - null_value
    sums_r = np.array([residual_r[cluster_id == g].sum() for g in groups])
    num_r, se_r = _wild_t_statistics(sums_r, n_g, n, weights, correction)
    good_r = se_r > 0
    t_star_r = num_r[good_r] / se_r[good_r]
    p_value = (float(np.mean(np.abs(t_star_r) >= abs(t_stat)))
               if np.isfinite(t_stat) and len(t_star_r) else None)

    return WildClusterResult(
        point_estimate=point, cluster_robust_se=se, t_statistic=float(t_stat),
        lower=point - q_hi * se, upper=point - q_lo * se,
        t_quantile_lower=q_lo, t_quantile_upper=q_hi,
        n_resamples=int(len(weights)), n_degenerate=int((~good).sum()),
        p_value_null_imposed=p_value, n_clusters=g_count,
        distinct_weight_vectors=2 ** g_count, exhaustive=exhaustive,
        quantile_convention=convention,
    )


# ---------------------------------------------------------------------------
# The pairs-cluster bootstrap. Whole clusters, drawn with replacement.
#
# Two intervals come out of one resample. The unrefined percentile interval,
# which is the estimator committed to be reported alongside the primary. And a
# studentised percentile-t interval, which is the small cluster refinement
# applied to a statistic that has no cluster-robust variance formula. A
# median, a fraction and a slope difference all lack one, so the studentising
# quantity is a delete-one-cluster jackknife standard error instead.
# ---------------------------------------------------------------------------

Statistic = Callable[[np.ndarray], float]


@dataclass
class PairsClusterResult:
    point_estimate: float
    percentile_lower: float
    percentile_upper: float
    jackknife_se: float
    studentised_lower: float
    studentised_upper: float
    t_quantile_lower: float
    t_quantile_upper: float
    n_resamples: int
    n_degenerate: int               # replicates with a zero or undefined jackknife SE
    n_undefined: int                # replicates where the statistic itself was undefined
    bootstrap_mean: float
    n_clusters: int
    percentile_convention: str
    studentised_convention: str


def _cluster_jackknife_se(members: Sequence[np.ndarray], stat: Statistic) -> float:
    """Delete-one-cluster jackknife standard error of `stat`.

    The deleted unit is a whole cluster, matching the resampling unit. A
    jackknife that deleted single pairs would be measuring within-cluster
    variation and would studentise by the wrong quantity, which is the same
    error in a different place as resampling pairs inside drawn clusters.
    """
    g_count = len(members)
    values = np.empty(g_count)
    for j in range(g_count):
        kept = np.concatenate([members[k] for k in range(g_count) if k != j])
        values[j] = stat(kept)
    if not np.all(np.isfinite(values)):
        return float("nan")
    centred = values - values.mean()
    return float(np.sqrt((g_count - 1) / g_count * float((centred ** 2).sum())))


def pairs_cluster_bootstrap(members: Sequence[np.ndarray], stat: Statistic,
                            rng, n_resamples: int, level: float,
                            studentise: bool = True) -> PairsClusterResult:
    """Resample WHOLE clusters with replacement, as many as there are, and
    take the statistic on the concatenation of every pair those clusters
    carry.

    A drawn cluster contributes every one of its pairs, once. Nothing selects
    pairs inside a drawn cluster and nothing fixes the number of pairs per
    cluster, so the resample size varies with which clusters were drawn. That
    variation is the between-cluster component the estimator exists to carry.
    A construction that removed it, by drawing pairs inside the drawn clusters
    or by taking a fixed number of pairs from each, would be the pair
    bootstrap wearing this function's name.
    """
    members = list(members)
    g_count = len(members)
    point = stat(np.concatenate(members))
    jack_se = _cluster_jackknife_se(members, stat) if studentise else float("nan")

    draws = rng.integers(0, g_count, size=(n_resamples, g_count))
    theta = np.empty(n_resamples)
    t_star = np.full(n_resamples, np.nan)
    n_degenerate = 0
    n_undefined = 0

    for b in range(n_resamples):
        picked = [members[c] for c in draws[b]]
        theta_b = stat(np.concatenate(picked))
        theta[b] = theta_b
        if not np.isfinite(theta_b):
            n_undefined += 1
            continue
        if studentise:
            se_b = _cluster_jackknife_se(picked, stat)
            if not np.isfinite(se_b) or se_b <= 0:
                n_degenerate += 1
            else:
                t_star[b] = (theta_b - point) / se_b

    finite_theta = np.sort(theta[np.isfinite(theta)])
    p_lo, p_hi, p_convention = quantiles_with_convention(finite_theta, level)

    if studentise and np.isfinite(jack_se) and jack_se > 0:
        finite_t = np.sort(t_star[np.isfinite(t_star)])
        q_lo, q_hi, s_convention = quantiles_with_convention(finite_t, level)
        s_lo, s_hi = point - q_hi * jack_se, point - q_lo * jack_se
    else:
        q_lo = q_hi = s_lo = s_hi = float("nan")
        s_convention = ("not computed: the statistic has no usable "
                        "delete-one-cluster jackknife standard error")

    return PairsClusterResult(
        point_estimate=float(point),
        percentile_lower=p_lo, percentile_upper=p_hi,
        jackknife_se=float(jack_se),
        studentised_lower=float(s_lo), studentised_upper=float(s_hi),
        t_quantile_lower=float(q_lo), t_quantile_upper=float(q_hi),
        n_resamples=n_resamples, n_degenerate=n_degenerate,
        n_undefined=n_undefined, bootstrap_mean=float(np.mean(finite_theta)),
        n_clusters=g_count,
        percentile_convention=p_convention, studentised_convention=s_convention,
    )


# ---------------------------------------------------------------------------
# The statistics themselves. Each takes an array of pair row positions, so
# every one of them can be handed straight to the cluster bootstrap above
# without a second implementation appearing inside the resampling loop.
# ---------------------------------------------------------------------------

def mean_statistic(values: np.ndarray) -> Statistic:
    return lambda idx: float(values[idx].mean())


def median_statistic(values: np.ndarray) -> Statistic:
    return lambda idx: float(np.median(values[idx]))


def trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
    """Symmetric trimming at `trim_fraction` per tail, then the mean of what
    is left. floor(n * f) values are removed from each end, so the two tails
    lose the same count and the trim is symmetric in count rather than in
    value."""
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must lie in [0, 0.5)")
    ordered = np.sort(np.asarray(values, dtype=float))
    cut = int(np.floor(len(ordered) * trim_fraction))
    kept = ordered[cut:len(ordered) - cut] if cut else ordered
    if len(kept) == 0:
        return float("nan")
    return float(kept.mean())


def trimmed_mean_statistic(values: np.ndarray, trim_fraction: float) -> Statistic:
    return lambda idx: trimmed_mean(values[idx], trim_fraction)


def win_fraction(differences: np.ndarray) -> Tuple[float, int, int]:
    """M07. The count of pairs in which the prior-on arm is closer to its
    target, over the count of NON-TIED pairs.

    A tie is an exact zero difference and it leaves the denominator as well as
    the numerator, which is what a sign test does. Returns
    (fraction, wins, non_tied).
    """
    differences = np.asarray(differences, dtype=float)
    wins = int((differences > 0).sum())
    non_tied = int((differences != 0).sum())
    if non_tied == 0:
        return float("nan"), wins, non_tied
    return wins / non_tied, wins, non_tied


def win_fraction_statistic(differences: np.ndarray) -> Statistic:
    return lambda idx: win_fraction(differences[idx])[0]


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """The least squares slope of y on x. Undefined, and returned as nan, when
    x does not vary: a bootstrap resample that happened to draw one cluster
    eleven times carries one requested target and supports no slope."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    denominator = float((xc ** 2).sum())
    if denominator == 0.0:
        return float("nan")
    return float((xc * (y - y.mean())).sum() / denominator)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(float((xc ** 2).sum()) * float((yc ** 2).sum())))
    if denominator == 0.0:
        return float("nan")
    return float((xc * yc).sum() / denominator)


def slope_difference_statistic(x: np.ndarray, y_on: np.ndarray,
                               y_off: np.ndarray) -> Statistic:
    """S3. The prior-on arm's target tracking slope minus the prior-off arm's,
    both fitted over the SAME matched pairs. The population is matched pairs
    and not admitted records, per M08's own computation note."""
    return lambda idx: ols_slope(x[idx], y_on[idx]) - ols_slope(x[idx], y_off[idx])


# ---------------------------------------------------------------------------
# M04. Distribution shape.
# ---------------------------------------------------------------------------

@dataclass
class DistributionShape:
    n: int
    mean: float
    sample_standard_deviation: float
    sample_skewness: float
    minimum: float
    maximum: float


def distribution_shape(values: np.ndarray) -> DistributionShape:
    """M04, on the sample convention throughout.

    the build plan fixes the formula because the build plan does not. The skewness is the
    third central moment divided by the cube of the sample standard deviation,
    both on the sample convention, so the third central moment carries the
    same (n - 1) denominator the sample variance does and the spread is
    ddof = 1. The superseded implementation computed skewness on the population
    convention while printing a sample spread beside it, which is two
    conventions in adjacent lines.

    The range is reported as a pair, the minimum and the maximum, and not as a
    width.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    centred = values - mean
    s = float(np.sqrt(float((centred ** 2).sum()) / (n - 1)))
    m3 = float((centred ** 3).sum()) / (n - 1)
    return DistributionShape(
        n=n, mean=mean, sample_standard_deviation=s,
        sample_skewness=(m3 / s ** 3) if s > 0 else float("nan"),
        minimum=float(values.min()), maximum=float(values.max()),
    )


# ---------------------------------------------------------------------------
# S4. The difference between the two arms' admission rates.
# ---------------------------------------------------------------------------

@dataclass
class ProportionDifference:
    admitted_a: int
    launched_a: int
    rate_a: float
    admitted_b: int
    launched_b: int
    rate_b: float
    difference: float
    pooled_rate: float
    standard_error: float
    z_statistic: float
    pairs_only_a_admitted: int
    pairs_only_b_admitted: int


def proportion_difference(admitted_a: int, launched_a: int,
                          admitted_b: int, launched_b: int,
                          pairs_only_a_admitted: int, pairs_only_b_admitted: int
                          ) -> ProportionDifference:
    """M19 and secondary outcome S4. A test of the difference between two
    proportions, which is what the pre-registration commits.

    The discordant pair counts are carried alongside because the two arms are
    NOT independent samples here. Every pair shares one latent code, so the two
    proportions are measured on the same units. The two-proportion test is the
    committed one and it is what the z statistic below is. The discordant
    counts are the paired view of the same data, reported beside it so a
    reader can see both rather than being handed one silently.

    pairs_only_a_admitted counts pairs in which arm A was admitted and arm B
    was not, so the pair lost its B member. The naming says which arm SURVIVED
    rather than which was lost, because the other convention inverts under
    reading and the two are one off-by-one apart in a table.

    M19's caution travels with the result and is not softened: unequal rates
    do not establish bias and equal rates do not establish its absence.
    """
    rate_a = admitted_a / launched_a
    rate_b = admitted_b / launched_b
    pooled = (admitted_a + admitted_b) / (launched_a + launched_b)
    se = float(np.sqrt(pooled * (1 - pooled) * (1 / launched_a + 1 / launched_b)))
    z = (rate_a - rate_b) / se if se > 0 else float("nan")
    return ProportionDifference(
        admitted_a=admitted_a, launched_a=launched_a, rate_a=rate_a,
        admitted_b=admitted_b, launched_b=launched_b, rate_b=rate_b,
        difference=rate_a - rate_b, pooled_rate=pooled, standard_error=se,
        z_statistic=float(z),
        pairs_only_a_admitted=pairs_only_a_admitted,
        pairs_only_b_admitted=pairs_only_b_admitted,
    )


# ---------------------------------------------------------------------------
# M03. Monte Carlo error, endpoints separately from the point estimate.
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloError:
    n_repetitions: int
    lower_endpoint_sd: float
    upper_endpoint_sd: float
    point_side_sd: float
    point_side_quantity: str
    lower_endpoints: List[float] = field(default_factory=list)
    upper_endpoints: List[float] = field(default_factory=list)


def monte_carlo_error(run: Callable[[int], Tuple[float, float, float]],
                      substreams: Sequence[int],
                      point_side_quantity: str) -> MonteCarloError:
    """Repeat a whole bootstrap on independent streams and measure how much
    its output moves.

    `run` takes a substream index and returns (lower, upper, point_side). The
    point-side quantity is named by the caller rather than assumed here,
    because it is a different thing for the two estimators and conflating them
    is exactly what the committed specification asks to be avoided.
    """
    lows, highs, points = [], [], []
    for s in substreams:
        lo, hi, pt = run(s)
        lows.append(lo)
        highs.append(hi)
        points.append(pt)
    return MonteCarloError(
        n_repetitions=len(substreams),
        lower_endpoint_sd=float(np.std(lows, ddof=1)),
        upper_endpoint_sd=float(np.std(highs, ddof=1)),
        point_side_sd=float(np.std(points, ddof=1)),
        point_side_quantity=point_side_quantity,
        lower_endpoints=[float(v) for v in lows],
        upper_endpoints=[float(v) for v in highs],
    )


# ===========================================================================
# B25. The metrics and diagnostics. 
#
# One estimator here is new. Everything else B25 reports is computed by a
# function already above, called on the population that metric names rather
# than reimplemented: M08 is ols_slope, M09 is pearson_correlation, and M10
# is M08 again on a regenerated set. That is why B24's arm difference in
# slope and B25's per-arm slopes are one computation reported at two levels
# and not two computations that agree.
# ===========================================================================

@dataclass
class SurrogateGap:
    """M22. The three quantities the build plan names, on one population."""
    n: int
    mean_absolute_difference: float
    mean_signed_difference: float
    correlation: float
    units: str


def surrogate_gap(predicted: np.ndarray, solver_value: np.ndarray,
                   units: str) -> SurrogateGap:
    """M22. The ensemble's prediction against the solver's value, on admitted
    generated shapes.

    The signed difference is oriented PREDICTED MINUS SOLVER, so a positive
    value means the surrogate read the shape as more efficient than the
    solver found it to be. That is the direction a generator exploiting the
    surrogate would push, which is what M22 exists to make visible, so the
    orientation is stated rather than left for a reader to infer from a sign.

    Both arrays are in the same units and the caller names them. The gap is
    read against M23, the ensemble's held-out error, which B13 computed in
    raw label units, so passing normalised values here would produce a
    number that cannot be compared with the reference it is meant to be read
    against.
    """
    predicted = np.asarray(predicted, dtype=float)
    solver_value = np.asarray(solver_value, dtype=float)
    if predicted.shape != solver_value.shape:
        raise ValueError("predicted and solver_value must have the same shape")
    difference = predicted - solver_value
    return SurrogateGap(
        n=len(difference),
        mean_absolute_difference=float(np.abs(difference).mean()),
        mean_signed_difference=float(difference.mean()),
        correlation=pearson_correlation(solver_value, predicted),
        units=units,
    )


def interval_reading(lower: float, upper: float) -> str:
    """The pre-registration's reading rule, applied in code so it cannot drift in the
    writing. An interval spanning zero is no detected difference. It is not
    evidence of no effect, and it is not a trend, a suggestion or a direction.
    """
    if lower <= 0.0 <= upper:
        return ("SPANS ZERO: no detected difference. The interval is inconclusive "
                "about the sign of the effect. This is not evidence of no effect. "
                "No equivalence test is pre-registered, so absence is not claimed.")
    if lower > 0.0:
        return ("EXCLUDES ZERO, entirely positive: the interval determines the "
                "sign, favouring the prior-on arm.")
    return ("EXCLUDES ZERO, entirely negative: the interval determines the "
            "sign, favouring the prior-off arm.")

