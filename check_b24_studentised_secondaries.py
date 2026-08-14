"""Falsification check on the interval estimator the four secondaries use.

THIS CHECK FAILED, AND IT IS RETAINED BECAUSE IT FAILED. Rerunning it will
fail again. It returned FAIL on the design B width agreement clause,
C_AGREE_TOLERANCE below, for both statistics tested. With no between-cluster
component the refined interval runs 2.82 times wider than the pair interval for
the median and 1.60 times wider for the slope, against a tolerance of 0.40
fixed in advance. No threshold was moved and no clause was deleted. The
consequence for reading the four secondaries is in RESULTS.txt section 4, and
the article reports the failure at section 3.4 and again at section 4.6.

WHAT IT TESTS. Three of the four pre-registered secondaries rest on a
studentised cluster bootstrap and on a delete-one-cluster jackknife standard
error. B24's other falsification check exercised the resampling unit and the
two estimators the primary rests on, and touched neither of those. This closes
that gap rather than arguing it away.

WHY COVERAGE AND NOT ONLY WIDTH. A studentised interval can be wide and still
be useless if the bootstrap t distribution it takes its quantiles from is not a
reasonable reference: it would be wide, wrongly placed, and would cover the
truth at the wrong rate, while every width comparison looked ordinary. The
slope difference's realised bootstrap t quantiles on the real data are
[-4.53, 1.93] against the primary's near-symmetric [-2.37, 2.36], which is the
specific doubt this check answers.

WHAT IT TESTS ON. Two statistics. The median, which is the plain case of the
mechanism, and the arm difference in tracking slope, which is where the doubt
is. The win fraction shares the same machinery and is not separately simulated;
that is a stated limit of this check rather than an implied one.

Six clauses, all six written before the script was run and none moved
afterwards. Two width clauses, three coverage clauses and one negative control.
The designs are built at the real study's shape, 11 clusters by 10 pairs, at
400 datasets per design and 399 resamples.

Run order      none. Optional, standalone, and outside the numbered steps.
Reads          nothing from the build. It runs its own simulation
Writes         nothing. The verdict goes to the console
Runtime        not recorded in the article
"""

from __future__ import annotations

import time

import numpy as np

import analysis
import dataset

# ---------------------------------------------------------------------------
# Fixed before running.
# ---------------------------------------------------------------------------

N_CLUSTERS = 11
PAIRS_PER_CLUSTER = 10
N_DATASETS = 400            # coverage is a proportion over this many datasets
N_RESAMPLES = 399           # (B+1)*alpha/2 = 10 exactly at 95 percent
LEVEL = 0.95

BETWEEN_SD = 1.0            # design A, a large deliberate between-cluster component
WITHIN_SD = 0.1             # design A, a small within-cluster one
FLAT_SD = 1.0               # design B, no between-cluster component at all

# The estimand is the SUPERPOPULATION value, which is what a cluster bootstrap
# targets when it treats clusters as sampled. Both designs are symmetric about
# it, so the true median and the true slope difference are both exactly zero
# and coverage is the fraction of datasets whose interval contains zero.
TRUE_VALUE = 0.0
SLOPE_ON = 0.8              # design for S3; the two arms share a true slope,
SLOPE_OFF = 0.8             # so the true DIFFERENCE is zero

# Width clauses, the same two the build plan's own check text states, applied here to
# the studentised estimator instead of to the mean.
C_WIDE_MIN = 2.0            # design A: refined cluster width over pair width
C_AGREE_TOLERANCE = 0.40    # design B: how far from parity the refined interval
                            # may sit. Wider than the 0.25 used for the
                            # unrefined percentile comparison, and stated in
                            # advance with its reason: a studentised interval
                            # taking its quantiles from an 11-cluster
                            # distribution is expected to sit somewhat wider
                            # than a 110-unit percentile interval even when
                            # there is nothing to be wide about.

# Coverage clauses.
C_COVERAGE_MIN = 0.85       # nominal is 0.95. Cameron, Gelbach and Miller report
                            # standard cluster-robust tests rejecting at about 10
                            # percent against a nominal 5 in this cluster range,
                            # which is coverage near 0.90, and the refinement is
                            # meant to improve on that. 0.85 is a floor a broken
                            # studentisation would miss badly and an imperfect
                            # working one would clear.
C_REFINED_NOT_WORSE = 0.03  # the refined interval's coverage may not fall more
                            # than this below the unrefined one's. With 400
                            # datasets the standard error of a coverage estimate
                            # near 0.9 is about 0.015, so 0.03 is roughly two of
                            # those and the clause does not fail on noise alone.
C_PAIR_COVERAGE_MAX = 0.85  # NEGATIVE CONTROL. On design A the pair bootstrap
                            # must undercover. A coverage measurement that
                            # nothing fails is not evidence.


# ---------------------------------------------------------------------------
# The comparator. Lives here and does not survive this file.
# ---------------------------------------------------------------------------

def pair_bootstrap(n_rows, stat, rng, n_resamples=N_RESAMPLES, level=LEVEL):
    """Pairs are the resampling unit and the clustering is ignored entirely."""
    theta = np.empty(n_resamples)
    for b in range(n_resamples):
        theta[b] = stat(rng.integers(0, n_rows, size=n_rows))
    finite = np.sort(theta[np.isfinite(theta)])
    return analysis.quantiles_with_convention(finite, level)[:2]


# ---------------------------------------------------------------------------
# The two designs.
# ---------------------------------------------------------------------------

def cluster_layout():
    cluster_id = np.repeat(np.arange(N_CLUSTERS), PAIRS_PER_CLUSTER)
    members = [np.flatnonzero(cluster_id == g) for g in range(N_CLUSTERS)]
    return cluster_id, members


def median_design(rng, between_sd, within_sd):
    """A paired difference series whose superpopulation median is zero."""
    cluster_id, members = cluster_layout()
    shift = (rng.normal(0.0, between_sd, size=N_CLUSTERS) if between_sd > 0
             else np.zeros(N_CLUSTERS))
    values = shift[cluster_id] + rng.normal(0.0, within_sd, size=len(cluster_id))
    return values, members


def slope_design(rng, between_sd, within_sd):
    """Two arms measured at the same requested targets, sharing a true slope,
    so the true arm difference in slope is zero.

    The between-cluster component enters as a cluster-level shift drawn
    INDEPENDENTLY for each arm. A shift shared by both arms would cancel in the
    difference and the design would carry no between-cluster component in the
    statistic being tested, which is the mistake this design has to avoid.
    """
    cluster_id, members = cluster_layout()
    x = np.linspace(0.0, 1.0, N_CLUSTERS)[cluster_id]
    if between_sd > 0:
        shift_on = rng.normal(0.0, between_sd, size=N_CLUSTERS)[cluster_id]
        shift_off = rng.normal(0.0, between_sd, size=N_CLUSTERS)[cluster_id]
    else:
        shift_on = shift_off = np.zeros(len(cluster_id))
    y_on = SLOPE_ON * x + shift_on + rng.normal(0.0, within_sd, size=len(cluster_id))
    y_off = SLOPE_OFF * x + shift_off + rng.normal(0.0, within_sd, size=len(cluster_id))
    return x, y_on, y_off, members


# ---------------------------------------------------------------------------
# One design, one statistic, N_DATASETS times.
# ---------------------------------------------------------------------------

def run(design, statistic_name, rng, between_sd, within_sd):
    n_rows = N_CLUSTERS * PAIRS_PER_CLUSTER
    widths = {"pair": [], "unrefined": [], "refined": []}
    covered = {"pair": 0, "unrefined": 0, "refined": 0}
    undefined_refined = 0

    for _ in range(N_DATASETS):
        if design == "median":
            values, members = median_design(rng, between_sd, within_sd)
            stat = analysis.median_statistic(values)
        else:
            x, y_on, y_off, members = slope_design(rng, between_sd, within_sd)
            stat = analysis.slope_difference_statistic(x, y_on, y_off)

        lo, hi = pair_bootstrap(n_rows, stat, rng)
        widths["pair"].append(hi - lo)
        covered["pair"] += int(lo <= TRUE_VALUE <= hi)

        result = analysis.pairs_cluster_bootstrap(
            members, stat, rng, n_resamples=N_RESAMPLES, level=LEVEL,
            studentise=True)

        widths["unrefined"].append(
            result.percentile_upper - result.percentile_lower)
        covered["unrefined"] += int(
            result.percentile_lower <= TRUE_VALUE <= result.percentile_upper)

        if np.isfinite(result.studentised_lower) and np.isfinite(result.studentised_upper):
            widths["refined"].append(
                result.studentised_upper - result.studentised_lower)
            covered["refined"] += int(
                result.studentised_lower <= TRUE_VALUE <= result.studentised_upper)
        else:
            undefined_refined += 1

    return {
        "width": {k: float(np.mean(v)) for k, v in widths.items()},
        "coverage": {k: covered[k] / N_DATASETS for k in covered},
        "undefined_refined": undefined_refined,
    }


def report(name, design_a, design_b, verdicts):
    print()
    print("=" * 74)
    print(f"{name}")
    print("=" * 74)
    for label, out in (("DESIGN A, large between-cluster component", design_a),
                       ("DESIGN B, no between-cluster component", design_b)):
        print(f"  {label}")
        print(f"    {'estimator':<12}{'mean width':>14}{'ratio to pair':>16}"
              f"{'coverage':>12}")
        for key in ("pair", "unrefined", "refined"):
            print(f"    {key:<12}{out['width'][key]:>14.6f}"
                  f"{out['width'][key] / out['width']['pair']:>16.4f}"
                  f"{out['coverage'][key]:>12.4f}")
        print(f"    refined interval undefined on "
              f"{out['undefined_refined']} of {N_DATASETS} datasets")
        print()

    def record(clause, ok, detail):
        verdicts.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {clause}: {detail}")

    r = design_a["width"]["refined"] / design_a["width"]["pair"]
    record("A, refined cluster interval is materially wider than the pair interval",
           r >= C_WIDE_MIN, f"ratio {r:.4f}, required >= {C_WIDE_MIN}")
    r = design_b["width"]["refined"] / design_b["width"]["pair"]
    record("B, the two agree once there is nothing to be wide about",
           abs(r - 1.0) <= C_AGREE_TOLERANCE,
           f"ratio {r:.4f}, required within {C_AGREE_TOLERANCE} of 1")
    c = design_a["coverage"]["refined"]
    record("A, refined coverage", c >= C_COVERAGE_MIN,
           f"{c:.4f}, required >= {C_COVERAGE_MIN}, nominal {LEVEL}")
    c = design_b["coverage"]["refined"]
    record("B, refined coverage", c >= C_COVERAGE_MIN,
           f"{c:.4f}, required >= {C_COVERAGE_MIN}, nominal {LEVEL}")
    gap = design_a["coverage"]["refined"] - design_a["coverage"]["unrefined"]
    record("A, the refinement does not cover worse than the unrefined interval",
           gap >= -C_REFINED_NOT_WORSE,
           f"refined minus unrefined {gap:+.4f}, "
           f"required >= {-C_REFINED_NOT_WORSE}")
    c = design_a["coverage"]["pair"]
    record("A, NEGATIVE CONTROL, the pair bootstrap undercovers",
           c <= C_PAIR_COVERAGE_MAX,
           f"{c:.4f}, required <= {C_PAIR_COVERAGE_MAX}")


def main():
    print("B24 second falsification check. The studentised cluster bootstrap the")
    print("secondaries rest on, tested for width AND for coverage.")
    print(f"  {N_CLUSTERS} clusters x {PAIRS_PER_CLUSTER} pairs, "
          f"{N_DATASETS} datasets per design, {N_RESAMPLES} resamples, "
          f"nominal {LEVEL}.")
    started = time.time()
    verdicts = []

    report("S1's estimator: the MEDIAN paired difference",
           run("median", "median", dataset.rng_for(24, 902), BETWEEN_SD, WITHIN_SD),
           run("median", "median", dataset.rng_for(24, 903), 0.0, FLAT_SD),
           verdicts)

    report("S3's estimator: the ARM DIFFERENCE IN TRACKING SLOPE",
           run("slope", "slope", dataset.rng_for(24, 904), BETWEEN_SD, WITHIN_SD),
           run("slope", "slope", dataset.rng_for(24, 905), 0.0, FLAT_SD),
           verdicts)

    print()
    print(f"Elapsed {time.time() - started:.1f} s.")
    print("OVERALL:", "PASS" if all(verdicts) else "FAIL")
    print()
    print("NOT COVERED by this check, stated rather than implied: S2's win")
    print("fraction is not separately simulated. It runs the same machinery on a")
    print("bounded 0/1 statistic, and a bounded statistic can behave differently")
    print("near its limits than an unbounded one. Its realised interval on the")
    print("real data sits well inside [0, 1], so it is not near a limit there.")
    return 0 if all(verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
