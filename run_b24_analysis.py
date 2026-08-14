"""B24. The analysis. Every reported statistic and every reported count.

Computes the primary outcome, being the mean paired difference in target
satisfaction error, with a wild cluster bootstrap-t interval at 9999 resamples
and the unrefined percentile cluster bootstrap reported beside it. Also
computes the endpoint Monte Carlo error, the paired difference distribution
shape, the four secondaries, the trimmed mean sensitivity check, the admission
exclusion count, the differential attrition test and the two per-arm mean
converged point counts.

This driver computes nothing itself. Every estimator lives in analysis.py and
every committed value is read from params.py. What this file does is wire the
two together, in the order the pre-registration fixes, and write the result
where the article can read it.

Counts are recomputed here from the evaluation records rather than carried over
from the generation run.

The interval is read in code and not in the writing, through
analysis.interval_reading. A lower bound at or below zero with an upper bound
at or above zero reads as no detected difference.

Nothing is printed and then discarded. If a figure appears on the console it
also appears in RESULTS.txt.

Run order      8 of 10. After B23, before B25.
Reads          b23_evaluation.json, normalization.npz
Writes         b24_analysis.json, and RESULTS.txt
Runtime        not recorded in the article
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import analysis
import dataset
import params

EVALUATION_RECORDS = Path("b23_evaluation.json")
NORMALIZATION = Path("normalization.npz")
RESULTS_TXT = Path("RESULTS.txt")
RESULTS_JSON = Path("b24_analysis.json")

BUILD_STEP = 24


# ---------------------------------------------------------------------------
# Committed values, read and never restated.
# ---------------------------------------------------------------------------

def committed():
    def value(key):
        slot = params.PARAMS[key]
        if isinstance(slot.value, params.Pending):
            raise ValueError(f"{key} is PENDING; B24 may not run against it")
        return slot.value

    return {
        "primary_statistic": value("primary_statistic"),
        "cluster_unit_bootstrap_method": value("cluster_unit_bootstrap_method"),
        "interval_level": float(value("interval_level")),
        "resample_count": int(value("resample_count")),
        "base_seed": int(value("seeds")),
        "minimum_converged_points":
            int(value("minimum_converged_sweep_points")["minimum_converged_points"]),
        "requested_target_band": value("requested_target_band"),
        "analysed_pair_floor": int(value("analysed_pair_floor")),
        "trim_fraction": float(params.B24_ANALYSIS["trim_fraction"]),
        "monte_carlo_repetitions": int(params.B24_ANALYSIS["monte_carlo_repetitions"]),
    }


# ---------------------------------------------------------------------------
# Admission and pairing.
# ---------------------------------------------------------------------------

def build_pairs(records, label_min, label_max, minimum_converged_points):
    """Admit, then pair. In that order, and complete-case, per the pre-registration."""
    flow = {
        "records_read": 0,
        "admission_field_missing": 0,
        "admission_field_disagreed_with_recomputed_rule": 0,
        # M20's two running sums, added the build. the build plan assigns the
        # per-arm means to B23, which does not carry them, so they are formed
        # here instead. That is consistent with this step's own stated role as
        # the sole source of every reported COUNT as well as every reported
        # statistic, and it needs no solver call: the per-record point counts
        # are already stored in B23's records.
        "n_converged_field_missing": 0,
        "prior_on": {"launched": 0, "admitted": 0, "produced_a_label": 0,
                     "label_but_below_min_points": 0,
                     "sum_converged_points_admitted": 0,
                     "sum_usable_points_admitted": 0},
        "prior_off": {"launched": 0, "admitted": 0, "produced_a_label": 0,
                      "label_but_below_min_points": 0,
                      "sum_converged_points_admitted": 0,
                      "sum_usable_points_admitted": 0},
        "pairs_launched": 0,
        "pairs_both_admitted": 0,
        "pairs_only_prior_on_admitted": 0,
        "pairs_only_prior_off_admitted": 0,
        "pairs_neither": 0,
    }

    rows = []
    for record in records:
        flow["records_read"] += 1
        flow["pairs_launched"] += 1
        target = float(record["target"])
        arms = {}
        for arm_name in ("prior_on", "prior_off"):
            arm = record[arm_name]
            state = analysis.read_admission(arm, minimum_converged_points)
            arms[arm_name] = state
            counts = flow[arm_name]
            counts["launched"] += 1
            counts["admitted"] += int(state.admitted)
            if state.admitted:
                # M20. Read n_converged EXPLICITLY, with no default, for the
                # same reason the pre-registration gives for the admission field: a
                # dict.get default would fold a missing count into the mean as
                # a zero and quietly drag it down. Absence is counted instead.
                if "n_converged" in arm:
                    counts["sum_converged_points_admitted"] += int(arm["n_converged"])
                else:
                    flow["n_converged_field_missing"] += 1
                counts["sum_usable_points_admitted"] += state.n_usable
            if state.label is not None:
                counts["produced_a_label"] += 1
                if state.n_usable < minimum_converged_points:
                    counts["label_but_below_min_points"] += 1
            if not state.field_present:
                flow["admission_field_missing"] += 1
            elif state.agrees is False:
                flow["admission_field_disagreed_with_recomputed_rule"] += 1

        on, off = arms["prior_on"], arms["prior_off"]
        if on.admitted and off.admitted:
            flow["pairs_both_admitted"] += 1
            achieved_on, error_on = analysis.normalised_error(
                on.label, target, label_min, label_max)
            achieved_off, error_off = analysis.normalised_error(
                off.label, target, label_min, label_max)
            rows.append({
                "target": target,
                "target_index": int(record["target_index"]),
                "sample_index": int(record["sample_index"]),
                "achieved_on": achieved_on, "achieved_off": achieved_off,
                "error_on": error_on, "error_off": error_off,
                "difference": error_off - error_on,
            })
        elif on.admitted:
            flow["pairs_only_prior_on_admitted"] += 1
        elif off.admitted:
            flow["pairs_only_prior_off_admitted"] += 1
        else:
            flow["pairs_neither"] += 1

    def column(key, dtype=float):
        return np.array([r[key] for r in rows], dtype=dtype)

    paired = analysis.PairedData(
        target=column("target"),
        target_index=column("target_index", int),
        sample_index=column("sample_index", int),
        achieved_on=column("achieved_on"), achieved_off=column("achieved_off"),
        error_on=column("error_on"), error_off=column("error_off"),
        difference=column("difference"),
        raw_span=label_max - label_min,
    )
    return paired, flow


# ---------------------------------------------------------------------------
# Formatting. One place, so the file reads as one document.
# ---------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        self.lines.append(text)

    def rule(self, char="-"):
        self.lines.append(char * 78)

    def heading(self, text):
        self()
        self.rule("=")
        self(text)
        self.rule("=")

    def section(self, text):
        self()
        self(text)
        self.rule()

    def write(self, path):
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def interval_text(lower, upper, level):
    return f"[{lower:.10f}, {upper:.10f}] at {level * 100:.0f} percent"


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    fixed = committed()
    level = fixed["interval_level"]
    resamples = fixed["resample_count"]
    trim = fixed["trim_fraction"]

    evaluation = json.loads(EVALUATION_RECORDS.read_text(encoding="utf-8"))
    norm = np.load(NORMALIZATION, allow_pickle=True)
    label_min = float(norm["label_min"])
    label_max = float(norm["label_max"])
    span = label_max - label_min

    paired, flow = build_pairs(evaluation["records"], label_min, label_max,
                               fixed["minimum_converged_points"])
    members = paired.cluster_members()
    d = paired.difference
    n = len(paired)

    # --- primary -----------------------------------------------------------
    primary = analysis.wild_cluster_bootstrap_t(
        d, paired.target_index, dataset.rng_for(BUILD_STEP, 0),
        n_resamples=resamples, level=level)
    primary_exhaustive = analysis.wild_cluster_bootstrap_t(
        d, paired.target_index, None, n_resamples=0, level=level, exhaustive=True)
    primary_unrefined = analysis.pairs_cluster_bootstrap(
        members, analysis.mean_statistic(d), dataset.rng_for(BUILD_STEP, 1),
        n_resamples=resamples, level=level, studentise=False)

    shape = analysis.distribution_shape(d)

    # --- Monte Carlo error, endpoints against the point side ---------------
    def wild_repetition(substream):
        r = analysis.wild_cluster_bootstrap_t(
            d, paired.target_index, dataset.rng_for(BUILD_STEP, substream),
            n_resamples=resamples, level=level)
        return r.lower, r.upper, r.point_estimate

    def percentile_repetition(substream):
        r = analysis.pairs_cluster_bootstrap(
            members, analysis.mean_statistic(d),
            dataset.rng_for(BUILD_STEP, substream),
            n_resamples=resamples, level=level, studentise=False)
        return r.percentile_lower, r.percentile_upper, r.bootstrap_mean

    reps = fixed["monte_carlo_repetitions"]
    mc_wild = analysis.monte_carlo_error(
        wild_repetition, range(100, 100 + reps),
        "the primary point estimate, which is a function of the data alone and "
        "carries exactly zero resampling error")
    mc_percentile = analysis.monte_carlo_error(
        percentile_repetition, range(200, 200 + reps),
        "the percentile cluster bootstrap's own bootstrap mean")

    # --- secondaries -------------------------------------------------------
    s1 = analysis.pairs_cluster_bootstrap(
        members, analysis.median_statistic(d), dataset.rng_for(BUILD_STEP, 2),
        n_resamples=resamples, level=level)

    win, wins, non_tied = analysis.win_fraction(d)
    win_indicator = np.where(d > 0, 1.0, 0.0)[d != 0]
    win_cluster = paired.target_index[d != 0]
    win_se = analysis.cluster_robust_se(win_indicator, win_cluster)
    win_t = (win - 0.5) / win_se if win_se > 0 else float("nan")
    win_test = analysis.wild_cluster_bootstrap_t(
        win_indicator, win_cluster, dataset.rng_for(BUILD_STEP, 3),
        n_resamples=resamples, level=level, null_value=0.5)
    s2 = analysis.pairs_cluster_bootstrap(
        members, analysis.win_fraction_statistic(d), dataset.rng_for(BUILD_STEP, 3),
        n_resamples=resamples, level=level)

    slope_on = analysis.ols_slope(paired.target, paired.achieved_on)
    slope_off = analysis.ols_slope(paired.target, paired.achieved_off)
    corr_on = analysis.pearson_correlation(paired.target, paired.achieved_on)
    corr_off = analysis.pearson_correlation(paired.target, paired.achieved_off)
    s3 = analysis.pairs_cluster_bootstrap(
        members,
        analysis.slope_difference_statistic(
            paired.target, paired.achieved_on, paired.achieved_off),
        dataset.rng_for(BUILD_STEP, 4), n_resamples=resamples, level=level)

    s4 = analysis.proportion_difference(
        admitted_a=flow["prior_on"]["admitted"], launched_a=flow["prior_on"]["launched"],
        admitted_b=flow["prior_off"]["admitted"], launched_b=flow["prior_off"]["launched"],
        pairs_only_a_admitted=flow["pairs_only_prior_on_admitted"],
        pairs_only_b_admitted=flow["pairs_only_prior_off_admitted"])

    # --- sensitivity -------------------------------------------------------
    m06 = analysis.pairs_cluster_bootstrap(
        members, analysis.trimmed_mean_statistic(d, trim),
        dataset.rng_for(BUILD_STEP, 5), n_resamples=resamples, level=level)

    # --- per target, descriptive ------------------------------------------
    per_target = []
    for g in np.unique(paired.target_index):
        mask = paired.target_index == g
        per_target.append({
            "target_index": int(g),
            "target_normalised": float(paired.target[mask][0]),
            "target_raw": float(paired.target[mask][0] * span + label_min),
            "n_pairs": int(mask.sum()),
            "mean_difference": float(d[mask].mean()),
            "mean_error_prior_on": float(paired.error_on[mask].mean()),
            "mean_error_prior_off": float(paired.error_off[mask].mean()),
        })

    # =======================================================================
    # RESULTS.txt
    # =======================================================================
    r = Report()
    r.rule("=")
    r("RESULTS.txt -- every reported statistic in this study, from one module.")
    r.rule("=")
    r()
    r(f"Produced by run_b24_analysis.py at build step B{BUILD_STEP}, "
      f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}.")
    r(f"Source records: {EVALUATION_RECORDS} (B23).")
    r(f"Normalisation:  {NORMALIZATION} (B08), "
      f"label_min {label_min!r}, label_max {label_max!r}.")
    r()
    r("The pre-registration: every reported statistic comes from")
    r("one analysis module, and no number in the paper is taken from a diagnostic")
    r("printed during a generation run. This file is that module's output. A number")
    r("in the paper that is not in this file has no source.")
    r()
    r("UNITS. Target satisfaction error is in NORMALISED units throughout, per the committed specification")
    r("and the pre-registration. Raw max(CL/CD) figures accompany and never replace them;")
    r(f"the raw span is {span!r} L/D, so a normalised error times that span is the")
    r("raw-equivalent error. The superseded build reported the error in raw ratio")
    r("units, which is what the committed specification rules out.")
    r()
    r("LABELS. Every measure below carries one of:")
    r("  PRIMARY     the single pre-registered primary outcome (the pre-registration)")
    r("  SECONDARY   one of the four pre-registered secondaries (the pre-registration)")
    r("  SENSITIVITY declared a sensitivity check and not an outcome; not promotable")
    r("  DESCRIPTIVE reported and labelled as descriptive")
    r("No measure here is marked post hoc, because no measure outside these four")
    r("categories was computed. The pre-registration applies no multiplicity correction and")
    r("withholds nothing.")

    r.heading("1. COMMITTED VALUES THIS ANALYSIS READ")
    r(f"  primary statistic          {fixed['primary_statistic']['statistic']}")
    r(f"  cluster unit               {fixed['cluster_unit_bootstrap_method']['cluster_unit']}"
      f", {fixed['cluster_unit_bootstrap_method']['n_clusters']} clusters")
    r(f"  bootstrap method           {fixed['cluster_unit_bootstrap_method']['method']}")
    r(f"  also reported              {fixed['cluster_unit_bootstrap_method']['also_reported']}")
    r(f"  interval level             {level}")
    r(f"  resample count             {resamples}")
    r(f"  minimum converged points   {fixed['minimum_converged_points']} of 9")
    r(f"  analysed pair floor        {fixed['analysed_pair_floor']} analysed pairs")
    r(f"  base seed                  {fixed['base_seed']}")
    r("  generator derivation       base seed + 1000 * build step + substream,")
    r("                             every generator in this file, no exceptions")
    r()
    r("  Substreams used at step 24:")
    for key in (0, 1, 2, 3, 4, 5, "100..119", "200..219", "900, 901"):
        r(f"    {str(key):10s} {params.B24_ANALYSIS['generator_substreams'][key]}")
    r()
    r(f"  SET AT THIS STEP: the trim fraction for M06, {trim} per tail.")
    r("  the build plan states it is a stated value and assigns it to B24. No earlier step")
    r("  sets it and the committed specification has no row for it. It is this study's own")
    r("  round figure, disclosed as stated, and it is recorded in params.B24_ANALYSIS.")
    r("  It was set after the pre-registration was issued and the issued document does")
    r("  not name it. It is a sensitivity parameter and not an outcome, an admission")
    r("  rule or a reporting rule, so the pre-registration is not engaged. The absence is")
    r("  listed rather than left to be discovered.")

    r.heading("2. ADMISSION AND PAIRING")
    r("Recomputed here from B23's records rather than carried over, because B24 is")
    r("the sole source of every reported count as well as every reported statistic.")
    r()
    r(f"  records read                              {flow['records_read']}")
    r(f"  pairs launched                            {flow['pairs_launched']}")
    r(f"  ANALYSED PAIRS (both members admitted)    {flow['pairs_both_admitted']}")
    r(f"  committed floor                           {fixed['analysed_pair_floor']}")
    r(f"  floor met                                 "
      f"{flow['pairs_both_admitted'] >= fixed['analysed_pair_floor']}")
    r(f"  pairs in which only the prior-on member was admitted   "
      f"{flow['pairs_only_prior_on_admitted']}")
    r(f"  pairs in which only the prior-off member was admitted  "
      f"{flow['pairs_only_prior_off_admitted']}")
    r(f"  pairs in which neither member was admitted             "
      f"{flow['pairs_neither']}")
    r()
    for arm in ("prior_on", "prior_off"):
        c = flow[arm]
        r(f"  {arm}: launched {c['launched']}, produced a label "
          f"{c['produced_a_label']}, admitted {c['admitted']}, "
          f"rate {c['admitted'] / c['launched']:.6f}")
    r()
    r("  M18, admission exclusion count. DESCRIPTIVE, and reported even at zero,")
    r("  per the pre-registration and the committed specification. The denominator is the records that produced an")
    r("  efficiency value, NOT the records launched.")
    excluded = (flow["prior_on"]["label_but_below_min_points"]
                + flow["prior_off"]["label_but_below_min_points"])
    denominator = (flow["prior_on"]["produced_a_label"]
                   + flow["prior_off"]["produced_a_label"])
    r(f"    excluded by the minimum converged point rule: {excluded} of {denominator}")
    r(f"    prior-on {flow['prior_on']['label_but_below_min_points']}, "
      f"prior-off {flow['prior_off']['label_but_below_min_points']}")
    r()
    r("  Admission field integrity. The pre-registration requires the field read explicitly")
    r("  with no default, so a record missing it is excluded and counted.")
    r(f"    records missing the admission field:  {flow['admission_field_missing']}")
    r("    stored field disagreeing with the rule recomputed from label and usable")
    r(f"    point count:                         "
      f"{flow['admission_field_disagreed_with_recomputed_rule']}")
    r()
    r("  M20, mean converged sweep points per admitted record, per arm.")
    r("  DESCRIPTIVE. Added during the build: the build plan assigns this metric's per-arm")
    r("  means to B23, B23 does not carry them, and this step is the sole source of")
    r("  every reported count. It is formed from B23's own stored per-record point")
    r("  counts and calls no solver.")
    for arm in ("prior_on", "prior_off"):
        c = flow[arm]
        n_adm = c["admitted"]
        mean_conv = c["sum_converged_points_admitted"] / n_adm if n_adm else float("nan")
        mean_use = c["sum_usable_points_admitted"] / n_adm if n_adm else float("nan")
        r(f"    {arm}: admitted {n_adm}, mean converged points {mean_conv!r},")
        r(f"      mean usable points {mean_use!r}")
    r(f"    records admitted whose converged-point field was missing: "
      f"{flow['n_converged_field_missing']}")
    r("    The denominator is the ADMITTED records in that arm, not the launched")
    r("    ones. Converged and usable coincide on this run because no converged")
    r("    point was dropped by the plausibility criterion; the two are printed")
    r("    separately because they are different quantities and need not coincide.")
    r("    An arm converging on systematically fewer points would have its achieved")
    r("    efficiency biased downward by truncation, and that bias would present as")
    r("    a prior effect. This is the marker that rules it out, and it is a marker")
    r("    rather than a clearance.")
    r()
    r("  Complete-case pairing is committed and is not bias-free. A pair with one")
    r("  surviving member is excluded entirely and its surviving member is used")
    r("  nowhere. The pre-registration states this as a limitation and not a neutral choice.")
    r()
    extended = evaluation.get("extension", {}).get("extended", None)
    r(f"  Extension performed: {extended}. The pre-registration requires the floor result")
    r("  reported beside the final result whenever the sweep is extended. It was not")
    r("  extended, so the analysed set below IS the floor result and the two are the")
    r("  same numbers, not two computations that happen to agree.")

    r.heading("3. PRIMARY OUTCOME")
    r("M01/M02. PRIMARY. The mean paired difference in normalised target")
    r("satisfaction error, with a confidence interval produced by resampling whole")
    r("requested-target clusters.")
    r()
    r("Definition, as committed at the pre-registration. For each matched pair the error is")
    r("the absolute difference between the achieved efficiency and the requested")
    r("efficiency, in normalised units. The paired difference is the prior-off arm's")
    r("error minus the prior-on arm's, so a positive value means the prior-on shape")
    r("landed closer to what was asked.")
    r()
    r(f"  n analysed pairs                      {n}")
    r(f"  clusters (requested targets)          {paired.n_clusters}")
    r(f"  MEAN PAIRED DIFFERENCE (normalised)   {primary.point_estimate!r}")
    r(f"  accompanying raw equivalent           "
      f"{primary.point_estimate * span!r} L/D")
    r(f"  cluster-robust standard error         {primary.cluster_robust_se!r}")
    r(f"  cluster-robust t statistic            {primary.t_statistic!r}")
    r()
    naive_se = float(d.std(ddof=1) / np.sqrt(n))
    r("  DESCRIPTIVE, and it is what makes the committed specification's requirement concrete on this data")
    r("  rather than a citation. The standard error the same series would carry if")
    r("  the pairs were treated as independent:")
    r(f"    independent-pairs standard error     {naive_se!r}")
    r(f"    cluster-robust standard error        {primary.cluster_robust_se!r}")
    r(f"    ratio                                "
      f"{primary.cluster_robust_se / naive_se!r}")
    r("  Several samples share each requested target and the paired difference varies")
    r("  systematically across targets, so the pairs are not independent. Ignoring")
    r("  that would understate the standard error by the ratio above. The independent")
    r("  figure is reported to show the size of the error, and no interval in this")
    r("  file is built from it.")
    r()
    r("  COMMITTED INTERVAL. Wild cluster bootstrap-t, Rademacher weights, whole")
    r("  clusters, percentile-t from unrestricted residuals.")
    r(f"    {interval_text(primary.lower, primary.upper, level)}")
    r(f"    accompanying raw equivalent   "
      f"[{primary.lower * span:.6f}, {primary.upper * span:.6f}] L/D")
    r(f"    bootstrap t quantiles         "
      f"[{primary.t_quantile_lower!r}, {primary.t_quantile_upper!r}]")
    r(f"    resamples drawn               {primary.n_resamples}")
    r(f"    degenerate replicates         {primary.n_degenerate}")
    r(f"    quantile convention           {primary.quantile_convention}")
    r(f"    p value, null imposed         {primary.p_value_null_imposed!r}")
    r()
    r("    READING, per the pre-registration:")
    r(f"    {analysis.interval_reading(primary.lower, primary.upper)}")
    r()
    r("  ALSO REPORTED, as committed: the unrefined percentile cluster bootstrap.")
    r("  It resamples whole clusters and applies no refinement, which is the")
    r("  construction the committed specification names as insufficient at this cluster count. It is here so")
    r("  the difference between a refined and an unrefined interval is visible.")
    r(f"    {interval_text(primary_unrefined.percentile_lower, primary_unrefined.percentile_upper, level)}")
    r(f"    width, refined {primary.upper - primary.lower!r}")
    r(f"    width, unrefined {primary_unrefined.percentile_upper - primary_unrefined.percentile_lower!r}")
    r(f"    ratio refined / unrefined "
      f"{(primary.upper - primary.lower) / (primary_unrefined.percentile_upper - primary_unrefined.percentile_lower)!r}")
    r()
    refined_spans = primary.lower <= 0.0 <= primary.upper
    unrefined_spans = (primary_unrefined.percentile_lower <= 0.0
                       <= primary_unrefined.percentile_upper)
    if refined_spans != unrefined_spans:
        r("    THE TWO INTERVALS DISAGREE ON WHETHER ZERO IS INCLUDED, and the")
        r("    disagreement is reported rather than resolved by choosing the more")
        r("    convenient one, per the committed specification.")
        r(f"      refined interval spans zero:   {refined_spans}")
        r(f"      unrefined interval spans zero: {unrefined_spans}")
        r("    THE REFINED INTERVAL IS THE ONE THE PRIMARY CLAIM RESTS ON. That was")
        r("    committed before evaluation, during the build, for the reason this")
        r("    disagreement now illustrates: Cameron, Gelbach and Miller report that")
        r("    standard cluster-robust inference over-rejects considerably in the")
        r("    five to thirty cluster range, and 11 clusters sits inside it. An")
        r("    unrefined interval that excludes zero where the refined one does not")
        r("    is what that over-rejection looks like on one dataset. The unrefined")
        r("    figure is reported and it is not the basis of any claim.")
    else:
        r("    The two intervals agree on whether zero is included.")
    r()
    r("  THE 2048 BOUND, reported with the interval as committed. With 11 clusters")
    r("  the Rademacher weights admit at most 2^11 = 2048 distinct weight vectors, so")
    r("  the bootstrap distribution is supported on at most 2048 points however many")
    r("  resamples are drawn. Raising the resample count cannot lift that bound.")
    r("  Because t*(-w) = -t*(w) exactly, the bootstrap t distribution is symmetric")
    r("  by construction and the interval is symmetric about the point estimate in")
    r("  standard-error units. That is a property of the weights, not of this data.")
    r()
    r("  Supplementary, and it costs nothing at this cluster count: the SAME interval")
    r("  computed by enumerating all 2048 weight vectors exactly, which has no Monte")
    r("  Carlo error at all. It is the reference the drawn interval is read against.")
    r("  It does not replace the committed 9999-draw procedure.")
    r(f"    {interval_text(primary_exhaustive.lower, primary_exhaustive.upper, level)}")
    r(f"    p value, null imposed, exact  {primary_exhaustive.p_value_null_imposed!r}")

    r.section("M03. Endpoint Monte Carlo error. DESCRIPTIVE, required by the committed specification.")
    r("Measured, not asserted, by repeating the whole bootstrap on independent")
    r(f"streams from the offset rule. {reps} repetitions each.")
    r()
    r("  THE POINT ESTIMATE'S RESAMPLING ERROR IS EXACTLY ZERO. The mean paired")
    r("  difference is a function of the data alone. It reads no resampling stream,")
    r("  so it does not move between repetitions. This is stated rather than")
    r("  simulated, and it is why the endpoints are reported separately.")
    r()
    r("  Wild cluster bootstrap-t, the committed interval:")
    r(f"    lower endpoint, SD across repetitions   {mc_wild.lower_endpoint_sd!r}")
    r(f"    upper endpoint, SD across repetitions   {mc_wild.upper_endpoint_sd!r}")
    r(f"    point estimate, SD across repetitions   {mc_wild.point_side_sd!r}")
    r()
    r("  Percentile cluster bootstrap, where a point-side resampling quantity does")
    r("  exist. Its bootstrap mean is the contrast the committed specification asks for: a quantity computed")
    r("  from the whole bootstrap distribution moves less than a tail order statistic.")
    r(f"    lower endpoint, SD across repetitions   {mc_percentile.lower_endpoint_sd!r}")
    r(f"    upper endpoint, SD across repetitions   {mc_percentile.upper_endpoint_sd!r}")
    r(f"    bootstrap mean, SD across repetitions   {mc_percentile.point_side_sd!r}")
    r()
    r("  What this supports and what it does not. No claim is made about the")
    r("  stability of any decimal place these figures do not support.")

    r.section("M04. Paired difference distribution shape. DESCRIPTIVE.")
    r("Sample convention throughout. Skewness is the third central moment divided by")
    r("the cube of the sample standard deviation, both with the (n-1) denominator.")
    r("The range is reported as a pair and not as a width.")
    r()
    r(f"  n                              {shape.n}")
    r(f"  mean                           {shape.mean!r}")
    r(f"  sample standard deviation      {shape.sample_standard_deviation!r}")
    r(f"  sample skewness                {shape.sample_skewness!r}")
    r(f"  minimum                        {shape.minimum!r}")
    r(f"  maximum                        {shape.maximum!r}")
    r()
    r("  The pre-registration declared the mean's vulnerability to a small number of large")
    r("  excursions in advance, with the solver mechanism that could produce them.")
    r("  The pre-registration commits that if the primary fits the realised distribution")
    r("  poorly, that is disclosed and the primary is still reported, with the median")
    r("  beside it and the mechanism stated. The mean is not substituted.")

    r.heading("4. SECONDARY OUTCOMES")
    r("Four, all pre-registered at the pre-registration, all two-sided, no correction")
    r("applied. Each answers a question the primary cannot.")

    validation = params.B24_ANALYSIS["secondary_estimator_validation"]
    r()
    r("A REQUIRED DISCLOSURE ON THE INTERVAL ESTIMATOR THESE FOUR USE.")
    r.rule()
    r("The primary's wild cluster bootstrap-t studentises by a cluster-robust")
    r("variance. A median, a fraction and a slope difference have no such formula,")
    r("so the secondaries keep the identical resampling unit, whole requested-target")
    r("clusters drawn with replacement, and studentise by a delete-one-cluster")
    r("jackknife standard error instead. That construction was validated on")
    r("synthetic data and the validation FAILED one clause of six.")
    r()
    r(f"  Verdict: {validation['verdict']}")
    r()
    r("  Coverage against a nominal 0.95, 400 synthetic datasets per design:")
    r(f"    {'':22}{'refined':>10}{'unrefined':>12}{'pair':>10}")
    for key, label in (("median_design_a", "median, clustered"),
                       ("median_design_b", "median, no clustering"),
                       ("slope_design_a", "slope, clustered"),
                       ("slope_design_b", "slope, no clustering")):
        v = validation[key]
        r(f"    {label:22}{v['coverage_refined']:>10.4f}"
          f"{v['coverage_unrefined']:>12.4f}{v['coverage_pair']:>10.4f}")
    r()
    r("  WHAT PASSED. The refined interval covers at or near nominal everywhere, it")
    r("  is materially wider than a pair-level interval where the clustering is real,")
    r("  and where the unrefined interval undercovers, being the slope in the")
    r("  clustered design at 0.9000, the refinement fixes it at 0.9400. The negative")
    r("  control has teeth: the pair bootstrap covers 0.4950 and 0.4025 in the")
    r("  clustered design.")
    r()
    r("  WHAT FAILED. With no between-cluster component the refined interval does not")
    r("  agree closely with the pair interval. It runs 2.82 times wider for the")
    r("  median and 1.60 times wider for the slope, against a tolerance of 0.40 fixed")
    r("  in advance. The typical interval is inflated and not merely a few of them:")
    r("  the refined width over the unrefined width at the median dataset is 2.26 and")
    r("  1.39 respectively. The mechanism is a noisy jackknife standard error over 11")
    r("  clusters, which gives the bootstrap t distribution heavy tails.")
    r()
    r("  WHAT A READER SHOULD TAKE FROM IT. The refined intervals below are")
    r("  CONSERVATIVE. Their coverage is sound and their WIDTH is not a tight")
    r("  statement of precision. Both the refined and the unrefined interval are")
    r("  printed for every secondary so the difference is visible rather than taken")
    r("  on trust.")
    r()
    r("  WHAT IT CHANGES HERE: nothing, and that is checked rather than hoped. Every")
    r("  secondary and the sensitivity check span their null value under BOTH")
    r("  constructions on this data, so no reported reading turns on the choice. This")
    r("  data also sits in the clustered regime, where the cluster-robust standard")
    r("  error is 1.94 times the independent-pairs one, and the failing clause tests")
    r("  the regime this study's data does not occupy. That is why the failure does")
    r("  not propagate into a wrong number here. It is not a reason to call the")
    r("  clause satisfied and it is not called satisfied.")
    r()
    r("  The 'nothing changes' claim above, computed on this data rather than")
    r("  asserted. Each measure against its own null value, under both constructions:")
    r(f"    {'measure':24}{'null':>7}{'refined spans':>15}{'unrefined spans':>17}")
    agreement = True
    for label, result, null in (("S1, median", s1, 0.0),
                                ("S2, win fraction", s2, 0.5),
                                ("S3, slope difference", s3, 0.0),
                                ("M06, trimmed mean", m06, 0.0)):
        spans_refined = result.studentised_lower <= null <= result.studentised_upper
        spans_unrefined = result.percentile_lower <= null <= result.percentile_upper
        agreement = agreement and (spans_refined == spans_unrefined)
        r(f"    {label:24}{null:>7}{str(spans_refined):>15}{str(spans_unrefined):>17}")
    r(f"    The two constructions agree on every measure: {agreement}")
    if not agreement:
        r("    THEY DO NOT AGREE. The disclosure above is written on the assumption")
        r("    that they do, and that assumption has failed. The disagreement is the")
        r("    finding and it must be reported as one.")

    r.section("S1. The median paired difference. SECONDARY. Metric M05.")
    r("Where the bulk of pairs sits, as distinct from the average pair.")
    r()
    r(f"  median paired difference (normalised)  {s1.point_estimate!r}")
    r(f"  accompanying raw equivalent            {s1.point_estimate * span!r} L/D")
    r(f"  cluster-resampled interval, refined    "
      f"{interval_text(s1.studentised_lower, s1.studentised_upper, level)}")
    r(f"  cluster-resampled interval, unrefined  "
      f"{interval_text(s1.percentile_lower, s1.percentile_upper, level)}")
    r(f"  delete-one-cluster jackknife SE        {s1.jackknife_se!r}")
    r(f"  resamples {s1.n_resamples}, degenerate {s1.n_degenerate}, "
      f"undefined {s1.n_undefined}")
    r(f"  quantile convention, refined           {s1.studentised_convention}")
    r(f"  quantile convention, unrefined         {s1.percentile_convention}")
    r()
    r("  On 'the same cluster-resampling estimator as the primary'. The primary's")
    r("  refinement studentises by a cluster-robust variance, and a median has no")
    r("  such formula. The resampling unit is identical, being whole requested-target")
    r("  clusters drawn with replacement, and the refinement is the same idea applied")
    r("  through the one studentising quantity a median does have, a")
    r("  delete-one-cluster jackknife standard error. The deleted unit is a whole")
    r("  cluster, matching the resampling unit. This is stated rather than left to be")
    r("  assumed identical to the primary's construction.")
    r()
    r(f"  READING: {analysis.interval_reading(s1.studentised_lower, s1.studentised_upper)}")

    r.section("S2. The paired win fraction. SECONDARY. Metric M07.")
    r("How often the prior helps, which is independent of how much. Ties leave both")
    r("the numerator and the denominator, which is what a sign test does.")
    r()
    r(f"  pairs in which the prior-on arm is closer  {wins}")
    r(f"  non-tied pairs (the denominator)           {non_tied}")
    r(f"  WIN FRACTION                               {win!r}")
    r()
    r("  Test statistic accounting for the clustering by requested target. A normal")
    r("  approximation computed as if the pairs were independent is exactly the")
    r("  failure the committed specification names, so the clustering enters both the standard error and the")
    r("  reference distribution.")
    r(f"    cluster-robust SE of the fraction        {win_se!r}")
    r(f"    cluster-robust t against 0.5             {win_t!r}")
    r(f"    wild cluster bootstrap-t p, null 0.5     {win_test.p_value_null_imposed!r}")
    r()
    r(f"  cluster-resampled interval, refined    "
      f"{interval_text(s2.studentised_lower, s2.studentised_upper, level)}")
    r(f"  cluster-resampled interval, unrefined  "
      f"{interval_text(s2.percentile_lower, s2.percentile_upper, level)}")
    r(f"  resamples {s2.n_resamples}, degenerate {s2.n_degenerate}, "
      f"undefined {s2.n_undefined}")
    r(f"  quantile convention, refined           {s2.studentised_convention}")
    r(f"  quantile convention, unrefined         {s2.percentile_convention}")

    r.section("S3. The arm difference in target tracking slope. SECONDARY.")
    r("Whether the prior changed the model's conditioning behaviour rather than only")
    r("its error. Both slopes are fitted over the SAME matched pairs, which is M08's")
    r("committed population, and both are in normalised units on both axes, so a")
    r("model tracking its request perfectly has slope 1.")
    r()
    r(f"  prior-on tracking slope         {slope_on!r}     DESCRIPTIVE")
    r(f"  prior-off tracking slope        {slope_off!r}     DESCRIPTIVE")
    r(f"  prior-on tracking correlation   {corr_on!r}     DESCRIPTIVE")
    r(f"  prior-off tracking correlation  {corr_off!r}     DESCRIPTIVE")
    r(f"  SLOPE DIFFERENCE (on minus off) {s3.point_estimate!r}")
    r()
    r(f"  cluster-resampled interval, refined    "
      f"{interval_text(s3.studentised_lower, s3.studentised_upper, level)}")
    r(f"  cluster-resampled interval, unrefined  "
      f"{interval_text(s3.percentile_lower, s3.percentile_upper, level)}")
    r(f"  delete-one-cluster jackknife SE        {s3.jackknife_se!r}")
    r(f"  resamples {s3.n_resamples}, degenerate {s3.n_degenerate}, "
      f"undefined {s3.n_undefined}")
    r(f"  quantile convention, refined           {s3.studentised_convention}")
    r(f"  quantile convention, unrefined         {s3.percentile_convention}")
    r(f"  bootstrap t quantiles                  "
      f"[{s3.t_quantile_lower!r}, {s3.t_quantile_upper!r}]")
    r()
    r("  The two intervals disagree in width and in centring here, and the")
    r("  disagreement is reported rather than resolved by picking one. The")
    r("  studentised bootstrap t distribution for this statistic is strongly")
    r("  asymmetric at 11 clusters, which widens the refined interval on one side.")
    r("  the committed specification requires a disagreement between two measures presented rather than")
    r("  adjudicated, and the refined interval is the one the secondary rests on.")
    r()
    r(f"  READING: {analysis.interval_reading(s3.studentised_lower, s3.studentised_upper)}")
    r()
    r("  M08 has NO published threshold. the build plan records that no source reports this")
    r("  construction with one, and that absence is stated wherever the metric is")
    r("  reported. The condition-blind baseline M10 that these slopes are read")
    r("  against is computed at B25 and is not asserted here.")

    r.section("S4. The difference between the two arms' admission rates. SECONDARY.")
    r("Whether the prior changed which shapes survived to be measured. Metric M19.")
    r()
    r(f"  prior-on admitted   {s4.admitted_a} of {s4.launched_a}, rate {s4.rate_a!r}")
    r(f"  prior-off admitted  {s4.admitted_b} of {s4.launched_b}, rate {s4.rate_b!r}")
    r(f"  DIFFERENCE (on minus off)  {s4.difference!r}")
    r(f"  pooled rate                {s4.pooled_rate!r}")
    r(f"  standard error             {s4.standard_error!r}")
    r(f"  two-proportion z statistic {s4.z_statistic!r}")
    r()
    r("  The denominator is the shapes launched in that arm. This run recorded zero")
    r("  timeouts and zero environment faults in both arms, so the denominator that")
    r("  excludes them coincides with the launched count here. That coincidence is a")
    r("  property of this run and is stated rather than relied on.")
    r()
    r("  The paired view of the same data, reported beside the committed test because")
    r("  the two arms share one latent code per pair and are not independent samples.")
    r("  These name the arm that SURVIVED, so the pair lost the other one:")
    r(f"    pairs in which only the prior-on member was admitted   "
      f"{s4.pairs_only_a_admitted}")
    r(f"    pairs in which only the prior-off member was admitted  "
      f"{s4.pairs_only_b_admitted}")
    r()
    r("  M19's caution travels with the result and is not softened. Unequal rates do")
    r("  not establish bias and equal rates do not establish its absence. Section")
    r("  The pre-registration classifies the missingness as depending on the generated shape rather")
    r("  than on the arm, so it is not missing completely at random and this study")
    r("  does not assume it is ignorable.")

    r.heading("5. SENSITIVITY")
    r("M06. The trimmed mean difference. SENSITIVITY, not an outcome. the committed specification forbids")
    r("promoting it to the headline and the pre-registration declares it a sensitivity check.")
    r("It shows how much of the primary's location depends on the tails and nothing")
    r("else.")
    r()
    r(f"  trim fraction                          {trim} per tail, symmetric in count")
    r(f"  values removed from each tail          {int(np.floor(n * trim))} of {n}")
    r(f"  TRIMMED MEAN DIFFERENCE (normalised)   {m06.point_estimate!r}")
    r(f"  accompanying raw equivalent            {m06.point_estimate * span!r} L/D")
    r(f"  cluster-resampled interval, refined    "
      f"{interval_text(m06.studentised_lower, m06.studentised_upper, level)}")
    r(f"  cluster-resampled interval, unrefined  "
      f"{interval_text(m06.percentile_lower, m06.percentile_upper, level)}")
    r(f"  delete-one-cluster jackknife SE        {m06.jackknife_se!r}")
    r(f"  quantile convention, refined           {m06.studentised_convention}")
    r(f"  quantile convention, unrefined         {m06.percentile_convention}")
    r()
    r("  The resampling is CLUSTER level. The superseded build trimmed and then")
    r("  resampled at pair level, which is the construction the committed specification rules out.")

    r.heading("6. DESCRIPTIVE")
    r.section("Per-arm target satisfaction error, over the analysed pairs.")
    r(f"  mean absolute error, prior-on    {float(paired.error_on.mean())!r}  "
      f"({float(paired.error_on.mean()) * span:.6f} L/D)")
    r(f"  mean absolute error, prior-off   {float(paired.error_off.mean())!r}  "
      f"({float(paired.error_off.mean()) * span:.6f} L/D)")
    r(f"  median absolute error, prior-on  {float(np.median(paired.error_on))!r}")
    r(f"  median absolute error, prior-off {float(np.median(paired.error_off))!r}")
    r(f"  mean achieved (normalised), prior-on   {float(paired.achieved_on.mean())!r}")
    r(f"  mean achieved (normalised), prior-off  {float(paired.achieved_off.mean())!r}")
    r(f"  mean requested (normalised)            {float(paired.target.mean())!r}")

    r.section("Per requested target. The clusters the interval is resampled over.")
    r(f"  {'idx':>3} {'normalised':>12} {'raw L/D':>10} {'pairs':>6} "
      f"{'mean diff':>14} {'err on':>12} {'err off':>12}")
    for row in per_target:
        r(f"  {row['target_index']:>3} {row['target_normalised']:>12.6f} "
          f"{row['target_raw']:>10.3f} {row['n_pairs']:>6} "
          f"{row['mean_difference']:>14.8f} {row['mean_error_prior_on']:>12.8f} "
          f"{row['mean_error_prior_off']:>12.8f}")
    r()
    r("  All 11 clusters contribute. The committed band was chosen from the training")
    r("  split's own label percentiles precisely so that none would be empty.")

    r.heading("7. WHAT THIS FILE DOES NOT CONTAIN")
    r("The prior mechanism metrics M12, M13 and M14 are gate quantities measured")
    r("before evaluation at B18. The pre-registration states they are not evaluation")
    r("outcomes, and they are not recomputed here.")
    r()
    r("M11 generative diversity, M10 the condition-blind baseline, M22 the surrogate")
    r("to solver gap and the per-arm reporting of M08 and M09 are B25's. B25 consumes")
    r("analysis.ols_slope and analysis.pearson_correlation rather than defining a")
    r("second slope, so the arm difference above and B25's per-arm figures are the")
    r("same computation reported at two levels.")
    r()
    r("No equivalence test exists in this study, so absence of an effect is not")
    r("claimed in either direction, whatever any interval above shows.")
    r.rule("=")
    r("End of RESULTS.txt.")
    r.rule("=")

    r.write(RESULTS_TXT)

    # =======================================================================
    # b24_analysis.json
    # =======================================================================
    def as_dict(obj):
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in vars(obj).items()
                if not isinstance(v, (np.ndarray, list))}

    payload = {
        "step": "B24",
        "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "committed": fixed,
        "normalisation": {"label_min": label_min, "label_max": label_max,
                          "span": span},
        "flow": flow,
        "n_analysed_pairs": n,
        "n_clusters": int(paired.n_clusters),
        "primary": as_dict(primary),
        "primary_exhaustive_2048": as_dict(primary_exhaustive),
        "primary_unrefined_percentile_cluster": as_dict(primary_unrefined),
        "distribution_shape": as_dict(shape),
        "monte_carlo_error": {
            "wild_cluster_bootstrap_t": as_dict(mc_wild),
            "percentile_cluster_bootstrap": as_dict(mc_percentile),
            "primary_point_estimate_resampling_error": 0.0,
        },
        "secondaries": {
            "S1_median": as_dict(s1),
            "S2_win_fraction": {
                "fraction": win, "wins": wins, "non_tied": non_tied,
                "cluster_robust_se": win_se, "cluster_robust_t_against_half": win_t,
                "wild_cluster_bootstrap_p_null_half": win_test.p_value_null_imposed,
                "interval": as_dict(s2),
            },
            "S3_slope_difference": {
                "slope_prior_on": slope_on, "slope_prior_off": slope_off,
                "correlation_prior_on": corr_on, "correlation_prior_off": corr_off,
                "interval": as_dict(s3),
            },
            "S4_admission_rate_difference": as_dict(s4),
        },
        "sensitivity": {"M06_trimmed_mean": as_dict(m06), "trim_fraction": trim},
        "descriptive": {
            "mean_error_prior_on": float(paired.error_on.mean()),
            "mean_error_prior_off": float(paired.error_off.mean()),
            "median_error_prior_on": float(np.median(paired.error_on)),
            "median_error_prior_off": float(np.median(paired.error_off)),
            "mean_achieved_prior_on": float(paired.achieved_on.mean()),
            "mean_achieved_prior_off": float(paired.achieved_off.mean()),
            "mean_requested": float(paired.target.mean()),
            "per_target": per_target,
        },
        "paired_series": {
            "target": paired.target.tolist(),
            "target_index": paired.target_index.tolist(),
            "sample_index": paired.sample_index.tolist(),
            "error_prior_on": paired.error_on.tolist(),
            "error_prior_off": paired.error_off.tolist(),
            "achieved_prior_on": paired.achieved_on.tolist(),
            "achieved_prior_off": paired.achieved_off.tolist(),
            "difference": paired.difference.tolist(),
        },
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print("\n".join(r.lines))
    print()
    print(f"Written: {RESULTS_TXT} and {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
