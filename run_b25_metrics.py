"""B25. The metrics and diagnostics.

Computes the per-arm target tracking slopes and correlations, the
condition-blind baseline, the generative diversity metric, the per-arm
evaluability rates and the surrogate to solver gap.

Three definitions matter here and each is fixed elsewhere. Diversity has one
definition, params.DIVERSITY_DEFINITION, read and not restated, and the metric
is the mean of the within-target statistic across the whole requested range.
The tracking slope is fitted over matched pairs, using
run_b24_analysis.build_pairs, called rather than reimplemented, so it runs over
the same population every other reported statistic uses. The condition-blind
baseline is computed rather than assumed: the requested target column is
shuffled by one permutation of the launched slots, the shapes are generated
again from the same latent codes, and all of them go through the solver.

Two standing checks run first and either can stop the step. B23's paired
geometry is regenerated from B23's own generation seed and must equal the
stored coefficients at a worst deviation of exactly 0.0. The surrogate
held-out error read from the parameter record must match that record's own
derivation text.

This driver decides nothing. Every committed value is read from params.py and
every estimator lives in analysis.py or evaluate.py.

Run order      9 of 10. After B24, before run_committed_training_history.py.
Reads          b23_evaluation.json, b24_analysis.json, committed_model.pt, the
               build artifacts, and the XFOIL binary
Writes         b25_metrics.json, b25_condition_blind_progress.jsonl, and its
               section appended to RESULTS.txt
Runtime        220 solver calls totalling 252.0 s, mean 1.146 s, maximum
               5.861 s, for the condition-blind baseline, per Table A3
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import analysis
import dataset
import evaluate
import model as model_mod
import params
import run_b16_weight_sweep as b16
import run_b24_analysis as b24

EVALUATION_RECORDS = Path("b23_evaluation.json")
B24_RESULTS = Path("b24_analysis.json")
NORMALIZATION = Path("normalization.npz")
MODEL_PATH = Path("committed_model.pt")
RESULTS_TXT = Path("RESULTS.txt")
RESULTS_JSON = Path("b25_metrics.json")
BASELINE_PROGRESS = Path("b25_condition_blind_progress.jsonl")

BUILD_STEP = 25
ARMS = ("prior_on", "prior_off")

# The one line of RESULTS.txt this file appends after, and the marker it
# writes so a rerun replaces its own section instead of appending a second.
END_LINE = "End of RESULTS.txt."
B25_MARKER = "B25. METRICS AND DIAGNOSTICS"


# ---------------------------------------------------------------------------
# Committed values, read and never restated.
# ---------------------------------------------------------------------------

def committed():
    def value(key):
        slot = params.PARAMS[key]
        if isinstance(slot.value, params.Pending):
            raise ValueError(f"{key} is PENDING; B25 may not run against it")
        return slot.value

    return {
        "diversity_definition": value("diversity_definition"),
        "requested_target_band": value("requested_target_band"),
        "samples_per_target": int(value("samples_per_target")["samples_per_target"]),
        "minimum_converged_points":
            int(value("minimum_converged_sweep_points")["minimum_converged_points"]),
        "base_seed": int(value("seeds")),
        "tracking_population": params.B25_METRICS["tracking_population"],
        "diversity_grid_at_b25": params.B25_METRICS["diversity_grid_at_b25"],
    }


def surrogate_heldout_reference():
    """M23, read from B13's own record and NOT recomputed here.

    the build plan assigns M23's computation to B13, and B13 computed it in
    the build on the validation split against the training split's own mean
    label. Recomputing it at B25 would be a second computation of a reported
    number, which is exactly what this step exists to stop. So the two
    figures are transcribed, and the transcription is checked against the
    parameter record's own derivation text rather than trusted, so a
    transcription that drifts from the record fails here instead of reaching
    the paper.
    """
    ensemble_mae, baseline_mae = 2.6538, 13.1704
    text = params.PARAMS["surrogate_members_training_schedule"].derivation
    for figure in (ensemble_mae, baseline_mae):
        if f"{figure:.4f}" not in text:
            raise ValueError(
                f"M23 transcription {figure} does not appear in the B13 record; "
                f"the transcription and the record have drifted apart")
    return {
        "ensemble_mean_absolute_error_raw": ensemble_mae,
        "training_split_mean_baseline_raw": baseline_mae,
        "units": "raw max(CL/CD)",
        "source": "the build plan B13, the build. Read from the parameter record, not "
                  "recomputed at B25.",
    }


# ---------------------------------------------------------------------------
# The model, and the check that this step reproduces B23's own generation.
# ---------------------------------------------------------------------------

def load_model(artifacts):
    cvae = model_mod.CVAE(artifacts.x_all.shape[1], artifacts.cond_all.shape[1],
                          b16.ARCH).to(model_mod.DTYPE)
    cvae.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    cvae.eval()
    return cvae


def regenerate_b23(cvae, artifacts, grid, n_samples):
    """Regenerate B23's own paired geometry from B23's own generation seed.

    Standing check, not a falsification check. M10 holds the latent codes
    fixed and moves only the target column, so the codes it reuses must be
    B23's codes and not merely codes drawn by the same rule. This regenerates
    them and requires the result to equal B23's stored coefficients exactly.
    A mismatch means the model, the seed rule or the generation path moved
    between the build and now, and the baseline would then differ from B23
    in two ways instead of one.
    """
    return evaluate.paired_generation(
        cvae, artifacts.reference_signature, b16.ARCH.latent_dim,
        model_mod.seed_int(23, 0), targets=grid, n_samples=n_samples)


def check_regeneration(pg, records):
    worst = 0.0
    for record in records:
        ti, si = record["target_index"], record["sample_index"]
        for arm, x in (("prior_on", pg.x_set), ("prior_off", pg.x_clear)):
            stored = np.asarray(record[arm]["standardised_coefficients"], dtype=float)
            worst = max(worst, float(np.max(np.abs(stored - x[ti, si].numpy()))))
    if worst != 0.0:
        raise ValueError(
            f"regenerating B23's geometry from seed_int(23, 0) does not reproduce "
            f"the stored coefficients: worst deviation {worst!r}. B25 refuses to "
            f"build a baseline whose only difference from B23 is meant to be the "
            f"target column.")
    return worst


# ---------------------------------------------------------------------------
# M11. Generative diversity.
# ---------------------------------------------------------------------------

def diversity(cvae, artifacts, grid, definition):
    """M11, on params.DIVERSITY_DEFINITION, on the COMMITTED diversity grid.

    What varies within a target is the latent code, drawn independently per
    sample. What is held fixed is the requested target and the flag. Both
    arms share each code, so the two arms' figures are measured on the same
    draws and a difference between them is the flag's doing.

    THE GRID IS evaluate.target_grid(), being the 11 evenly spaced normalised
    targets from 0.0 to 1.0 inclusive that the pre-registration of the issued
    pre-registration fixes as a committed value. It is the same grid B16's
    sweep and B18's gate evaluate this definition on, so all three steps'
    figures are comparable point for point and no grid disclosure is needed.

    CORRECTED in the build. This step previously evaluated M11 on the
    requested target band instead, reasoning from M11's own phrase "across
    the whole requested range" and from B25's own logic text. That reading is
    defensible on those two sentences and it departed from a value section
    the pre-registration commits, which the pre-registration does not permit. The committed grid
    governs. Nothing else at this step moved: the requested target band is
    still what M08, M09, M10, M16 and M22 use, and the band remains the grid
    F04 and the paired analysis run on.
    """
    n_samples = int(definition["n_samples_per_target"])
    pg = evaluate.paired_generation(
        cvae, artifacts.reference_signature, b16.ARCH.latent_dim,
        model_mod.seed_int(BUILD_STEP, 0), targets=grid, n_samples=n_samples)
    per_target = {arm: evaluate.within_target_spread(x).numpy()
                  for arm, x in (("prior_on", pg.x_set), ("prior_off", pg.x_clear))}
    return {
        "n_targets": len(grid),
        "n_samples_per_target": n_samples,
        "targets": [float(t) for t in grid],
        "generation_seed": model_mod.seed_int(BUILD_STEP, 0),
        "within_target_statistic": definition["within_target_statistic"],
        "across_target_reduction": definition["across_target_reduction"],
        "grid": "the committed diversity grid, 11 evenly spaced normalised "
                "targets from 0.0 to 1.0 inclusive (the pre-registration)",
        "per_target": {arm: [float(v) for v in per_target[arm]] for arm in ARMS},
        "mean_across_range": {arm: float(per_target[arm].mean()) for arm in ARMS},
        "difference_on_minus_off": float(per_target["prior_on"].mean()
                                         - per_target["prior_off"].mean()),
    }


# ---------------------------------------------------------------------------
# M08 and M09. Tracking, over matched pairs.
# ---------------------------------------------------------------------------

def tracking(paired):
    """M08 and M09, per arm, over the matched pairs.

    ols_slope and pearson_correlation are analysis.py's, called here on the
    population M08 names. B24 called the same two functions on the same
    population to form its secondary S3, so the arm difference reported
    there and the per-arm figures reported here are one computation read at
    two levels, and the driver checks that rather than asserting it.

    Both axes are normalised, so a model tracking its request perfectly has
    slope exactly 1.
    """
    out = {"n_pairs": len(paired), "population": "matched pairs"}
    for arm, achieved in (("prior_on", paired.achieved_on),
                          ("prior_off", paired.achieved_off)):
        out[arm] = {
            "slope": analysis.ols_slope(paired.target, achieved),
            "correlation": analysis.pearson_correlation(paired.target, achieved),
        }
        # F04 plots signed error against requested target and draws the fitted
        # trend through it. the build plan states that trend's slope is M08's slope
        # less one. It is arithmetic on a number already computed, not a
        # second fit, and it is labelled as derived.
        out[arm]["f04_trend_slope_derived"] = out[arm]["slope"] - 1.0
    return out


def check_against_b24(tracked, b24_blob):
    """The one-computation claim, checked on this data rather than asserted."""
    s3 = b24_blob["secondaries"]["S3_slope_difference"]
    recorded = {
        "prior_on": {"slope": s3["slope_prior_on"], "correlation": s3["correlation_prior_on"]},
        "prior_off": {"slope": s3["slope_prior_off"], "correlation": s3["correlation_prior_off"]},
    }
    mismatches = []
    for arm in ARMS:
        for key in ("slope", "correlation"):
            if tracked[arm][key] != recorded[arm][key]:
                mismatches.append(
                    f"{arm}.{key}: B25 {tracked[arm][key]!r} against B24 {recorded[arm][key]!r}")
    return {
        "identical_to_b24": not mismatches,
        "mismatches": mismatches,
        "note": "B24 formed its secondary S3 from these same two slopes. Equality "
                "here is bit-for-bit and is the evidence that one computation is "
                "being reported at two levels rather than two computations agreeing.",
    }


# ---------------------------------------------------------------------------
# M10. The condition-blind baseline. Shuffle the target column, regenerate.
# ---------------------------------------------------------------------------

def shuffled_column(grid, n_samples):
    """One permutation of the launched slots' requested targets.

    A permutation and not a fresh draw, so the multiset of requested targets
    is exactly the one B23 launched and the cluster sizes are unchanged. Slots
    the permutation happens to leave alone are left alone; removing them would
    make this something other than a shuffle, and their count is reported.
    """
    original = np.repeat(np.asarray(grid, dtype=float), n_samples)
    perm = dataset.rng_for(BUILD_STEP, 1).permutation(len(original))
    shuffled = original[perm]
    return original, shuffled, perm, int((perm == np.arange(len(perm))).sum())


def baseline_header(grid, n_samples, shuffled, min_points, timeout, n_points):
    return {
        "step": "B25",
        "metric": "M10, the condition-blind baseline slope",
        "generation_seed": model_mod.seed_int(23, 0),
        "shuffle_stream": f"dataset.rng_for({BUILD_STEP}, 1)",
        "model": str(MODEL_PATH),
        "targets": [float(t) for t in grid],
        "samples_per_target": n_samples,
        "conditioning_targets": [float(t) for t in shuffled],
        "minimum_converged_points": min_points,
        "timeout_seconds": timeout,
        "n_points_per_surface": n_points,
    }


def load_baseline_progress(path, header):
    """Resumable, and it refuses to resume onto a different run. Same
    construction as B23's, for the same reason: a rerun under a different
    seed, shuffle or model must refuse rather than mix two runs into one
    record."""
    done, records = set(), []
    if not os.path.exists(path):
        return done, records
    with open(path) as fh:
        first = fh.readline()
        if not first.strip():
            return done, records
        stored = json.loads(first)
        if stored.get("kind") != "header":
            raise ValueError(f"{path} does not start with a run header; refusing to resume")
        for key in ("generation_seed", "model", "targets", "conditioning_targets",
                    "samples_per_target", "minimum_converged_points"):
            if stored["header"][key] != header[key]:
                raise ValueError(
                    f"{path} was written for a different run: {key} differs. "
                    f"Refusing to resume. Move that file aside for a fresh run.")
        for line in fh:
            if line.strip():
                record = json.loads(line)
                done.add(record["slot"])
                records.append(record)
    return done, records


def run_condition_blind(cvae, artifacts, pg, grid, n_samples, min_points,
                        progress_path, verbose=True):
    """Generate at the shuffled targets from B23's own codes, and solve.

    Every record carries two target fields and they are not the same thing.

      conditioning_target  what the model READ. The shuffled value.
      target               what the slot was ORIGINALLY asked for, and what
                           the slope is fitted against. The model never saw it.

    That separation is the metric. Fitting achieved efficiency on the column
    the model read would measure tracking again and return something near
    M08, which is not a chance reference.
    """
    original, shuffled, perm, n_fixed = shuffled_column(grid, n_samples)
    settings = evaluate.solver_settings()
    timeout = evaluate.committed_timeout()
    bounds = evaluate.plausibility_bounds()
    n_points = params.B19_CONSISTENCY_GATE["gate_one"]["n_points_per_surface"]

    header = baseline_header(grid, n_samples, shuffled, min_points, timeout, n_points)
    z_flat = pg.z.reshape(-1, pg.z.shape[-1])
    x_set, x_clear = evaluate.decode_both_arms(
        cvae, z_flat, shuffled, artifacts.reference_signature)

    done, records = load_baseline_progress(progress_path, header)
    fresh = not os.path.exists(progress_path)
    if fresh:
        with open(progress_path, "w") as fh:
            fh.write(json.dumps({"kind": "header", "header": header}) + "\n")

    if verbose:
        print(f"M10. condition-blind baseline: {len(original)} slots, "
              f"{2 * len(original)} solver calls")
        print(f"  slots the permutation left in place: {n_fixed} of {len(original)}")
        print(f"  already done: {len(done)} ({'fresh run' if fresh else 'resuming'})",
              flush=True)

    with open(progress_path, "a") as fh:
        for slot in range(len(original)):
            if slot in done:
                continue
            ti, si = divmod(slot, n_samples)
            row = {
                "slot": slot, "target_index": ti, "sample_index": si,
                "target": float(original[slot]),
                "conditioning_target": float(shuffled[slot]),
            }
            for arm, x in (("prior_on", x_set), ("prior_off", x_clear)):
                upper, lower = evaluate.standardised_to_coefficients(
                    x[slot], artifacts.std_stats)
                rec = evaluate.evaluate_coefficients(
                    upper[0], lower[0], name=f"b25_m10_{arm}_slot{slot}",
                    bounds=bounds, settings=settings, timeout_seconds=timeout,
                    n_points_per_surface=n_points)
                row[arm] = {
                    "status": rec.status, "plausible": rec.plausible,
                    "reason": rec.reason, "n_converged": rec.n_converged,
                    "n_usable": rec.n_usable, "label": rec.label,
                    "elapsed_seconds": rec.elapsed_seconds,
                    "admitted": analysis.is_admitted(rec.label, rec.n_usable, min_points),
                }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            records.append(row)
            done.add(slot)
            if verbose and (slot + 1) % 10 == 0:
                both = sum(1 for r in records
                           if r["prior_on"]["admitted"] and r["prior_off"]["admitted"])
                print(f"  slot {slot + 1}/{len(original)}: {both} pairs analysed so far",
                      flush=True)

    records.sort(key=lambda r: r["slot"])
    return records, header, n_fixed, perm


def condition_blind_metric(records, header, n_fixed, label_min, label_max, min_points):
    """M10 itself. Admit, pair, then fit achieved on the ORIGINAL column.

    The admission rule and the pairing are B24's, called and not rebuilt, so
    the baseline population is formed by the same rule as the population M08
    is fitted over. A baseline formed under a looser rule would not be M08
    recomputed; it would be a different statistic wearing the name.
    """
    paired, flow = b24.build_pairs(records, label_min, label_max, min_points)
    out = {
        "n_slots": len(records),
        "n_fixed_points_of_the_permutation": n_fixed,
        "flow": flow,
        "n_analysed_pairs": len(paired),
        "population": "matched pairs of the condition-blind run",
        "fitted_against": "the ORIGINAL requested target column, which the model "
                          "never received",
        "conditioned_on": "the shuffled target column",
        "generation_seed": header["generation_seed"],
        "shuffle_stream": header["shuffle_stream"],
    }
    for arm, achieved in (("prior_on", paired.achieved_on),
                          ("prior_off", paired.achieved_off)):
        out[arm] = {
            "slope": analysis.ols_slope(paired.target, achieved),
            "correlation": analysis.pearson_correlation(paired.target, achieved),
        }
    return out, paired


# ---------------------------------------------------------------------------
# M22. The surrogate to solver gap, on admitted generated shapes.
# ---------------------------------------------------------------------------

def surrogate_to_solver(records, artifacts, label_norm, min_points):
    """M22. The ensemble's prediction against the solver's value.

    In RAW max(CL/CD), because M23 is the reference this gap is read against
    and B13 computed M23 in raw units. The ensemble predicts a normalised
    label, so the prediction is carried across by the same B08 artifact the
    solver value is normalised by everywhere else.

    The population is every ADMITTED generated shape from B23, both arms
    pooled, which is M22's own wording. The per-arm split below is the same
    three quantities on each arm's own subset and is labelled as such.
    """
    rows = {arm: {"x": [], "solver": []} for arm in ARMS}
    n_launched = 0
    for record in records:
        n_launched += 1
        for arm in ARMS:
            state = analysis.read_admission(record[arm], min_points)
            if state.admitted:
                rows[arm]["x"].append(record[arm]["standardised_coefficients"])
                rows[arm]["solver"].append(float(state.label))

    def gap(x_list, solver_list):
        x = torch.tensor(np.asarray(x_list, dtype=float), dtype=model_mod.DTYPE)
        with torch.no_grad():
            predicted_norm = artifacts.ensemble.predict_mean(x).numpy()
        predicted = dataset.denormalize_label(predicted_norm, label_norm)
        solver = np.asarray(solver_list, dtype=float)
        g = analysis.surrogate_gap(predicted, solver, "raw max(CL/CD)")
        return {
            "n": g.n,
            "mean_absolute_difference": g.mean_absolute_difference,
            "mean_signed_difference": g.mean_signed_difference,
            "correlation": g.correlation,
            "units": g.units,
        }

    pooled_x = rows["prior_on"]["x"] + rows["prior_off"]["x"]
    pooled_solver = rows["prior_on"]["solver"] + rows["prior_off"]["solver"]
    return {
        "population": "every admitted generated shape from B23, both arms pooled",
        "denominator_note": f"{len(pooled_solver)} admitted shapes of "
                            f"{2 * n_launched} launched",
        "signed_orientation": "predicted minus solver; positive means the surrogate "
                              "read the shape as more efficient than the solver found it",
        "pooled": gap(pooled_x, pooled_solver),
        "per_arm": {arm: gap(rows[arm]["x"], rows[arm]["solver"]) for arm in ARMS},
        "reference": surrogate_heldout_reference(),
    }


# ---------------------------------------------------------------------------
# M16. Validity, per arm, with the denominator stated.
# ---------------------------------------------------------------------------

def validity(evaluation_blob, b24_blob):
    """M16, and the admitted fraction B25's own logic text asks for.

    Neither count is recounted here. The per-arm attrition counts are B23's,
    read from its own record, and the admission counts are B24's, read from
    its own record, because B24 states in its own words that it is the sole
    source of every reported count. What B25 does is form the RATE and state
    the denominator, which is the one thing M16 requires and neither of those
    two steps carries.

    the committed specification requires the timeout's effect on every denominator shown, and the committed specification
    requires the denominator stated. So the denominator is the launched count
    less the timeouts and the environment faults, and both exclusions are
    printed even where they are zero.
    """
    out = {
        "denominator_rule": "launched, less timeouts, less environment faults "
                            "(the committed specification)",
        "counts_read_from": "B23 for the attrition categories, B24 for the "
                            "admission counts. Neither is recounted at B25.",
    }
    for arm in ARMS:
        flow = evaluation_blob["attrition_flow"][arm]
        admission = b24_blob["flow"][arm]
        denominator = flow["launched"] - flow["timeout"] - flow["environment_fault"]
        out[arm] = {
            "launched": flow["launched"],
            "timeout": flow["timeout"],
            "environment_fault": flow["environment_fault"],
            "plausibility_rejected": flow["plausibility_rejected"],
            "denominator": denominator,
            "produced_a_label": flow["produced_a_label"],
            "evaluability_rate": flow["produced_a_label"] / denominator,
            "admitted": admission["admitted"],
            "admitted_fraction_of_launched": admission["admitted"] / admission["launched"],
        }
    return out


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def append_to_results(section_lines, path=RESULTS_TXT):
    """Append B25's section to the file B24 wrote.

    Idempotent on rerun: an existing B25 section is replaced rather than
    followed by a second one. B24's own driver rewrites RESULTS.txt from
    scratch, so B24 must run before B25 and a rerun of B24 drops this
    section, which is the correct behaviour and not a loss: the section is
    regenerated by rerunning B25.
    """
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    marker = next((i for i, line in enumerate(existing) if B25_MARKER in line), None)
    if marker is not None:
        keep = existing[:marker - 1] if marker > 0 else []
    else:
        end = next((i for i, line in enumerate(existing) if line.strip() == END_LINE), None)
        keep = existing[:end - 1] if end is not None and end > 0 else existing
    while keep and not keep[-1].strip():
        keep.pop()
    tail = ["=" * 78, END_LINE, "=" * 78]
    path.write_text("\n".join(keep + section_lines + tail) + "\n", encoding="utf-8")


def build_section(fixed, produced_at, regen_worst, div, tracked, agreement,
                  blind, gap, valid, band):
    r = b24.Report()
    r.heading(f"{B25_MARKER}")
    r()
    r("Produced by run_b25_metrics.py at build step B25, " + produced_at + ".")
    r("Source records: b23_evaluation.json (B23). Pairing: run_b24_analysis.build_pairs.")
    r("Appended to the file B24 wrote. Nothing above this line is altered.")
    r()
    r("Every metric below is computed once and in one place. Three definitions")
    r("changed from the superseded build and each change is stated where it lands.")
    r()
    r("  M08 M09  tracking slope and correlation, per arm, over MATCHED PAIRS")
    r("  M10      the condition-blind baseline, COMPUTED and not assumed")
    r("  M11      generative diversity, ONE definition, averaged across the range")
    r("  M16      evaluability per arm, with the denominator stated")
    r("  M22      the surrogate to solver gap on admitted generated shapes")
    r()
    r("STANDING CHECK, run before anything below was computed. B23's own paired")
    r("geometry was regenerated from B23's own generation seed and compared against")
    r("the coefficients B23 stored. Worst deviation over all 220 shapes: "
      f"{regen_worst:.1e}.")
    r("M10 holds those codes fixed and moves only the target column, so this has to")
    r("be exact for the baseline to differ from B23 in one way rather than two.")

    # --- M11 ---------------------------------------------------------------
    r.section("M11. GENERATIVE DIVERSITY. Requirement (the committed specification).")
    r("ONE definition, params.DIVERSITY_DEFINITION, read and not restated. The")
    r("superseded build computed two different quantities under this word, at")
    r("different sample counts and at different single conditions.")
    r()
    r("  within-target statistic  mean pairwise Euclidean distance in")
    r("                           standardised CST coefficient space, across the")
    r("                           samples at that target")
    r(f"  across-target reduction  {div['across_target_reduction']}")
    r(f"  samples per target       {div['n_samples_per_target']}")
    r(f"  targets                  {div['n_targets']}, the committed diversity grid,")
    r("                           0.0 to 1.0 normalised, inclusive (the pre-registration)")
    r(f"  generation seed          {div['generation_seed']}")
    r()
    r("  Both arms share every latent code, so the two figures below are measured")
    r("  on the same draws and their difference is the flag's doing.")
    r()
    r(f"  MEAN ACROSS THE COMMITTED GRID, prior-on    "
      f"{div['mean_across_range']['prior_on']}")
    r(f"  MEAN ACROSS THE COMMITTED GRID, prior-off   "
      f"{div['mean_across_range']['prior_off']}")
    r(f"  difference, on minus off                    {div['difference_on_minus_off']}")
    r()
    r("  Per grid target. This is F06's series, and the two means above are")
    r("  the across-range mean F06 marks on it.")
    r()
    r("    idx   normalised    raw L/D      prior-on     prior-off")
    for i, t in enumerate(div["targets"]):
        raw = band["raw_low"] + (t - band["normalised_low"]) / (
            band["normalised_high"] - band["normalised_low"]) * (
            band["raw_high"] - band["raw_low"])
        r(f"    {i:3d}   {t:10.6f} {raw:10.3f}   {div['per_target']['prior_on'][i]:11.6f}   "
          f"{div['per_target']['prior_off'][i]:11.6f}")
    r()
    r(f"  READING. The prior-on arm's diversity is "
      f"{div['mean_across_range']['prior_on']:.6f} and the prior-off arm's")
    r(f"  is {div['mean_across_range']['prior_off']:.6f}. M11 exists to make that")
    r("  visible, because a collapsed model and a well-behaved model can share the")
    r("  same mean distance to the reference and no other reported quantity")
    r("  separates them. Whether any difference is uniform across the range or")
    r("  concentrated somewhere in it is what F06 shows and the mean cannot.")
    r()
    r("  GRID. ONE grid, and it is the committed one. The pre-registration fixes the")
    r("  diversity grid as 11 evenly spaced normalised targets from 0.0 to 1.0")
    r("  inclusive, which is the full conditioning range and NOT the requested")
    r("  target band. B16's sweep, B18's gate and this step all evaluate the one")
    r("  definition on that one grid, so their figures are comparable point for")
    r("  point and no cross-grid caution is needed.")
    r()
    r("  WHAT THIS MEANS FOR THE RANGE COVERED, stated because it is a real cost of")
    r("  the committed value. The grid spans 0.0 to 1.0 while the requested target")
    r("  band stops at 0.552208, so roughly half the grid's points sit at targets")
    r("  this study never requests. The figure therefore describes the model's")
    r("  generative spread across its whole conditioning range rather than across")
    r("  the band alone. That is what the pre-registration committed and it is reported as")
    r("  committed. It is NOT recomputed on the band, because a second quantity")
    r("  sharing this word is exactly what the committed specification forbid.")

    # --- M08 M09 -----------------------------------------------------------
    r.section("M08 AND M09. TARGET TRACKING, PER ARM. Requirement and convention.")
    r("Fitted over MATCHED PAIRS, which is the population every other reported")
    r("statistic in this study uses. The superseded build fitted over admitted")
    r("records, a different and larger population.")
    r()
    r(f"  population   {tracked['population']}, n = {tracked['n_pairs']}")
    r("  both axes are normalised, so perfect tracking is slope exactly 1")
    r()
    for arm in ARMS:
        r(f"  {arm:10s} slope        {tracked[arm]['slope']}")
        r(f"  {arm:10s} correlation  {tracked[arm]['correlation']}")
    r()
    r("  F04's fitted trend, DERIVED and not separately fitted. F04 plots signed")
    r("  error against requested target; that trend's slope is M08's slope less one,")
    r("  so the figure and the number above are the same quantity in two forms.")
    for arm in ARMS:
        r(f"    {arm:10s} {tracked[arm]['f04_trend_slope_derived']}")
    r()
    r(f"  ONE COMPUTATION, CHECKED. Identical to B24's S3 inputs: "
      f"{agreement['identical_to_b24']}")
    r("  B24 formed its secondary S3, the arm difference in slope, from these same")
    r("  two slopes. The comparison above is bit-for-bit.")
    if agreement["mismatches"]:
        for line in agreement["mismatches"]:
            r(f"    MISMATCH {line}")
    r()
    r("  M08 HAS NO PUBLISHED THRESHOLD. the build plan records that no source reports")
    r("  this construction with one, and that absence is stated wherever the metric")
    r("  appears. What the slopes are read against is M10 below, which is measured.")

    # --- M10 ---------------------------------------------------------------
    r.section("M10. CONDITION-BLIND BASELINE SLOPE. Requirement.")
    r("COMPUTED. The superseded build printed an assumed baseline as literal text and")
    r("computed nothing. This one cost 220 solver calls.")
    r()
    r("Construction. The requested target COLUMN was shuffled by one permutation of")
    r("the 110 launched slots, and the shapes were generated again from B23's own")
    r("latent codes. Everything else is held identical to B23: same model, same")
    r("decode resolution, same plausibility filter, same solver settings and timeout,")
    r("same admission rule, same complete-case pairing.")
    r()
    r("  THE FIT IS AGAINST THE ORIGINAL COLUMN, which the model never received.")
    r("  The model was CONDITIONED ON the shuffled column. Fitting on the column the")
    r("  model read would measure tracking again and return something near M08,")
    r("  which is not a chance reference.")
    r()
    r(f"  slots launched                          {blind['n_slots']}")
    r(f"  slots the permutation left in place     "
      f"{blind['n_fixed_points_of_the_permutation']}")
    r(f"  ANALYSED PAIRS of the baseline run      {blind['n_analysed_pairs']}")
    r(f"  generation seed (B23's own)             {blind['generation_seed']}")
    r(f"  shuffle stream                          {blind['shuffle_stream']}")
    r()
    for arm in ARMS:
        r(f"  {arm:10s} BASELINE SLOPE        {blind[arm]['slope']}")
        r(f"  {arm:10s} baseline correlation  {blind[arm]['correlation']}")
    r()
    r("  Read against M08 above:")
    for arm in ARMS:
        r(f"    {arm:10s} tracking {tracked[arm]['slope']:.6f}   "
          f"baseline {blind[arm]['slope']:.6f}")
    r()
    r("  Baseline run attrition, per arm, launched to admitted:")
    for arm in ARMS:
        f = blind["flow"][arm]
        r(f"    {arm:10s} launched {f['launched']}, produced a label "
          f"{f['produced_a_label']}, admitted {f['admitted']},")
        r(f"    {'':10s} label but below the minimum point count "
          f"{f['label_but_below_min_points']}")
    r(f"    pairs both admitted {blind['flow']['pairs_both_admitted']}, "
      f"only prior-on {blind['flow']['pairs_only_prior_on_admitted']}, "
      f"only prior-off {blind['flow']['pairs_only_prior_off_admitted']}, "
      f"neither {blind['flow']['pairs_neither']}")
    r()
    r("  M10 HAS NO PUBLISHED CONSTRUCTION. R7 Q4 records a clean not-found: no")
    r("  source builds a deliberately unconditioned or shuffled generator as a chance")
    r("  reference. The closest published anchor is performance blind rather than")
    r("  condition blind, and that distinction is stated rather than elided. This")
    r("  construction is this study's own and is disclosed as its own.")

    # --- M22 ---------------------------------------------------------------
    r.section("M22. SURROGATE TO SOLVER GAP. Requirement.")
    r("On admitted generated shapes, in RAW max(CL/CD), which is the unit M23 is in.")
    r()
    r(f"  population   {gap['population']}")
    r(f"  denominator  {gap['denominator_note']}")
    r("  sign         predicted minus solver, so a positive value means the")
    r("               surrogate read the shape as more efficient than the solver")
    r("               found it to be")
    r()
    p = gap["pooled"]
    r(f"  MEAN ABSOLUTE DIFFERENCE   {p['mean_absolute_difference']}  ({p['units']})")
    r(f"  MEAN SIGNED DIFFERENCE     {p['mean_signed_difference']}")
    r(f"  CORRELATION                {p['correlation']}")
    r(f"  n                          {p['n']}")
    r()
    r("  The same three quantities on each arm's own subset:")
    for arm in ARMS:
        a = gap["per_arm"][arm]
        r(f"    {arm:10s} n {a['n']:4d}   mean abs {a['mean_absolute_difference']:.6f}   "
          f"mean signed {a['mean_signed_difference']:+.6f}   corr {a['correlation']:.6f}")
    r()
    ref = gap["reference"]
    r("  M23, the reference this gap is read against. READ from B13's record and NOT")
    r("  recomputed here, because the build plan assigns its computation to B13 and this")
    r("  step computes each quantity once.")
    r(f"    ensemble held-out mean absolute error   "
      f"{ref['ensemble_mean_absolute_error_raw']}  ({ref['units']})")
    r(f"    training-split-mean baseline            "
      f"{ref['training_split_mean_baseline_raw']}")
    r("    source                                  the build plan B13, the build")
    r()
    r("  A gap is only interpretable against what the surrogate achieves on data it")
    r("  was fitted near. The two figures above are what F07 draws as a band.")
    r()
    ratio = p["mean_absolute_difference"] / ref["ensemble_mean_absolute_error_raw"]
    r("  READING, held to what the figures support and no further. The gap on")
    r(f"  generated shapes is {ratio:.2f} times the held-out reference, so the surrogate")
    r("  is materially less accurate on the shapes the generator produced than on data")
    r("  it was fitted near. The mean signed difference is "
      f"{p['mean_signed_difference']:+.6f}, so the")
    if p["mean_signed_difference"] < 0:
        r("  surrogate reads generated shapes as LESS efficient than the solver finds")
        r("  them. That is the opposite direction from the failure M22 exists to detect:")
        r("  a generator exploiting blind spots in the model it optimised against would")
        r("  show the surrogate OVER-predicting.")
    else:
        r("  surrogate reads generated shapes as MORE efficient than the solver finds")
        r("  them, which is the direction a generator exploiting the model it optimised")
        r("  against would produce.")
    r("  What this does not establish is that no exploitation occurred anywhere. It is")
    r(f"  a mean over {p['n']} shapes, and a concentration at the top of the efficiency")
    r("  range would show in F07 rather than in any of the three scalars above.")

    # --- M16 ---------------------------------------------------------------
    r.section("M16. EVALUABILITY RATE, PER ARM. Requirement (the committed specification).")
    r(f"Denominator rule: {valid['denominator_rule']}.")
    r("Counts read from B23 for the attrition categories and from B24 for the")
    r("admission counts. Neither is recounted at B25. B25 forms the rate and states")
    r("the denominator, which is the one thing M16 requires that neither step carries.")
    r()
    for arm in ARMS:
        v = valid[arm]
        r(f"  {arm}")
        r(f"    launched                           {v['launched']}")
        r(f"    timeouts, excluded below           {v['timeout']}")
        r(f"    environment faults, excluded below {v['environment_fault']}")
        r(f"    plausibility rejected              {v['plausibility_rejected']}")
        r(f"    DENOMINATOR                        {v['denominator']}")
        r(f"    produced a usable label            {v['produced_a_label']}")
        r(f"    EVALUABILITY RATE                  {v['evaluability_rate']}")
        r(f"    admitted, B24's count              {v['admitted']}")
        r(f"    admitted fraction of launched      {v['admitted_fraction_of_launched']}")
    r()
    r("  Zero counts are printed rather than omitted. This run recorded no timeouts")
    r("  and no environment faults in either arm, so the denominator that excludes")
    r("  them coincides with the launched count here. That is a property of this run")
    r("  and is stated rather than relied on.")

    # --- what is not here --------------------------------------------------
    r.section("WHAT THIS SECTION DOES NOT CONTAIN")
    r("M23 is B13's and is read above, not recomputed. M12, M13 and M14 are gate")
    r("quantities measured before evaluation at B18. M01 through M07, M17, M18, M19,")
    r("M20 and M21 are B24's and B23's and are above this section. M15 is B09's, M24")
    r("is B02's, M25 is B07's, M26 is B16's, M27 is B20's and M28 is B15's.")
    r()
    r("No number in this section appears anywhere else in the build, and no number")
    r("above it is recomputed here. The one exception is stated where it occurs: the")
    r("per-arm tracking slopes are the same two values B24 formed its secondary S3")
    r("from, and the equality is checked bit-for-bit rather than assumed.")
    return r.lines


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main(verbose=True):
    fixed = committed()
    band = fixed["requested_target_band"]
    min_points = fixed["minimum_converged_points"]

    evaluation = json.loads(EVALUATION_RECORDS.read_text(encoding="utf-8"))
    b24_blob = json.loads(B24_RESULTS.read_text(encoding="utf-8"))
    norm = np.load(NORMALIZATION, allow_pickle=True)
    label_min, label_max = float(norm["label_min"]), float(norm["label_max"])

    artifacts = model_mod.load_build_artifacts(".")
    cvae = load_model(artifacts)
    grid = evaluate.requested_target_grid()
    n_samples = fixed["samples_per_target"]

    pg = regenerate_b23(cvae, artifacts, grid, n_samples)
    regen_worst = check_regeneration(pg, evaluation["records"])
    if verbose:
        print(f"standing check: B23 geometry regenerated exactly "
              f"(worst deviation {regen_worst:.1e})", flush=True)

    # M11 is evaluated on the COMMITTED diversity grid, evaluate.target_grid(),
    # which is the 11 evenly spaced normalised targets from 0.0 to 1.0 inclusive
    # that the pre-registration fixes. It is NOT the requested target band `grid` that
    # every other quantity at this step uses. See diversity()'s own docstring.
    div = diversity(cvae, artifacts, evaluate.target_grid(),
                    fixed["diversity_definition"])

    paired, _ = b24.build_pairs(evaluation["records"], label_min, label_max, min_points)
    tracked = tracking(paired)
    agreement = check_against_b24(tracked, b24_blob)

    records, header, n_fixed, perm = run_condition_blind(
        cvae, artifacts, pg, grid, n_samples, min_points, BASELINE_PROGRESS, verbose)
    blind, _ = condition_blind_metric(records, header, n_fixed,
                                      label_min, label_max, min_points)

    gap = surrogate_to_solver(evaluation["records"], artifacts,
                              artifacts.label_norm, min_points)
    valid = validity(evaluation, b24_blob)

    produced_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    section = build_section(fixed, produced_at, regen_worst, div, tracked,
                            agreement, blind, gap, valid, band)
    append_to_results(section)

    RESULTS_JSON.write_text(json.dumps({
        "step": "B25",
        "produced_at": produced_at,
        "committed": fixed,
        "regeneration_check_worst_deviation": regen_worst,
        "M11_generative_diversity": div,
        "M08_M09_tracking": tracked,
        "one_computation_check_against_b24": agreement,
        "M10_condition_blind_baseline": blind,
        "M10_permutation": [int(i) for i in perm],
        "M22_surrogate_to_solver_gap": gap,
        "M16_validity": valid,
    }, indent=1), encoding="utf-8")

    if verbose:
        print()
        print("\n".join(section))
        print()
        print(f"Written: {RESULTS_TXT} (appended) and {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    raise SystemExit(main(verbose=not a.quiet))
