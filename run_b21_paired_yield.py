"""B21. Measure the paired yield on a pilot and set the launch target.

Runs a pilot and measures the fraction of launched pairs in which both members
cleared admission, then divides the committed floor of analysed pairs by that
yield to derive the launch target.

The yield is measured at the pair level, by counting pairs in which both
members cleared admission over the pairs launched. It is not derived from the
single-shape admission rate. The two differ whenever the two members' failures
are not perfectly correlated, and the single-shape rate is the optimistic one,
so deriving from it would understate the launch target while every number
still looked reasonable. Both rates are computed and both are recorded, so the
gap between them is visible rather than asserted.

The floor is params.PARAMS['analysed_pair_floor'], written before this pilot
ran. This driver reads it. The launch target is derived from the floor and the
measured yield and is not chosen. The driver raises if the measured yield is at
or below zero, because the launch target would then be undefined.

No analysis of any evaluation outcome happens here. The pilot records admission
and nothing else: no generated label is compared against any target, and no
paired difference is formed.

On the build of record the yield was 0.990909, being 109 of 110 pilot pairs,
giving an inflation factor of 1.00917 and a launch target of 101 pairs,
realised as 11 targets by 10 samples.

Run order      6 of 10. After B20, before B23.
Reads          the build artifacts, committed_model.pt, and the XFOIL binary
Writes         b21_paired_yield.json
Runtime        220 solver calls. No total recorded in the article
"""

import json
import math

import numpy as np
import torch

import analysis
import evaluate
import model as model_mod
import params
import run_b16_weight_sweep as b16

OUT_PATH = "b21_paired_yield.json"
MODEL_PATH = "committed_model.pt"

# Pilot size. A stated choice at this step, disclosed rather than derived:
# there is no rule anywhere in the build plan for how large a yield pilot must be.
# 10 samples at each target of the committed band is 110 pairs and 220 solver
# calls, which is large enough that the yield's own standard error is a few
# percentage points and small enough to run in one pass. The realised
# standard error is recorded below with the estimate, so the precision of the
# inflation factor is reported rather than assumed.
#
# The target count is NOT set here. It comes from the committed
# requested_target_band, because the yield has to be measured on the grid the
# run will actually use. An earlier pilot ran on the [0, 1]
# diversity grid before the band was committed, measured 0.627273, and
# derived a launch target of 160 pairs. That measurement belongs to that grid
# and is superseded outright by this one, not averaged with it.
N_SAMPLES_PER_TARGET = 10


def load_committed_model(A):
    cvae = model_mod.CVAE(A.x_all.shape[1], A.cond_all.shape[1], b16.ARCH).to(model_mod.DTYPE)
    cvae.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    cvae.eval()
    return cvae


def run(out_path=OUT_PATH, verbose=True):
    floor_slot = params.PARAMS["analysed_pair_floor"]
    if isinstance(floor_slot.value, params.Pending):
        raise ValueError("analysed_pair_floor is PENDING; it must be stated before the pilot")
    floor = int(floor_slot.value)

    min_points = params.PARAMS["minimum_converged_sweep_points"].value
    if isinstance(min_points, params.Pending):
        raise ValueError("minimum_converged_sweep_points is PENDING; B20 has not run")
    min_points = int(min_points["minimum_converged_points"])

    A = model_mod.load_build_artifacts(".")
    cvae = load_committed_model(A)
    settings = evaluate.solver_settings()
    timeout = evaluate.committed_timeout()
    bounds = evaluate.plausibility_bounds()
    n_points_per_surface = params.B19_CONSISTENCY_GATE["gate_one"]["n_points_per_surface"]

    grid = evaluate.requested_target_grid()
    n_targets = len(grid)
    band = params.PARAMS["requested_target_band"].value

    generation_seed = model_mod.seed_int(21, 0)
    pg = evaluate.paired_generation(cvae, A.reference_signature, b16.ARCH.latent_dim,
                                     generation_seed, targets=grid,
                                     n_samples=N_SAMPLES_PER_TARGET)

    n_pairs = n_targets * N_SAMPLES_PER_TARGET
    if verbose:
        print(f"requested target band (committed): "
              f"[{band['normalised_low']:.6f}, {band['normalised_high']:.6f}] "
              f"normalised = [{band['raw_low']:.3f}, {band['raw_high']:.3f}] raw L/D")
        print(f"floor (stated before any pilot): {floor} analysed pairs")
        print(f"admission: label present AND usable points >= {min_points} "
              f"(of {len(settings.alphas())} requested angles)")
        print(f"pilot: {n_targets} targets x {N_SAMPLES_PER_TARGET} samples "
              f"= {n_pairs} pairs, {2 * n_pairs} solver calls")
        print(f"generation seed {generation_seed}, timeout {timeout:g}s", flush=True)

    records = []
    for ti in range(n_targets):
        for si in range(N_SAMPLES_PER_TARGET):
            row = {"target_index": ti, "sample_index": si,
                   "target": float(pg.targets[ti])}
            for arm, x in (("prior_on", pg.x_set), ("prior_off", pg.x_clear)):
                upper, lower = evaluate.standardised_to_coefficients(x[ti, si], A.std_stats)
                rec = evaluate.evaluate_coefficients(
                    upper[0], lower[0], name=f"b21_{arm}_t{ti}_s{si}",
                    bounds=bounds, settings=settings, timeout_seconds=timeout,
                    n_points_per_surface=n_points_per_surface,
                )
                row[arm] = {
                    "status": rec.status, "plausible": rec.plausible,
                    "plausibility_reason": rec.plausibility_reason,
                    "n_converged": rec.n_converged, "n_usable": rec.n_usable,
                    "label": rec.label,
                    "implausible_reasons": rec.implausible_reasons,
                    "elapsed_seconds": rec.elapsed_seconds,
                    "admitted": analysis.is_admitted(rec.label, rec.n_usable, min_points),
                }
            records.append(row)
        if verbose:
            done = (ti + 1) * N_SAMPLES_PER_TARGET
            both = sum(1 for r in records if r["prior_on"]["admitted"] and r["prior_off"]["admitted"])
            print(f"  target {ti + 1}/{n_targets} (t={pg.targets[ti]:.4f}): "
                  f"{both}/{done} pairs admitted so far", flush=True)

    # --- the yield, counted at the pair level ------------------------------
    both_admitted = sum(1 for r in records
                        if r["prior_on"]["admitted"] and r["prior_off"]["admitted"])
    pair_yield = both_admitted / n_pairs

    # --- the single-shape rate, recorded for comparison only ---------------
    shapes = [r[arm] for r in records for arm in ("prior_on", "prior_off")]
    single_admitted = sum(1 for s in shapes if s["admitted"])
    single_rate = single_admitted / len(shapes)

    per_arm = {}
    for arm in ("prior_on", "prior_off"):
        adm = sum(1 for r in records if r[arm]["admitted"])
        per_arm[arm] = {"admitted": adm, "launched": n_pairs, "rate": adm / n_pairs}

    # --- the launch target, derived --------------------------------------
    if pair_yield <= 0.0:
        raise ValueError(
            "measured pair yield is exactly zero; the launch target is undefined and "
            "is reported as such rather than inflated by a division by zero."
        )
    launch_pairs = math.ceil(floor / pair_yield)
    inflation = 1.0 / pair_yield
    yield_se = math.sqrt(pair_yield * (1 - pair_yield) / n_pairs)

    # --- attrition flow, per arm (M17's shape, at pilot scale) -------------
    def flow(arm):
        rs = [r[arm] for r in records]
        return {
            "launched": len(rs),
            "plausibility_rejected": sum(1 for s in rs if not s["plausible"]),
            "timeout": sum(1 for s in rs if s["status"] == "timeout"),
            "points_dropped_as_implausible": sum(
                s["n_converged"] - s["n_usable"] for s in rs),
            "environment_fault": sum(1 for s in rs if s["status"] == "environment_fault"),
            "failed": sum(1 for s in rs if s["status"] == "failed"),
            "partially_converged": sum(1 for s in rs if s["status"] == "partially_converged"),
            "converged": sum(1 for s in rs if s["status"] == "converged"),
            "produced_a_label": sum(1 for s in rs if s["label"] is not None),
            "label_but_below_min_points": sum(
                1 for s in rs if s["label"] is not None and not s["admitted"]),
            "admitted": sum(1 for s in rs if s["admitted"]),
        }

    flows = {arm: flow(arm) for arm in ("prior_on", "prior_off")}

    if verbose:
        print(f"\npairs launched            : {n_pairs}")
        print(f"pairs with BOTH admitted  : {both_admitted}")
        print(f"PAIR YIELD                : {pair_yield:.6f} "
              f"(standard error {yield_se:.4f} on {n_pairs} pairs)")
        print(f"single-shape rate         : {single_rate:.6f} "
              f"({single_admitted}/{len(shapes)}), recorded for comparison, NOT used")
        print(f"  gap, single minus pair  : {single_rate - pair_yield:.6f}")
        for arm, v in per_arm.items():
            print(f"  {arm:10s} admitted   : {v['admitted']}/{v['launched']} = {v['rate']:.4f}")
        print(f"\nfloor                     : {floor} analysed pairs")
        print(f"inflation factor 1/yield  : {inflation:.6f}")
        print(f"LAUNCH TARGET             : ceil({floor} / {pair_yield:.6f}) "
              f"= {launch_pairs} pairs launched")
        print(f"                            = {2 * launch_pairs} generated shapes, "
              f"{2 * launch_pairs} solver calls")
        print("\nper-arm attrition flow:")
        for arm, f in flows.items():
            print(f"  {arm}: " + ", ".join(f"{k} {v}" for k, v in f.items()))

    blob = {
        "step": "B21",
        "pilot": {
            "n_targets": n_targets, "n_samples_per_target": N_SAMPLES_PER_TARGET,
            "n_pairs": n_pairs, "generation_seed": generation_seed,
            "model": MODEL_PATH,
            "target_grid": pg.targets.tolist(),
            "requested_target_band": band,
            "target_grid_note": (
                "The COMMITTED requested target band (slot requested_target_band), "
                "read from the parameter record. This is NOT the [0, 1] diversity "
                "grid B16 and B18 generate at; that grid is unchanged and "
                "evaluate.target_grid still returns it, so B16's sweep table and "
                "B18's gate stay reproducible."
            ),
        },
        "admission_rule": {
            "minimum_converged_points": min_points,
            "n_requested_angles": len(settings.alphas()),
            "definition": "a record is admitted only if it has a label AND its "
                          "USABLE point count (converged points surviving the committed specification's "
                          "physical plausibility filter) >= minimum_converged_points",
        },
        "pair_yield": pair_yield,
        "pair_yield_standard_error": yield_se,
        "pairs_both_admitted": both_admitted,
        "single_shape_admission_rate": single_rate,
        "single_shape_admitted": single_admitted,
        "single_shape_launched": len(shapes),
        "per_arm_admission": per_arm,
        "attrition_flow": flows,
        "floor_analysed_pairs": floor,
        "inflation_factor": inflation,
        "launch_target_pairs": launch_pairs,
        "records": records,
    }
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(blob, fh, indent=1)
        if verbose:
            print(f"\nwritten to {out_path}")
    return blob


if __name__ == "__main__":
    run()
