"""B23. The paired generation and evaluation run.

Retrains the zero-weight control from B18's recorded weight vector and training
seed, then generates matched pairs, solves every shape, and records the
attrition flow. This driver decides nothing: it reads the committed launch
target, the committed requested target band, the committed admission rule and
the committed model, and runs them.

The pairing rule is the whole point of the design. One latent code is drawn per
target and sample index and is passed to both arms; the arms differ only in the
conditioning block and the flag. That construction lives in
evaluate.paired_generation and is not reimplemented here, so B18's gate, B21's
pilot and this run all pair the same way.

Every generated shape lands in exactly one status, so kept plus every discard
category equals attempted. A record is admitted when a label is present and at
least the committed minimum of the 9 sweep points are usable. The floor test
requires at least the committed number of pairs with both members admitted.

Resumable. Every completed pair is appended to the progress file as one JSON
line, and a rerun skips the pairs already recorded. The run header is written
as the first line and checked on resume, so a rerun under a different seed,
grid, model, sample count or admission rule refuses rather than silently mixing
two runs into one record.

No extension happens here. This driver launches exactly the committed launch
target and stops.

Run order      7 of 10. After B21, before B24.
Reads          committed_model.pt, b18_gate.json, b21_paired_yield.json, the
               build artifacts, and the XFOIL binary
Writes         b23_evaluation.json, b23_evaluation_progress.jsonl
Runtime        220 solver calls totalling 265.3 s, mean 1.206 s, maximum
               6.912 s, per Table A3
"""

import argparse
import json
import os

import numpy as np
import torch

import analysis
import evaluate
import model as model_mod
import params
import run_b16_weight_sweep as b16

PROGRESS_PATH = "b23_evaluation_progress.jsonl"
OUT_PATH = "b23_evaluation.json"
MODEL_PATH = "committed_model.pt"

ARMS = ("prior_on", "prior_off")


def committed_run_config():
    """Every number this run uses, read from the parameter record. A driver
    that restates one of these is a second copy of a committed value."""
    sp = params.PARAMS["samples_per_target"].value
    if isinstance(sp, params.Pending):
        raise ValueError("samples_per_target is PENDING; B21 has not run")
    floor = params.PARAMS["analysed_pair_floor"].value
    if isinstance(floor, params.Pending):
        raise ValueError("analysed_pair_floor is PENDING; it is a committed floor")
    mp = params.PARAMS["minimum_converged_sweep_points"].value
    if isinstance(mp, params.Pending):
        raise ValueError("minimum_converged_sweep_points is PENDING; B20 has not run")
    return {
        "samples_per_target": int(sp["samples_per_target"]),
        "launch_target_pairs": int(sp["launch_target_pairs"]),
        "launched_pairs": int(sp["launched_pairs_at_that_grid"]),
        "floor_analysed_pairs": int(floor),
        "minimum_converged_points": int(mp["minimum_converged_points"]),
        "requested_target_band": params.PARAMS["requested_target_band"].value,
    }


def load_progress(path, header):
    """Returns the set of (target_index, sample_index) already recorded and
    the records themselves. Refuses to resume onto a different run."""
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
        for key in ("generation_seed", "model", "targets", "samples_per_target",
                    "minimum_converged_points"):
            if stored["header"][key] != header[key]:
                raise ValueError(
                    f"{path} was written for a different run: {key} is "
                    f"{stored['header'][key]!r} there and {header[key]!r} here. "
                    f"Refusing to resume. Move that file aside if a fresh run is wanted."
                )
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            done.add((rec["target_index"], rec["sample_index"]))
            records.append(rec)
    return done, records


def flow(records, arm):
    """The per-arm reason-annotated flow B23 requires. Every launched
    shape lands in exactly one status, and the discard reasons are kept."""
    rs = [r[arm] for r in records]
    return {
        "launched": len(rs),
        "plausibility_rejected": sum(1 for s in rs if not s["plausible"]),
        "converged": sum(1 for s in rs if s["status"] == "converged"),
        "partially_converged": sum(1 for s in rs if s["status"] == "partially_converged"),
        "failed": sum(1 for s in rs if s["status"] == "failed"),
        "timeout": sum(1 for s in rs if s["status"] == "timeout"),
        "environment_fault": sum(1 for s in rs if s["status"] == "environment_fault"),
        "points_dropped_as_implausible": sum(s["n_converged"] - s["n_usable"] for s in rs),
        "produced_a_label": sum(1 for s in rs if s["label"] is not None),
        "label_but_below_min_points": sum(
            1 for s in rs if s["label"] is not None and not s["admitted"]),
        "admitted": sum(1 for s in rs if s["admitted"]),
    }


def summarise(records, cfg, header):
    both = sum(1 for r in records
               if r["prior_on"]["admitted"] and r["prior_off"]["admitted"])
    shapes = [r[arm] for r in records for arm in ARMS]
    single = sum(1 for s in shapes if s["admitted"])
    per_arm = {arm: {"admitted": sum(1 for r in records if r[arm]["admitted"]),
                     "launched": len(records)} for arm in ARMS}
    for v in per_arm.values():
        v["rate"] = v["admitted"] / v["launched"] if v["launched"] else float("nan")

    per_target = []
    for ti, t in enumerate(header["targets"]):
        rows = [r for r in records if r["target_index"] == ti]
        per_target.append({
            "target_index": ti, "target": t, "pairs_launched": len(rows),
            "pairs_both_admitted": sum(
                1 for r in rows if r["prior_on"]["admitted"] and r["prior_off"]["admitted"]),
        })

    return {
        "step": "B23",
        "header": header,
        "committed": cfg,
        "pairs_launched": len(records),
        "analysed_pairs": both,
        "floor_analysed_pairs": cfg["floor_analysed_pairs"],
        "floor_met": both >= cfg["floor_analysed_pairs"],
        "shortfall": max(0, cfg["floor_analysed_pairs"] - both),
        "single_shape_admission_rate": single / len(shapes) if shapes else float("nan"),
        "single_shape_admitted": single,
        "single_shape_launched": len(shapes),
        "per_arm_admission": per_arm,
        "attrition_flow": {arm: flow(records, arm) for arm in ARMS},
        "per_target": per_target,
        "extension": {
            "extended": False,
            "note": "No extension was performed by this run. The pre-registration permits "
                    "extension above the floor and requires its trigger, the result "
                    "visible at the decision point, and the resulting size to be "
                    "reported with it. None of that is this driver's to decide.",
        },
        "records": records,
    }


def run(progress_path=PROGRESS_PATH, out_path=OUT_PATH, verbose=True):
    cfg = committed_run_config()
    A = model_mod.load_build_artifacts(".")

    cvae = model_mod.CVAE(A.x_all.shape[1], A.cond_all.shape[1], b16.ARCH).to(model_mod.DTYPE)
    cvae.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    cvae.eval()

    settings = evaluate.solver_settings()
    timeout = evaluate.committed_timeout()
    bounds = evaluate.plausibility_bounds()
    n_points = params.B19_CONSISTENCY_GATE["gate_one"]["n_points_per_surface"]
    min_points = cfg["minimum_converged_points"]

    grid = evaluate.requested_target_grid()
    n_targets = len(grid)
    n_samples = cfg["samples_per_target"]
    generation_seed = model_mod.seed_int(23, 0)

    header = {
        "generation_seed": generation_seed,
        "model": MODEL_PATH,
        "targets": [float(t) for t in grid],
        "samples_per_target": n_samples,
        "minimum_converged_points": min_points,
        "timeout_seconds": timeout,
        "n_points_per_surface": n_points,
        "n_requested_angles": len(settings.alphas()),
        "requested_target_band": cfg["requested_target_band"],
        "seed_rule": "base seed + 1000 * build step + substream, at step 23, substream 0",
    }

    pg = evaluate.paired_generation(cvae, A.reference_signature, b16.ARCH.latent_dim,
                                     generation_seed, targets=grid, n_samples=n_samples)

    done, records = load_progress(progress_path, header)
    fresh = not os.path.exists(progress_path)
    if fresh:
        with open(progress_path, "w") as fh:
            fh.write(json.dumps({"kind": "header", "header": header}) + "\n")

    n_pairs = n_targets * n_samples
    if verbose:
        band = cfg["requested_target_band"]
        print(f"B23. paired generation and evaluation, committed launch target")
        print(f"  band          : [{band['normalised_low']:.6f}, "
              f"{band['normalised_high']:.6f}] normalised "
              f"= [{band['raw_low']:.3f}, {band['raw_high']:.3f}] raw max(CL/CD)")
        print(f"  launch        : {n_targets} targets x {n_samples} samples "
              f"= {n_pairs} pairs, {2 * n_pairs} solver calls")
        print(f"  floor         : {cfg['floor_analysed_pairs']} analysed pairs "
              f"(derived launch target {cfg['launch_target_pairs']} pairs)")
        print(f"  admission     : label present AND usable points >= {min_points} "
              f"of {len(settings.alphas())}")
        print(f"  generation seed {generation_seed}, timeout {timeout:g}s, model {MODEL_PATH}")
        print(f"  already done  : {len(done)} of {n_pairs} pairs "
              f"({'fresh run' if fresh else 'resuming'})", flush=True)

    with open(progress_path, "a") as fh:
        for ti in range(n_targets):
            for si in range(n_samples):
                if (ti, si) in done:
                    continue
                row = {"target_index": ti, "sample_index": si,
                       "target": float(pg.targets[ti])}
                for arm, x in (("prior_on", pg.x_set), ("prior_off", pg.x_clear)):
                    upper, lower = evaluate.standardised_to_coefficients(x[ti, si], A.std_stats)
                    rec = evaluate.evaluate_coefficients(
                        upper[0], lower[0], name=f"b23_{arm}_t{ti}_s{si}",
                        bounds=bounds, settings=settings, timeout_seconds=timeout,
                        n_points_per_surface=n_points,
                    )
                    row[arm] = {
                        "status": rec.status,
                        "plausible": rec.plausible,
                        "plausibility_reason": rec.plausibility_reason,
                        "reason": rec.reason,
                        "n_converged": rec.n_converged,
                        "n_usable": rec.n_usable,
                        "n_requested": rec.n_requested,
                        "label": rec.label,
                        "polar": None if rec.polar is None else rec.polar.tolist(),
                        "implausible_reasons": rec.implausible_reasons,
                        "elapsed_seconds": rec.elapsed_seconds,
                        "admitted": analysis.is_admitted(rec.label, rec.n_usable, min_points),
                        "standardised_coefficients": x[ti, si].tolist(),
                    }
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                records.append(row)
                done.add((ti, si))
            if verbose:
                both = sum(1 for r in records
                           if r["prior_on"]["admitted"] and r["prior_off"]["admitted"])
                print(f"  target {ti + 1}/{n_targets} (t={pg.targets[ti]:.4f}): "
                      f"{both}/{len(records)} pairs analysed so far", flush=True)

    records.sort(key=lambda r: (r["target_index"], r["sample_index"]))
    blob = summarise(records, cfg, header)

    if out_path:
        with open(out_path, "w") as f:
            json.dump(blob, f, indent=1)

    if verbose:
        print(f"\npairs launched              : {blob['pairs_launched']}")
        print(f"ANALYSED PAIRS              : {blob['analysed_pairs']}")
        print(f"committed floor             : {blob['floor_analysed_pairs']}")
        verdict = "YES" if blob["floor_met"] else f"NO, short by {blob['shortfall']}"
        print(f"floor met                   : {verdict}")
        for arm, v in blob["per_arm_admission"].items():
            print(f"  {arm:10s} admitted     : {v['admitted']}/{v['launched']} = {v['rate']:.4f}")
        print("\nper-arm flow:")
        for arm, f in blob["attrition_flow"].items():
            print(f"  {arm}: " + ", ".join(f"{k} {v}" for k, v in f.items()))
        print("\npairs analysed per target:")
        for t in blob["per_target"]:
            print(f"  t={t['target']:.4f}: {t['pairs_both_admitted']}/{t['pairs_launched']}")
        if out_path:
            print(f"\n-> {out_path}")
    return blob


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", default=PROGRESS_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    run(progress_path=a.progress, out_path=a.out, verbose=not a.quiet)
