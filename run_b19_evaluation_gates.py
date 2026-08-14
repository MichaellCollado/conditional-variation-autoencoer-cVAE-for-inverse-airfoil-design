"""B19. The two evaluation gates, run before any generated shape is evaluated.

Gate zero is solver responsiveness. It solves one known airfoil,
seeds/e387.dat, through the same path at the committed operating point and
timeout, with the coordinates read raw and not round-tripped through the CST
fit, and requires a converged status on all 9 requested angles. A responding
binary, a partial sweep, or a written-but-empty polar all fail.

Gate one is pipeline consistency. Twenty-five dataset rows whose labels are
already known are pushed back through the same decode and solver path the
evaluation will use, and the recomputed labels are compared against the stored
ones. Five rows come from each of the five families, at the 0th, 25th, 50th,
75th and 100th percentile of that family's own stored label. The selection is
deterministic and draws no random number. Every one of the 25 must satisfy a
relative difference at or below 0.01; one row outside it fails the gate, and
the failing rows are named.

The two gates are deliberately separable. A broken fit or decode fails gate
one and leaves gate zero passing, which is information a single combined gate
would destroy.

Both thresholds are in params.B19_CONSISTENCY_GATE, fixed before either gate
was run. This driver reads them and defines none.

Run order      4 of 10. After B18, before B20. First solver call of the run.
Reads          dataset.npz, population.npz, the build artifacts, and the
               XFOIL binary
Writes         b19_gates.json
Runtime        gate zero, 1 solver call at 0.728 s on the build of record
"""

import json

import numpy as np

import evaluate
import params

GATE = params.B19_CONSISTENCY_GATE
OUT_PATH = "b19_gates.json"
KNOWN_AIRFOIL = "seeds/e387.dat"


def load_rows():
    """The stored dataset: its coefficients, its labels, its families. Read
    from the artifacts on disk, not reconstructed."""
    ds = np.load("dataset.npz", allow_pickle=True)
    pop = np.load("population.npz", allow_pickle=True)
    row_index = ds["row_index"]
    return {
        "upper": pop["upper_coefficients"][row_index],
        "lower": pop["lower_coefficients"][row_index],
        "labels": ds["label"],
        "family": ds["family"],
        "order": int(pop["cst_order"]),
    }


def run(rows=None, out_path=OUT_PATH, verbose=True):
    if rows is None:
        rows = load_rows()

    settings = evaluate.solver_settings()
    timeout = evaluate.committed_timeout()
    bounds = evaluate.plausibility_bounds()
    n_points = GATE["gate_one"]["n_points_per_surface"]
    tolerance = GATE["gate_one"]["tolerance_relative"]
    n_per_family = GATE["gate_one"]["n_rows"] // len(set(rows["family"].tolist()))

    if verbose:
        print(f"operating point: Re={settings.reynolds:g} M={settings.mach:g} "
              f"Ncrit={settings.ncrit:g} alpha {settings.alpha_start:g}"
              f"..{settings.alpha_end:g} step {settings.alpha_step:g} "
              f"N={settings.n_panels} iter={settings.iter_limit}")
        print(f"timeout: {timeout:g}s   decode: {n_points} points/surface   "
              f"tolerance: {tolerance:g} relative")

    # --- gate zero ---------------------------------------------------------
    g0 = evaluate.gate_zero(KNOWN_AIRFOIL, settings, timeout)
    if verbose:
        print(f"\ngate zero  {KNOWN_AIRFOIL}: status={g0.status} "
              f"{g0.n_converged}/{g0.n_requested} in {g0.elapsed_seconds:.3f}s -> "
              f"{'PASS' if g0.passed else 'FAIL'}")

    # --- gate one ----------------------------------------------------------
    row_indices = evaluate.select_gate_one_rows(rows["family"], rows["labels"],
                                                 n_per_family)
    if verbose:
        print(f"\ngate one  {len(row_indices)} rows: {list(map(int, row_indices))}")

    g1 = evaluate.gate_one(
        row_indices, rows["upper"], rows["lower"], rows["labels"], rows["family"],
        bounds=bounds, settings=settings, timeout_seconds=timeout,
        n_points_per_surface=n_points, tolerance_relative=tolerance,
    )

    if verbose:
        print(f"{'row':>5} {'family':>8} {'stored':>10} {'recomputed':>11} "
              f"{'rel diff':>10} {'conv':>5}  ok")
        for r in g1.rows:
            rec = "None" if r.recomputed_label is None else f"{r.recomputed_label:10.4f}"
            rel = "n/a" if r.relative_difference is None else f"{r.relative_difference:10.3e}"
            print(f"{r.row:5d} {r.family:>8} {r.stored_label:10.4f} {rec:>11} "
                  f"{rel:>10} {r.n_converged:5d}  {'y' if r.within_tolerance else 'N'}")
        print(f"\nworst relative difference: {g1.max_relative_difference:.6e} "
              f"against tolerance {tolerance:g}")
        print(f"gate one -> {'PASS' if g1.passed else 'FAIL'}"
              + (f", failing rows {g1.failing_rows}" if g1.failing_rows else ""))

    verdict = g0.passed and g1.passed
    blob = {
        "step": "B19",
        "thresholds_fixed_at": GATE["fixed_at"],
        "verdict": "PASS" if verdict else "FAIL",
        "gate_zero": {
            "threshold": GATE["gate_zero"]["threshold"],
            "airfoil": g0.airfoil, "status": g0.status,
            "n_converged": g0.n_converged, "n_requested": g0.n_requested,
            "elapsed_seconds": g0.elapsed_seconds, "reason": g0.reason,
            "passed": g0.passed,
        },
        "gate_one": {
            "tolerance_relative": tolerance,
            "n_rows": g1.n_rows,
            "n_points_per_surface": n_points,
            "row_selection": GATE["gate_one"]["row_selection"],
            "max_relative_difference": g1.max_relative_difference,
            "failing_rows": g1.failing_rows,
            "passed": g1.passed,
            "rows": [vars(r) for r in g1.rows],
        },
        "solver_settings": params.PARAMS["solver_operating_point_settings"].value,
        "per_call_timeout": timeout,
    }
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(blob, fh, indent=2)
        if verbose:
            print(f"\nwritten to {out_path}")
    return g0, g1, blob


if __name__ == "__main__":
    g0, g1, _ = run()
    print(f"\nB19 VERDICT: {'PASS' if (g0.passed and g1.passed) else 'FAIL'}")
