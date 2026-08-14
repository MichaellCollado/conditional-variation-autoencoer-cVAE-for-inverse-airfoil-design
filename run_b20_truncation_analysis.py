"""B20. Set the admission rule from the truncation analysis.

Measures how much a maximum taken from the first k converged points
understates the maximum over the whole sweep, across every fully converged
sweep in the dataset, and selects the smallest k that satisfies both committed
tolerances.

The order is the step. params.B20_TRUNCATION_TOLERANCE was written before this
file existed and before any bias was computed. This driver reads it, computes
the bias table, and applies the rule mechanically. The count is whatever the
tolerance returns. If the tolerance admits no count below the full sweep
length, the full sweep length is committed and that is reported. Nothing here
relaxes a tolerance to reach a smaller number.

On the build of record it selected 8 of 9. Seven satisfies the mean clause at
0.0053 and fails the tail clause at 0.0442; eight satisfies both, at 0.0018
and 0.0133. The rule turns on the tail clause.

Note the input. This driver reads B07's own stored sweep records, in
b07_labelling_progress.jsonl, because it needs every converged point of every
sweep and dataset.npz stores only the label each sweep produced.

Run order      5 of 10. After B19, before B21.
Reads          b07_labelling_progress.jsonl
Writes         b20_truncation.json
Runtime        not recorded in the article
"""

import json

import numpy as np

import analysis
import params

TOL = params.B20_TRUNCATION_TOLERANCE
OUT_PATH = "b20_truncation.json"
PROGRESS = "b07_labelling_progress.jsonl"


def load_fully_converged_sweeps():
    """Every fully converged sweep in the dataset, from B07's own stored
    records. A sweep is fully converged when every requested angle converged,
    which is what the build plan's 'every fully converged sweep' names. Partially
    converged sweeps are excluded here on purpose: the bias at count k is
    defined against the whole sweep, and a sweep that never had a whole sweep
    cannot supply that reference.
    """
    polars, rows, families = [], [], []
    with open(PROGRESS) as fh:
        for line in fh:
            r = json.loads(line)
            if r["polar"] is None:
                continue
            if r["n_converged"] != r["n_requested"]:
                continue
            polars.append(np.array(r["polar"], dtype=float))
            rows.append(r["row_index"])
            families.append(r["family"])
    return polars, np.array(rows), np.array(families)


def run(out_path=OUT_PATH, verbose=True):
    polars, rows, families = load_fully_converged_sweeps()
    n_points = polars[0].shape[0]
    if any(p.shape[0] != n_points for p in polars):
        raise ValueError("fully converged sweeps do not all have the same point count")

    if verbose:
        print("TOLERANCE, fixed before this analysis ran:")
        print(f"  mean absolute relative bias      <= {TOL['mean_relative_bias']:g}")
        print(f"  {TOL['upper_percentile']:g}th percentile absolute bias   "
              f"<= {TOL['upper_percentile_absolute_relative_bias']:g}")
        print(f"\n{len(polars)} fully converged sweeps, {n_points} points each")
        from collections import Counter
        print("  per family:", dict(sorted(Counter(families.tolist()).items())))

    candidates = list(range(1, n_points + 1))
    table = analysis.truncation_bias_table(polars, candidates, TOL["upper_percentile"])

    if verbose:
        print(f"\n{'k':>3} {'mean signed':>13} {'mean abs':>11} "
              f"{'p' + str(int(TOL['upper_percentile'])) + ' abs':>11} {'worst abs':>11} "
              f"{'unaffected':>11}")
        for r in table:
            print(f"{r.k:3d} {r.mean_signed:13.6f} {r.mean_absolute:11.6f} "
                  f"{r.upper_percentile_absolute:11.6f} {r.worst_absolute:11.6f} "
                  f"{r.n_sweeps_unaffected:6d}/{r.n_sweeps:<4d}")

    selected = analysis.select_minimum_point_count(
        table, TOL["mean_relative_bias"], TOL["upper_percentile_absolute_relative_bias"])
    if selected is None:
        selected = n_points
        note = ("No candidate below the full sweep length satisfied both tolerances. "
                "The full sweep length is committed and this is reported as the "
                "finding; no tolerance was relaxed.")
    else:
        note = ("Selected mechanically as the smallest candidate satisfying both "
                "stated tolerances.")

    peaks = analysis.peak_angle_distribution(polars)
    alphas, counts = np.unique(peaks, return_counts=True)

    if verbose:
        print(f"\nSELECTED minimum converged point count: {selected}")
        print(note)
        chosen = next(r for r in table if r.k == selected)
        print(f"  at k={selected}: mean abs {chosen.mean_absolute:.6f}, "
              f"p{int(TOL['upper_percentile'])} abs {chosen.upper_percentile_absolute:.6f}, "
              f"worst abs {chosen.worst_absolute:.6f}")
        print("\npeak angle distribution (alpha at max CL/CD, degrees):")
        for a, c in zip(alphas, counts):
            print(f"  {a:5.1f}  {c:5d}  {100 * c / len(peaks):5.1f}%")

    blob = {
        "step": "B20",
        "tolerance_fixed_at": TOL["fixed_at"],
        "tolerance": {
            "mean_relative_bias": TOL["mean_relative_bias"],
            "upper_percentile": TOL["upper_percentile"],
            "upper_percentile_absolute_relative_bias":
                TOL["upper_percentile_absolute_relative_bias"],
        },
        "selection_rule": TOL["selection_rule"],
        "n_fully_converged_sweeps": len(polars),
        "sweep_length": n_points,
        "bias_table": [vars(r) for r in table],
        "selected_minimum_converged_points": int(selected),
        "selection_note": note,
        "peak_angle_distribution": {
            "alpha": alphas.tolist(), "count": counts.tolist(),
        },
    }
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(blob, fh, indent=2)
        if verbose:
            print(f"\nwritten to {out_path}")
    return table, selected, blob


if __name__ == "__main__":
    run()
