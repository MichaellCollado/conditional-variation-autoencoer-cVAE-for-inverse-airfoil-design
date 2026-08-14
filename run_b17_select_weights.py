"""B17. Select the prior weight, and by the same rule the other two.

Applies params.WEIGHT_SELECTION_RULE to the sweep table alone. The rule was
written before B16's sweep existed, which is the whole of its protection: a
selection rule cannot be shaped by the table it is applied to if it predates
the table. This driver reads the rule and defines none of it.

The rule is the highest score of effect gain less reconstruction cost less
diversity cost, on min-max normalised components, with an exact tie going to
the smaller weight. It carries no tolerance constant by design.

This file reads the sweep table and nothing else. It reads no solver result,
because none exists at this point in the run order.

B17 is the one step with no falsification check. Recomputing the decision from
the recorded table catches a transcription error and nothing else, because a
selection rule that is wrong reproduces itself exactly on recomputation. The
mitigation is procedural and it is the ordering above.

Run order      2 of 10. After B16, before B18.
Reads          sweep/sweep_table.json
Writes         b17_selection.json
Runtime        not recorded in the article
"""

import json

import params

TABLE_PATH = "sweep/sweep_table.json"
OUT_PATH = "b17_selection.json"

# Each swept weight's effect measure, lower being better, naming what that
# weight is FOR. Fixed in WEIGHT_SELECTION_RULE before the sweep ran.
EFFECT_COLUMN = {
    "avian": "mean_distance_prior_on",
    "target": "val_target",
    "spread": "val_spread",
}
SWEPT = ["target", "spread", "avian"]


def normalise(values, invert):
    """Map to [0, 1] on the ladder's own observed range. invert=True gives
    (max - v) / range, invert=False gives (v - min) / range. A zero range
    yields 0.0 for every candidate, which is the correct reading of a
    quantity the ladder does not move."""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0.0:
        return [0.0] * len(values)
    return [((hi - v) if invert else (v - lo)) / span for v in values]


def main():
    table = json.load(open(TABLE_PATH))
    rows = table["rows"]
    selection = {}

    for name in SWEPT:
        ladder = sorted([r for r in rows if r["swept_weight"] == name],
                        key=lambda r: r["candidate"])
        cand = [r["candidate"] for r in ladder]
        R = [r["val_reconstruction"] for r in ladder]
        D = [r["diversity_prior_on"] for r in ladder]
        E = [r[EFFECT_COLUMN[name]] for r in ladder]

        recon_cost = normalise(R, invert=False)
        diversity_cost = normalise(D, invert=True)
        effect_gain = normalise(E, invert=True)
        score = [g - rc - dc for g, rc, dc in zip(effect_gain, recon_cost, diversity_cost)]

        best = max(score)
        tied = [c for c, s in zip(cand, score) if s == best]
        chosen = min(tied)

        print(f"\n{name.upper()} weight    effect measure = {EFFECT_COLUMN[name]}")
        print(f"  {'candidate':>13} {'R':>10} {'D':>9} {'E':>12} "
              f"{'recon_c':>9} {'div_c':>9} {'gain':>9} {'score':>9}")
        for i, c in enumerate(cand):
            mark = " <-- selected" if c == chosen else ""
            print(f"  {c:>13.6g} {R[i]:>10.6f} {D[i]:>9.4f} {E[i]:>12.6f} "
                  f"{recon_cost[i]:>9.4f} {diversity_cost[i]:>9.4f} "
                  f"{effect_gain[i]:>9.4f} {score[i]:>9.4f}{mark}")
        if len(tied) > 1:
            print(f"  exact tie among {tied}; smallest taken")

        selection[name] = {
            "selected": chosen,
            "effect_column": EFFECT_COLUMN[name],
            "ladder": cand,
            "val_reconstruction": R,
            "diversity_prior_on": D,
            "effect_measure": E,
            "recon_cost": recon_cost,
            "diversity_cost": diversity_cost,
            "effect_gain": effect_gain,
            "score": score,
            "tied_at_best": tied,
            "selected_row": next(r for r in ladder if r["candidate"] == chosen),
        }

    out = {
        "selected_by_sweep": {k: v["selected"] for k, v in selection.items()},
        "set_as_stated_values": {
            "divergence_weight": params.PARAMS["reconstruction_divergence_weighting"].value["divergence_weight"],
            "safeguard_weight": params.PARAMS["safeguard_weight"].value,
            "reconstruction_weight": params.PARAMS["reconstruction_divergence_weighting"].value["reconstruction_weight"],
        },
        "rule_fixed_at": params.WEIGHT_SELECTION_RULE["fixed_at"],
        "no_evaluation_result_seen": (
            "No solver evaluation result has been seen at this point. The evaluation "
            "path does not exist in this build: evaluate.py contains paired generation "
            "and the B18 gate only, B19's consistency gates and B23's paired evaluation "
            "run are not built, and the solver has not been called since B07's dataset "
            "labelling pass. This selection read sweep/sweep_table.json and nothing else."
        ),
        "detail": selection,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)

    print("\nCOMMITTED WEIGHTS")
    for k, v in out["set_as_stated_values"].items():
        print(f"  {k:22s} = {v:<14.6g} stated value, not selected from the table")
    for k, v in out["selected_by_sweep"].items():
        print(f"  {k + '_weight':22s} = {v:<14.6g} selected from its ladder")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
