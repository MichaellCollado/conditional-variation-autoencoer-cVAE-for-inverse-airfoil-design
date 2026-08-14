"""Regenerate the committed model's training history with its components logged.

Figure A1 plots the per-epoch validation objective components of the committed
run, and Figure A2 plots the per-dimension divergence on the validation split.
B18 trained the committed model and recorded its best epoch, its best selection
metric and its live dimension count, but no build step stored either the
per-epoch components or the per-dimension values. A figure cannot be drawn from
a quantity nothing stored, so the committed run is repeated here with them
logged.

This is a regeneration and not a new run. The training seed, the weights, the
architecture, the split, the frozen surrogate ensemble and the reference are
all read from what B17 and B18 committed. The only difference is that
model.train_cvae is called with log_components=True, which adds validation
split term evaluations after each epoch and draws no random number.

The regeneration is not assumed to reproduce. It is checked, and this script
exits without writing if any check fails: the best epoch must equal B18's
recorded value, the best selection metric must equal it bit for bit, the live
dimension count must equal it, and every tensor of the reproduced best
checkpoint must equal committed_model.pt bit for bit.

Run order      10 of 10. After B25, before run_figures.py.
Reads          b17_selection.json, b18_gate.json, committed_model.pt, and the
               build artifacts
Writes         committed_training_history.json
Runtime        not recorded in the article. It reproduces one 150 epoch
               training run
"""

import json

import numpy as np
import torch

import model as model_mod
import run_b16_weight_sweep as b16

OUT_PATH = "committed_training_history.json"


def main():
    gate = json.load(open("b18_gate.json"))
    sel = json.load(open("b17_selection.json"))

    committed = model_mod.ObjectiveWeights(
        divergence_weight=sel["set_as_stated_values"]["divergence_weight"],
        safeguard_weight=sel["set_as_stated_values"]["safeguard_weight"],
        target_weight=sel["selected_by_sweep"]["target"],
        spread_weight=sel["selected_by_sweep"]["spread"],
        avian_weight=sel["selected_by_sweep"]["avian"],
    )
    if committed.__dict__ != gate["committed_weights"]:
        raise SystemExit("weights read here differ from the weights B18 recorded")

    training_seed = gate["training_seed"]
    if training_seed != model_mod.seed_int(18, 0):
        raise SystemExit("recorded training seed does not match the offset rule")

    A = model_mod.load_build_artifacts(".")
    print(f"retraining the committed model at seed {training_seed}, "
          f"{b16.ARCH.epochs} epochs", flush=True)

    res = model_mod.train_cvae(
        A.x_all, A.cond_all, A.train_idx, A.val_idx, A.ensemble, A.safeguard,
        A.safeguard_bounds, A.reference_signature, A.region_extent,
        committed, b16.ARCH, training_seed=training_seed,
        liveness_threshold=b16.LIVENESS_THRESHOLD, log_components=True,
    )

    stored = torch.load("committed_model.pt", weights_only=False)
    tensors_identical = (
        set(stored.keys()) == set(res.best_state.keys())
        and all(bool(torch.equal(stored[k], res.best_state[k])) for k in stored)
    )

    checks = {
        "best_epoch": (res.best_epoch, gate["committed_best_epoch"]),
        "best_selection_metric": (res.best_selection_metric,
                                  gate["committed_best_selection_metric"]),
        "n_live_dimensions": (res.n_live_dimensions, gate["committed_n_live_dimensions"]),
        "best_checkpoint_bit_for_bit": (tensors_identical, True),
    }
    ok = True
    for name, (got, want) in checks.items():
        agree = got == want
        ok = ok and agree
        print(f"  {name:30s} regenerated {got!r}  recorded {want!r}  "
              f"{'MATCH' if agree else 'DIFFERS'}")
    if not ok:
        raise SystemExit("the regeneration does not reproduce B18's committed run; "
                         "no artifact written and no figure may be drawn from it")

    per_dim = [float(v) for v in np.asarray(res.per_dimension_divergence)]
    out = {
        "produced_by": "run_committed_training_history.py, the build",
        "what_this_is": (
            "The committed model's own B18 training run, repeated with the "
            "validation-split objective components logged. Verified against "
            "b18_gate.json's recorded best epoch, best selection metric, live "
            "dimension count, and against committed_model.pt bit for bit."
        ),
        "training_seed": training_seed,
        "weights": committed.__dict__,
        "architecture": {
            "latent_dim": b16.ARCH.latent_dim, "hidden_width": b16.ARCH.hidden_width,
            "depth": b16.ARCH.depth, "learning_rate": b16.ARCH.learning_rate,
            "epochs": b16.ARCH.epochs, "batch_size": b16.ARCH.batch_size,
            "warmup_epochs": b16.ARCH.warmup_epochs,
        },
        "liveness_threshold": b16.LIVENESS_THRESHOLD,
        "best_epoch": res.best_epoch,
        "best_selection_metric": res.best_selection_metric,
        "final_selection_metric": res.final_selection_metric,
        "n_live_dimensions": res.n_live_dimensions,
        "per_dimension_divergence": per_dim,
        "per_dimension_divergence_note": (
            "Computed at the FINAL epoch's validation-split posterior, which is "
            "where model.train_cvae computes it and where B18's recorded live "
            "dimension count comes from. It is not the selected checkpoint's."
        ),
        "history": res.history,
        "verification": {k: {"regenerated": v[0], "recorded": v[1]} for k, v in checks.items()},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
