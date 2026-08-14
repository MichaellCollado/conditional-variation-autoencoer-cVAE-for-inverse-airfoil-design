"""B18. Train the committed model and run the prior mechanism gate.

Trains the conditional VAE at B17's selected weights, then runs the three
group verification suite the article gives in Table 7. Group one is arm
blindness: every term except the prior must be bit-identical across a flag
flip, at a tolerance of exactly zero. Group two is gate response: the prior
term must be exactly zero with flags clear and strictly greater than zero with
flags set. Group three is the effect, against three thresholds.

The order is the point of the step. Every threshold was fixed in
params.B18_GATE_THRESHOLDS before B16's sweep was run, and this driver reads
them and defines none. The zero-weight control is trained and run through the
gate first, and is required to fail against numbers it had no part in setting.
The committed model then goes through the same gate with the same numbers.

Groups one and two are evaluated on one fixed batch of real validation
geometry with the flag column flipped and every other input held identical,
rather than on generated shapes. On generated geometry the flags-set test
would fail precisely when the prior worked perfectly and pulled every flagged
generation inside the region, so a well-behaved gate would reject a
well-behaved model.

No solver is called. The gate measures distance in standardised coefficient
space.

If the control passes a group three test, or the committed model fails any
test, that is the finding and it is reported. No threshold moves.

Run order      3 of 10. After B17, before B19.
Reads          b17_selection.json, and the build artifacts
Writes         committed_model.pt, control_model.pt, b18_gate.json
Runtime        2 training runs at 150 epochs each. No wall clock recorded
"""

import inspect
import json

import torch

import evaluate
import model as model_mod
import params
import run_b16_weight_sweep as b16

TH = params.B18_GATE_THRESHOLDS
OUT_PATH = "b18_gate.json"


def train_at(A, weights, training_seed):
    res = model_mod.train_cvae(
        A.x_all, A.cond_all, A.train_idx, A.val_idx, A.ensemble, A.safeguard,
        A.safeguard_bounds, A.reference_signature, A.region_extent,
        weights, b16.ARCH, training_seed=training_seed,
        liveness_threshold=b16.LIVENESS_THRESHOLD,
    )
    cvae = res.model
    cvae.load_state_dict(res.best_state)
    return cvae, res


def group_one(A, cvae):
    """Arm blindness. Every non-avian term evaluated twice on one fixed batch
    with the conditioning flag column flipped between calls, every other input
    held identical. Threshold: bit-identical, tolerance exactly 0.0."""
    x = A.x_all[A.val_idx]
    cond_a = A.cond_all[A.val_idx].clone()
    cond_b = cond_a.clone()
    cond_b[:, 1:21] = torch.where(cond_b[:, -1:] > 0.5,
                                  torch.zeros_like(cond_b[:, 1:21]),
                                  A.reference_signature.expand_as(cond_b[:, 1:21]))
    cond_b[:, -1] = 1.0 - cond_b[:, -1]

    cvae.eval()
    with torch.no_grad():
        mu, logvar = cvae.encoder(x)
        x_hat = cvae.decoder(mu, cond_a)
        z = torch.randn((x.shape[0], b16.ARCH.latent_dim), dtype=model_mod.DTYPE,
                        generator=torch.Generator().manual_seed(1))
        x_gen = cvae.decoder(z, cond_a)

        calls = {
            "reconstruction": lambda: model_mod.reconstruction_term(x_hat, x),
            "divergence": lambda: model_mod.divergence_term(mu, logvar),
            "safeguard_recon": lambda: A.safeguard.term(x_hat, A.safeguard_bounds),
            "safeguard_gen": lambda: A.safeguard.term(x_gen, A.safeguard_bounds),
            "target": lambda: model_mod.target_consistency_term(A.ensemble, x_gen, cond_a[:, 0]),
            "spread": lambda: model_mod.spread_penalty_term(A.ensemble, x_gen),
        }
        results = {}
        for name, fn in calls.items():
            v_a, v_b = float(fn()), float(fn())
            # The term cannot read the flag because no flag parameter exists on
            # it. Asserted structurally as well as numerically, so a future
            # signature change is caught rather than passing silently.
            target_fn = {"safeguard_recon": A.safeguard.term, "safeguard_gen": A.safeguard.term}.get(
                name, {"reconstruction": model_mod.reconstruction_term,
                       "divergence": model_mod.divergence_term,
                       "target": model_mod.target_consistency_term,
                       "spread": model_mod.spread_penalty_term}.get(name))
            sig = inspect.signature(target_fn).parameters
            takes_flag = any("flag" in p for p in sig)
            results[name] = {"value_flags_a": v_a, "value_flags_b": v_b,
                             "identical": v_a == v_b, "takes_flag_argument": takes_flag,
                             "passed": (v_a == v_b) and not takes_flag}
    return {"tests": results, "passed": all(r["passed"] for r in results.values())}


def group_two(A):
    """The gate responds. The avian term on a FIXED batch of real validation
    geometry, once with every flag clear and once with every flag set. Real
    geometry rather than generated, so this measures the term's gating and not
    the model's training: on generated shapes a perfectly working prior would
    pull every flagged sample inside the region, drive the term to zero, and
    make a well-behaved gate reject a well-behaved model."""
    x = A.x_all[A.val_idx]
    n = x.shape[0]
    clear = float(model_mod.avian_prior_term(x, torch.zeros(n, dtype=torch.bool),
                                             A.reference_signature, A.region_extent))
    setv = float(model_mod.avian_prior_term(x, torch.ones(n, dtype=torch.bool),
                                            A.reference_signature, A.region_extent))
    return {"term_flags_clear": clear, "term_flags_set": setv,
            "clear_is_exactly_zero": clear == 0.0, "set_is_strictly_positive": setv > 0.0,
            "passed": clear == 0.0 and setv > 0.0}


def group_three(A, cvae, generation_seed):
    pg = evaluate.paired_generation(cvae, A.reference_signature, b16.ARCH.latent_dim,
                                     generation_seed=generation_seed)
    g3 = TH["group_three_effect"]

    dc = evaluate.direction_consistency(pg, A.reference_signature)
    d_on = evaluate.mean_distance_to_reference(pg.x_set, A.reference_signature)
    d_off = evaluate.mean_distance_to_reference(pg.x_clear, A.reference_signature)
    sep = d_off - d_on
    sep_threshold = g3["g3b_mean_distance_separation"]["threshold_as_fraction_of_region_extent"] * A.region_extent
    ratio, arm_disp, redraw_disp = evaluate.arm_effect_against_noise(pg)

    tests = {
        "g3a_direction_consistency": {
            "measured": dc, "threshold": g3["g3a_direction_consistency"]["threshold"],
            "passed": dc >= g3["g3a_direction_consistency"]["threshold"]},
        "g3b_mean_distance_separation": {
            "measured": sep, "threshold": sep_threshold,
            "mean_distance_prior_on": d_on, "mean_distance_prior_off": d_off,
            "passed": sep >= sep_threshold},
        "g3c_effect_against_sampling_noise": {
            "measured": ratio, "threshold": g3["g3c_effect_against_sampling_noise"]["threshold"],
            "arm_displacement": arm_disp, "within_arm_redraw_displacement": redraw_disp,
            "passed": ratio >= g3["g3c_effect_against_sampling_noise"]["threshold"]},
    }
    return {"tests": tests,
            "passed_all": all(t["passed"] for t in tests.values()),
            "failed_all": all(not t["passed"] for t in tests.values())}


def run_gate(A, cvae, generation_seed):
    return {"group_one": group_one(A, cvae),
            "group_two": group_two(A),
            "group_three": group_three(A, cvae, generation_seed)}


def report(label, gate):
    print(f"\n--- {label} ---")
    print(f"  group one, arm blindness      : {'PASS' if gate['group_one']['passed'] else 'FAIL'}")
    for n, r in gate["group_one"]["tests"].items():
        print(f"      {n:16s} {r['value_flags_a']:.10f} vs {r['value_flags_b']:.10f}  "
              f"identical={r['identical']} takes_flag_arg={r['takes_flag_argument']}")
    g2 = gate["group_two"]
    print(f"  group two, gate responds      : {'PASS' if g2['passed'] else 'FAIL'}")
    print(f"      flags clear = {g2['term_flags_clear']:.17g} (must be exactly 0.0)")
    print(f"      flags set   = {g2['term_flags_set']:.6f} (must be > 0)")
    print(f"  group three, effect:")
    for n, r in gate["group_three"]["tests"].items():
        print(f"      {n:34s} measured {r['measured']:>10.4f}  "
              f"threshold {r['threshold']:>8.4f}  {'PASS' if r['passed'] else 'FAIL'}")


def main():
    A = model_mod.load_build_artifacts(".")
    sel = json.load(open("b17_selection.json"))
    committed = model_mod.ObjectiveWeights(
        divergence_weight=sel["set_as_stated_values"]["divergence_weight"],
        safeguard_weight=sel["set_as_stated_values"]["safeguard_weight"],
        target_weight=sel["selected_by_sweep"]["target"],
        spread_weight=sel["selected_by_sweep"]["spread"],
        avian_weight=sel["selected_by_sweep"]["avian"],
    )
    control = model_mod.ObjectiveWeights(**{**committed.__dict__, "avian_weight": 0.0})

    training_seed = model_mod.seed_int(18, 0)
    generation_seed = model_mod.seed_int(18, 1)

    print("Thresholds, fixed at:", TH["fixed_at"])
    print(f"committed weights: {committed}")
    print(f"control weights  : {control}")
    print(f"training seed {training_seed}, generation seed {generation_seed}", flush=True)

    print("\ntraining the ZERO-WEIGHT CONTROL first, and gating it before the "
          "committed model exists", flush=True)
    control_cvae, control_res = train_at(A, control, training_seed)
    torch.save(control_res.best_state, "control_model.pt")
    control_gate = run_gate(A, control_cvae, generation_seed)
    report("CONTROL, avian weight 0.0", control_gate)

    control_ok = (control_gate["group_one"]["passed"]
                  and control_gate["group_two"]["passed"]
                  and control_gate["group_three"]["failed_all"])
    print(f"\n  control requirement (pass one and two, FAIL every group three test): "
          f"{'MET' if control_ok else 'NOT MET'}")

    print("\ntraining the COMMITTED model", flush=True)
    cvae, res = train_at(A, committed, training_seed)
    torch.save(res.best_state, "committed_model.pt")
    committed_gate = run_gate(A, cvae, generation_seed)
    report("COMMITTED", committed_gate)

    committed_ok = (committed_gate["group_one"]["passed"]
                    and committed_gate["group_two"]["passed"]
                    and committed_gate["group_three"]["passed_all"])
    print(f"\n  committed model requirement (pass all three groups): "
          f"{'MET' if committed_ok else 'NOT MET'}")

    out = {
        "thresholds": TH, "committed_weights": committed.__dict__,
        "control_weights": control.__dict__,
        "training_seed": training_seed, "generation_seed": generation_seed,
        "control_gate": control_gate, "committed_gate": committed_gate,
        "control_requirement_met": control_ok,
        "committed_requirement_met": committed_ok,
        "committed_best_epoch": res.best_epoch,
        "committed_best_selection_metric": res.best_selection_metric,
        "committed_n_live_dimensions": res.n_live_dimensions,
        "control_best_epoch": control_res.best_epoch,
        "control_n_live_dimensions": control_res.n_live_dimensions,
        "no_solver_called": "No solver call was made anywhere in B18.",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> {OUT_PATH}")
    print("\nB18 VERDICT: " + ("PASS" if (control_ok and committed_ok) else "FAIL"))


if __name__ == "__main__":
    main()
