"""B16. The prior weight sensitivity sweep.

Sweeps each of the three selectable weights one at a time with the other two
at zero, 21 distinct training runs in all, and records per candidate the
validation reconstruction, the mean distance to the avian reference per arm,
the diversity statistic and the live latent dimension count.

Selects nothing. It writes the table; B17 applies the rule to it. The rule
itself is params.WEIGHT_SELECTION_RULE, fixed before this sweep was run, and
this driver reads it rather than defining one.

Each ladder is scaled to the term's own magnitude at initialisation, so the
driver raises rather than proceeding if a term's value at initialisation is
exactly zero, because the ladder scale would then be undefined. Every run
writes to a path carrying its own weight and the driver refuses to overwrite
an existing path, so no run of a sweep can silently replace another.

The diversity figures here sit on the sweep's internal grid of 11 evenly
spaced normalised targets from 0.0 to 1.0. That is not the requested target
band committed later, and figures on the two grids are not comparable point
for point.

Run order      1 of 10. First retained driver. Before B17.
Reads          the build artifacts, through model.load_build_artifacts
Writes         sweep/sweep_table.json, and per candidate
               sweep/history_<key>.json and sweep/checkpoint_<key>.pt
Runtime        21 training runs in 309 s on the machine in Table A3
"""

import json
import os
import time

import numpy as np
import torch

import dataset as dataset_mod
import evaluate
import model as model_mod
import params

SWEEP_DIR = "sweep"
TABLE_PATH = os.path.join(SWEEP_DIR, "sweep_table.json")

LADDER_MULTIPLIERS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
DIVERGENCE_LADDER = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]

# The reference vector every sweep holds the non-swept weights at. Divergence
# at its committed stated value; every auxiliary weight at zero, so each swept
# weight is a clean perturbation of the plain conditional VAE. The safeguard's
# reference is immaterial and provably so, since its term is identically zero.
REFERENCE = {"divergence": 1.0, "safeguard": 0.0, "target": 0.0, "spread": 0.0, "avian": 0.0}

# Avian is swept LAST, so nothing about its ladder can be read back into the
# design of the ladders before it.
SWEPT_ORDER = ["divergence", "target", "spread", "avian"]

ARCH = model_mod.CVAEArchitecture(
    latent_dim=8, hidden_width=64, depth=2, learning_rate=1e-3,
    epochs=150, batch_size=64, warmup_epochs=20,
)
LIVENESS_THRESHOLD = 0.01


def weights_from(vec):
    return model_mod.ObjectiveWeights(
        divergence_weight=vec["divergence"], safeguard_weight=vec["safeguard"],
        target_weight=vec["target"], spread_weight=vec["spread"],
        avian_weight=vec["avian"],
    )


def measure_initialisation_terms(A, training_seed):
    """recon_0 and T_0 for every term, at the untrained model's
    initialisation, over the whole training split in one batch, at the
    sweep's own training seed. u_T = recon_0 / T_0 sets each auxiliary
    ladder. Reproducible: the sequence of RNG draws below is fixed."""
    torch.manual_seed(training_seed)
    net = model_mod.CVAE(A.x_all.shape[1], A.cond_all.shape[1], ARCH).to(model_mod.DTYPE)
    x_tr, c_tr = A.x_all[A.train_idx], A.cond_all[A.train_idx]
    flag_tr = c_tr[:, -1] > 0.5

    mu, logvar = net.encoder(x_tr)
    z = net.reparameterize(mu, logvar)
    x_hat = net.decoder(z, c_tr)
    z_gen = torch.randn((x_tr.shape[0], ARCH.latent_dim), dtype=model_mod.DTYPE)
    x_gen = net.decoder(z_gen, c_tr)

    with torch.no_grad():
        return {
            "reconstruction": float(model_mod.reconstruction_term(x_hat, x_tr)),
            "divergence": float(model_mod.divergence_term(mu, logvar)),
            "safeguard": float(A.safeguard.term(x_gen, A.safeguard_bounds)),
            "target": float(model_mod.target_consistency_term(A.ensemble, x_gen, c_tr[:, 0])),
            "spread": float(model_mod.spread_penalty_term(A.ensemble, x_gen)),
            "avian": float(model_mod.avian_prior_term(x_gen, flag_tr,
                                                      A.reference_signature, A.region_extent)),
        }


def build_ladders(init_terms):
    recon0 = init_terms["reconstruction"]
    ladders = {"divergence": list(DIVERGENCE_LADDER)}
    for name in ("target", "spread", "avian"):
        t0 = init_terms[name]
        if t0 == 0.0:
            raise ValueError(
                f"T_0 for the {name} term is exactly zero, so u_T is undefined and this "
                f"weight cannot be swept. Per WEIGHT_SELECTION_RULE the sweep stops and "
                f"the fact is reported. No value is substituted."
            )
        u = recon0 / t0
        ladders[name] = [m * u for m in LADDER_MULTIPLIERS]
    return ladders


def columns_at_checkpoint(A, cvae, generation_seed):
    """Every column the selection rule reads, measured at the SELECTED
    checkpoint on the validation split. The caller must already have loaded
    the best state into cvae."""
    x_val, cond_val = A.x_all[A.val_idx], A.cond_all[A.val_idx]
    cvae.eval()
    with torch.no_grad():
        mu, logvar = cvae.encoder(x_val)
        x_hat = cvae.decoder(mu, cond_val)
        recon = float(model_mod.reconstruction_term(x_hat, x_val))
        div = float(model_mod.divergence_term(mu, logvar))

        z_gen = torch.randn((x_val.shape[0], ARCH.latent_dim), dtype=model_mod.DTYPE,
                            generator=torch.Generator().manual_seed(generation_seed))
        x_gen = cvae.decoder(z_gen, cond_val)
        safe = float(A.safeguard.term(x_gen, A.safeguard_bounds))
        targ = float(model_mod.target_consistency_term(A.ensemble, x_gen, cond_val[:, 0]))
        spr = float(model_mod.spread_penalty_term(A.ensemble, x_gen))

        per_dim = model_mod.per_dimension_divergence(mu, logvar)
        n_live = int(np.sum(per_dim > LIVENESS_THRESHOLD))

    pg = evaluate.paired_generation(cvae, A.reference_signature, ARCH.latent_dim,
                                     generation_seed=generation_seed)
    return {
        "val_reconstruction": recon,
        "val_divergence": div,
        "val_safeguard": safe,
        "val_target": targ,
        "val_spread": spr,
        "mean_distance_prior_on": evaluate.mean_distance_to_reference(pg.x_set, A.reference_signature),
        "mean_distance_prior_off": evaluate.mean_distance_to_reference(pg.x_clear, A.reference_signature),
        "diversity_prior_on": evaluate.generative_diversity(pg.x_set),
        "diversity_prior_off": evaluate.generative_diversity(pg.x_clear),
        "n_live_dimensions": n_live,
        "per_dimension_divergence": [float(v) for v in per_dim],
    }


def main():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    A = model_mod.load_build_artifacts(".")

    # The conditioning layout evaluate.build_conditioning assumes must be the
    # layout B10 actually committed. Checked against the artifact, not assumed.
    sig = A.reference_signature
    probe = evaluate.build_conditioning(torch.tensor([0.5], dtype=model_mod.DTYPE), sig, True)
    set_rows = A.cond_all[A.cond_all[:, -1] > 0.5]
    assert torch.equal(probe[0, 1:21], sig), "signature block layout disagrees with B10"
    assert torch.equal(set_rows[0, 1:21], sig), "B10 artifact block is not the signature"
    assert float(probe[0, -1]) == 1.0, "flag column sense disagrees with B10"

    training_seed = model_mod.seed_int(16, 0)
    generation_seed = model_mod.seed_int(16, 1)
    print(f"training seed {training_seed}, generation seed {generation_seed}", flush=True)

    init_terms = measure_initialisation_terms(A, training_seed)
    ladders = build_ladders(init_terms)
    recon0 = init_terms["reconstruction"]
    print("\ninitialisation term values and the u_T each ladder is scaled by:")
    for k, v in init_terms.items():
        u = "n/a" if k in ("reconstruction", "divergence", "safeguard") else f"{recon0 / v:.6g}"
        print(f"  {k:16s} = {v:.6e}   u_T = {u}", flush=True)
    print("\nladders:")
    for k in SWEPT_ORDER:
        print(f"  {k:12s} {['%.6g' % c for c in ladders[k]]}", flush=True)

    # One run per DISTINCT weight vector. The reference vector is shared by four
    # ladder rows (divergence at 1.0, and target/spread/avian each at 0.0) and is
    # trained once. Those rows are the same run and say so.
    cache = {}
    rows = []
    t_start = time.time()

    for name in SWEPT_ORDER:
        for value in ladders[name]:
            vec = dict(REFERENCE)
            vec[name] = value
            key = "d{divergence:.10g}_s{safeguard:.10g}_t{target:.10g}_p{spread:.10g}_a{avian:.10g}".format(**vec)

            if key not in cache:
                history_path = os.path.join(SWEEP_DIR, f"history_{key}.json")
                if os.path.exists(history_path):
                    raise FileExistsError(
                        f"{history_path} already exists. B16 refuses to overwrite a "
                        f"training history; that is the exact failure this step exists "
                        f"to remove. Move or delete the sweep directory to rerun."
                    )
                t0 = time.time()
                res = model_mod.train_cvae(
                    A.x_all, A.cond_all, A.train_idx, A.val_idx, A.ensemble,
                    A.safeguard, A.safeguard_bounds, A.reference_signature,
                    A.region_extent, weights_from(vec), ARCH,
                    training_seed=training_seed, liveness_threshold=LIVENESS_THRESHOLD,
                )
                cvae = res.model
                cvae.load_state_dict(res.best_state)
                cols = columns_at_checkpoint(A, cvae, generation_seed)
                with open(history_path, "w") as f:
                    json.dump({
                        "weights": vec, "training_seed": training_seed,
                        "best_epoch": res.best_epoch,
                        "best_selection_metric": res.best_selection_metric,
                        "final_selection_metric": res.final_selection_metric,
                        "history": res.history,
                    }, f, indent=1)
                torch.save(res.best_state, os.path.join(SWEEP_DIR, f"checkpoint_{key}.pt"))
                cache[key] = {
                    "weights": vec, "history_path": history_path,
                    "checkpoint_path": os.path.join(SWEEP_DIR, f"checkpoint_{key}.pt"),
                    "best_epoch": res.best_epoch,
                    "best_selection_metric": res.best_selection_metric,
                    **cols,
                }
                print(f"  [{time.time() - t_start:7.1f}s] {name}={value:<12.6g} "
                      f"R={cols['val_reconstruction']:.6f} "
                      f"E_div={cols['val_divergence']:.4f} "
                      f"d_on={cols['mean_distance_prior_on']:.4f} "
                      f"D={cols['diversity_prior_on']:.4f} "
                      f"live={cols['n_live_dimensions']} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            else:
                print(f"  [{time.time() - t_start:7.1f}s] {name}={value:<12.6g} "
                      f"reference run, already trained, shared", flush=True)

            rows.append({"swept_weight": name, "candidate": value, **cache[key]})

    with open(TABLE_PATH, "w") as f:
        json.dump({
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "reference_vector": REFERENCE,
            "initialisation_terms": init_terms,
            "ladders": ladders,
            "architecture": ARCH.__dict__,
            "diversity_definition": params.DIVERSITY_DEFINITION,
            "n_distinct_runs": len(cache),
            "rows": rows,
        }, f, indent=1)

    print(f"\n{len(rows)} table rows from {len(cache)} distinct runs "
          f"in {time.time() - t_start:.0f}s -> {TABLE_PATH}", flush=True)


if __name__ == "__main__":
    main()
