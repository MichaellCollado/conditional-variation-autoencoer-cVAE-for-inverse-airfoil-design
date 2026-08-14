"""The surrogate ensemble, the conditional VAE, the objective, and the loader.

Holds everything that shapes the trained artifact: the frozen surrogate
ensemble, the six objective terms, the training loop, checkpoint selection,
and the single artifact loader every driver goes through.

The six terms are reconstruction, latent divergence, the geometric safeguard,
target consistency, the ensemble spread penalty, and the avian prior.
avian_prior_term is the only one that reads the flag. The other five are
structurally unable to, because none of their term functions accepts a flag
argument at all.

load_build_artifacts is the one loader. It opens dataset.npz, population.npz,
conditioning.npz, split.npz, standardization.npz, normalization.npz,
avian_signature.npz and surrogate_ensemble.pt; derives the safeguard bounds
from the training split; and runs five standing diagnostics that read and
raise, compute nothing any caller uses, and change no value.

Two things worth knowing about the numbers. The surrogate's held-out error
baseline is the training split's mean label, not the whole dataset's. The
safeguard bounds are re-derived from the training population's own measured
range at load, so they are stored as constants nowhere.

No scipy. Every statistic this file needs, it computes itself.

Called by run_b16_weight_sweep.py, run_b18_prior_mechanism_gate.py,
run_b21_paired_yield.py, run_b23_paired_evaluation.py, run_b25_metrics.py,
run_committed_training_history.py and run_figures.py.

Public API
    MLP, seed_int
    build_surrogate_ensemble, train_surrogate_member, SurrogateEnsemble,
        SurrogateArchitecture, load_surrogate_ensemble,
        mean_absolute_error_raw, training_split_mean_baseline_error
    reconstruction_term, divergence_term, target_consistency_term,
        spread_penalty_term, avian_prior_term, squared_hinge
    TorchSafeguard, derive_safeguard_bounds, SafeguardBounds
    ObjectiveWeights, total_objective
    CVAE, CVAEArchitecture, Encoder, Decoder
    train_cvae, TrainingResult, selection_metric_on, per_dimension_divergence
    load_build_artifacts, BuildArtifacts, StandingDiagnosticError
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import geometry
import dataset as dataset_mod

DTYPE = torch.float64


def seed_int(build_step: int, substream: int) -> int:
    """One integer draw from this build's own offset-ruled numpy stream
    (dataset.rng_for), used to seed torch. torch takes an integer seed, not
    a numpy Generator, so this is the one place a torch stream is derived
    from the recorded base seed and offset rule rather than seeded
    independently."""
    rng = dataset_mod.rng_for(build_step, substream)
    return int(rng.integers(0, 2**31 - 1))


# ---------------------------------------------------------------------------
# B13. Surrogate ensemble.
#
# Predicts the normalised label from geometry alone (the standardised CST
# coefficient vector, 20 columns at the committed order 9) -- not from the
# full conditioning array, which carries the target and the avian block and
# would let the surrogate see what it is being asked to predict.
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_width: int, depth: int, out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_width))
            layers.append(nn.ReLU())
            d = hidden_width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SurrogateArchitecture:
    member_count: int
    hidden_width: int
    depth: int
    learning_rate: float
    epochs: int
    batch_size: int


def train_surrogate_member(member_seed: int, x_train: torch.Tensor, y_train: torch.Tensor,
                            arch: SurrogateArchitecture) -> MLP:
    torch.manual_seed(member_seed)
    model = MLP(x_train.shape[1], arch.hidden_width, arch.depth, 1).to(DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=arch.learning_rate)
    batch_gen = torch.Generator().manual_seed(member_seed)
    n = x_train.shape[0]
    for _ in range(arch.epochs):
        perm = torch.randperm(n, generator=batch_gen)
        for start in range(0, n, arch.batch_size):
            idx = perm[start:start + arch.batch_size]
            xb, yb = x_train[idx], y_train[idx]
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            opt.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@dataclass
class SurrogateEnsemble:
    members: List[MLP]
    architecture: SurrogateArchitecture

    def predict_each(self, x: torch.Tensor) -> torch.Tensor:
        """(n_members, n_rows). Members are frozen (requires_grad_(False)
        on their own parameters) but this forward pass stays differentiable
        with respect to x, which is what B14's target and spread terms need
        -- gradient must reach the generator that produced x, not the
        surrogate."""
        return torch.stack([m(x).squeeze(-1) for m in self.members], dim=0)

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_each(x).mean(dim=0)

    def predict_spread(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample spread across members, sample convention (ddof=1),
        consistent with this build's other dispersion statistics (B11's
        cohens_d in dataset.py, and the build plan's stated sample convention)."""
        preds = self.predict_each(x)
        if preds.shape[0] < 2:
            raise ValueError("spread requires at least 2 members")
        return preds.std(dim=0, unbiased=True)


def build_surrogate_ensemble(x_train: torch.Tensor, y_train: torch.Tensor,
                              arch: SurrogateArchitecture, build_step: int = 13) -> SurrogateEnsemble:
    members = [
        train_surrogate_member(seed_int(build_step, i), x_train, y_train, arch)
        for i in range(arch.member_count)
    ]
    return SurrogateEnsemble(members=members, architecture=arch)


def mean_absolute_error_raw(pred_normalized: torch.Tensor, true_raw: np.ndarray,
                             norm: dataset_mod.LabelNormalization) -> float:
    pred_raw = dataset_mod.denormalize_label(pred_normalized.detach().numpy(), norm)
    return float(np.mean(np.abs(pred_raw - true_raw)))


def training_split_mean_baseline_error(train_raw_labels: np.ndarray, val_raw_labels: np.ndarray) -> float:
    """CORRECTION 1: the baseline the ensemble's held-out error
    is read against is the TRAINING split's own mean label, not the full
    dataset's. Predicting a statistic computed over rows the model will
    later be judged against (including validation rows) would leak
    validation-set information into the "trivial" baseline itself."""
    baseline = float(train_raw_labels.mean())
    return float(np.mean(np.abs(val_raw_labels - baseline)))


# ---------------------------------------------------------------------------
# B14. Objective terms.
# ---------------------------------------------------------------------------

def reconstruction_term(x_hat: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
    """Squared error across standardised dimensions (summed over the 20
    columns), averaged per batch (mean over rows)."""
    return torch.mean(torch.sum((x_hat - x_real) ** 2, dim=1))


def divergence_term(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed form KL against the unit Gaussian, averaged per batch."""
    return torch.mean(-0.5 * torch.sum(1.0 + logvar - mu ** 2 - torch.exp(logvar), dim=1))


def squared_hinge(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) ** 2


@dataclass
class SafeguardBounds:
    thickness_lower_bound: float
    curvature_upper_bound: float
    margin_fraction: float
    n_grid_points: int
    edge_margin: float
    derivation: str


def derive_safeguard_bounds(training_upper_coefficients: np.ndarray,
                             training_lower_coefficients: np.ndarray,
                             order: int, n_grid_points: int, edge_margin: float,
                             margin_fraction: float) -> SafeguardBounds:
    """CORRECTION 2: re-derived from the TRAINING population's
    own measured range, not inherited -- the superseded threshold rested
    on a figure that exists in no supplied file.

    CORRECTION 4: B14 re-opened to fix a measurement-path
    defect found by the build's pre-flight scan. This function used to
    decode each row onto a 160-point cosine grid with decode_airfoil and
    then np.interp the result onto the 200-point interior grid, while
    TorchSafeguard.term -- the thing the bound is actually applied by --
    evaluates the CST basis directly at the interior grid with no
    interpolation. The two paths disagreed by 1.10x on the training
    population's curvature maximum (9.697e-05 interpolated against
    8.779e-05 exact), so the committed bound was derived on one
    measurement and enforced on another.

    The fix is to derive on the path the term uses. geometry.decode_surface
    already evaluates class times basis at whatever abscissae it is given,
    so passing it the interior grid directly is the same computation
    TorchSafeguard performs, in numpy. No new helper, and the
    n_points_per_surface argument is gone because nothing interpolates any
    more.

    This tightens curvature_upper_bound and leaves every verdict unchanged,
    because the safeguard term is identically zero on this build either way
    (0 of 785 training rows violate either bound on the exact path). It is
    fixed because a threshold derived on one measurement and applied on
    another is a defect whether or not it currently bites.

    Mirrors B03's own plausibility-bound derivation exactly: decode each
    training row's own CST coefficients (no perturbation, this is the
    population that already exists), measure max thickness and max
    absolute second difference of camber on the SAME interior grid B03
    already committed (n_grid_points, edge_margin, both from params.py's
    pre_solver_filter_thresholds slot), and apply the SAME margin_fraction
    B03 already justified and disclosed (0.20) -- reused rather than a
    freshly invented number, since it is already a stated, justified
    choice at this same kind of bound.

    Only the two bounds the safeguard actually reads are computed:
    thickness_lower_bound (penalised when thickness falls BELOW it) and
    curvature_upper_bound (penalised when curvature rises ABOVE it) --
    the same asymmetry B03's own filter uses (thickness has both a floor
    and a ceiling; camber/curvature only ever has a ceiling in this build).
    """
    max_thicknesses = []
    max_curvatures = []
    grid = geometry._interior_grid(n_grid_points, edge_margin)
    for u, l in zip(training_upper_coefficients, training_lower_coefficients):
        upper_y = geometry.decode_surface(u, grid)
        lower_y = geometry.decode_surface(l, grid)
        thickness = upper_y - lower_y
        camber = 0.5 * (upper_y + lower_y)
        max_thicknesses.append(float(np.max(thickness)))
        max_curvatures.append(float(np.max(np.abs(np.diff(camber, n=2)))))

    thickness_min = min(max_thicknesses)
    curvature_max = max(max_curvatures)
    thickness_lower_bound = thickness_min * (1.0 - margin_fraction)
    curvature_upper_bound = curvature_max * (1.0 + margin_fraction)

    derivation = (
        f"Training population only ({len(max_thicknesses)} rows), decoded from their own "
        f"stored CST coefficients (order {order}, no perturbation -- this is the existing "
        f"population, not a fresh draw), measured on the same interior grid B03 already "
        f"committed (n_grid_points={n_grid_points}, edge_margin={edge_margin}), by "
        f"evaluating the CST basis DIRECTLY at that grid -- the same path "
        f"TorchSafeguard.term applies the bound on, with no intermediate decode grid and "
        f"no interpolation. "
        f"min(max_thickness) over the training population = {thickness_min:.6f}; "
        f"max(max_abs_camber_2nd_diff) over the training population = {curvature_max:.6f}. "
        f"thickness_lower_bound = min(max_thickness) * (1 - margin) = {thickness_min:.6f} * "
        f"(1 - {margin_fraction}) = {thickness_lower_bound:.6f}. "
        f"curvature_upper_bound = max(max_abs_camber_2nd_diff) * (1 + margin) = "
        f"{curvature_max:.6f} * (1 + {margin_fraction}) = {curvature_upper_bound:.6f}. "
        f"margin_fraction = {margin_fraction}, reused from the already-committed and "
        f"already-justified B03 margin (params.py slot pre_solver_filter_thresholds), not "
        f"a freshly invented figure for this step."
    )

    return SafeguardBounds(
        thickness_lower_bound=thickness_lower_bound,
        curvature_upper_bound=curvature_upper_bound,
        margin_fraction=margin_fraction,
        n_grid_points=n_grid_points,
        edge_margin=edge_margin,
        derivation=derivation,
    )


class TorchSafeguard:
    """A differentiable decode-and-measure, mirroring geometry.py's own
    fit/decode/measure functions exactly (same class function, same
    Bernstein basis, same interior grid, same second-difference curvature
    proxy), so the safeguard reads the same notion of "shape" the rest of
    the pipeline does. The only reason this is reimplemented rather than
    calling geometry.py directly is differentiability: geometry.py's
    decode is numpy, and the safeguard must backpropagate into the
    generator that produced the coefficients it measures.

    Takes standardised coefficients and destandardises internally with the
    SAME artifact geometry.py's own standardize/destandardize use (B08),
    not a separately fit scale.

    Takes NO conditioning input at all -- only the coefficient vector being
    measured. This is exactly what B14's own falsification check verifies:
    flipping the flag column of a batch cannot change this term's value,
    because this class never sees the flag column in the first place.
    """

    def __init__(self, order: int, n_grid_points: int, edge_margin: float,
                 std_stats: geometry.StandardizationStats):
        grid = geometry._interior_grid(n_grid_points, edge_margin)
        c = geometry.class_function(grid)
        basis = geometry.bernstein_basis(grid, order)
        design = basis * c[:, None]  # (n_grid_interior, order + 1)
        self.design = torch.tensor(design, dtype=DTYPE)
        self.mean = torch.tensor(std_stats.mean, dtype=DTYPE)
        self.std = torch.tensor(std_stats.std, dtype=DTYPE)
        self.order = order

    def decode_thickness_curvature(self, standardized_coefficients: torch.Tensor
                                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = standardized_coefficients * self.std + self.mean
        n_c = self.order + 1
        upper_c = raw[:, :n_c]
        lower_c = raw[:, n_c:2 * n_c]
        upper_y = upper_c @ self.design.T  # (batch, n_grid_interior)
        lower_y = lower_c @ self.design.T
        thickness = upper_y - lower_y
        camber = 0.5 * (upper_y + lower_y)
        d1 = camber[:, 1:] - camber[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        max_thickness = thickness.max(dim=1).values
        max_curvature = d2.abs().max(dim=1).values
        return max_thickness, max_curvature

    def term(self, standardized_coefficients: torch.Tensor, bounds: SafeguardBounds) -> torch.Tensor:
        max_t, max_c = self.decode_thickness_curvature(standardized_coefficients)
        t_term = squared_hinge(bounds.thickness_lower_bound - max_t)
        c_term = squared_hinge(max_c - bounds.curvature_upper_bound)
        return torch.mean(t_term + c_term)


def target_consistency_term(ensemble: SurrogateEnsemble, x_gen: torch.Tensor,
                             target_normalized: torch.Tensor) -> torch.Tensor:
    """Squared error between the ensemble mean prediction and the requested
    target, both normalised. Reads geometry and the target column only --
    never the flag."""
    pred = ensemble.predict_mean(x_gen)
    return torch.mean((pred - target_normalized) ** 2)


def spread_penalty_term(ensemble: SurrogateEnsemble, x_gen: torch.Tensor) -> torch.Tensor:
    """Mean per-sample spread across ensemble members. Reads geometry only."""
    return torch.mean(ensemble.predict_spread(x_gen))


def avian_prior_term(x_gen: torch.Tensor, flag_set: torch.Tensor,
                      reference_signature: torch.Tensor, region_extent: float) -> torch.Tensor:
    """The one term that reads the flag. Squared hinge on distance above
    the region extent, summed over flagged rows and divided by the flagged
    count. flag_set: bool tensor, True where the row's flag is SET (the
    same sense B10's conditioning array and B14's plan text both use --
    1.0/True = signature present, the avian pull applies). Exactly zero
    when no row in the batch is flagged, by construction of the sum over
    an empty set, guarded explicitly against 0/0 rather than left to
    produce nan."""
    n_flagged = int(flag_set.sum().item())
    if n_flagged == 0:
        return torch.zeros((), dtype=DTYPE)
    diff = x_gen[flag_set] - reference_signature
    dist = torch.linalg.norm(diff, dim=1)
    hinge = squared_hinge(dist - region_extent)
    return torch.sum(hinge) / n_flagged


@dataclass
class ObjectiveWeights:
    divergence_weight: float
    safeguard_weight: float
    target_weight: float
    spread_weight: float
    avian_weight: float


def total_objective(recon: torch.Tensor, div: torch.Tensor,
                     safeguard_recon: torch.Tensor, safeguard_gen: torch.Tensor,
                     target: torch.Tensor, spread: torch.Tensor, avian: torch.Tensor,
                     weights: ObjectiveWeights, divergence_schedule_multiplier: float) -> torch.Tensor:
    """Reconstruction plus weighted divergence plus each weighted term.
    Reconstruction carries an implicit weight of 1; every other term is
    scaled by its own committed weight, recorded in params.py. The safeguard is applied to both passes' output
    (it takes no conditioning input, so there is nothing pass-specific
    about it) and both applications share the one safeguard weight."""
    return (
        recon
        + weights.divergence_weight * divergence_schedule_multiplier * div
        + weights.safeguard_weight * (safeguard_recon + safeguard_gen)
        + weights.target_weight * target
        + weights.spread_weight * spread
        + weights.avian_weight * avian
    )


# ---------------------------------------------------------------------------
# B15. Conditional VAE, training loop, checkpoint selection.
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_width: int, depth: int, latent_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_width))
            layers.append(nn.ReLU())
            d = hidden_width
        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(d, latent_dim)
        self.logvar_head = nn.Linear(d, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.mu_head(h), self.logvar_head(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, conditioning_dim: int, hidden_width: int,
                 depth: int, out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        d = latent_dim + conditioning_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_width))
            layers.append(nn.ReLU())
            d = hidden_width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, cond], dim=1))


@dataclass
class CVAEArchitecture:
    latent_dim: int
    hidden_width: int
    depth: int
    learning_rate: float
    epochs: int
    batch_size: int
    warmup_epochs: int


class CVAE(nn.Module):
    def __init__(self, geometry_dim: int, conditioning_dim: int, arch: CVAEArchitecture):
        super().__init__()
        self.encoder = Encoder(geometry_dim, arch.hidden_width, arch.depth, arch.latent_dim)
        self.decoder = Decoder(arch.latent_dim, conditioning_dim, arch.hidden_width,
                                arch.depth, geometry_dim)
        self.arch = arch

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor,
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, dtype=std.dtype, generator=generator)
        return mu + eps * std


@dataclass
class TrainingResult:
    model: CVAE
    best_state: Dict[str, torch.Tensor]
    best_epoch: int
    best_selection_metric: float
    final_state: Dict[str, torch.Tensor]
    final_selection_metric: float
    history: List[dict]
    per_dimension_divergence: np.ndarray
    n_live_dimensions: int
    liveness_threshold: float


def per_dimension_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> np.ndarray:
    """Mean, over rows, of the per-dimension KL contribution
    0.5*(mu_d^2 + exp(logvar_d) - 1 - logvar_d). One value per latent
    dimension."""
    per_dim = 0.5 * (mu ** 2 + torch.exp(logvar) - 1.0 - logvar)
    return per_dim.mean(dim=0).detach().numpy()


def train_cvae(x_all: torch.Tensor, cond_all: torch.Tensor,
                train_idx: np.ndarray, val_idx: np.ndarray,
                ensemble: SurrogateEnsemble, safeguard: TorchSafeguard,
                safeguard_bounds: SafeguardBounds,
                reference_signature: torch.Tensor, region_extent: float,
                weights: ObjectiveWeights, arch: CVAEArchitecture,
                training_seed: int, liveness_threshold: float = 0.01,
                log_components: bool = False) -> TrainingResult:
    """Two passes per batch, per B15's own logic:

    Pass one reconstructs REAL geometry (encode -> reparameterise -> decode
    at the row's own stored conditioning) with reconstruction, divergence
    and the safeguard. No avian term.

    Pass two generates from FRESH codes (sampled independently of the
    encoder) at the SAME targets and the SAME stored flags as the batch,
    applying the safeguard, the target term, the spread penalty and the
    flag gated avian term.

    `log_components` is off by default and changes nothing when off. Turned
    on, each epoch's history entry additionally carries every objective term
    evaluated UNWEIGHTED on the validation split, which is what F11 plots.
    Added during the build because no build step recorded the components and a
    figure cannot be drawn from a quantity nothing stored. It draws no random
    number and touches no optimiser state, so a run with it on is bit
    identical to the same run with it off; the driver that uses it verifies
    that against B18's own recorded values rather than asserting it.

    Each epoch, the selection metric (CORRECTION 3: unweighted, never
    scaled by any objective weight) is evaluated on the validation split,
    using a FIXED set of freshly-sampled codes drawn once before training
    and reused every epoch -- so an epoch-to-epoch change in the metric
    reflects the model, not a different random draw. The checkpoint at the
    best metric is kept; the final epoch's weights are also kept aside,
    for the falsification check's own comparison.
    """
    torch.manual_seed(training_seed)
    geometry_dim = x_all.shape[1]
    conditioning_dim = cond_all.shape[1]
    model = CVAE(geometry_dim, conditioning_dim, arch).to(DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=arch.learning_rate)

    x_train, cond_train = x_all[train_idx], cond_all[train_idx]
    x_val, cond_val = x_all[val_idx], cond_all[val_idx]
    target_val = cond_val[:, 0]

    val_gen = torch.Generator().manual_seed(training_seed + 1)
    z_fresh_val = torch.randn((x_val.shape[0], arch.latent_dim), dtype=DTYPE, generator=val_gen)

    train_gen = torch.Generator().manual_seed(training_seed)
    n_train = x_train.shape[0]

    history: List[dict] = []
    best_metric = float("inf")
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] = {}
    final_state: Dict[str, torch.Tensor] = {}
    final_metric = float("inf")
    final_mu = final_logvar = None

    for epoch in range(arch.epochs):
        model.train()
        schedule_mult = min(1.0, epoch / max(1, arch.warmup_epochs))
        perm = torch.randperm(n_train, generator=train_gen)
        for start in range(0, n_train, arch.batch_size):
            idx = perm[start:start + arch.batch_size]
            xb, condb = x_train[idx], cond_train[idx]
            target_b = condb[:, 0]
            flag_b = condb[:, -1] > 0.5

            opt.zero_grad()

            mu, logvar = model.encoder(xb)
            z = model.reparameterize(mu, logvar)
            x_hat = model.decoder(z, condb)
            recon = reconstruction_term(x_hat, xb)
            div = divergence_term(mu, logvar)
            safeguard_recon = safeguard.term(x_hat, safeguard_bounds)

            z_gen = torch.randn((xb.shape[0], arch.latent_dim), dtype=DTYPE)
            x_gen = model.decoder(z_gen, condb)
            safeguard_gen = safeguard.term(x_gen, safeguard_bounds)
            target_term = target_consistency_term(ensemble, x_gen, target_b)
            spread_term = spread_penalty_term(ensemble, x_gen)
            avian_term = avian_prior_term(x_gen, flag_b, reference_signature, region_extent)

            loss = total_objective(recon, div, safeguard_recon, safeguard_gen,
                                    target_term, spread_term, avian_term,
                                    weights, schedule_mult)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            mu_v, logvar_v = model.encoder(x_val)
            x_hat_v = model.decoder(mu_v, cond_val)
            recon_v = reconstruction_term(x_hat_v, x_val)
            x_gen_v = model.decoder(z_fresh_val, cond_val)
            target_v = target_consistency_term(ensemble, x_gen_v, target_val)
            selection_metric = float(recon_v.item() + target_v.item())

        entry = {
            "epoch": epoch,
            "val_selection_metric": selection_metric,
            "divergence_schedule_multiplier": schedule_mult,
        }
        if log_components:
            with torch.no_grad():
                flag_v = cond_val[:, -1] > 0.5
                entry["val_components"] = {
                    "reconstruction": float(recon_v.item()),
                    "divergence": float(divergence_term(mu_v, logvar_v).item()),
                    "safeguard_reconstruction_pass":
                        float(safeguard.term(x_hat_v, safeguard_bounds).item()),
                    "safeguard_generation_pass":
                        float(safeguard.term(x_gen_v, safeguard_bounds).item()),
                    "target_consistency": float(target_v.item()),
                    "ensemble_spread": float(spread_penalty_term(ensemble, x_gen_v).item()),
                    "avian_prior": float(avian_prior_term(
                        x_gen_v, flag_v, reference_signature, region_extent).item()),
                }
        history.append(entry)

        if selection_metric < best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == arch.epochs - 1:
            final_state = {k: v.clone() for k, v in model.state_dict().items()}
            final_metric = selection_metric
            final_mu, final_logvar = mu_v, logvar_v

    per_dim_div = per_dimension_divergence(final_mu, final_logvar)
    n_live = int(np.sum(per_dim_div > liveness_threshold))

    return TrainingResult(
        model=model, best_state=best_state, best_epoch=best_epoch,
        best_selection_metric=best_metric, final_state=final_state,
        final_selection_metric=final_metric, history=history,
        per_dimension_divergence=per_dim_div, n_live_dimensions=n_live,
        liveness_threshold=liveness_threshold,
    )


def load_surrogate_ensemble(path) -> SurrogateEnsemble:
    """Reloads the frozen B13 ensemble from surrogate_ensemble.pt. Members
    come back frozen (requires_grad_(False)) exactly as they were saved, so
    a reloaded ensemble is the same object B13 committed and is never
    retrained between arms or between sweep points."""
    blob = torch.load(path, weights_only=False)
    arch = SurrogateArchitecture(**blob["architecture"])
    members: List[MLP] = []
    for state in blob["members"]:
        in_dim = state["net.0.weight"].shape[1]
        member = MLP(in_dim, arch.hidden_width, arch.depth, 1).to(DTYPE)
        member.load_state_dict(state)
        member.eval()
        for p in member.parameters():
            p.requires_grad_(False)
        members.append(member)
    return SurrogateEnsemble(members=members, architecture=arch)


@dataclass
class BuildArtifacts:
    """Everything a training run reads, loaded once from the artifacts the
    earlier steps committed.

    Stated need for this function existing at all: from B16 onward the
    build trains many models (one per sweep point, plus the committed model
    and the B18 zero-weight control), and every one of them must read the
    same split, the same standardisation, the same frozen ensemble, the
    same reference signature and the same region extent. A second loader
    written in a second driver is how two of those quietly stop being the
    same. One loader, read by every driver.
    """
    x_all: torch.Tensor                     # (n_rows, 20) standardised geometry
    cond_all: torch.Tensor                  # (n_rows, 22) conditioning (B10)
    train_idx: np.ndarray
    val_idx: np.ndarray
    family: np.ndarray
    ensemble: SurrogateEnsemble
    safeguard: "TorchSafeguard"
    safeguard_bounds: SafeguardBounds
    reference_signature: torch.Tensor       # (20,) standardised avian signature
    region_extent: float
    std_stats: geometry.StandardizationStats
    label_norm: dataset_mod.LabelNormalization
    order: int


# ---------------------------------------------------------------------------
# Standing load diagnostics, D01 to D05. 
#
# the committed specification specifies six diagnostics that run on every execution and
# stay, distinct from the falsification checks that run once and are deleted.
# the committed specification retires the verification suite as a file and says a
# surviving check becomes a standing gate rather than a test module. Both are
# satisfied by putting them here, in the one loader every driver reads, which is
# also the "at load" point the build plan names for D03 and D04 by name. D06 lives in
# solver.run_polar, beside the timer it checks.
#
# These RAISE. They compute nothing that any caller reads and they change no
# value, so a run with them present is bit identical to a run without them. That
# was verified during the build rather than assumed, over the whole of this
# function's return value and over a real solver call.
#
# Every threshold is recorded here and in the reproducibility appendix.
# ---------------------------------------------------------------------------

STANDING_LOAD_DIAGNOSTICS = {
    # D01. The moments of the standardised geometry on the row set the artifact
    # was derived from. 1e-10 is the tolerance B08's own check used.
    "d01_moment_tolerance": 1e-10,
    "d01_round_trip_tolerance": 1e-10,
    # D02. Stored artifacts against a fresh derivation from the current dataset.
    # Exact equality holds on the environment of record and was measured at
    # the build at a worst deviation of exactly 0.0. A tolerance rather than
    # exact equality is used so that a library upgrade moving a last bit cannot
    # make the loader refuse to load, while a genuinely stale artifact, which
    # differs by a row set rather than by an ulp, is still caught by orders of
    # magnitude.
    "d02_relative_tolerance": 1e-12,
    "d02_absolute_tolerance": 1e-15,
    # D04 compares copies rather than computations, so it is exact.
    # D05. The spread across ensemble members must not be degenerate. If
    # reseeding ever failed the members would be identical and the spread
    # penalty would be a constant zero while every prediction stayed ordinary.
    "d05_max_zero_spread_fraction": 0.01,
}


class StandingDiagnosticError(RuntimeError):
    """Raised when a standing load diagnostic fails. It is not caught anywhere
    in this build. A failure here means an artifact every reported number rests
    on is not what it claims to be, and continuing would produce numbers that
    look ordinary and measure something else."""


def _standing_load_diagnostics(x_std, raw, labels, family, train_idx, val_idx,
                                cond_array, std_stats, label_norm,
                                std_npz, norm_npz, avian_npz, ensemble) -> None:
    """D01 to D05, run on every load. Reads only, raises on failure."""
    t = STANDING_LOAD_DIAGNOSTICS

    def fail(which, detail):
        raise StandingDiagnosticError(f"{which}: {detail}")

    # --- D01. Standardisation and normalisation round trip -----------------
    on_rows = x_std[train_idx]
    worst_mean = float(np.abs(on_rows.mean(axis=0)).max())
    if worst_mean > t["d01_moment_tolerance"]:
        fail("D01", f"standardised column mean departs from zero by {worst_mean:.3e} "
                    f"on the derivation row set, tolerance {t['d01_moment_tolerance']:g}")
    worst_spread = float(np.abs(on_rows.std(axis=0, ddof=0) - 1.0).max())
    if worst_spread > t["d01_moment_tolerance"]:
        fail("D01", f"standardised column spread departs from one by {worst_spread:.3e}, "
                    f"tolerance {t['d01_moment_tolerance']:g}")
    round_trip = dataset_mod.denormalize_label(
        dataset_mod.normalize_label(labels, label_norm), label_norm)
    worst_trip = float(np.abs(round_trip - labels).max())
    if worst_trip > t["d01_round_trip_tolerance"]:
        fail("D01", f"label normalise then denormalise departs by {worst_trip:.3e}, "
                    f"tolerance {t['d01_round_trip_tolerance']:g}")

    # --- D02. Artifact freshness -------------------------------------------
    rtol, atol = t["d02_relative_tolerance"], t["d02_absolute_tolerance"]
    fresh = geometry.derive_standardization_stats(raw[train_idx], "freshness check")
    if not np.allclose(fresh.mean, std_npz["mean"], rtol=rtol, atol=atol):
        fail("D02", "the stored standardisation mean is not what the current dataset "
                    "and split produce; standardization.npz is stale")
    if not np.allclose(fresh.std, std_npz["std"], rtol=rtol, atol=atol):
        fail("D02", "the stored standardisation spread is not what the current dataset "
                    "and split produce; standardization.npz is stale")
    if not (np.isclose(labels[train_idx].min(), float(norm_npz["label_min"]),
                       rtol=rtol, atol=atol)
            and np.isclose(labels[train_idx].max(), float(norm_npz["label_max"]),
                           rtol=rtol, atol=atol)):
        fail("D02", "the stored normalisation bounds are not the current training "
                    "split's own label range; normalization.npz is stale")
    # The two artifacts derived FROM the standardisation. A change in
    # standardization.npz leaves both stale while their own files look fine.
    fresh_sig = geometry.standardize(avian_npz["raw_signature"], std_stats)
    if not np.allclose(fresh_sig, avian_npz["standardized_signature"],
                       rtol=rtol, atol=atol):
        fail("D02", "the stored standardised avian signature is not the stored raw "
                    "signature under the current standardisation; avian_signature.npz "
                    "is stale against standardization.npz")
    distance = np.linalg.norm(x_std - fresh_sig, axis=1)
    # The avian family is identified rather than named, by the row nearest the
    # reference, which B09 established is the avian seed's own row at distance
    # exactly zero. The count comparison below would fail if that were wrong.
    is_avian = family == family[int(np.argmin(distance))]
    fresh_extent = float(np.percentile(distance[is_avian], float(avian_npz["percentile"])))
    if not np.isclose(fresh_extent, float(avian_npz["extent"]), rtol=rtol, atol=atol):
        fail("D02", f"the stored region extent {float(avian_npz['extent'])!r} is not the "
                    f"current population's own percentile {fresh_extent!r}; "
                    f"avian_signature.npz is stale")
    inside = distance <= float(avian_npz["extent"])
    counts = (int(np.sum(inside & is_avian)), int(np.sum(inside & ~is_avian)))
    stored_counts = (int(avian_npz["n_avian_inside"]), int(avian_npz["n_non_avian_inside"]))
    if counts != stored_counts:
        fail("D02", f"region separation counts recompute to {counts} against the stored "
                    f"{stored_counts}")

    # --- D03. Split integrity ----------------------------------------------
    if len(np.intersect1d(train_idx, val_idx)) != 0:
        fail("D03", f"the training and validation index sets overlap on "
                    f"{len(np.intersect1d(train_idx, val_idx))} rows")
    union = np.sort(np.concatenate([train_idx, val_idx]))
    if not np.array_equal(union, np.arange(len(labels))):
        fail("D03", f"the split does not cover every row exactly once: {len(union)} "
                    f"indices over {len(labels)} rows, "
                    f"{len(union) - len(np.unique(union))} duplicated")
    absent = [f for f in np.unique(family) if not np.any(family[val_idx] == f)]
    if absent:
        fail("D03", f"stratification collapsed: {absent} absent from validation")

    # --- D04. Conditioning assembly integrity ------------------------------
    signature = avian_npz["standardized_signature"]
    n_sig = len(signature)
    flag = cond_array[:, -1] > 0.5
    block = cond_array[:, 1:1 + n_sig]
    if not np.array_equal(block[flag], np.broadcast_to(signature, block[flag].shape)):
        fail("D04", "a flag-set row does not carry the reference signature exactly; "
                    "the flag may be read with inverted sense, which would make every "
                    "reported quantity measure the opposite of what it names")
    if not np.array_equal(block[~flag], np.zeros_like(block[~flag])):
        fail("D04", "a flag-clear row does not carry the null block exactly; see D04's "
                    "inverted-flag note above")
    if not np.array_equal(cond_array[:, 0], dataset_mod.normalize_label(labels, label_norm)):
        fail("D04", "conditioning column zero is not the normalised label")

    # --- D05. Ensemble spread non-degeneracy --------------------------------
    with torch.no_grad():
        member_predictions = torch.stack(
            [m(torch.tensor(x_std[val_idx], dtype=DTYPE)).squeeze(-1)
             for m in ensemble.members])
    spread = member_predictions.std(dim=0, unbiased=True).numpy()
    zero_fraction = float((spread == 0.0).mean())
    if zero_fraction > t["d05_max_zero_spread_fraction"]:
        fail("D05", f"{zero_fraction:.4f} of validation rows have exactly zero spread "
                    f"across ensemble members, above {t['d05_max_zero_spread_fraction']:g}. "
                    f"Reseeding has failed and the spread penalty is a constant zero")


def load_build_artifacts(root=".", n_grid_points: int = 200, edge_margin: float = 0.005,
                          margin_fraction: float = 0.20) -> BuildArtifacts:
    import os

    def p(name):
        return os.path.join(root, name)

    ds = np.load(p("dataset.npz"), allow_pickle=True)
    pop = np.load(p("population.npz"), allow_pickle=True)
    cond_npz = np.load(p("conditioning.npz"), allow_pickle=True)
    split_npz = np.load(p("split.npz"), allow_pickle=True)
    std_npz = np.load(p("standardization.npz"), allow_pickle=True)
    norm_npz = np.load(p("normalization.npz"), allow_pickle=True)
    avian_npz = np.load(p("avian_signature.npz"), allow_pickle=True)

    row_index = ds["row_index"]
    order = int(pop["cst_order"])
    raw = np.concatenate(
        [pop["upper_coefficients"][row_index], pop["lower_coefficients"][row_index]], axis=1
    )

    std_stats = geometry.StandardizationStats(
        mean=std_npz["mean"], std=std_npz["std"],
        row_set_description=str(std_npz["row_set_description"]),
    )
    label_norm = dataset_mod.LabelNormalization(
        label_min=float(norm_npz["label_min"]), label_max=float(norm_npz["label_max"]),
        row_set_description=str(norm_npz["row_set_description"]),
    )

    x_std = geometry.standardize(raw, std_stats)
    train_idx = split_npz["train_idx"]
    val_idx = split_npz["val_idx"]

    bounds = derive_safeguard_bounds(
        pop["upper_coefficients"][row_index][train_idx],
        pop["lower_coefficients"][row_index][train_idx],
        order=order, n_grid_points=n_grid_points, edge_margin=edge_margin,
        margin_fraction=margin_fraction,
    )

    artifacts = BuildArtifacts(
        x_all=torch.tensor(x_std, dtype=DTYPE),
        cond_all=torch.tensor(cond_npz["array"], dtype=DTYPE),
        train_idx=train_idx, val_idx=val_idx, family=ds["family"],
        ensemble=load_surrogate_ensemble(p("surrogate_ensemble.pt")),
        safeguard=TorchSafeguard(order, n_grid_points, edge_margin, std_stats),
        safeguard_bounds=bounds,
        reference_signature=torch.tensor(avian_npz["standardized_signature"], dtype=DTYPE),
        region_extent=float(avian_npz["extent"]),
        std_stats=std_stats, label_norm=label_norm, order=order,
    )

    # D01 to D05, on every load. They read and raise, and change nothing above.
    _standing_load_diagnostics(
        x_std=x_std, raw=raw, labels=ds["label"], family=ds["family"],
        train_idx=train_idx, val_idx=val_idx, cond_array=cond_npz["array"],
        std_stats=std_stats, label_norm=label_norm,
        std_npz=std_npz, norm_npz=norm_npz, avian_npz=avian_npz,
        ensemble=artifacts.ensemble,
    )
    return artifacts


def selection_metric_on(model: CVAE, x_val: torch.Tensor, cond_val: torch.Tensor,
                         z_fresh_val: torch.Tensor, ensemble: SurrogateEnsemble) -> float:
    """Recomputes the unweighted selection metric for a given model state
    and a given fixed set of validation codes. Used by B15's falsification
    check to verify a reloaded checkpoint reproduces its recorded metric."""
    model.eval()
    with torch.no_grad():
        mu_v, _ = model.encoder(x_val)
        x_hat_v = model.decoder(mu_v, cond_val)
        recon_v = reconstruction_term(x_hat_v, x_val)
        x_gen_v = model.decoder(z_fresh_val, cond_val)
        target_v = target_consistency_term(ensemble, x_gen_v, cond_val[:, 0])
        return float(recon_v.item() + target_v.item())
