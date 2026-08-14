"""Paired generation, the evaluation path, and the two evaluation gates.

Holds the pairing construction, the distance and diversity measures the
mechanism gate reads, the solver-facing evaluation path, and gates zero and
one.

The pairing rule is the point of the design. One latent code is drawn per
target and sample index and is passed to both arms. The arms differ in the
conditioning block and the flag and in nothing else, so any difference between
them is the flag's doing and not a different draw.

Paired generation lives here rather than in model.py because three steps need
the identical construction and must not each grow their own: the weight
sensitivity sweep, the mechanism gate, and the paired evaluation run.

Gate zero solves one known airfoil, seeds/e387.dat, through the same path with
its coordinates read raw and not round-tripped through the fit, and requires a
converged status on all 9 requested angles. Gate one pushes 25 dataset rows of
already known label back through the same decode and solver path and requires
a relative difference at or below 0.01 on every one of them.

This file calls the solver, through solver.py and only through solver.py.

Called by run_b16_weight_sweep.py, run_b18_prior_mechanism_gate.py,
run_b19_evaluation_gates.py, run_b21_paired_yield.py,
run_b23_paired_evaluation.py, run_b25_metrics.py and run_figures.py.

Public API
    target_grid, requested_target_grid, build_conditioning
    paired_generation, decode_both_arms, PairedGeneration
    mean_distance_to_reference, direction_consistency,
        arm_effect_against_noise, within_target_spread, generative_diversity
    solver_settings, committed_timeout, plausibility_bounds,
        standardised_to_coefficients, label_from_polar
    evaluate_coefficients, admitted, ShapeRecord
    gate_zero, GateZeroResult
    gate_one, select_gate_one_rows, GateOneResult, GateOneRow
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

import geometry
import params
import solver as solver_mod

DTYPE = torch.float64


def target_grid(n_targets: int = None) -> np.ndarray:
    """The DIVERSITY grid, spanning the full normalised label range [0, 1].

    This is the internal grid of params.DIVERSITY_DEFINITION, which B16's
    sweep and B18's gate generate on. It is NOT the requested target band.
    It stays at [0, 1] and does not read the band, so B16's sweep table and
    B18's gate remain exactly reproducible now that the band is committed.
    Changing this function would silently move a diversity figure that four
    committed weights were selected against.
    """
    if n_targets is None:
        n_targets = params.DIVERSITY_DEFINITION["n_targets"]
    return np.linspace(0.0, 1.0, n_targets)


def requested_target_grid() -> np.ndarray:
    """The REQUESTED TARGET grid, from the committed band (the committed specification).

    This is what B21's pilot and B23's run generate at. Read from the
    parameter record rather than restated, and evenly spaced with both
    endpoints inclusive, per the band's own committed spacing.
    """
    band = params.PARAMS["requested_target_band"].value
    if isinstance(band, params.Pending):
        raise ValueError("requested_target_band is PENDING; nothing may generate at it")
    return np.linspace(band["normalised_low"], band["normalised_high"],
                       band["n_targets"])


def build_conditioning(targets: torch.Tensor, signature: torch.Tensor,
                        flag_set: bool) -> torch.Tensor:
    """Assembles conditioning rows in exactly B10's committed layout: column
    0 the normalised target, columns 1..20 the standardised avian signature
    where the flag is set and zero where it is clear, column 21 the flag
    itself at 1.0 set and 0.0 clear. Built here rather than reused from
    dataset.assemble_conditioning because that function assembles rows for
    the DATASET, one per labelled shape, and this assembles rows for
    REQUESTED targets that have no dataset row. The layout is the same and
    is asserted against the dataset artifact by B16's own driver."""
    n = targets.shape[0]
    n_sig = signature.shape[0]
    cond = torch.zeros((n, 1 + n_sig + 1), dtype=DTYPE)
    cond[:, 0] = targets
    if flag_set:
        cond[:, 1:1 + n_sig] = signature
        cond[:, 1 + n_sig] = 1.0
    return cond


@dataclass
class PairedGeneration:
    """x_set and x_clear are (n_targets, n_samples, geometry_dim), in
    STANDARDISED coefficient space, which is the space the decoder emits in
    and the space the region extent is defined in. Element [t, s] of each
    was decoded from the SAME latent code z[t, s]."""
    targets: np.ndarray
    x_set: torch.Tensor
    x_clear: torch.Tensor
    z: torch.Tensor


def decode_both_arms(cvae, z: torch.Tensor, flat_targets, signature: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The pairing rule itself. One latent code per row, decoded twice at the
    same target, once with the flag set and once with it clear.

    Stated need for it being a function rather than four lines inside
    paired_generation. B25's condition-blind baseline (M10) shuffles the
    target column, so its rows no longer share a target in groups and the
    grid-shaped API below cannot express them. Extracting the pairing keeps
    one implementation of it rather than letting B25 grow a second.
    """
    t = torch.as_tensor(np.asarray(flat_targets, dtype=float), dtype=DTYPE)
    cond_set = build_conditioning(t, signature, flag_set=True)
    cond_clear = build_conditioning(t, signature, flag_set=False)
    cvae.eval()
    with torch.no_grad():
        return cvae.decoder(z, cond_set), cvae.decoder(z, cond_clear)


def paired_generation(cvae, signature: torch.Tensor, latent_dim: int,
                       generation_seed: int, n_targets: int = None,
                       n_samples: int = None, targets: np.ndarray = None
                       ) -> PairedGeneration:
    """Pass `targets` explicitly to generate at the requested target band.
    Left unset, the diversity grid is used, which is what B16 and B18 call
    with and must keep calling with."""
    if n_samples is None:
        n_samples = params.DIVERSITY_DEFINITION["n_samples_per_target"]

    if targets is not None:
        grid = np.asarray(targets, dtype=float)
        n_targets = len(grid)
    else:
        if n_targets is None:
            n_targets = params.DIVERSITY_DEFINITION["n_targets"]
        grid = target_grid(n_targets)
    gen = torch.Generator().manual_seed(generation_seed)
    z = torch.randn((n_targets * n_samples, latent_dim), dtype=DTYPE, generator=gen)

    x_set, x_clear = decode_both_arms(cvae, z, np.repeat(grid, n_samples), signature)

    shape = (n_targets, n_samples, x_set.shape[1])
    return PairedGeneration(
        targets=grid,
        x_set=x_set.reshape(shape),
        x_clear=x_clear.reshape(shape),
        z=z.reshape((n_targets, n_samples, latent_dim)),
    )


def mean_distance_to_reference(x: torch.Tensor, signature: torch.Tensor) -> float:
    """M12 for one arm. Mean Euclidean distance in standardised coefficient
    space from every generated shape to the fixed avian reference."""
    return float(torch.linalg.norm(x - signature, dim=-1).mean())


def within_target_spread(x: torch.Tensor) -> torch.Tensor:
    """Mean pairwise Euclidean distance among the samples at each target.
    Returns one value per target. This single quantity is both the
    within-target statistic of DIVERSITY_DEFINITION and the denominator of
    M14, because 'the displacement produced by redrawing the latent code at
    the same target' is exactly the distance between two samples that share
    a target and differ only in their code.

    Public rather than private because B25 reports the per-target
    values as well as their mean, because F06 plots the statistic at each
    requested target and the across-range mean is drawn on top of them.
    """
    n_targets, n_samples, _ = x.shape
    if n_samples < 2:
        raise ValueError("mean pairwise distance needs at least 2 samples per target")
    d = torch.cdist(x, x)  # (n_targets, n_samples, n_samples)
    iu = torch.triu_indices(n_samples, n_samples, offset=1)
    return d[:, iu[0], iu[1]].mean(dim=1)


def generative_diversity(x: torch.Tensor) -> float:
    """M11, per params.DIVERSITY_DEFINITION. One definition, averaged across
    the conditioning range rather than taken at a single target."""
    return float(within_target_spread(x).mean())


def direction_consistency(pg: PairedGeneration, signature: torch.Tensor) -> float:
    """M13. The fraction of pairs in which the prior-on shape is STRICTLY
    closer to the avian reference than its prior-off counterpart. Strict, so
    an exact tie counts against the prior rather than for it."""
    d_set = torch.linalg.norm(pg.x_set - signature, dim=-1)
    d_clear = torch.linalg.norm(pg.x_clear - signature, dim=-1)
    return float((d_set < d_clear).double().mean())


def arm_effect_against_noise(pg: PairedGeneration) -> Tuple[float, float, float]:
    """M14. Returns (ratio, arm_displacement, redraw_displacement).

    Numerator: the mean displacement between the two arms' shapes at matched
    target and sample index, which share a latent code.

    Denominator: the mean displacement produced by redrawing the code within
    one arm at the same target. Implementation note, fixed before the gate
    ran and recorded rather than left to inference: the threshold text says
    'within a single arm', and the two arms may differ in spread, so the
    denominator is the MEAN of the two arms' within-target mean pairwise
    distances. Taking the prior-on arm alone would shrink the denominator
    exactly when the prior collapses generation, which would make the test
    easier in the one case it most needs to be hard. The mean is the neutral
    choice between the two.
    """
    arm = float(torch.linalg.norm(pg.x_set - pg.x_clear, dim=-1).mean())
    redraw = float(0.5 * (within_target_spread(pg.x_set).mean()
                          + within_target_spread(pg.x_clear).mean()))
    if redraw == 0.0:
        raise ValueError(
            "within-arm redraw displacement is exactly zero; the ratio is undefined. "
            "This is a total generative collapse and is reported, not divided through."
        )
    return arm / redraw, arm, redraw


# ---------------------------------------------------------------------------
# B19. The evaluation path.
#
# One path from a coefficient vector to a record. B19's gate one, B21's pilot
# and B23's full run all call it, so a generated shape and a stored dataset
# row travel the same code and a divergence between them cannot hide in a
# second implementation. That is the whole point of gate one, and it would be
# defeated by writing the gate its own decode-and-solve.
# ---------------------------------------------------------------------------

XFOIL_BINARY = Path("XFOIL6.99") / "xfoil.exe"
STAGE_DIR = Path("_stage")
SCRATCH_DIR = Path("_scratch")


def solver_settings() -> solver_mod.SolverSettings:
    """The committed B04 operating point, read from the parameter
    record rather than restated. A driver that hand-builds SolverSettings is
    a second copy of the operating point and is how two runs quietly stop
    being the same run."""
    value = params.PARAMS["solver_operating_point_settings"].value
    if isinstance(value, params.Pending):
        raise ValueError("solver_operating_point_settings is PENDING; B04 has not run")
    return solver_mod.SolverSettings(**value)


def committed_timeout() -> float:
    """The committed B05 per-call timeout, read from the record."""
    value = params.PARAMS["per_call_timeout"].value
    if isinstance(value, params.Pending):
        raise ValueError("per_call_timeout is PENDING; B05 has not run")
    return float(value)


def plausibility_bounds() -> geometry.PlausibilityBounds:
    """The committed B03 bounds, read from the record."""
    value = params.PARAMS["pre_solver_filter_thresholds"].value
    if isinstance(value, params.Pending):
        raise ValueError("pre_solver_filter_thresholds is PENDING; B03 has not run")
    return geometry.PlausibilityBounds(
        thickness_upper=value["thickness_upper"],
        thickness_lower=value["thickness_lower"],
        camber_second_difference_bound=value["camber_second_difference_bound"],
        margin_fraction=value["margin_fraction"],
        n_grid_points=value["n_grid_points"],
        edge_margin=value["edge_margin"],
        derivation="read from params.PARAMS['pre_solver_filter_thresholds']",
    )


def label_from_polar(polar: Optional[np.ndarray]) -> Optional[float]:
    """The efficiency label: the maximum of lift over drag across the
    converged points of one sweep, in the fixed cruise-regime operating point
    B04 committed. Returns None when the sweep produced no usable point, so a
    missing label is a None and never a sentinel number that could be
    mistaken for a measurement.

    Only converged rows reach here: solver._parse_polar writes one row per
    angle XFOIL actually accumulated, and an angle that did not converge
    produces no row at all.
    """
    if polar is None or len(polar) == 0:
        return None
    return float(np.max(polar[:, 1] / polar[:, 2]))


def standardised_to_coefficients(x_std, std_stats: geometry.StandardizationStats
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Undo B08's standardisation and split the 20-column vector back into
    the upper and lower CST coefficient blocks, in the concatenation order
    B08 committed: upper first, then lower."""
    if isinstance(x_std, torch.Tensor):
        x_std = x_std.detach().cpu().numpy()
    raw = geometry.destandardize(np.atleast_2d(np.asarray(x_std, dtype=float)), std_stats)
    half = raw.shape[1] // 2
    return raw[:, :half], raw[:, half:]


@dataclass
class ShapeRecord:
    """One shape's full passage through the evaluation path, with every
    reason kept. the committed specification requires the flow to be reason-annotated, which means
    the reason has to survive at the record and not only in a running count.

    n_converged and n_usable are separate on purpose. n_converged is what the
    solver reported. n_usable is what survives the committed specification's post-convergence
    physical plausibility filter. The two differ exactly when that filter
    binds, and reporting its affected rate, which the committed specification requires, needs both.
    Admission and the label are both computed on n_usable.
    """
    name: str
    plausible: bool
    plausibility_reason: str
    status: str                       # a SolveStatus value, or "plausibility_rejected"
    reason: Optional[str]
    n_converged: int
    n_usable: int
    n_requested: int
    label: Optional[float]
    polar: Optional[np.ndarray]       # the usable points only
    implausible_reasons: List[str]
    elapsed_seconds: float


PLAUSIBILITY_REJECTED = "plausibility_rejected"


def evaluate_coefficients(upper_coefficients: np.ndarray, lower_coefficients: np.ndarray,
                           name: str, bounds: geometry.PlausibilityBounds,
                           settings: solver_mod.SolverSettings, timeout_seconds: float,
                           n_points_per_surface: int,
                           xfoil_binary=XFOIL_BINARY, stage_dir=STAGE_DIR,
                           scratch_dir=SCRATCH_DIR) -> ShapeRecord:
    """Decode, filter, solve, label. The single evaluation path.

    The plausibility filter runs first and the solver is skipped when it
    rejects, matching B07's own order, so a shape that could never have
    entered the dataset cannot enter the evaluation set either.
    """
    import analysis

    decoded = geometry.decode_airfoil(upper_coefficients, lower_coefficients,
                                       n_points_per_surface=n_points_per_surface)
    verdict = geometry.plausibility_filter(decoded.upper, decoded.lower, bounds)
    if not verdict.accepted:
        return ShapeRecord(
            name=name, plausible=False, plausibility_reason=verdict.reason,
            status=PLAUSIBILITY_REJECTED,
            reason=f"rejected by the B03 plausibility filter: {verdict.reason}",
            n_converged=0, n_usable=0, n_requested=len(settings.alphas()),
            label=None, polar=None, implausible_reasons=[], elapsed_seconds=0.0,
        )

    result = solver_mod.run_polar_on_coords(
        decoded.x, decoded.y, name, settings, timeout_seconds,
        xfoil_binary=xfoil_binary, stage_dir=stage_dir, scratch_dir=scratch_dir,
    )

    # the committed specification. Drop any purportedly converged point that is not physically
    # meaningful, before the label's maximum can select it.
    pp = analysis.physical_plausibility(result.polar)
    usable = None if result.polar is None else result.polar[pp.keep]

    return ShapeRecord(
        name=name, plausible=True, plausibility_reason=verdict.reason,
        status=result.status.value, reason=result.reason,
        n_converged=result.n_converged, n_usable=0 if usable is None else len(usable),
        n_requested=result.n_requested,
        label=label_from_polar(usable), polar=usable,
        implausible_reasons=pp.reasons, elapsed_seconds=result.elapsed_seconds,
    )


def admitted(record: ShapeRecord, min_converged_points: int) -> bool:
    """B24's admission rule, applied at the record. A record is admitted only
    if it has a label AND cleared the committed minimum converged point
    count. Both conditions are read explicitly and neither defaults.

    min_converged_points is a required argument with no default on purpose.
    It is committed at B20, and a default here would let a caller admit
    records against a number nothing chose.

    The comparison itself lives in analysis.is_admitted, because the target
    file structure makes analysis.py the sole source of the admission rule.
    This is the ShapeRecord-shaped caller of it, not a second copy.

    The count passed is n_usable, not n_converged. A point the the committed specification filter
    dropped is not a converged point the admission rule may count, since the
    label was not computed from it either.
    """
    import analysis
    return analysis.is_admitted(record.label, record.n_usable, min_converged_points)


# ---------------------------------------------------------------------------
# B19. Gate zero and gate one. Thresholds come from
# params.B19_CONSISTENCY_GATE, fixed before either gate ran.
# ---------------------------------------------------------------------------

@dataclass
class GateZeroResult:
    passed: bool
    airfoil: str
    status: str
    n_converged: int
    n_requested: int
    elapsed_seconds: float
    reason: Optional[str]


def gate_zero(known_airfoil_path, settings: solver_mod.SolverSettings,
               timeout_seconds: float, xfoil_binary=XFOIL_BINARY,
               stage_dir=STAGE_DIR) -> GateZeroResult:
    """Solve one known airfoil and require a converged result, not merely a
    responding binary.

    The airfoil's raw digitised coordinates are solved directly, with no CST
    round trip, so gate zero shares no failure mode with gate one. A broken
    fit or decode would fail gate one and leave gate zero untouched, which is
    what makes the two gates separable evidence rather than one gate run
    twice.
    """
    known_airfoil_path = Path(known_airfoil_path)
    result = solver_mod.run_polar(known_airfoil_path, settings, timeout_seconds,
                                   xfoil_binary=xfoil_binary, stage_dir=stage_dir)
    passed = (result.status is solver_mod.SolveStatus.CONVERGED
              and result.n_converged == result.n_requested)
    return GateZeroResult(
        passed=passed, airfoil=known_airfoil_path.name, status=result.status.value,
        n_converged=result.n_converged, n_requested=result.n_requested,
        elapsed_seconds=result.elapsed_seconds, reason=result.reason,
    )


def select_gate_one_rows(family: np.ndarray, labels: np.ndarray,
                          n_per_family: int) -> np.ndarray:
    """The gate one row set, per B19_CONSISTENCY_GATE['gate_one']
    ['row_selection']. Within each family, the rows at evenly spaced ranks of
    that family's own sorted label, from its minimum to its maximum
    inclusive. Deterministic and RNG-free, so the row set is reconstructible
    from dataset.npz alone."""
    chosen: List[int] = []
    for fam in sorted(set(family.tolist())):
        positions = np.flatnonzero(family == fam)
        ordered = positions[np.argsort(labels[positions], kind="stable")]
        ranks = np.unique(np.round(
            np.linspace(0, len(ordered) - 1, n_per_family)).astype(int))
        chosen.extend(int(ordered[r]) for r in ranks)
    return np.array(sorted(chosen), dtype=int)


@dataclass
class GateOneRow:
    row: int
    family: str
    stored_label: float
    recomputed_label: Optional[float]
    relative_difference: Optional[float]
    within_tolerance: bool
    status: str
    n_converged: int
    elapsed_seconds: float


@dataclass
class GateOneResult:
    passed: bool
    tolerance_relative: float
    n_rows: int
    rows: List[GateOneRow] = field(default_factory=list)
    failing_rows: List[int] = field(default_factory=list)
    max_relative_difference: float = float("nan")


def gate_one(row_indices: np.ndarray, upper_coefficients: np.ndarray,
              lower_coefficients: np.ndarray, stored_labels: np.ndarray,
              family: np.ndarray, bounds: geometry.PlausibilityBounds,
              settings: solver_mod.SolverSettings, timeout_seconds: float,
              n_points_per_surface: int, tolerance_relative: float,
              xfoil_binary=XFOIL_BINARY, stage_dir=STAGE_DIR,
              scratch_dir=SCRATCH_DIR) -> GateOneResult:
    """Decode each named row from its stored coefficients, solve it through
    the same path a generated shape will take, and compare the recomputed
    label against the stored one.

    Every array is passed in rather than loaded here, so the falsification
    check can hand this function a scratch copy with one label deliberately
    altered and watch the gate fail on that row.

    A row that fails to produce a label at all counts as a failing row. It is
    not a small deviation and it is not silently skipped; a stored label that
    the same path can no longer reproduce is exactly the inconsistency the
    gate exists to find.
    """
    rows: List[GateOneRow] = []
    failing: List[int] = []
    worst = 0.0

    for i in row_indices:
        i = int(i)
        record = evaluate_coefficients(
            upper_coefficients[i], lower_coefficients[i], name=f"gate1_row{i}",
            bounds=bounds, settings=settings, timeout_seconds=timeout_seconds,
            n_points_per_surface=n_points_per_surface,
            xfoil_binary=xfoil_binary, stage_dir=stage_dir, scratch_dir=scratch_dir,
        )
        stored = float(stored_labels[i])
        if record.label is None:
            rel, ok = None, False
        else:
            rel = abs(record.label - stored) / abs(stored)
            ok = rel <= tolerance_relative
            worst = max(worst, rel)
        if not ok:
            failing.append(i)
        rows.append(GateOneRow(
            row=i, family=str(family[i]), stored_label=stored,
            recomputed_label=record.label, relative_difference=rel,
            within_tolerance=ok, status=record.status,
            n_converged=record.n_converged, elapsed_seconds=record.elapsed_seconds,
        ))

    return GateOneResult(
        passed=(len(failing) == 0), tolerance_relative=tolerance_relative,
        n_rows=len(rows), rows=rows, failing_rows=failing,
        max_relative_difference=worst,
    )
