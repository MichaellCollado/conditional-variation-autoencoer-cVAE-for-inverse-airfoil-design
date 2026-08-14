"""Shape representation. CST fit, decode, closure, plausibility filter.

The one place a shape is defined. The dataset, the prior and the evaluation
path all read from here, so the three cannot drift into three different
notions of what a shape is.

Holds the class shape transformation at an order the caller passes in and
which is never hard-coded here; the fit from digitised coordinates to
coefficients and the decode back; structural trailing-edge closure; the
pre-solver plausibility filter and the derivation of its bounds from the
seeds' own measured range; and the standardisation statistics, derived over
the training split alone.

The class-function exponents are N1 = 0.5 and N2 = 1.0, the standard
round-nose pointed-tail airfoil form. Trailing edge thickness is zero, forced
by setting both surfaces' trailing-edge ordinates to their shared midpoint,
because XFOIL otherwise collapses an unclosed trailing edge and changes the
shape the solver evaluates from the shape the model emitted.

fit_surface reports its residual maximum and RMS in chord units and applies
no pass mark.

Called by dataset.py, prior.py, model.py, evaluate.py and run_figures.py.

Public API
    read_seed_dat, load_seed_library, SeedGeometry, Surface
    class_function, bernstein_basis, cosine_spaced_grid
    fit_surface, SurfaceFit
    decode_surface, decode_airfoil, DecodedAirfoil
    measure_thickness_and_camber, plausibility_filter,
        derive_plausibility_bounds, PlausibilityBounds, PlausibilityVerdict
    derive_standardization_stats, standardize, destandardize,
        StandardizationStats
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Reading and splitting seed coordinate files
# ---------------------------------------------------------------------------

@dataclass
class Surface:
    """One airfoil surface, x increasing from the leading edge (0) to the
    trailing edge (1). Every function below assumes this ordering."""
    x: np.ndarray
    y: np.ndarray


@dataclass
class SeedGeometry:
    name: str
    path: str
    upper: Surface
    lower: Surface


def read_seed_dat(path) -> SeedGeometry:
    """Read one Selig-format coordinate file.

    Selig format, confirmed by inspecting the five attached seed files
    directly rather than assumed: a name line, then a single closed loop
    starting at the trailing edge (x=1), running over the upper surface to
    the leading edge (the point of minimum x), then over the lower surface
    back to the trailing edge (x=1).

    Splits the loop at the point of minimum x, and reverses the upper
    surface segment so both surfaces come back with x increasing from the
    leading edge to the trailing edge.

    Raises ValueError on anything that does not look like this format.
    Does not guess at malformed input.
    """
    path = Path(path)
    lines = path.read_text(errors="strict").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty file")
    name = lines[0].strip()

    xs, ys = [], []
    for line_number, raw in enumerate(lines[1:], start=2):
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(
                f"{path}:{line_number}: expected two numbers, got {stripped!r}"
            )
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: could not parse {stripped!r}") from exc
        xs.append(x)
        ys.append(y)

    if len(xs) < 6:
        raise ValueError(f"{path}: only {len(xs)} coordinate points, too few to be an airfoil")

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    le_index = int(np.argmin(xs))
    if le_index == 0 or le_index == len(xs) - 1:
        raise ValueError(
            f"{path}: minimum-x point is the first or last point (index {le_index}). "
            f"This does not look like a single-loop Selig-format file starting at "
            f"the trailing edge. Not proceeding on an assumed layout; inspect the file."
        )

    upper_x = xs[: le_index + 1][::-1]
    upper_y = ys[: le_index + 1][::-1]
    lower_x = xs[le_index:]
    lower_y = ys[le_index:]

    if not np.all(np.diff(upper_x) >= -1e-9):
        raise ValueError(f"{path}: upper surface x is not monotonic after split and reverse")
    if not np.all(np.diff(lower_x) >= -1e-9):
        raise ValueError(f"{path}: lower surface x is not monotonic after split")

    return SeedGeometry(
        name=name,
        path=str(path),
        upper=Surface(x=upper_x, y=upper_y),
        lower=Surface(x=lower_x, y=lower_y),
    )


def load_seed_library(directory) -> List[SeedGeometry]:
    """Read every .dat file in `directory` as a seed. Fails loudly on the
    first unreadable file rather than silently skipping it."""
    directory = Path(directory)
    paths = sorted(directory.glob("*.dat"))
    if not paths:
        raise ValueError(f"{directory}: no .dat files found")
    return [read_seed_dat(p) for p in paths]


# ---------------------------------------------------------------------------
# B02. CST class and shape functions, fit, and decode
# ---------------------------------------------------------------------------

# Standard CST class-function exponents for a round-nose, pointed-tail
# airfoil (Kulfan 2008). Fixed by the locked choice of CST as the shape
# parameterisation; not an open value in the committed specification.
CLASS_N1 = 0.5
CLASS_N2 = 1.0


def class_function(x: np.ndarray, n1: float = CLASS_N1, n2: float = CLASS_N2) -> np.ndarray:
    """C(x) = x^n1 * (1-x)^n2. Zero at x=0 and at x=1 for the standard
    airfoil exponents."""
    x = np.asarray(x, dtype=float)
    return np.power(x, n1) * np.power(1.0 - x, n2)


def bernstein_basis(x: np.ndarray, order: int) -> np.ndarray:
    """Bernstein polynomial basis of the given order, evaluated at x.
    Column i is C(order, i) * x^i * (1-x)^(order-i).

    `order` is the CST order (author's choice, unset in params.py).
    Never defaulted here; every caller must supply it.
    """
    if order < 0:
        raise ValueError("order must be >= 0")
    x = np.asarray(x, dtype=float)
    basis = np.empty((x.shape[0], order + 1), dtype=float)
    for i in range(order + 1):
        basis[:, i] = comb(order, i) * np.power(x, i) * np.power(1.0 - x, order - i)
    return basis


@dataclass
class SurfaceFit:
    """te_offset is measured directly from the raw data (this surface's own
    ordinate at x=1) and used only to compute the fit residual against the
    real seed. It is not part of the coefficient vector, and decode() does
    not use it: decode always closes the trailing edge to zero, per the committed specification."""
    order: int
    coefficients: np.ndarray
    te_offset: float
    residual_max: float
    residual_rms: float


def fit_surface(surface: Surface, order: int) -> SurfaceFit:
    """Fit a CST shape-function coefficient vector to one surface.

    Logic:
      1. subtract the linear trailing-edge term from the ordinates
      2. build the class-and-shape basis at the surface's own abscissae
      3. solve least squares for the coefficients
      4. return coefficients and the residual maximum and RMS, in chord
         units, no pass mark applied.
    """
    x, y = surface.x, surface.y
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("surface.x and surface.y must be 1-D arrays of equal length")

    te_offset = float(y[-1])
    target = y - x * te_offset

    c = class_function(x)
    basis = bernstein_basis(x, order)
    design = basis * c[:, None]

    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)

    fitted = design @ coefficients
    residual = target - fitted

    return SurfaceFit(
        order=order,
        coefficients=coefficients,
        te_offset=te_offset,
        residual_max=float(np.max(np.abs(residual))),
        residual_rms=float(np.sqrt(np.mean(residual ** 2))),
    )


def cosine_spaced_grid(n_points: int) -> np.ndarray:
    """Cosine-spaced abscissae on [0, 1], clustered near both ends."""
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    beta = np.linspace(0.0, np.pi, n_points)
    return 0.5 * (1.0 - np.cos(beta))


def decode_surface(coefficients: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Evaluate one surface's ordinate at x_grid from its CST coefficients.
    Uses no linear trailing-edge term: the decoded representation is always
    zero-thickness at the trailing edge, per the committed specification."""
    order = coefficients.shape[0] - 1
    c = class_function(x_grid)
    basis = bernstein_basis(x_grid, order)
    return c * (basis @ coefficients)


@dataclass
class DecodedAirfoil:
    """A full loop: trailing edge, over the upper surface to the leading
    edge, over the lower surface, back to the trailing edge. Matches the
    seed file convention."""
    x: np.ndarray
    y: np.ndarray
    upper: Surface
    lower: Surface


def decode_airfoil(upper_coefficients: np.ndarray, lower_coefficients: np.ndarray,
                    n_points_per_surface: int) -> DecodedAirfoil:
    """Decode a full airfoil from CST coefficients for the upper and lower
    surfaces.

    Logic:
      1. build a cosine-spaced abscissa grid
      2. evaluate class times shape (the linear trailing-edge term is zero,
         per the committed specification) per surface
      3. set both trailing-edge ordinates to their shared midpoint
      4. return the ordered surface loop
    """
    x_grid = cosine_spaced_grid(n_points_per_surface)

    upper_y = decode_surface(upper_coefficients, x_grid)
    lower_y = decode_surface(lower_coefficients, x_grid)

    te_mid = 0.5 * (upper_y[-1] + lower_y[-1])
    upper_y = upper_y.copy()
    lower_y = lower_y.copy()
    upper_y[-1] = te_mid
    lower_y[-1] = te_mid

    loop_x = np.concatenate([x_grid[::-1], x_grid[1:]])
    loop_y = np.concatenate([upper_y[::-1], lower_y[1:]])

    return DecodedAirfoil(
        x=loop_x,
        y=loop_y,
        upper=Surface(x=x_grid, y=upper_y),
        lower=Surface(x=x_grid, y=lower_y),
    )


# ---------------------------------------------------------------------------
# B03. Plausibility filter
# ---------------------------------------------------------------------------

@dataclass
class PlausibilityBounds:
    """The three stated bounds, derived from the seed population's own
    measured range with a stated margin. Build with
    derive_plausibility_bounds(); do not hand-construct with invented
    numbers."""
    thickness_upper: float
    thickness_lower: float
    camber_second_difference_bound: float
    margin_fraction: float
    n_grid_points: int
    edge_margin: float
    derivation: str


@dataclass
class PlausibilityVerdict:
    accepted: bool
    reason: str  # "ok", "crossing", "thickness_high", "thickness_low", "camber_kink"


# How far from each end of the chord the interior grid stays away, on both
# the leading- and trailing-edge side. Excludes x=0 and x=1 themselves,
# where thickness is zero by construction and a crossing or curvature test
# is meaningless, and stays clear of a second problem found while running
# this on the actual seed files: raw digitized coordinate data does not
# reach exactly x=0. The closest recorded leading-edge point across the
# five seeds is E387 at x=0.00044 chord. A grid point any closer to 0 than
# that falls outside every seed's own recorded range, so np.interp clamps
# both surfaces to their nearest recorded point instead of interpolating,
# which can make upper and lower read as exactly equal there and be
# misread as a crossing. EDGE_MARGIN is set an order of magnitude above
# that closest recorded point, so this does not happen on any seed, and it
# is applied identically whether the grid is measuring a seed or a
# decoded, generated shape.
EDGE_MARGIN = 0.005


def _interior_grid(n_grid_points: int, edge_margin: float = EDGE_MARGIN) -> np.ndarray:
    """The common interior grid the filter interpolates both surfaces onto,
    restricted to [edge_margin, 1 - edge_margin]."""
    full = cosine_spaced_grid(n_grid_points)
    return full[(full >= edge_margin) & (full <= 1.0 - edge_margin)]


def measure_thickness_and_camber(upper: Surface, lower: Surface, grid: np.ndarray):
    """Interpolate both surfaces onto `grid`; return thickness(x),
    camber(x), and the interpolated ordinates themselves."""
    upper_y = np.interp(grid, upper.x, upper.y)
    lower_y = np.interp(grid, lower.x, lower.y)
    thickness = upper_y - lower_y
    camber = 0.5 * (upper_y + lower_y)
    return thickness, camber, upper_y, lower_y


def plausibility_filter(upper: Surface, lower: Surface, bounds: PlausibilityBounds
                         ) -> PlausibilityVerdict:
    """B03's filter.

    Logic:
      1. interpolate both surfaces onto a common interior grid
      2. reject if the surfaces cross anywhere
      3. reject if maximum thickness is above its upper bound or below its
         lower bound
      4. reject if the second difference of camber exceeds its bound
         anywhere
      5. return the verdict and the reason
    """
    grid = _interior_grid(bounds.n_grid_points, bounds.edge_margin)
    thickness, camber, upper_y, lower_y = measure_thickness_and_camber(upper, lower, grid)

    if np.any(upper_y <= lower_y):
        return PlausibilityVerdict(accepted=False, reason="crossing")

    max_thickness = float(np.max(thickness))
    if max_thickness > bounds.thickness_upper:
        return PlausibilityVerdict(accepted=False, reason="thickness_high")
    if max_thickness < bounds.thickness_lower:
        return PlausibilityVerdict(accepted=False, reason="thickness_low")

    second_difference = np.diff(camber, n=2)
    if np.any(np.abs(second_difference) > bounds.camber_second_difference_bound):
        return PlausibilityVerdict(accepted=False, reason="camber_kink")

    return PlausibilityVerdict(accepted=True, reason="ok")


def derive_plausibility_bounds(seeds: Sequence[SeedGeometry], margin_fraction: float,
                                n_grid_points: int = 200,
                                edge_margin: float = EDGE_MARGIN) -> PlausibilityBounds:
    """Derive the three stated bounds from the seed population's own
    measured range, with a stated margin.

    Measures max thickness and the peak absolute second difference of
    camber directly from each seed's raw coordinates, not through a CST
    fit/decode round trip; this needs no committed CST order to run, which
    is the straightforward option, per the build plan's governing principle.

    margin_fraction widens the thickness band on both sides, and the
    camber bound on the high side, so every seed clears its own bound with
    room to spare, rather than sitting exactly on the line it is tested
    against. The margin is stated here explicitly; it is not inherited
    from the prior build, which recorded no supported figure for it.
    """
    if not seeds:
        raise ValueError("no seeds supplied")
    if margin_fraction < 0:
        raise ValueError("margin_fraction must be >= 0")

    grid = _interior_grid(n_grid_points, edge_margin)

    max_thicknesses = []
    max_abs_second_diffs = []
    per_seed = []
    for seed in seeds:
        thickness, camber, _, _ = measure_thickness_and_camber(seed.upper, seed.lower, grid)
        mt = float(np.max(thickness))
        md = float(np.max(np.abs(np.diff(camber, n=2))))
        max_thicknesses.append(mt)
        max_abs_second_diffs.append(md)
        per_seed.append((seed.name, mt, md))

    thickness_min = min(max_thicknesses)
    thickness_max = max(max_thicknesses)
    camber_max = max(max_abs_second_diffs)

    thickness_upper = thickness_max * (1.0 + margin_fraction)
    thickness_lower = thickness_min * (1.0 - margin_fraction)
    camber_bound = camber_max * (1.0 + margin_fraction)

    per_seed_text = "; ".join(
        f"{n}: max_thickness={mt:.6f}, max_abs_camber_2nd_diff={md:.6f}"
        for n, mt, md in per_seed
    )

    derivation = (
        f"Measured directly from raw seed coordinates (no CST round trip), on "
        f"a {n_grid_points}-point cosine grid restricted to "
        f"[{edge_margin}, {1.0 - edge_margin}] chord, across {len(seeds)} seeds. "
        f"Per-seed figures: {per_seed_text}. "
        f"thickness_upper = max(max_thickness) * (1 + margin) "
        f"= {thickness_max:.6f} * (1 + {margin_fraction}) = {thickness_upper:.6f}. "
        f"thickness_lower = min(max_thickness) * (1 - margin) "
        f"= {thickness_min:.6f} * (1 - {margin_fraction}) = {thickness_lower:.6f}. "
        f"camber_second_difference_bound = max(max_abs_camber_2nd_diff) * (1 + margin) "
        f"= {camber_max:.6f} * (1 + {margin_fraction}) = {camber_bound:.6f}. "
        f"margin_fraction = {margin_fraction}, chosen at B03 as a round, stated "
        f"figure, wide enough that every seed clears its own bound with room "
        f"for a perturbed sample, and not inherited from the prior build. "
        f"The camber bound has no corresponding lower-side widening: a smoother "
        f"(lower second-difference) camber than any seed is not implausible, "
        f"so only the crossing and thickness checks have a lower side."
    )

    return PlausibilityBounds(
        thickness_upper=thickness_upper,
        thickness_lower=thickness_lower,
        camber_second_difference_bound=camber_bound,
        margin_fraction=margin_fraction,
        n_grid_points=n_grid_points,
        edge_margin=edge_margin,
        derivation=derivation,
    )


# ---------------------------------------------------------------------------
# B08. Standardisation of the CST coefficient representation. 
# Held here, not in dataset.py, per the target file structure table: the
# standardised representation is shared by the dataset build (B08), the
# avian prior's distance computation (prior.py, B09), and the evaluation
# path when it re-derives a generated shape's distance to the reference.
# One implementation in the one file the representation itself already
# lives in prevents those three from drifting into different notions of
# "standardised".
#
# This module does not choose the row set the statistics are computed
# over -- that choice (author's choice, disclosed) is made by B08's
# driver, which selects the coefficient rows before calling
# derive_standardization_stats. Nothing here defaults or infers a row set.
# ---------------------------------------------------------------------------

@dataclass
class StandardizationStats:
    """Per-column population mean and standard deviation (ddof=0), so that
    applying this back to the exact row set it was derived from reproduces
    mean 0 and std 1 on every column, to the precision B08's falsification
    check requires (1e-10)."""
    mean: np.ndarray
    std: np.ndarray
    row_set_description: str


def derive_standardization_stats(coefficients: np.ndarray, row_set_description: str
                                  ) -> StandardizationStats:
    """coefficients: (n_rows, n_columns), already restricted by the caller
    to whatever row set was chosen. Raises rather than silently guarding
    against a zero-spread column: dividing by a zero spread is undefined,
    and a real zero-spread column would be a finding to investigate, not a
    case to paper over with an epsilon."""
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be a 2-D (rows, columns) array")
    mean = coefficients.mean(axis=0)
    std = coefficients.std(axis=0, ddof=0)
    if np.any(std == 0.0):
        raise ValueError(
            "at least one column has zero spread over the given row set; "
            "z-scoring it would divide by zero. Not silently handled."
        )
    return StandardizationStats(mean=mean, std=std, row_set_description=row_set_description)


def standardize(coefficients: np.ndarray, stats: StandardizationStats) -> np.ndarray:
    """coefficients may be a single row (1-D, n_columns) or a batch (2-D,
    n_rows x n_columns); numpy broadcasting against the 1-D mean/std
    handles both without a separate code path."""
    return (coefficients - stats.mean) / stats.std


def destandardize(standardized: np.ndarray, stats: StandardizationStats) -> np.ndarray:
    return standardized * stats.std + stats.mean
