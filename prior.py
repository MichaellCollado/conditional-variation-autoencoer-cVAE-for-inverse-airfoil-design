"""The avian reference and the region extent the prior measures against.

Holds the fit of the seagull section to a CST reference, the distance from a
population to that reference in standardised coefficient space, and the
derivation of the region extent as a percentile of the avian family's own
distances, applied to that subset alone so it describes the family's spread
rather than the dataset's.

This module produced avian_signature.npz, which the rest of the build reads.
It is not imported at run time by any module or driver, and it is kept because
it is the code that derived a committed value the article reports.

The prior term itself is not here. It is model.avian_prior_term, a squared
hinge on distance above the region extent, and it is the only term in the
objective that reads the flag.

The reference's spanwise resolution is mixed, because the source splits it that
way: the shape comes from span-averaged coefficients and the magnitude from
envelope equations at a single station. This build performs no spanwise
averaging of its own, since seagull.dat is already one two-dimensional cross
section. The two statements are about different things.

The region extent is stored at full precision and unrounded, so the value the
model reads is the value the article discloses.

Public API
    AvianReference, fit_avian_reference
    distances_to_reference
    RegionExtent, derive_region_extent
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import geometry


@dataclass
class AvianReference:
    seed_name: str
    order: int
    raw_signature: np.ndarray           # (2*(order+1),), upper then lower, unstandardised
    standardized_signature: np.ndarray  # same shape, standardised with the geometry artifact


def fit_avian_reference(seagull_path, order: int,
                         std_stats: geometry.StandardizationStats) -> AvianReference:
    """Fits the seagull seed directly from its own coordinate file,
    at the committed CST order, and standardises the resulting coefficient
    vector with the SAME artifact the dataset's own geometry uses (B08) --
    not a separately fit or separately scaled version of it, which is what
    'standardise the signature with the same artifact the geometry uses'
    (B09's own logic text) requires.

    No spanwise averaging is performed HERE: seagull.dat is read as the
    single two-dimensional cross-section it already is. The section's own
    resolution is mixed and the averaging is upstream of this build; see
    the module docstring and the params.py slot
    spanwise_resolution_avian_section.
    """
    seed = geometry.read_seed_dat(seagull_path)
    fit_u = geometry.fit_surface(seed.upper, order)
    fit_l = geometry.fit_surface(seed.lower, order)
    raw_signature = np.concatenate([fit_u.coefficients, fit_l.coefficients])
    standardized_signature = geometry.standardize(raw_signature, std_stats)
    return AvianReference(
        seed_name=seed.name, order=order,
        raw_signature=raw_signature,
        standardized_signature=standardized_signature,
    )


def distances_to_reference(standardized_coefficients: np.ndarray,
                            reference: AvianReference) -> np.ndarray:
    """Euclidean distance, in standardised coefficient space, from every
    row of standardized_coefficients (n_rows, n_columns) to the reference
    signature. Works identically for the whole population or a subset;
    the caller decides which rows to pass in."""
    diff = standardized_coefficients - reference.standardized_signature
    return np.linalg.norm(diff, axis=1)


@dataclass
class RegionExtent:
    percentile: float
    extent: float                    # full float64 precision, unrounded
    n_avian_inside: int
    n_avian_total: int
    n_non_avian_inside: int
    n_non_avian_total: int


def derive_region_extent(distances: np.ndarray, is_avian: np.ndarray,
                          percentile: float) -> RegionExtent:
    """Region extent is the stated percentile of the AVIAN SUBSET's OWN
    distances to the reference -- not of the whole population's distances
    (B09's own logic text: 'the stated percentile of the avian subset's
    distances'). Stored unrounded: np.percentile's float64 return value is
    kept exactly as computed, with no formatting or rounding applied
    anywhere between here and params.py."""
    avian_distances = distances[is_avian]
    extent = np.percentile(avian_distances, percentile)
    inside = distances <= extent
    return RegionExtent(
        percentile=percentile,
        extent=float(extent),
        n_avian_inside=int(np.sum(inside & is_avian)),
        n_avian_total=int(np.sum(is_avian)),
        n_non_avian_inside=int(np.sum(inside & ~is_avian)),
        n_non_avian_total=int(np.sum(~is_avian)),
    )
