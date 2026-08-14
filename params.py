"""The parameter record. Every committed value of the study, in one place.

This is the parameter file PAPER.md names under Code availability. It holds
each committed parameter with the value the pipeline reads, the rule by which
the value was set, and the derivation that produced it. Where the article
rounds a value, it is rounded from the full-precision figure here.

Nothing in this file is a result. It records what was fixed, not what was
measured.

WHAT IS IN HERE.

  PARAMS
      33 parameter slots. Each is a ParamSlot carrying a value, the treatment
      it was set under, its derivation, and the build step that committed it.
  DIVERSITY_DEFINITION
      The one diversity definition used everywhere the word appears.
  WEIGHT_SELECTION_RULE
      The rule that selected the prior term weight, fixed before the sweep it
      was applied to existed.
  B18_GATE_THRESHOLDS
      The mechanism gate, all three groups.
  B19_CONSISTENCY_GATE
      Gate zero and gate one, including the airfoil gate zero solves.
  B20_TRUNCATION_TOLERANCE
      The tolerance that sets the minimum converged point count.
  B24_ANALYSIS
      The primary outcome, its estimator and its resampling.
  B25_METRICS
      The reported metrics and their definitions.
  KNOWN_DISCREPANCIES
      14 recorded disagreements, kept rather than resolved silently.

WHAT IS NOT IN HERE, AND WHERE IT IS INSTEAD.

  Selected checkpoint, epoch 146 and six live latent dimensions.
      Measured at training. Recorded in b18_gate.json.
  Geometric safeguard bounds, the thickness floor and the curvature ceiling.
      Derived at load by model.derive_safeguard_bounds from the training
      split. The derivation and the realised values are in the notes below.
  Realised flag-clear fraction, 0.095723.
      Measured at the flag draw. Recorded in flag_assignment.npz.
  Secondaries width agreement tolerance, 0.40.
      A module constant, C_AGREE_TOLERANCE, in
      check_b24_studentised_secondaries.py.

The B step numbers in the notes below are this repository own build
vocabulary, and are how the drivers are named. PAPER.md carries no build step
numbering, so nothing here is cited to it. "The committed specification"
names the study target specification, the document that fixed these
requirements before the build ran; it is not part of the published
repository.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Any


class Treatment(Enum):
    STATED_VALUE = "stated_value"
    RANGE_WITH_RULE = "range_with_rule"
    AUTHORS_CHOICE = "authors_choice"


class Pending:
    """Sentinel for a value that has not been set. Never compare a Pending
    value to a number, and never let one reach a computation silently --
    code that needs a real value from a PENDING slot should raise, not
    substitute a default."""

    def __repr__(self):
        return "PENDING"

    def __bool__(self):
        # A pending slot is falsy on purpose, so `if slot.value:` reads
        # naturally as "is this set", but this must never be relied on to
        # silently skip a required check.
        return False


PENDING = Pending()


@dataclass
class ParamSlot:
    """One committed parameter. `value` is what the pipeline reads."""

    key: str
    treatment: Treatment
    note: str
    # Required, non-empty, when treatment is RANGE_WITH_RULE. Records the
    # selection rule text itself, not the number the rule produced.
    rule_text: Optional[str] = None
    # Required, non-empty, when treatment is AUTHORS_CHOICE. Records what is
    # disclosed about the choice.
    disclosure_obligation: Optional[str] = None
    # Which build step committed the value.
    set_at: str = "UNRESOLVED"
    value: Any = field(default_factory=lambda: PENDING)
    # What was measured, what rule or margin was applied, and the numbers
    # that resulted.
    derivation: Optional[str] = None

    def __post_init__(self):
        if self.treatment is Treatment.RANGE_WITH_RULE and not self.rule_text:
            raise ValueError(f"{self.key}: range-with-rule slot must carry rule_text")
        if self.treatment is Treatment.AUTHORS_CHOICE and not self.disclosure_obligation:
            raise ValueError(f"{self.key}: author's-choice slot must carry disclosure_obligation")


# ---------------------------------------------------------------------------
# Rules and thresholds fixed during the build, BEFORE the artifacts they will be
# applied to exist.
#
# Two things are written here ahead of time on purpose, and the order is the
# point of them:
#
#   WEIGHT_SELECTION_RULE is fixed before B16's sweep is run. B17 is the one
#   step in the build plan with no falsification check, because a selection
#   rule that is wrong reproduces itself exactly on recomputation. The only
#   mitigation available is procedural: the rule cannot be shaped by the
#   table it will be applied to if it is written down before that table
#   exists. That is what this constant is.
#
#   B18_GATE_THRESHOLDS is fixed before B18's gate is run, and before the
#   zero-weight control that must fail against it is trained. Tightening a
#   threshold until one specific failing model is rejected is calibration
#   against an outcome. Here the numbers are committed first and the control
#   is required to fail against numbers it had no part in setting.
#
# Neither constant carries any measured value. Both are read by the drivers
# rather than restated in them.
# ---------------------------------------------------------------------------

DIVERSITY_DEFINITION = {
    "n_targets": 11,
    "target_grid": "11 evenly spaced normalised targets, 0.0 to 1.0 inclusive",
    "n_samples_per_target": 20,
    "within_target_statistic": "mean pairwise Euclidean distance in standardised "
    "CST coefficient space across the samples at that target",
    "across_target_reduction": "arithmetic mean over the 11 targets",
    "text": (
        "ONE definition, stated in full and averaged across the conditioning "
        "range rather than taken at a single condition. What varies: the latent "
        "code, drawn independently per sample. What is held fixed: the requested "
        "target and the flag. At each of 11 evenly spaced normalised targets from 0.0 "
        "to 1.0 inclusive, 20 samples are generated, one per independently drawn "
        "latent code. The statistic across those 20 samples is their mean pairwise "
        "Euclidean distance in standardised CST coefficient space. The metric is the "
        "arithmetic mean of that statistic over the 11 targets. "
        "\n"
        "Where this is set, and why it is not set at B25: the build plan assigns the "
        "diversity definition to B25, but B16's own logic text requires the sweep to "
        "record 'the diversity statistic defined at B25', and B25's prerequisite "
        "(B23) is far downstream of B16. the committed specification forbid two definitions sharing "
        "the word. So the definition is fixed once, here, at the first step that "
        "consumes it, and B25 consumes this same definition unchanged rather than "
        "introducing a second one. Recorded as a departure from the build plan's own "
        "step assignment, not as a silent choice. "
        "\n"
        "The 11-target grid is the sweep's and the gate's own internal grid over the "
        "full normalised label range. It is NOT the requested target band (slot "
        "requested_target_band, still UNRESOLVED) and does not commit that band."
    ),
}


WEIGHT_SELECTION_RULE = {
    "fixed_at": "Before B16's sweep was run in this build. AMENDED once "
    "during the build, still before any sweep was run here. The "
    "amendment and its cause are recorded in full under "
    "'amendment_record' below rather than replaced silently, because a "
    "rule that changes without a visible reason is the same failure "
    "B17's procedural mitigation exists to prevent.",

    "weights_selected_by_sweep": ("target_weight", "spread_weight", "avian_weight"),
    "weights_set_as_stated_values": ("divergence_weight", "safeguard_weight"),

    "amendment_record": (
        "THE FIRST VERSION OF THIS RULE WAS DEFECTIVE. It swept five weights and "
        "selected each as the strongest effect admissible under a budget of 1.10 * R*, "
        "where R* was the smallest validation reconstruction on that weight's own "
        "ladder. Two faults, both derivable without any sweep table, and the second "
        "confirmed against a table before this build's own sweep was run. "
        "\n"
        "FAULT ONE, general. For any term with a non-zero gradient, R* is attained "
        "where that term's weight is zero, because switching a competing term off is "
        "what minimises reconstruction. Every non-zero weight then buys its effect "
        "with reconstruction, measured against a baseline that paid nothing for it. A "
        "fixed 10 percent budget against a term-off baseline can therefore admit "
        "nothing but zero. The rule was biased toward selecting zero for every weight "
        "it governed. For the avian weight this is not a tuning wart: a selected value "
        "of zero would make the committed model identical to B18's own zero-weight "
        "control, so the gate would compare a model against itself and group three "
        "would fail for both. The study's central open value would be zero by "
        "construction of the rule rather than by evidence. "
        "\n"
        "FAULT TWO, specific to the divergence weight. The first version used 'lower "
        "effect measure is better' uniformly, with the divergence term as the "
        "divergence weight's own measure. Low divergence is posterior collapse, which "
        "is a pathology and not the term doing its job, so the direction was wrong for "
        "that one weight and right for the others. Evidence, from a divergence sweep "
        "run on this pipeline under the first version of the rule: it selected weight "
        "0.0, the only admissible row, with validation KL of 106.4 nats per row and "
        "flag-set diversity of 0.927 against 3.08 at weight 0.01 and 4.26 at weight "
        "0.1. A divergence weight of zero is a conditional autoencoder, not a "
        "conditional VAE, and the model family is LOCKED. Every downstream quantity in "
        "this build is measured on generations from fresh N(0, I) codes, and at weight "
        "zero those codes decode off-distribution. "
        "\n"
        "The divergence sweep table that produced that finding is NOT a selection "
        "input under the amended rule and does not become one. The divergence weight "
        "is now set as a stated value on the locked model family, which needs no table "
        "at all. The corrected selection form below was checked against that same "
        "table only to confirm it does not carry fault one, and it does not: it "
        "returns 0.1 rather than 0.0 there. That check governs no committed value, "
        "since the divergence weight is no longer selected by any rule. "
        "\n"
        "The amendment was made before any run of B16 in this build. No sweep table "
        "for the target, spread or avian weights existed when it was written, and no "
        "evaluation path had been run."
    ),

    "text": (
        "SCOPE. B17's own text sets the prior weight 'and by the same procedure the "
        "target, spread and safeguard weights'. The RECONSTRUCTION weight is not open: "
        "B14's total objective gives it an implicit weight of 1 and it is the scale "
        "every other weight is expressed against. Of the five remaining weights, three "
        "are selected by sweep and two are set as stated values, for reasons that need "
        "no table and are recorded here before the sweep runs. "
        "\n"
        "DIVERGENCE WEIGHT, stated value 1.0. A divergence weight of 1.0 is the exact "
        "evidence lower bound, the beta = 1 case, which is the plain conditional VAE "
        "the LOCKED model family names. It is committed on that ground alone. Its "
        "ladder is still run and reported as sensitivity evidence under the committed specification, but the "
        "table is evidence and not a selector, and the committed value does not move "
        "with it. Disclosure that belongs with it: the reported ladder need not have "
        "its best trade at 1.0, and where it does not, that is stated rather than used "
        "to move the value. Recorded mismatch: the committed specification gives this row the "
        "treatment RANGE WITH RULE, and it is set here as a stated value, because the "
        "only rule available would have to be applied to a table and no table may "
        "override a locked element. See KNOWN_DISCREPANCIES. "
        "\n"
        "SAFEGUARD WEIGHT, stated value 1.0. The safeguard term is identically zero on "
        "this build, so its gradient is identically zero and no sweep can identify its "
        "weight: every candidate produces the same objective and the same trained "
        "model, bit for bit. This was measured before the sweep, not discovered in it. "
        "On the exact measurement path the term itself uses, 0 of 785 training rows "
        "violate the thickness floor and 0 of 785 violate the curvature ceiling, and "
        "the term evaluates to exactly 0.0 at the untrained model's initialisation. "
        "The decoder emits standardised coefficients and begins near the population "
        "mean, which is a plausible shape by construction, so it never enters the "
        "region the hinge penalises. The weight is set to 1.0 and REPORTED together "
        "with the measurement that the safeguard never binds, which is what the committed specification "
        "requires: a weight that changes the objective is reported, and a weight that "
        "provably does not is reported as that. The bounds are NOT retightened to make "
        "the term active. Choosing a threshold so that a term starts binding is "
        "calibration against an outcome, which is the specific failure B18 exists to "
        "avoid. the build plan gives this row the treatment STATED VALUE, so setting it as "
        "one resolves the treatment mismatch B14 and B17 had opened against that "
        "table, in that table's favour. "
        "\n"
        "SWEPT WEIGHTS. Target, spread and avian. One at a time. Each is swept across "
        "its own six-point ladder while every other weight is held at the reference "
        "vector: divergence 1.0, safeguard 0.0, target 0.0, spread 0.0, avian 0.0. The "
        "reference for divergence is its committed stated value. The reference for the "
        "safeguard is immaterial and provably so, since its term is identically zero; "
        "0.0 is used so the reference vector reads as the plain conditional VAE. Every "
        "auxiliary reference is 0.0, so each swept weight is a clean perturbation of "
        "that plain model rather than a perturbation of an arbitrary mixture. The "
        "known cost of a one-at-a-time design is stated rather than hidden: the three "
        "swept weights are each selected in the absence of the other two, the "
        "committed model runs all three together, and no interaction is measured. The "
        "avian weight is swept LAST, so that nothing about its ladder can be read back "
        "into the design of the two ladders before it. "
        "\n"
        "Every run uses the same training seed, the same split (B12), the same frozen "
        "surrogate ensemble (B13), the same architecture (B15) and the same unweighted "
        "checkpoint selection metric (B15). Only the one weight moves. "
        "\n"
        "LADDERS. The three swept terms differ from each other in raw magnitude by "
        "several orders, so a common nominal ladder would not be a common ladder in "
        "effect. Each ladder is scaled to its own term: for weight w on term T, "
        "ladder = [0, 0.01*u_T, 0.1*u_T, 1.0*u_T, 10.0*u_T, 100.0*u_T], where "
        "u_T = recon_0 / T_0, and recon_0 and T_0 are the reconstruction term and term "
        "T evaluated at the untrained model's initialisation over the whole training "
        "split in one batch, at the sweep's own training seed. u_T is the weight at "
        "which term T would contribute as much as reconstruction does at "
        "initialisation, so every ladder runs from negligible, through parity, to "
        "clearly too much. If any T_0 is exactly zero, u_T is undefined and that "
        "weight cannot be swept; it is set as a stated value with the measurement that "
        "made it unsweepable, which is what happened to the safeguard weight above. "
        "The divergence sensitivity ladder is nominal, 0, 0.01, 0.1, 1.0, 10.0, 100.0, "
        "since beta = 1 is a real anchor to read it against. "
        "\n"
        "SWEEP TABLE. One row per run. Every column the selection rule reads is in the "
        "table, so the rule is applied to the table alone. Columns, all measured at "
        "that run's OWN selected checkpoint on the validation split: validation "
        "reconstruction; mean distance to the avian reference for the flag-set arm; "
        "the same for the flag-clear arm; flag-set generative diversity per "
        "DIVERSITY_DEFINITION; the validation divergence, safeguard, "
        "target-consistency and ensemble-spread term values; the best epoch; the live "
        "latent dimension count; and the path of that run's own training history file. "
        "\n"
        "SELECTION (B17), applied to the sweep table alone, before any evaluation path "
        "has been run. For each swept weight, over its own ladder, three quantities "
        "are formed. Each is normalised to that ladder's own observed range, so all "
        "three are dimensionless and lie in [0, 1], and no tolerance constant is "
        "needed anywhere. Where a denominator is zero the quantity is 0 for every "
        "candidate, which is the correct reading of something the ladder does not "
        "move. "
        "\n"
        " recon_cost(c)     = (R(c) - min R) / (max R - min R) "
        " diversity_cost(c) = (max D - D(c)) / (max D - min D) "
        " effect_gain(c)    = (max E - E(c)) / (max E - min E) "
        "\n"
        "R is validation reconstruction, D is flag-set diversity, and E is that "
        "weight's own effect measure, lower being better, naming what the weight is "
        "FOR: avian -> mean distance to the avian reference on the flag-set arm; "
        "target -> the validation target-consistency term; spread -> the validation "
        "mean ensemble spread. "
        "\n"
        " score(c) = effect_gain(c) - recon_cost(c) - diversity_cost(c) "
        "\n"
        "The SELECTED value is the candidate with the highest score. Exact ties go to "
        "the smaller weight. "
        "\n"
        "Why this form. It charges a weight for both of its costs and credits it for "
        "what it buys, on one scale, and it is the trade the build plan's F08 says the "
        "sweep exists to make legible. It carries no invented tolerance, which matters "
        "because the literature scan records that no airfoil or aerodynamic source reports a weight "
        "selection procedure for a shape prior term at all, so there is no published "
        "figure to adopt and every constant would have been this study's own. It "
        "cannot degenerate to zero by construction of the anchor, because zero scores "
        "effect_gain = 0 and wins only when every non-zero candidate genuinely costs "
        "more than it buys. Diversity is charged separately from reconstruction "
        "because reconstruction is measured on the reconstruction pass and a model can "
        "reconstruct well while its generations collapse, so reconstruction alone does "
        "not cover collapse. "
        "\n"
        "Known weakness, stated rather than patched: min-max normalisation is set by "
        "the ladder's endpoints, so one extreme candidate compresses the rest. The "
        "ladders span four decades deliberately, so the top candidate is expected to "
        "be clearly too much and to sit at recon_cost near 1. That is what makes the "
        "trade visible; it also means the selected value is a choice among the "
        "ladder's own points and not a continuous optimum. "
        "\n"
        "WHAT B17 RECORDS. For each swept weight, the selected value, its ladder row, "
        "and the full score table with all three normalised components per candidate. "
        "For each stated-value weight, the value and the ground it rests on. And an "
        "explicit statement that no solver evaluation result had been seen at the "
        "point of selection."
    ),
}


B18_GATE_THRESHOLDS = {
    "fixed_at": "Before B18's gate was run and before the zero-weight "
    "control it must reject was trained.",

    # Groups one and two are evaluated on a FIXED batch of real validation-split
    # geometry, not on generated shapes. Stated here because it decides what the
    # tests can mean. On generated geometry, group two's 'positive with flags set'
    # test would fail precisely when the prior worked perfectly and pulled every
    # flagged generation inside the region, so a well-behaved gate would reject a
    # well-behaved model. On fixed real geometry the two tests measure the term's
    # gating and nothing else, which is what groups one and two are for. This is
    # the same construction B14's own falsification check used.
    "group_one_arm_blindness": {
        "measurement": "Each non-avian term (reconstruction, divergence, safeguard on "
        "the reconstruction pass, safeguard on the generation pass, "
        "target consistency, ensemble spread) evaluated twice on one "
        "fixed batch, with the conditioning array's flag column flipped "
        "between the two calls and every other input held identical.",
        "threshold": "Bit-identical across the flip. Tolerance exactly 0.0, not a "
        "small number.",
        "verdict_rule": "PASS if every non-avian term returns exactly equal values.",
    },
    "group_two_gate_responds": {
        "measurement": "The avian prior term on the same fixed batch of real "
        "validation geometry, once with every flag clear and once with "
        "every flag set.",
        "threshold_flags_clear": "Exactly 0.0. Tolerance exactly 0.0.",
        "threshold_flags_set": "Strictly greater than 0.0.",
        "verdict_rule": "PASS if both hold.",
    },

    # Group three is measured on paired generation: one latent code per (target,
    # sample), decoded twice from the same code, once with the flag set (prior-on)
    # and once with it clear (prior-off), over the DIVERSITY_DEFINITION target grid.
    # No solver is called. This is not the evaluation path (B19, B23).
    "group_three_effect": {
        "n_targets": 11,
        "n_samples_per_target": 20,
        "g3a_direction_consistency": {
            "measurement": "M13. The fraction of pairs in which the prior-on shape is "
            "strictly closer to the avian reference, in standardised "
            "coefficient space, than its prior-off counterpart.",
            "threshold": 0.60,
            "verdict_rule": "PASS if the fraction is >= 0.60.",
            "basis": "Chance is 0.50. 0.60 is a round margin above chance, chosen "
            "before any measurement. the build plan records M13 as having no "
            "specification entry and no published source, so no external "
            "threshold exists to adopt.",
        },
        "g3b_mean_distance_separation": {
            "measurement": "M12. The mean distance to the avian reference on the "
            "prior-off arm minus the same on the prior-on arm.",
            "threshold_as_fraction_of_region_extent": 0.10,
            "verdict_rule": "PASS if the difference is >= 0.10 * the committed region "
            "extent (slot inactive_region_extent). Stated as a "
            "fraction of the extent rather than as a bare distance so "
            "the threshold carries the scale the prior itself is "
            "defined on.",
            "basis": "A round tenth of the region radius, chosen before any "
            "measurement. M12 has no specification entry and no published "
            "source.",
        },
        "g3c_effect_against_sampling_noise": {
            "measurement": "M14. The mean displacement between the two arms' shapes at "
            "matched target and sample index, divided by the mean "
            "displacement produced by redrawing the latent code within "
            "a single arm at the same target.",
            "threshold": 0.25,
            "verdict_rule": "PASS if the ratio is >= 0.25.",
            "basis": "The arm difference must be at least a quarter of the model's own "
            "sampling variation to be read as an effect rather than as noise. "
            "A round figure, chosen before any measurement. M14 has no "
            "specification entry and no published source.",
        },
    },

    "control_requirement": (
        "A model trained at avian weight zero, with every other committed weight "
        "unchanged, is run through this same gate with these same thresholds. It is "
        "required to PASS group one and group two and to FAIL every test in group "
        "three. The thresholds are fixed above, before that control is trained. If the "
        "control passes any group three test, or if the committed model fails any "
        "test, that is reported as the finding. No threshold is moved."
    ),
    "disclosure": (
        "Every threshold above is the author's choice under a disclosure requirement. "
        "the build plan assigns them that treatment; the committed specification has no row for them "
        "(see KNOWN_DISCREPANCIES). None has a published basis: the build plan records M12, "
        "M13 and M14 as existing on the locked design's authority with no specification "
        "entry and no source of this shape found."
    ),
}


B19_CONSISTENCY_GATE = {
    "fixed_at": "Before either gate was run. B19's own text does not "
    "demand this ordering the way B18's does, but a tolerance chosen "
    "after the deviations are seen is the same defect B18 exists to "
    "prevent, so it is committed first here as well.",

    "gate_zero": {
        "measurement": "One known airfoil, solved through solver.run_polar at the "
        "committed B04 operating point and the committed B05 timeout. "
        "The airfoil is seeds/e387.dat, read as raw digitised "
        "coordinates and NOT round-tripped through the CST fit, so "
        "gate zero tests the solver and its environment alone and "
        "shares no failure mode with the CST path gate one tests.",
        "threshold": "Status must be CONVERGED, meaning every one of the 9 requested "
        "angles converged. A responding binary, a partial sweep, or a "
        "written-but-empty polar all FAIL.",
        "verdict_rule": "PASS only on status == converged and n_converged == 9.",
    },

    "gate_one": {
        "n_rows": 25,
        "row_selection": (
            "Five rows from each of the five families, 25 in total. Within each "
            "family the five are taken at the 0th, 25th, 50th, 75th and 100th "
            "percentile of that family's OWN stored label, by rank on the sorted "
            "label, with ties broken by row order. Deterministic, with no RNG draw, "
            "so the gate row set is reconstructible from dataset.npz alone. The plan "
            "asks only for rows 'spanning the families'; spanning each family's label "
            "range as well costs nothing and makes a label-dependent path divergence "
            "visible, which a family-only spread would not."
        ),
        "n_points_per_surface": 160,
        "n_points_note": (
            "The decode resolution the evaluation path uses, fixed here as a stated "
            "value. 160 matches dataset.one_draw's own resolution, which is what the "
            "B06 sampler measured plausibility on, so the evaluation path and the "
            "dataset construction path decode at the same resolution rather than at "
            "two. Every shape B19 and B23 solve is decoded at this resolution, "
            "generated and stored alike."
        ),
        "tolerance_relative": 0.01,
        "verdict_rule": (
            "PASS only if EVERY gate row satisfies "
            "|recomputed_label - stored_label| / stored_label <= 0.01. One row "
            "outside it fails the gate, and the failing rows are named."
        ),
        "basis": (
            "Author's choice under a disclosure requirement, and this study's own. "
            "the literature scan records a clean not-found on both halves: no published source "
            "re-runs labelled shapes through the same pipeline to confirm stored "
            "labels, and no published tolerance for such a check exists. There is "
            "nothing to adopt, so a figure is stated and disclosed as stated. "
            "\n"
            "Relative rather than absolute, because the stored labels span 50.499 to "
            "172.174 and a fixed L/D allowance would be a different requirement at "
            "the two ends of that range. "
            "\n"
            "Why 1 percent is wide enough not to fail on nothing: the recomputation "
            "is a deterministic re-solve of the same coordinates through the same "
            "binary, so the expected deviation is zero, and 1 percent leaves room "
            "for a decode-resolution difference between this path and the one B07 "
            "ran without swallowing anything larger. "
            "\n"
            "Why 1 percent is narrow enough to catch what the gate is for: the "
            "failures this gate exists to detect are path divergences, not noise. "
            "Upper and lower coefficients applied to the wrong surfaces, a different "
            "operating point, a lift or drag column read from the wrong position, or "
            "a stale stored label all move max(CL/CD) by tens of percent. None would "
            "hide inside 1 percent. "
            "\n"
            "Calibration for the reader, from values this build has already "
            "committed and not from this gate's own output: 1 percent of the label "
            "range (121.6752 L/D) is 1.217 L/D, against a committed surrogate "
            "ensemble held-out mean absolute error of 2.6538 L/D. A deviation inside "
            "this tolerance is smaller than an error the pipeline already carries "
            "elsewhere and reports."
        ),
    },

    "disclosure": (
        "Both thresholds above are the author's choice under a disclosure "
        "requirement. the build plan assigns the consistency gate tolerance that "
        "treatment; the committed specification has no row for it (see KNOWN_DISCREPANCIES), "
        "so it is recorded as a module-level constant here on the same precedent as "
        "B18_GATE_THRESHOLDS, the per-seed count, the flag-clear fraction and the "
        "safeguard bounds."
    ),
}


B20_TRUNCATION_TOLERANCE = {
    "fixed_at": "The build, BEFORE the truncation analysis was run and before any "
    "candidate point count's bias was computed. the committed specification and B20 both require "
    "this ordering by name: the tolerance is stated first and the count "
    "follows from it mechanically, rather than the count being chosen and "
    "the tolerance written to fit it.",

    "quantity": (
        "Truncation bias at candidate minimum converged point count k, over every "
        "FULLY converged sweep in the dataset. For one sweep, "
        "bias_k = (max(CL/CD) over the first k converged points) - (max(CL/CD) over "
        "the whole sweep), which is at most zero, since truncating a maximum can only "
        "lower it. Reported relative: rel_bias_k = bias_k / (max over the whole "
        "sweep). Relative rather than absolute because the labels span 50.499 to "
        "172.174 and a fixed L/D allowance would be a different requirement at the "
        "two ends."
    ),

    "mean_relative_bias": 0.010,
    "upper_percentile": 95.0,
    "upper_percentile_absolute_relative_bias": 0.020,

    "selection_rule": (
        "The committed minimum converged point count is the SMALLEST k for which "
        "BOTH of the following hold over the fully converged sweeps: the mean "
        "absolute relative bias is at most 0.010, and the 95th percentile of the "
        "absolute relative bias is at most 0.020. Applied mechanically to the bias "
        "table. If no k below the full sweep length satisfies both, the full sweep "
        "length is committed and that is reported as the finding."
    ),

    "basis": (
        "Author-stated and this study's own. the literature scan records a clean not-found: no "
        "published source states a minimum converged point count, or any equivalent "
        "rule, for accepting a maximum taken from a partially converged sweep, and "
        "the build plan retired the superseded build's threshold, so nothing is inherited. "
        "\n"
        "The two figures are round and are stated as round. 1 percent on the mean "
        "and 2 percent on the upper tail, so the typical sweep's admitted label is "
        "within 1 percent of the label the full sweep would have given and the "
        "unlucky one in twenty is within 2 percent. The tail is allowed twice the "
        "mean because a maximum's truncation bias is one-sided and skewed by "
        "construction, so a tail figure equal to the mean would be a stricter "
        "requirement than it reads as. "
        "\n"
        "Calibration for the reader, from values already committed and not from this "
        "analysis's own output: 1 percent of the label range (121.6752 L/D) is 1.217 "
        "L/D, against a committed surrogate ensemble held-out mean absolute error of "
        "2.6538 L/D. A truncation bias inside the mean tolerance is smaller than an "
        "error the pipeline already carries elsewhere and reports. This is context "
        "for reading the figure, not the derivation of it; no committed quantity "
        "determines these two numbers and neither is presented as following practice."
    ),
}


B24_ANALYSIS = {
    "fixed_at": "The build, at B24, which is the step the build plan assigns the trim "
    "fraction to and the step the build plan's own schedule assigns the "
    "generator derivation to. Recorded here as a module-level constant on "
    "the same precedent as B18_GATE_THRESHOLDS, B19_CONSISTENCY_GATE and "
    "B20_TRUNCATION_TOLERANCE, none of which is a the committed specification row "
    "either. Two things below were set at this step and everything else "
    "B24 uses was committed before B23 ran.",

    "trim_fraction": 0.10,
    "trim_fraction_basis": (
        "M06's trim fraction. the build plan says in its own words that the trim fraction "
        "is a stated value and assigns the computation to B24. No step before B24 sets "
        "it, no the committed specification row covers it, and the issued pre-registration "
        "declares trimmed means as a sensitivity check without naming a fraction. So "
        "it is stated here and disclosed as stated. "
        "\n"
        "0.10 per tail, 0.20 in total, symmetric in COUNT: floor(n * 0.10) values are "
        "removed from each end. A round conventional figure, this study's own, not "
        "adopted from any source and not presented as following practice. "
        "\n"
        "WHAT IT MAY AND MAY NOT DO. the committed specification forbids promoting a trimmed mean to the "
        "headline and the pre-registration summarised at Appendix A.1 declares it a sensitivity check and not an "
        "outcome. It is reported and it is not promoted. Its only job is to show how "
        "much of the primary's location depends on the tails. "
        "\n"
        "STATED HONESTLY, because it is the same failure class B22's check already "
        "carries on five slots: this value was set AFTER the pre-registration was "
        "issued, and the issued document does not name it. It could not have been set "
        "earlier without a step to set it at, and it is not an outcome, a primary, an "
        "admission rule or a reporting rule, so the pre-registration's prohibition on "
            "in-place amendment is not engaged by recording it. It is nevertheless a number in the "
        "analysis that the issued document does not carry, and it is listed as such "
        "rather than left to be discovered."
    ),

    "generator_substreams": {
        "rule": "dataset.rng_for(24, substream), which is the one base seed 20260806 "
        "plus 1000 times the build step number plus the substream index. Every "
        "generator B24 uses is derived from that single rule and no generator "
        "is constructed any other way. One substream per estimator, so "
        "re-running one interval does not perturb another's draw.",
        0: "primary outcome, wild cluster bootstrap-t, the committed interval",
        1: "primary outcome, unrefined percentile cluster bootstrap, reported alongside",
        2: "S1, the median paired difference",
        3: "S2, the paired win fraction",
        4: "S3, the arm difference in target tracking slope",
        5: "M06, the trimmed mean, sensitivity only",
        "100..119": "the 20 independent repetitions that measure the wild cluster "
        "bootstrap-t's endpoint Monte Carlo error (M03)",
        "200..219": "the 20 independent repetitions that measure the percentile "
        "cluster bootstrap's endpoint and point-side Monte Carlo error (M03)",
        "900, 901": "the two synthetic designs of B24's falsification check, which "
        "touched no evaluation record and was deleted after passing",
    },

    "secondary_estimator_validation": {
        "what_was_run": (
            "check_b24_studentised_secondaries.py, the build. Two synthetic designs "
            "at the real study's shape, 11 clusters by 10 pairs, 400 datasets per "
            "design, 399 resamples, nominal 95 percent. Design A carries a large "
            "between-cluster component and design B carries none. The estimand is the "
            "superpopulation value and both designs are symmetric about it, so the "
            "true median and the true slope difference are both exactly zero and "
            "coverage is the fraction of datasets whose interval contains zero. "
            "Two statistics were simulated, the median (S1) and the arm difference in "
            "tracking slope (S3). S2's win fraction was NOT separately simulated and "
            "that is stated as a limit of the check rather than left implied."
        ),
        "provenance_caveat": (
            "STATED BECAUSE IT BEARS ON HOW THE RESULT MAY BE USED. This check was "
            "written AFTER S3's bootstrap t quantiles were seen on the real data and "
            "looked unusual. Its data are synthetic and carry no evaluation outcome, "
            "but the decision to run it was triggered by an outcome. It is therefore "
            "used to CHARACTERISE and DISCLOSE the estimator and it is NOT used to "
            "reselect which interval any secondary rests on. Reselecting an estimator "
            "on evidence gathered because a result looked odd is the failure the "
            "pre-registration exists to prevent, and synthetic data does not make it "
            "safe."
        ),
        "verdict": "FAIL on one clause of six, for both statistics. Reported as a "
        "failure. No threshold was moved and no clause was deleted.",
        "median_design_a": {"coverage_refined": 0.9250, "coverage_unrefined": 0.9325,
            "coverage_pair": 0.4950,
            "width_refined_over_pair": 11.9755,
            "width_unrefined_over_pair": 3.0049},
        "median_design_b": {"coverage_refined": 0.9325, "coverage_unrefined": 0.9100,
            "coverage_pair": 0.9375,
            "width_refined_over_pair": 2.8180,
            "width_refined_over_unrefined_median_dataset": 2.263,
            "fraction_of_datasets_refined_is_narrower": 0.10},
        "slope_design_a": {"coverage_refined": 0.9400, "coverage_unrefined": 0.9000,
            "coverage_pair": 0.4025,
            "width_refined_over_pair": 5.7054,
            "width_unrefined_over_pair": 3.7263},
        "slope_design_b": {"coverage_refined": 0.9575, "coverage_unrefined": 0.9375,
            "coverage_pair": 0.9500,
            "width_refined_over_pair": 1.6007,
            "width_refined_over_unrefined_median_dataset": 1.386,
            "fraction_of_datasets_refined_is_narrower": 0.18},
        "reading": (
            "WHAT PASSED. The refined interval covers at or near nominal on both "
            "designs and for both statistics, 0.925 to 0.9575 against a nominal 0.95. "
            "It is materially wider than a pair-level interval where the clustering is "
            "real, which is the first clause of the build plan's own check text. Where the "
            "unrefined interval undercovers, which is the slope on design A at 0.90, "
            "the refinement fixes it, at 0.94. And the negative control has teeth: the "
            "pair bootstrap covers 0.4950 and 0.4025 on design A, so a coverage "
            "measurement that nothing fails is not what is being reported here. "
            "\n"
            "WHAT FAILED, and it is the build plan's own second clause. With NO "
            "between-cluster component the refined interval does not agree closely "
            "with the pair interval. It sits 2.82 times wider for the median and 1.60 "
            "times wider for the slope, against a tolerance of 0.40 stated in advance. "
            "This is not a heavy tail dragging a mean: the ratio of the refined width "
            "to the unrefined width at the MEDIAN dataset is 2.26 for the median and "
            "1.39 for the slope, so the typical interval is inflated and not just a "
            "few of them. The plan says in its own words that an estimator which is "
            "always wider is also wrong, and on this evidence this one is wider than "
            "it needs to be whenever there is nothing to be wide about. "
            "\n"
            "THE MECHANISM, measured rather than asserted. The studentising quantity "
            "is a delete-one-cluster jackknife standard error over 11 clusters. On a "
            "no-clustering design it is a noisy estimate, and when a resample's SE "
            "lands small the ratio it divides explodes, so the bootstrap t "
            "distribution acquires heavy tails and its outer quantiles run far beyond "
            "a normal reference. On one inspected design B dataset about 3 percent of "
            "replicates returned a jackknife SE below a quarter of the original SE. "
            "The effect is strongest for the median, which is the least smooth of the "
            "statistics involved. No source was consulted for this; it is what this "
            "build measured on its own estimator. "
            "\n"
            "WHAT IT CHANGES IN THIS PAPER: NOTHING, and that is checked rather than "
            "hoped. Every secondary and the sensitivity check span their null value "
            "under BOTH constructions on the real data, so no reported reading turns "
            "on the choice. The real data also sits in the clustered regime, where the "
            "cluster-robust standard error is 1.94 times the independent-pairs one, "
            "and that is design A rather than design B. The failing clause tests a "
            "regime this study's data does not occupy. That is a reason the failure "
            "does not propagate into a wrong number here. It is NOT a reason to call "
            "the clause satisfied, and it is not called satisfied. "
            "\n"
            "WHAT IS DISCLOSED IN THE PAPER. The refined intervals on the secondaries "
            "are conservative, measurably so, and their WIDTH should not be read as a "
            "tight statement of precision. Their coverage is sound. Both the refined "
            "and the unrefined interval are reported for every secondary so a reader "
            "can see the difference rather than take the narrower or the wider on "
            "trust."
        ),
    },

    "monte_carlo_repetitions": 20,
    "monte_carlo_basis": (
        "the committed specification and M03 require the Monte Carlo error of the interval ENDPOINTS reported "
        "separately from the Monte Carlo error of the point estimate. Both are measured "
        "here rather than asserted, by repeating the whole bootstrap on 20 independent "
        "streams from the offset rule above and taking the sample standard deviation "
        "across repetitions. 20 is this study's own figure, stated as such. "
        "\n"
        "The two quantities are genuinely different objects here and are labelled so "
        "they cannot be conflated. The PRIMARY POINT ESTIMATE, the mean paired "
        "difference, is a function of the data alone: it reads no resampling stream and "
        "its Monte Carlo error is exactly zero, which is stated rather than simulated. "
        "The point-side quantity that does carry resampling error is the percentile "
        "cluster bootstrap's own bootstrap mean, and that is what its figure reports. "
        "The endpoints of both estimators are tail order statistics and carry the "
        "larger error, which is the contrast the committed specification exists to make visible and the reason "
        "the resample count was chosen for an interval rather than for a standard error."
    ),
}


B25_METRICS = {
    "fixed_at": "The build, at B25. Recorded here as a module-level constant on the "
    "same precedent as B18_GATE_THRESHOLDS, B19_CONSISTENCY_GATE, "
    "B20_TRUNCATION_TOLERANCE and B24_ANALYSIS, none of which is a the build plan "
    "section 14 row either. Two of B25's three open items were already "
    "committed elsewhere: the diversity sample count and statistic are "
    "DIVERSITY_DEFINITION, fixed at B16, and are consumed unchanged. What "
    "is set here is the tracking population, the grid the diversity "
    "statistic is evaluated on at this step, and the construction of the "
    "condition-blind baseline.",

    "tracking_population": "matched pairs",
    "tracking_population_basis": (
        "M08's own computation note states the population in its own words: 'The "
        "population is matched pairs, not admitted records.' the build plan's schedule gives "
        "the tracking population the treatment STATED VALUE and assigns it to B25, and "
        "the committed specification has no row for it (see KNOWN_DISCREPANCIES). It is stated "
        "here. "
        "\n"
        "The matched pairs are the 109 pairs in which BOTH members cleared admission, "
        "which is the identical population every other reported statistic in this study "
        "uses, and the identical set B24 computed the primary outcome on. B25 does not "
        "rebuild it: it calls run_b24_analysis.build_pairs, so there is one pairing and "
        "not two that agree. "
        "\n"
        "THE SUPERSEDED BUILD FITTED OVER ADMITTED RECORDS, which is a larger and "
        "differently shaped population than the one every other statistic used. On this "
        "data those two populations differ by one shape, so the correction is small "
        "here; it is made because the populations are not the same object, not because "
        "the difference is large."
    ),

    "diversity_grid_at_b25": ("the committed diversity grid, 11 evenly spaced "
        "normalised targets from 0.0 to 1.0 inclusive"),
    "diversity_grid_basis": (
        "ONE GRID, AND IT IS THE COMMITTED ONE. The issued pre-registration, summarised at Appendix A.1," "fixes the diversity grid as 20 samples at each of 11 evenly "
        "spaced normalised targets from 0.0 to 1.0 inclusive, and states in its own "
        "words that this is the full conditioning range and not the requested target "
        "band. That is DIVERSITY_DEFINITION's grid, it is what evaluate.target_grid() "
        "returns, and B16's sweep, B18's gate and B25 all evaluate the one definition "
        "on it. Their figures are therefore comparable point for point. "
        "\n"
        "CORRECTED IN SESSION 14, and recorded rather than replaced silently. B25 "
        "previously evaluated M11 on the requested target band. The reasoning was "
        "M11's own phrase 'across the whole requested range' together with B25's own "
        "logic text 'at each requested target', and on those two sentences alone it is "
        "a defensible reading. It was nevertheless a departure from a value the issued "
            "pre-registration commits, and the pre-registration does not permit a committed value to be "
        "amended in place. The committed grid governs and M11 was recomputed on it. "
        "\n"
        "THE COST OF THE COMMITTED VALUE, stated because it is real and is not "
        "removed by following the commitment. The grid spans 0.0 to 1.0 while the "
        "requested band covers the training split's 5th to 95th label percentile and "
        "stops at 0.5522 normalised. Roughly half the grid's points therefore sit at "
        "targets this study never requests, so M11 describes the model's generative "
        "spread across its whole conditioning range rather than across the band alone. "
        "That is what was committed and it is reported as committed. "
        "\n"
        "WHAT WAS NOT DONE. The band figure is not also reported. Two quantities "
        "sharing the word diversity is exactly what the committed specification forbid, and it is the "
        "defect the build plan retired the superseded build's two diversity computations for. "
        "One definition, one grid, one number. "
        "\n"
        "WHAT DID NOT MOVE. The requested target band remains the grid every other "
        "quantity at B25 uses, being M08, M09, M10, M16 and M22, and it remains the "
        "population the paired analysis and F04 run on. Only M11 and F06 sit on the "
        "committed diversity grid."
    ),

    "condition_blind_baseline": {
        "construction": (
            "M10. The requested target COLUMN is shuffled and the generations are "
            "produced again. Everything else is held identical to B23: the same "
            "committed model, the same latent codes from the same generation seed, the "
            "same decode resolution, the same plausibility filter, the same solver "
            "settings and timeout, the same admission rule and the same complete-case "
            "pairing. The model therefore receives a target unrelated to the slot it "
            "is filling, which is what M10 asks for in its own words. "
            "\n"
            "THE FIT IS AGAINST THE ORIGINAL REQUESTED COLUMN, not the shuffled one. "
            "That is the whole construction. Fitting achieved efficiency on the "
            "shuffled column the model actually read would measure tracking again and "
            "return something near M08, which is not a chance reference. Fitting it on "
            "the original column, which the model never saw, is what a slope of no "
            "conditioning looks like on this pipeline."
        ),
        "shuffle_rule": (
            "One permutation of the 110 launched slots, drawn from "
            "dataset.rng_for(25, 1). A permutation rather than a fresh draw, so the "
            "multiset of requested targets is exactly the one B23 launched and the "
            "cluster sizes are unchanged. A fixed point of the permutation, being a "
            "slot whose shuffled target equals its original one, is left alone: "
            "removing them would make the shuffle something other than a shuffle, and "
            "the count of them is reported."
        ),
        "basis": (
            "the literature scan records a CLEAN NOT-FOUND. No published source constructs a "
            "deliberately unconditioned or shuffled generator as a chance reference. "
            "The closest published anchor, arXiv:2302.02913 sections 9.2 and 9.5, is "
            "performance blind rather than condition blind, and the distinction is "
            "stated rather than elided. This construction is therefore this study's "
            "own and is disclosed as its own. "
            "\n"
            "THE SUPERSEDED BUILD PRINTED AN ASSUMED BASELINE AS LITERAL TEXT and "
            "computed nothing. That value is void. What replaces it is a measurement "
            "with its own solver calls and its own attrition, and if the measured "
            "baseline is not near zero, that is the finding and it is reported."
        ),
    },

    "generator_substreams": {
        "rule": "dataset.rng_for(25, substream), and model.seed_int(25, substream) "
        "where a torch stream is needed, which is the one base seed 20260806 "
        "plus 1000 times the build step number plus the substream index. Same "
        "rule as every other step in this build.",
        0: "M11 generative diversity, the latent codes for the 11 requested targets by "
        "20 samples, both arms sharing each code",
        1: "M10 condition-blind baseline, the permutation of the requested target column",
        900: "the diversity clauses of B25's falsification check, which touched no "
        "evaluation record and was deleted after passing",
    },

    "condition_blind_reuses_b23_codes": (
        "M10's generation reuses B23's own latent codes, drawn from "
        "model.seed_int(23, 0), rather than drawing fresh ones at step 25. Stated "
        "because it is a choice and not an oversight. The baseline exists to isolate "
        "what the CONDITION contributes, so the codes are held fixed and the target "
        "column is the only thing that moves between B23's generation and this one. "
        "Drawing fresh codes would change two things at once. The driver verifies the "
        "reuse rather than assuming it, by regenerating B23's own geometry from that "
        "seed and requiring it to match the stored coefficients exactly before the "
        "shuffle is applied."
    ),
}


# ---------------------------------------------------------------------------
# The 33 slots of the committed specification, in the table's own order.
# ---------------------------------------------------------------------------

PARAMS = {

    "cst_order": ParamSlot(
        key="cst_order",
        treatment=Treatment.AUTHORS_CHOICE,
        note="Justification type must be named",
        disclosure_obligation=(
            "Name which of the three recurring justification types supports the "
            "chosen order: matching a target pressure or force prediction as order "
            "increases until it plateaus, naming one of Kulfan's originally proposed "
            "discretisation levels, or empirical fitting-error convergence against a "
            "reference database."
        ),
        set_at="B02 Set at B06 by empirical fitting-error convergence (order 5). "
        "Re-derived during the build, superseding that result, by the committed specification's "
        "force-prediction convergence method (order 9) after the "
        "author asked for order 8 'to better match literature' -- not "
        "itself a committed-specification justification type -- and, on that being flagged, "
        "chose to actually run the force-convergence method rather "
        "than hand-pick a value.",
        value=9,
        derivation=(
            "SUPERSEDED RESULT (kept here for the record, not used): "
            "empirical fitting-error convergence (the committed specification's third named type) "
            "against the five real seeds, orders 3-16, 10% sustained-plateau "
            "rule, gave order 5 (derive_cst_order.py). Full table and rule "
            "text are in BUILD_LOG.md. "
            "\n"
            "CURRENT RESULT: the committed specification's first named justification type, matching a "
            "target force prediction as order increases until it plateaus. "
            "Target force prediction: max(CL/CD) over the committed 0-8 "
            "degree alpha sweep at the committed B04 operating point -- the "
            "same efficiency metric used everywhere else in this "
            "build, not an arbitrary pick for this run. Method "
            "(derive_cst_order_force_convergence.py): fit and solve all five "
            "real seeds at each candidate order 3 through 14 (30s diagnostic "
            "timeout, not the committed the committed specification value). Rule stated before "
            "running: worst-case (max-across-seeds) relative change in "
            "max(CL/CD) from the previous order must stay below a stated 2%, "
            "sustained for the rest of the searched range -- same discipline "
            "as the fitting-error-convergence run, for the same reason (a "
            "single noisy step should not be read as a plateau). "
            "\n"
            "Raw per-seed max(L/D) by order: 3: e387=96.135 s1223=86.713 "
            "sd7003=68.601 seagull=99.748 sg6043=116.014; 4: e387=95.838 "
            "s1223=89.565 sd7003=68.775(partial conv) seagull=91.460 "
            "sg6043=114.517; 5: e387=96.401 s1223=87.366 sd7003=69.357 "
            "seagull=94.181 sg6043=117.757; 6: e387=98.948 "
            "s1223=83.965(partial) sd7003=68.229 seagull=100.266 "
            "sg6043=117.603; 7: e387=98.698 s1223=87.227 sd7003=68.142 "
            "seagull=100.668 sg6043=117.374; 8: e387=97.811(partial) "
            "s1223=87.477(partial) sd7003=66.734 seagull=98.354 "
            "sg6043=117.224; 9: e387=97.912 s1223=84.298 sd7003=65.288 "
            "seagull=97.586 sg6043=117.460; 10: e387=98.448 s1223=84.508 "
            "sd7003=65.458 seagull=98.076 sg6043=117.364; 11: e387=98.227 "
            "s1223=85.239 sd7003=66.297 seagull=98.922 sg6043=117.495; "
            "12: e387=97.907 s1223=85.681 sd7003=66.523 seagull=98.911 "
            "sg6043=117.279; 13: e387=97.897 s1223=85.622 sd7003=66.297 "
            "seagull=98.773 sg6043=117.339; 14: e387=98.116 s1223=84.966 "
            "sd7003=66.190 seagull=180.732(anomaly, see below) sg6043=117.339. "
            "\n"
            "Worst-case relative order-to-order change: 3->4 8.31%, 4->5 "
            "2.97%, 5->6 6.46%, 6->7 3.89%, 7->8 2.30%, 8->9 3.63%, 9->10 "
            "0.55%, 10->11 1.28%, 11->12 0.52%, 12->13 0.34%, 13->14 82.98%. "
            "\n"
            "ORDER 14 EXCLUDED, investigated not assumed: seagull's max(L/D) "
            "jumps from 98.77 (order 13) to 180.73 (order 14), driven by CD "
            "roughly halving across the entire alpha sweep (e.g. alpha=0 CL/"
            "CD/L-D: order 13 1.109/0.01296/85.6, order 14 1.094/0.00661/"
            "165.5 -- CL essentially unchanged, CD nearly halved, at every "
            "alpha, not one outlier point). Checked directly against the "
            "geometry before accepting or rejecting this: order 13 vs 14 "
            "seagull fits are near-identical (upper residual_max 0.000310 "
            "vs 0.000269 chord; max|camber second-difference| 0.000014 vs "
            "0.000013 on the B03 interior grid; thickness range identical "
            "to 5 decimal places; no added wiggle in a curvature-proxy scan). "
            "A geometry change this small does not credibly explain a 2x "
            "drag change; this is read as XFOIL's transition-location "
            "prediction flipping between two states on a near-imperceptible "
            "surface perturbation, not a real CST-order effect on the shape "
            "-- and it is isolated to seagull alone (no other seed shows any "
            "comparable jump at any searched order; sg6043 in fact repeats "
            "exactly, 117.339 at both order 13 and 14, consistent with its "
            "order-14 coefficient converging to ~0 rather than any "
            "instability). The order 13->14 step is excluded from the "
            "plateau search on this basis, and the rule is re-applied over "
            "the diagnosed-reliable range, orders 3-13, where the metric is "
            "smooth for every seed. "
            "\n"
            "Within orders 3-13: every consecutive step from 9->10 onward "
            "stays under 2% for the rest of that range (9->10 0.55%, 10->11 "
            "1.28%, 11->12 0.52%, 12->13 0.34%); order 8->9 itself is 3.63%, "
            "above threshold, which is why 9 and not 8 is chosen. "
            "\n"
            "Chosen order: 9. This happens to be close to the value the "
            "author initially asked for (8) and to typical literature CST "
            "orders, but it was not chosen for that reason -- it is what "
            "this stated rule returns on this data, reported as such; the "
            "closeness to 8 is a coincidence worth noting, not the "
            "justification."
        ),
    ),

    "cst_fit_error_acceptance": ParamSlot(
        key="cst_fit_error_acceptance",
        treatment=Treatment.STATED_VALUE,
        note="Two non-interchangeable reference points disclosed",
        set_at="B02 measured these. The slot was left PENDING from the build to "
        "the build and is written here, during the build, from a fresh "
        "recomputation at the committed order rather than from memory of "
        "what B02 reported. No value moved; the slot simply never carried "
        "one.",
        value={
            "measure": "per-surface residual in ordinate, chord-normalised",
            "order": 9,
            "acceptance_threshold": None,
            "per_seed": {
                "E387":                      {"upper_max": 4.350e-04, "upper_rms": 1.438e-04,
                    "lower_max": 3.924e-03, "lower_rms": 7.369e-04},
                "S1223":                     {"upper_max": 8.114e-04, "upper_rms": 2.336e-04,
                    "lower_max": 3.090e-03, "lower_rms": 7.905e-04},
                "SD7003-085-88":             {"upper_max": 3.943e-03, "upper_rms": 7.191e-04,
                    "lower_max": 1.264e-04, "lower_rms": 3.767e-05},
                "Seagull (Liu et al. 2006)": {"upper_max": 5.931e-04, "upper_rms": 1.488e-04,
                    "lower_max": 5.694e-04, "lower_rms": 1.412e-04},
                "SG6043":                    {"upper_max": 3.082e-04, "upper_rms": 1.212e-04,
                    "lower_max": 1.586e-03, "lower_rms": 4.112e-04},
            },
        },
        derivation=(
            "Metric M24. The maximum and root mean square residual of each seed's "
            "fit, per surface, in chord units, at the committed CST order (9). "
            "Recomputed during the build by geometry.fit_surface against the five real "
            "files in seeds/, so the figures here are the fit the pipeline actually "
            "uses and not a transcription. "
            "\n"
            "NO ACCEPTANCE THRESHOLD IS CLAIMED, which is the committed specification's own treatment and is "
            "the reason acceptance_threshold is None rather than a number. the committed specification "
            "discloses two reference points and records that they are not "
            "interchangeable. Wind tunnel manufacturing tolerance (Algorithms "
            "11(10):163, 2018, ten year window) and a distributional 1e-5 to 1e-4 "
            "band across a reference database (CSTO doi:10.2514/6.2024-2140, five "
            "year window). the committed specification resolves the closest source to the CSTO band, being "
            "the five year window source and database wide, and adopts NEITHER as a "
            "pass mark. Neither is adopted here. "
            "\n"
            "Stated plainly, because it would otherwise have to be inferred from the "
            "table: these residuals sit ABOVE the CSTO band. The worst surface "
            "maxima are 3.9e-03 chord on SD7003's upper surface and 3.9e-03 on "
            "E387's lower, and the root mean square figures run from 3.8e-05 to "
            "7.9e-04, so only one of the ten surfaces has an RMS inside 1e-04. This "
            "is reported as it stands. It is not a failure against a pass mark, "
            "because the committed specification adopts no pass mark, and presenting the CSTO band as one "
            "after the fact would be adopting a threshold this specification "
            "declined to adopt. What it does mean is that the shared order-9 basis "
            "represents these five seeds to a few parts in a thousand of chord and "
            "not to a few parts in a hundred thousand, and any claim about the "
            "representation's fidelity is scoped to that. "
            "\n"
            "The figure that matters most for this study is the avian anchor's, "
            "because the anchor's fit quality decides how faithfully the reference "
            "signature is expressed, and the region extent and the prior term are "
            "both defined against that signature. The seagull section is the BEST "
            "fitted of the five on both surfaces (upper max 5.931e-04, lower max "
            "5.694e-04, and the only seed whose two surfaces agree to within 4 "
            "percent of each other). Recorded as an observation, not as a claim that "
            "the order was chosen for it; order 9 came from the committed specification's force-prediction "
            "convergence rule over all five seeds jointly (see slot cst_order)."
        ),
    ),

    "perturbation_scale": ParamSlot(
        key="perturbation_scale",
        treatment=Treatment.RANGE_WITH_RULE,
        note="Sampler bound required, not only rejection",
        rule_text=(
            "Stated on the sampler as a per-coefficient interval relative to that "
            "seed's own coefficient. Its width is the smallest value whose accepted "
            "population spans a stated camber and thickness range. The rule is "
            "applied before any label is computed (the committed specification; B06 logic)."
        ),
        set_at="B06",
        value=0.05,
        derivation=(
            "REBUILT after cst_order changed from 5 to 9 (see "
            "that slot's derivation): a perturbation width is meaningless "
            "independent of the coefficient space it is drawn in, so B06 was "
            "rerun in full against the order-9 seed fits. The order-5/width-"
            "0.07 result this slot previously carried no longer applies to "
            "anything downstream and is superseded outright, not kept as an "
            "alternate reading. "
            "\n"
            "The bound (dataset.bounded_perturbation): every accepted coefficient "
            "satisfies perturbed_i = seed_i + delta_i, delta_i drawn i.i.d. "
            "Uniform(-width, +width), applied independently to every entry of "
            "both the upper- and lower-surface coefficient vectors (10 "
            "coefficients per surface at order 9, up from 6 at order 5). This "
            "is a bound by construction of the draw -- delta_i cannot land "
            "outside [-width, +width] -- not a rejection test applied after "
            "an unbounded draw. The plausibility filter (the committed specification, B03) is applied "
            "afterward, as a separate second-stage rejection on the "
            "resulting decoded shape; it does not define or enforce the "
            "coefficient-space bound itself. "
            "\n"
            "Width selection (run_b06_seed_library_sampler.py, "
            "dataset.select_perturbation_width), same rule and same target "
            "ranges as before (both are order-independent, measured from raw "
            "seed coordinates, not from any CST fit): target thickness range "
            "(0.068024, 0.145590) chord, target camber range (0.014569, "
            "0.100276), coverage tolerance 5% of each target's span, "
            "candidate widths 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, "
            "0.15, 0.20 (ascending), 300 trials per seed per candidate. "
            "\n"
            "Results at order 9: widths 0.005 through 0.03 did not cover the "
            "target range. Width 0.05 covered it: 1498 of 1500 trial draws "
            "accepted (99.9%), achieved thickness range (0.068359, "
            "0.144181) against target (0.068024, 0.145590), achieved camber "
            "range (0.008392, 0.109980) against target (0.014569, "
            "0.100276) -- both edges reached within tolerance. 0.05 is the "
            "smallest searched width that covers the target range at order "
            "9 and is the committed value. (Order 9 needed a narrower width "
            "than order 5's 0.07 to cover the same target range -- with 4 "
            "more coefficients sharing the same per-coefficient interval, "
            "a smaller width already produces comparable decoded-shape "
            "diversity; not unexpected, not investigated further since the "
            "search is empirical either way.) "
            "\n"
            "Committed population (build_population, at width=0.05, "
            "per_seed_count=200, base seed via dataset.rng_for(6, 1)): 1005 "
            "rows total (5 unperturbed seeds + 1000 perturbed, 200 per "
            "family). Rejected-draw counts during the committed build: "
            "e387 0, s1223 0, sd7003 1, seagull 0, sg6043 0 -- acceptance "
            "against the plausibility filter is essentially total at this "
            "width, consistent with the 99.9% search-trial rate. Written to "
            "population.npz, overwriting the order-5 population. "
            "\n"
            "Falsification check (per the build plan, run and deleted after "
            "passing, on the order-9 population): every one of the 1000 "
            "perturbed rows' coefficients satisfied abs(perturbed - seed) "
            "<= 0.05 exactly (max escape 0.0). Nearest-seed-by-distance "
            "agreed with the recorded family label on all 1005 rows (0 "
            "disagreements, against a stated 5% limit)."
        ),
    ),

    "pre_solver_filter_thresholds": ParamSlot(
        key="pre_solver_filter_thresholds",
        treatment=Treatment.STATED_VALUE,
        note="Criteria and rejection count reported",
        set_at="B03",
        # Set by derive_b03_bounds.py, run against the five real seed
        # files in data/seeds/. Criteria: no crossing between surfaces on
        # the interior grid; maximum thickness inside [thickness_lower,
        # thickness_upper]; peak absolute second difference of camber on
        # the interior grid at or below camber_second_difference_bound.
        # Rejection count against these bounds is not yet reported here;
        # that is a B06/B07 sampling-time count, not a B03 fact.
        value={
            "thickness_upper": 0.14558990124579074,
            "thickness_lower": 0.0680239706227634,
            "camber_second_difference_bound": 0.0003794664188666495,
            "margin_fraction": 0.20,
            "n_grid_points": 200,
            "edge_margin": 0.005,
        },
        derivation=(
            "Measured directly from raw seed coordinates (no CST round trip), on "
            "a 200-point cosine grid restricted to [0.005, 0.995] chord, across "
            "the 5 seeds in data/seeds/ (e387, s1223, sd7003, seagull, sg6043). "
            "Per-seed max_thickness / max_abs_camber_2nd_diff: "
            "E387: 0.090694 / 0.000202; S1223: 0.121325 / 0.000316; "
            "SD7003-085-88: 0.085030 / 0.000226; "
            "Seagull (Liu et al. 2006): 0.097466 / 0.000064; "
            "SG6043: 0.100123 / 0.000096. "
            "thickness_upper = max(max_thickness) * 1.20 = 0.121325 * 1.20 = 0.145590. "
            "thickness_lower = min(max_thickness) * 0.80 = 0.085030 * 0.80 = 0.068024. "
            "camber_second_difference_bound = max(max_abs_camber_2nd_diff) * 1.20 "
            "= 0.000316 * 1.20 = 0.000379. "
            "margin_fraction = 0.20, an author's choice made at this step: a round, "
            "stated figure, wide enough that every seed clears its own bound with "
            "room for a perturbed sample, and not inherited from the prior build. "
            "edge_margin = 0.005 chord excluded from each end of the measurement "
            "grid: raw digitized seed coordinates do not reach x=0 or x=1 exactly "
            "(closest recorded leading-edge point across the five seeds is E387 at "
            "x=0.00044), and a grid point closer to an edge than the data reaches "
            "makes np.interp clamp instead of interpolate, which read as a false "
            "'crossing' on E387 and SD7003 before this margin was added. Verified: "
            "with these bounds, all five seeds pass their own filter "
            "(derive_b03_bounds.py, run on the actual seed files)."
        ),
    ),

    "trailing_edge_thickness": ParamSlot(
        key="trailing_edge_thickness",
        treatment=Treatment.STATED_VALUE,
        note="Closure stated. the build plan states the value itself: zero.",
        set_at="B02 (structural: decode() always closes the trailing edge; "
        "this slot is not read by geometry.py, it records the fact). "
        "Written into the slot during the build; the value was never in "
        "doubt and the slot simply never carried it.",
        value=0.0,
        derivation=(
            "Zero. the committed specification states the value itself, so there is nothing to select and "
            "nothing to derive. The trailing edge is closed by setting the CST "
            "trailing edge thickness parameter to zero, and the closure is stated "
            "here rather than left to be read off the code. "
            "\n"
            "Enforced in two places in geometry.py, both structural. decode_surface "
            "evaluates class times shape with no linear trailing edge term added, so "
            "the term is absent rather than set small. decode_airfoil then sets both "
            "surfaces' trailing edge ordinates to their shared midpoint explicitly. "
            "\n"
            "Recorded at B02 and worth keeping visible: the class function's own "
            "value at x=1 is exactly zero for any coefficients, given the standard "
            "airfoil exponents, which makes that explicit closure step a no-op in "
            "this implementation. It is kept anyway, so the committed specification's requirement is visible "
            "and checked in code rather than relied on implicitly, and B02's own "
            "falsification check asserted on every decode that the two trailing edge "
            "ordinates are equal, not merely close. "
            "\n"
            "Why the requirement is load-bearing rather than cosmetic: the committed specification records "
            "that XFOIL will forcibly collapse an unclosed trailing edge, which can "
            "produce an abrupt closure and change the shape the solver actually "
            "evaluates from the shape the model emitted. Closing it upstream means "
            "the solver never has to. the committed specification also records a clean not-found, that no "
            "source quantifies an XFOIL convergence failure rate against trailing "
            "edge gap size for a CST decoded generated shape (R2 not-found 4), so "
            "the gap is set to zero rather than to a small tolerated value."
        ),
    ),

    "dataset_size_and_split": ParamSlot(
        key="dataset_size_and_split",
        treatment=Treatment.STATED_VALUE,
        note="Both counts distinguished; split provenance stated",
        set_at="B07 (distinct-shape and shape-condition-pair counts, the committed specification); "
        "B12 (validation fraction / split provenance, the committed specification), the build",
        value={
            "distinct_labelled_shapes": 982,
            "attempted": 1005,
            "validation_fraction": 0.20,
            "train_rows": 785,
            "val_rows": 197,
        },
        derivation=(
            "B07's own count, the committed specification's distinct-shape half only -- the shape-"
            "condition-pair count the committed specification also asks for does not exist until "
            "B23 pairs generated shapes against requested targets, and is "
            "not guessed here. From run_b07_labelling_pass.py + "
            "finalize_b07_labelling.py on the committed order-9 population "
            "(1005 rows: 5 unperturbed seeds + 1000 perturbed, per B06) at "
            "the committed 7.56s timeout: 982 of 1005 attempted shapes were "
            "labelled (0 discarded on the plausibility re-check -- expected, "
            "since B06 already filtered at build time; 23 timed out; 0 true "
            "failures; 0 environment faults). Per family: e387 188/201 kept "
            "(13 timeout), s1223 200/201 (1 timeout), sd7003 198/201 (3 "
            "timeout), seagull 195/201 (6 timeout, avian family), sg6043 "
            "201/201 (0 timeout). Label range 50.499 to 172.174 (max(CL/CD), "
            "the committed specification). Reconciliation verified directly: kept + every discard "
            "category (0+23+0+0) = 1005 = attempted. Falsification check "
            "PASS (10 sampled rows spanning the label range, recomputed from "
            "each row's stored polar by an independently written expression, "
            "matched to floating-point-echo precision on every row; all 982 "
            "kept rows independently confirmed to carry a positive stored "
            "converged point count); check script deleted after passing, "
            "per the build plan. "
            "\n"
            "B12, validation-fraction half, the build: validation_fraction "
            "= 0.20, a stated value -- no selection rule is named for it "
            "anywhere in the reviewed plan text (unlike the perturbation "
            "width at B06 or the timeout at B05), so it is disclosed as a "
            "conventional, round figure rather than presented as derived. "
            "Split (dataset.split_dataset), stratified by family: within "
            "each family, shuffle under one recorded seed "
            "(dataset.rng_for(12, 0), i.e. base seed 20260806 + 1000*12 + "
            "0), then take round(0.20 * family_count) rows as that family's "
            "validation share. Per family (total -> validation): e387 188 "
            "-> 38, s1223 200 -> 40, sd7003 198 -> 40, seagull 195 -> 39, "
            "sg6043 201 -> 40. Totals: 785 train, 197 validation, 982 "
            "overall. Written to split.npz (train_idx, val_idx, "
            "validation_fraction, base_seed, rng_step, rng_substream); this "
            "one split is what B13 and B15 must both read, per the build plan's "
            "own instruction, and nothing downstream constructs its own. "
            "Falsification check PASS: train/val index sets disjoint; their "
            "union covers all 982 rows exactly once; every family's "
            "validation share is within 0.03 of 0.20; every family appears "
            "in validation (all five, by construction of the per-family "
            "split, not merely by chance). Check script deleted after "
            "passing, per the build plan."
        ),
    ),

    "geometry_standardisation_statistics": ParamSlot(
        key="geometry_standardisation_statistics",
        treatment=Treatment.AUTHORS_CHOICE,
        note="Leakage claim scoped to the choice",
        disclosure_obligation=(
            "State which row set (training split alone, or the full dataset) the "
            "per-column mean and spread were computed over, and scope any "
            "no-leakage claim to what that choice actually supports."
        ),
        set_at="B08, the build",
        value={
            "row_set": "training split only (B12), 785 of 982 dataset rows",
            "n_columns": 20,
            "label_min": 50.49872773536895,
            "label_max": 172.17391304347828,
        },
        derivation=(
            "Row set chosen: the training split alone (785 of 982 rows, from "
            "B12's split.npz), not the full dataset. Justification: computing "
            "the scaling constants over rows the model later validates on "
            "would leak validation-set distributional information into "
            "training-time scaling, a standard leakage concern; excluding "
            "those rows avoids it directly rather than requiring a scoped "
            "defence of why it is harmless. Both the label normalisation and "
            "the geometry standardisation use this same row set -- the build plan's "
            "own B08 text names one row-set choice governing both operations, "
            "not two independent choices. "
            "\n"
            "Label normalisation (dataset.derive_label_normalization, min-max "
            "over the training split): label_min=50.49872773536895, "
            "label_max=172.17391304347828 (full precision, unrounded -- the "
            "same value the model reads is the value recorded here). "
            "normalize_label maps this range to [0, 1]; denormalize_label is "
            "its exact inverse. "
            "\n"
            "Geometry standardisation (geometry.derive_standardization_stats, "
            "per-column population mean and std, ddof=0, over the training "
            "split): 20 columns (upper CST coefficients at order 9, then "
            "lower, concatenated -- the same convention "
            "dataset.nearest_seed_family already used). No column had zero "
            "spread over the training split; the function would have raised "
            "rather than silently guarding against that. Written to "
            "standardization.npz (mean, std, row_set_description, n_rows, "
            "n_columns) and normalization.npz (label_min, label_max, "
            "row_set_description, n_rows). "
            "\n"
            "Leakage claim this scopes to: no validation-label or "
            "validation-geometry information reaches the normalisation or "
            "standardisation constants. It says nothing about test-time "
            "generative behaviour, which these constants do not touch. "
            "\n"
            "Falsification check PASS (run_b08_normalisation.py, deleted "
            "after passing): standardising the training split against its "
            "own artifact gave max |column mean| = 1.117e-15 and max "
            "|column std - 1| = 1.776e-15, both far inside the build plan's 1e-10 "
            "tolerance; normalising then denormalising every one of the 982 "
            "labels (not only the training rows) returned to itself with "
            "max deviation 1.421e-14."
        ),
    ),

    "spanwise_resolution_avian_section": ParamSlot(
        key="spanwise_resolution_avian_section",
        treatment=Treatment.STATED_VALUE,
        note="Averaged or single station, said explicitly",
        set_at="B09, the build. Wording corrected during the build, at the "
        "author's direction, after the paper's Method draft surfaced a "
        "contradiction between this slot and slot "
        "avian_section_geometry_figures. No value changed and no "
        "artifact was regenerated; the two slots now state one "
        "position instead of two.",
        value="mixed: span-averaged shape, single-station magnitude",
        derivation=(
            "THE RESOLUTION IS MIXED, and this slot previously read 'single "
            "station', which described the pipeline and not the "
            "reconstruction. Both facts are true and they are about "
            "different things, which is why the earlier wording read as a "
            "contradiction against slot avian_section_geometry_figures. "
            "\n"
            "THE RECONSTRUCTION, which is what the committed specification asks about. The shape "
            "comes from Liu et al. (2006)'s span-averaged coefficients, "
            "averaged over that paper's own stated valid range for the "
            "seagull, 2y/b = 0.166 to 0.772, beyond which the primary "
            "feathers separate and no single continuous airfoil exists. The "
            "magnitude, being maximum camber and maximum thickness, comes "
            "from that paper's envelope equations at the single station "
            "2y/b = 0.4. Equation (1) is the mean Birnbaum-Glauert camber "
            "line distribution and Equation (2) is the thickness "
            "distribution. The source splits shape and magnitude that way "
            "itself, so no single-resolution reconstruction of this section "
            "exists to be chosen. This matches the build plan's the committed specification resolution "
            "record and the build plan's own instruction, both of which state the "
            "resolution as mixed. "
            "\n"
            "THE PIPELINE, which is a separate fact and is what the earlier "
            "wording described. seeds/seagull.dat is one two-dimensional "
            "cross-section of 300 points, not a set of spanwise stations. "
            "prior.py's fit_avian_reference fits that single file directly "
            "with geometry.fit_surface at the committed CST order (9). No "
            "spanwise averaging is performed anywhere in THIS BUILD, because "
            "the averaging happened upstream, when the section was "
            "reconstructed. The absence of averaging code is therefore not "
            "evidence that the reconstruction is single-station, and stating "
            "it without that qualification is what produced the "
            "contradiction."
        ),
    ),

    "avian_section_geometry_figures": ParamSlot(
        key="avian_section_geometry_figures",
        treatment=Treatment.STATED_VALUE,
        note=(
            "Live conflict with the source, must reconcile. Per the build plan, this is "
            "already status 'resolved by reconciliation' at the specification "
            "level: shape from the span-averaged coefficients, magnitude from the "
            "envelope equations at a single station, computed from the seed "
            "coordinate file rather than asserted. Not a numeric slot in the "
            "usual sense; recorded here because the committed specification lists it."
        ),
        set_at="B02 (fit) and B09 (reference signature); reported per the "
        "the committed specification resolution record. Written into the slot during the build, "
        "recomputed from seeds/seagull.dat rather than transcribed.",
        value={
            "source_file": "seeds/seagull.dat",
            "n_raw_coordinate_points": 300,
            "spanwise_resolution": "mixed: span-averaged shape, single-station magnitude",
            "max_thickness": 0.097466,
            "max_thickness_x_over_c": 0.2017,
            "max_camber": 0.100276,
            "max_camber_x_over_c": 0.4409,
            "max_abs_camber_second_difference": 6.418739e-05,
            "measurement_grid": "the committed B03 interior grid, 200 cosine-spaced "
            "points restricted to [0.005, 0.995] chord",
        },
        derivation=(
            "COMPUTED FROM THE COORDINATES THE PIPELINE READS, which is the whole of "
            "the committed specification's requirement. Measured during the build by "
            "geometry.measure_thickness_and_camber against seeds/seagull.dat on the "
            "committed B03 interior grid, the same grid the plausibility filter and "
            "the safeguard bounds are measured on. No figure here is taken from the "
            "source's text. "
            "\n"
            "Maximum thickness 0.097466 chord at x/c 0.2017. Maximum camber 0.100276 "
            "chord at x/c 0.4409. The max thickness figure is the same one B03's own "
            "bound derivation recorded for this seed, and the max camber figure is "
            "the same one B06's width-selection target range used, so the three "
            "steps that touch this geometry agree by construction rather than by "
            "coincidence. "
            "\n"
            "THE RECONCILIATION the committed specification REQUIRES, stated in the form the committed specification fixes. The "
            "status of this entry is resolved by reconciliation and not live "
            "conflict. The reconstruction is mixed in spanwise resolution, with the "
            "shape from Liu et al.'s span-averaged coefficients and the magnitude "
            "from their envelope equations at a single station, which is how the "
            "source itself splits them. The apparent conflict recorded in earlier "
            "revisions does not exist. The figures a research return attributed to "
            "the seagull's own section are the COMMON maximum camber and thickness "
            "the source imposes across the sections in its S1223 comparison, which "
            "is a shared normalisation scale and not that section's own values. The "
            "transcription was accurate and the attribution was not. "
            "\n"
            "What the paper therefore says, in one sentence, per the committed specification: it cites Liu "
            "et al. 2006 for the coefficients, the envelope equations and the "
            "station, and states that the comparison amplitudes appearing in that "
            "source are a shared scale, so a reader who finds them does not conclude "
            "this reconstruction is wrong. "
            "\n"
            "the committed specification travels with this and is not restated here beyond the pointer: the "
            "seagull to S1223 resemblance is cited to Liu et al. 2006 and not to "
            "Ananda and Selig, and the resemblance is qualified as a statement about "
            "shape after normalisation to a common maximum camber and thickness, not "
            "about the sections as they stand. "
            "\n"
            "Spanwise resolution is MIXED and is recorded separately in slot "
            "spanwise_resolution_avian_section, which carries the full statement. "
            "In short: the shape is span-averaged, the magnitude is taken at the "
            "single station 2y/b = 0.4, and the averaging is upstream of this "
            "build. seeds/seagull.dat is one two-dimensional cross-section of 300 "
            "points, so THIS BUILD performs no spanwise averaging, which is a fact "
            "about the pipeline and not about the reconstruction's resolution."
        ),
    ),

    "efficiency_target_definition": ParamSlot(
        key="efficiency_target_definition",
        treatment=Treatment.STATED_VALUE,
        note="One of four documented forms, named",
        set_at="B07 (the label is computed there as the maximum of lift "
        "over drag across converged points, within the fixed "
        "cruise-regime operating point the committed specification sets at B04). Written into "
        "the slot during the build; the definition has governed every "
        "label in this build since B07 and the slot never carried it.",
        value={
            "definition": "max(CL / CD) over the converged points of one alpha sweep",
            "sweep": "alpha 0 to 8 degrees inclusive, step 1 degree, 9 requested points",
            "operating_point": "the committed B04 point, Re 300000, M 0, Ncrit 9",
            "t18_form": "a maximum over an angle sweep, evaluated at a named cruise "
            "condition",
            "units": "dimensionless lift-to-drag ratio",
        },
        derivation=(
            "THE FORM, NAMED, which is the committed specification's whole requirement. the committed specification records four "
            "documented forms and states that no source names any one as the field "
            "default, so the obligation is to say which is used. "
            "\n"
            "This build uses the first form, a maximum over an angle sweep, "
            "evaluated at the fourth form's named cruise condition. The label of a "
            "shape is max(CL/CD) across the converged points of a single alpha sweep "
            "from 0 to 8 degrees in 1 degree steps, at the committed B04 operating "
            "point. "
            "\n"
            "THE DIFFERENCE FROM THE FIRST FORM AS PUBLISHED, stated rather than "
            "elided. the committed specification's first form is a maximum over a JOINTLY OPTIMISED angle "
            "sweep, where the angle is optimised together with the design. Here the "
            "sweep is fixed in advance and identical for every shape, and the "
            "maximum is taken over that fixed grid. That makes the label cheaper and "
            "makes it comparable across shapes without an inner optimisation, and it "
            "means the reported figure is a maximum over nine stated angles rather "
            "than over a continuum. A shape whose true peak lies between two grid "
            "angles is labelled at the better of the two neighbours, which "
            "understates it. No correction is applied and the grid is reported. "
            "\n"
            "Where the label is computed, in one place: evaluate.label_from_polar, "
            "which B07's dataset pass, B19's consistency gate, B21's pilot and B23's "
            "run all call. Only converged points reach it, because "
            "solver._parse_polar writes one row per angle XFOIL actually "
            "accumulated. A sweep producing no usable point returns None and never a "
            "sentinel number. "
            "\n"
            "The realised label range over the 982 labelled dataset rows is 50.499 "
            "to 172.174, and B08's min-max normalisation is derived on the training "
            "split alone against that scale. "
            "\n"
            "Two entries that scope any reading of this label and are not restated "
            "here beyond the pointer. the committed specification and slot minimum_converged_sweep_points "
            "govern when a maximum taken from a PARTIAL sweep is still admitted, "
            "committed at 8 of 9 points. the committed specification governs the single extreme but "
            "converged result, which is retained with the solver mechanism named as "
            "the caveat and with no statistical outlier test applied."
        ),
    ),

    "conditioning_composition": ParamSlot(
        key="conditioning_composition",
        treatment=Treatment.STATED_VALUE,
        note="Redundant block disclosed with evidence",
        set_at="B10, the build",
        value={
            "n_columns": 22,
            "column_0": "normalised target (the row's own normalised label)",
            "columns_1_to_20": "standardised avian signature block",
            "column_21": "flag: 1.0 = set (signature block present), "
            "0.0 = clear (zero block present)",
        },
        derivation=(
            "dataset.assemble_conditioning builds one (982, 22) array: "
            "column 0 is the normalised label (B08's normalize_label); "
            "columns 1..20 are the standardised avian signature (B09's "
            "standardized_signature) where the flag is set, or all zero "
            "where clear; column 21 is the flag itself, 1.0 when set and "
            "0.0 when clear -- the same sense B14's 'flag gated avian term' "
            "will read. The redundant block this row's note names is the "
            "signature/zero block sitting alongside an explicit flag column "
            "rather than being inferred from whether the block is all-zero; "
            "the evidence that the redundancy is not silently miswired is "
            "this step's own falsification check (below), which reads the "
            "block and the flag independently and confirms they agree on "
            "every one of the 982 rows. Written to conditioning.npz."
        ),
    ),

    "null_representation": ParamSlot(
        key="null_representation",
        treatment=Treatment.STATED_VALUE,
        note="Composite justification disclosed as composite",
        set_at="B10, the build",
        value="zero block (20 columns, all 0.0) plus the separate flag column",
        derivation=(
            "a zero block with a "
            "separate indicator flag. No sentinel constant (which would risk "
            "being confused with a real standardised coordinate landing near "
            "that value by chance) and no gating layer (a learned "
            "multiplicative gate is exactly the composite machinery the committed specification "
            "would require a separate justification for, which this choice "
            "avoids needing at all -- both were retired). The flag column "
            "alone is what the model conditions its reading of the block on; "
            "the zero block itself carries no information beyond 'not the "
            "signature'. Implemented in dataset.assemble_conditioning; "
            "verified by this step's falsification check to land on the "
            "correct rows on every one of the 982 rows, not merely by "
            "construction."
        ),
    ),

    "reconstruction_divergence_weighting": ParamSlot(
        key="reconstruction_divergence_weighting",
        treatment=Treatment.RANGE_WITH_RULE,
        note="Schedule reported",
        rule_text=(
            "Every auxiliary loss weight carries a selection procedure stated "
            "before any evaluation run. A weight with no literature value "
            "is accompanied by a reported sensitivity sweep (the committed specification, convention). "
            "\n"
            "RESOLVED during the build, having stood UNRESOLVED since B01. The "
            "reconstruction weight is not open after all: B14's total objective "
            "gives it an implicit weight of 1 and it is the scale every other "
            "weight is expressed against, so there is nothing to select. The "
            "divergence weight's FULL VALUE is open, and no step in the reviewed "
            "plan named where it is set -- B15 fixes only the warm-up schedule "
            "shape that multiplies it. It is carried into B16's sweep and B17's "
            "selection under the same rule as the other four weights, on B17's "
            "own 'and by the same procedure' authority, rather than being guessed "
            "or left to a default. No model can be trained at all without it, so "
            "leaving it unset was not an option B16 had. "
            "\n"
            "The full rule is the module-level constant WEIGHT_SELECTION_RULE in "
            "this file, fixed before B16's sweep was run. This slot does not "
            "restate it; there is one copy. Under the AMENDED rule the divergence "
            "weight is not swept at all. It is set as a stated value, below."
        ),
        set_at="Before B16. Set as a stated value on the locked model "
        "family, not selected from any table. See "
        "WEIGHT_SELECTION_RULE['amendment_record'] for why the swept form "
        "was withdrawn.",
        value={"reconstruction_weight": 1.0, "divergence_weight": 1.0},
        derivation=(
            "reconstruction_weight = 1.0, implicit in B14's total objective, which "
            "adds reconstruction unweighted and scales every other term. It is the "
            "scale the other weights are expressed against, so there is nothing to "
            "select. Recorded explicitly because the committed specification requires every weight that "
            "changes the objective to be reported, and an implicit 1 that appears "
            "in no list is precisely the failure the committed specification names. "
            "\n"
            "divergence_weight = 1.0. A divergence weight of 1.0 is the exact "
            "evidence lower bound, the beta = 1 case, which is the plain "
            "conditional VAE that this study's LOCKED model family names. It is "
            "committed on that ground and needs no sweep table. "
            "\n"
            "Why it is not selected by the sweep rule, recorded rather than "
            "quietly dropped: applying the first version of that rule to a real "
            "divergence ladder on this pipeline returned weight 0.0 as the only "
            "admissible row, at 106.4 nats validation KL per row and flag-set "
            "diversity 0.927 against 4.26 at weight 0.1. Weight zero is a "
            "conditional autoencoder, and every downstream quantity in this build "
            "is measured on generations from fresh N(0, I) codes, which at that "
            "weight decode off-distribution. A selection rule cannot be permitted "
            "to override a locked element, so the weight is set rather than "
            "selected. "
            "\n"
            "The divergence ladder is still RUN and REPORTED as sensitivity "
            "evidence under the committed specification, at 0, 0.01, 0.1, 1.0, 10.0, 100.0. It is evidence "
            "and not a selector. If its best trade does not sit at 1.0, that is "
            "reported as it stands and the committed value does not move. "
            "\n"
            "Treatment mismatch, recorded not resolved: the committed specification gives "
            "this row RANGE WITH RULE and it is set here as a stated value. See "
            "KNOWN_DISCREPANCIES."
        ),
    ),

    "prior_term_weight": ParamSlot(
        key="prior_term_weight",
        treatment=Treatment.RANGE_WITH_RULE,
        note="Not selected from the outcome",
        rule_text=(
            "The selection rule is fixed in this file before the weight "
            "sensitivity sweep (B16) exists. The rule is then applied to the "
            "sweep table alone, at B17, without reference to any evaluation-path "
            "result (the committed specification). The weight is not selected after the ablation "
            "outcome is seen; any change made after that point is disclosed. "
            "\n"
            "The full rule is the module-level constant WEIGHT_SELECTION_RULE in "
            "this file, written during the build before the sweep was run and before "
            "any sweep table, training history or evaluation result existed. B01 "
            "left this text as a statement of the rule's PROPERTIES (fixed early, "
            "applied to the table alone) without the rule's own content, since B01 "
            "predated the objective terms the rule has to read. The build wrote "
            "the content, still ahead of the sweep, which is what the build plan's "
            "procedural mitigation for B17 actually requires. Recorded as written "
            "later than B01, not backdated."
        ),
        set_at="B17 (the build), under WEIGHT_SELECTION_RULE",
        value=5.862918756788936,
        derivation=(
            "Selected by WEIGHT_SELECTION_RULE, applied to sweep/sweep_table.json "
            "alone. The rule was written and shown to the author before B16 was "
            "run, and amended once, still before B16 was run. No solver "
            "evaluation result had been seen at the point of selection, and none "
            "has been seen at any point in this build since B07's dataset "
            "labelling pass. Full score table in b17_selection.json. "
            "\n"
            "Ladder, scaled by u_avian = recon_0 / avian_0 = 20.22021 / 3.448830 = "
            "5.862918756788936, so the selected value is exactly parity: the weight "
            "at which the avian term contributes as much as reconstruction did at "
            "the untrained model's initialisation. Candidates 0, 0.0586292, "
            "0.586292, 5.86292, 58.6292, 586.292. "
            "\n"
            "Scores (effect_gain - recon_cost - diversity_cost, each normalised to "
            "the ladder's own range): 0.000000, -0.096545, 0.098563, 0.184391, "
            "-0.291845, -1.029619. Selected 5.862918756788936 at score 0.184391, a "
            "single candidate with no tie. Its components: effect_gain 1.0000 (it "
            "attains the ladder's smallest mean distance to the reference on the "
            "prior-on arm, 1.170410 against 3.827637 at weight zero), recon_cost "
            "0.0092, diversity_cost 0.8065. "
            "\n"
            "The trade this represents, stated rather than hidden: the committed "
            "weight buys the full available reduction in distance to the reference "
            "for almost no reconstruction cost, and pays for it in generative "
            "diversity, which falls from 3.6386 at weight zero to 1.0278 here. "
            "That is a large diversity cost and it is the reported cost of the "
            "committed weight, not a footnote. The two candidates either side "
            "score 0.098563 and -0.291845, so the selection is not knife-edge but "
            "the margin is not wide either. "
            "\n"
            "Weights above parity buy nothing further: mean distance at 58.6292 is "
            "1.286437 and at 586.292 is 1.269190, both WORSE than at parity, while "
            "reconstruction degrades from 1.173257 to 3.217894 and 9.648997. The "
            "prior saturates at parity on this ladder."
        ),
    ),

    "target_and_spread_weights": ParamSlot(
        key="target_and_spread_weights",
        treatment=Treatment.RANGE_WITH_RULE,
        note="",
        rule_text=(
            "Same procedure as the prior weight: rule fixed before the sweep "
            "exists, applied to the sweep table at B17 (the committed specification). The full rule "
            "is the module-level constant WEIGHT_SELECTION_RULE in this file. "
            "Each of these two weights has its own ladder and its own effect "
            "measure under that rule (target -> the validation target-consistency "
            "term; spread -> the validation mean ensemble spread), and each is "
            "selected independently of the other."
        ),
        set_at="B17 (the build), under WEIGHT_SELECTION_RULE",
        value={"target_weight": 99.09702261385384,
            "spread_weight": 54.307616878345584},
        derivation=(
            "Both selected by WEIGHT_SELECTION_RULE, applied to "
            "sweep/sweep_table.json alone, each on its own ladder and independently "
            "of the other. Full score tables in b17_selection.json. No solver "
            "evaluation result had been seen at the point of selection. "
            "\n"
            "TARGET WEIGHT. Ladder scaled by u_target = recon_0 / target_0 = "
            "20.22021 / 0.02040446 = 990.9702261385384; candidates 0, 9.9097, "
            "99.097, 990.97, 9909.7, 99097. Effect measure: the validation "
            "target-consistency term. Scores -0.000687, 0.841500, 0.939098, "
            "0.817012, 0.366103, -1.855921. Selected 99.09702261385384, which is "
            "0.1 * u_target, at score 0.939098. Components: effect_gain 0.9901 "
            "(target term 0.000280 against 0.020587 at weight zero, a factor of "
            "73), recon_cost 0.0004, diversity_cost 0.0506. This weight is close "
            "to free: it improves target satisfaction by nearly two orders of "
            "magnitude for essentially no reconstruction and little diversity. "
            "\n"
            "SPREAD WEIGHT. Ladder scaled by u_spread = recon_0 / spread_0 = "
            "20.22021 / 0.03723273 = 543.0761687834558; candidates 0, 5.43076, "
            "54.3076, 543.076, 5430.76, 54307.6. Effect measure: the validation "
            "mean ensemble spread. Scores -0.000637, 0.148800, 0.675362, 0.394611, "
            "-1.081618, -1.082744. Selected 54.307616878345584, which is 0.1 * "
            "u_spread, at score 0.675362. Components: effect_gain 0.7634 (spread "
            "0.006285 against 0.022395 at weight zero), recon_cost 0.0034, "
            "diversity_cost 0.0846. "
            "\n"
            "Known cost of the one-at-a-time design, stated not hidden: each of "
            "these two was selected with the other two swept weights at zero, and "
            "the committed model runs all three together with the avian weight. No "
            "interaction between them is measured anywhere in this build."
        ),
    ),

    "safeguard_weight": ParamSlot(
        key="safeguard_weight",
        treatment=Treatment.STATED_VALUE,
        note="Must appear in the reported objective",
        set_at="Before B16. Set as a stated value, which resolves the "
        "treatment mismatch B14 and B17 had opened against the build plan's table "
        "in that table's favour. The weight is not swept because it "
        "provably cannot be: its term is identically zero on this build, so "
        "every candidate yields the same objective and the same trained "
        "model bit for bit.",
        value=1.0,
        derivation=(
            "MEASURED BEFORE THE SWEEP, not discovered in it. On the exact "
            "measurement path TorchSafeguard.term itself uses, over the 785-row "
            "training population: 0 rows violate thickness_lower_bound "
            "(0.055426321765484482), and 0 rows violate curvature_upper_bound "
            "(0.00010535145925184528). The worst training shape's curvature is "
            "8.779e-05, so the bound sits exactly 1.200x above it, which is the "
            "margin itself. Evaluated at the untrained model's initialisation over "
            "the whole training split, the safeguard term is exactly 0.0 on both "
            "the reconstruction pass and the generation pass. "
            "\n"
            "Mechanism: the decoder emits STANDARDISED coefficients and begins near "
            "zero in that space, which is the training population's own mean shape "
            "and plausible by construction, so it never enters the region the "
            "hinge penalises. The term's gradient is therefore identically zero and "
            "the weight changes nothing. This is why a sweep over it produced "
            "bit-identical validation reconstruction at every candidate. "
            "\n"
            "Value committed: 1.0. Reported together with the measurement above, "
            "which is what the committed specification asks for -- a weight that changes the objective is "
            "reported, and a weight that provably does not is reported as that. "
            "The safeguard is a guard that did not bind on this build, and it is "
            "presented as one. "
            "\n"
            "The bounds are NOT retightened to make the term active. Choosing a "
            "threshold so that a term starts binding is calibration against an "
            "outcome, which is the specific failure B18 exists to prevent. "
            "\n"
            "DEFECT FOUND AND FIXED, B14 re-opened at the author's direction, "
            "the build. The bound had been derived on a different measurement path "
            "from the one it is applied on. model.derive_safeguard_bounds decoded "
            "each row onto a 160-point cosine grid and then interpolated onto the "
            "200-point interior grid, giving a training-population curvature "
            "maximum of 9.697e-05, while TorchSafeguard evaluates the CST basis "
            "directly at the interior grid with no interpolation, giving 8.779e-05. "
            "The two differed by 1.10x and the committed bound carried the "
            "interpolated figure. "
            "\n"
            "Fix: derive_safeguard_bounds now calls geometry.decode_surface at the "
            "interior grid directly, which is the same computation TorchSafeguard "
            "performs, so the bound is derived and enforced through one path. The "
            "n_points_per_surface argument is removed, since nothing interpolates "
            "any more. Verified: the numpy derivation path and the torch "
            "enforcement path now agree to 8.3e-17 on thickness and 5.6e-17 on "
            "curvature across all 785 training rows. "
            "\n"
            "Superseded values, recorded and not carried: thickness_lower_bound "
            "0.055425, curvature_upper_bound 0.000116. Committed values: "
            "0.055426321765484482 and 0.00010535145925184528. The curvature bound "
            "tightens by 1.10x and the thickness bound is unchanged to five "
            "decimals. No verdict changes: 0 of 785 training rows violate either "
            "bound before or after, and the safeguard term remains identically "
            "zero, so nothing downstream of B14 moves. It was fixed because a "
            "threshold derived on one measurement and applied on another is a "
            "defect whether or not it currently bites."
        ),
    ),

    "inactive_region_extent": ParamSlot(
        key="inactive_region_extent",
        treatment=Treatment.AUTHORS_CHOICE,
        note="No published derivation exists",
        disclosure_obligation=(
            "State the percentile used, the space the distance is measured in "
            "(standardised CST coefficient space), and that no published "
            "derivation procedure was found. State the rounding, if any, "
            "applied to the stored value."
        ),
        set_at="B09, the build",
        value=1.3003770283589529,
        derivation=(
            "Percentile: 90.0. No published derivation procedure exists for "
            "this quantity (the committed specification records this as a double not-found); the "
            "figure is a round, disclosed choice, not searched to produce a "
            "particular downstream separation. Chosen as wide enough to "
            "include the large majority of the avian family's own perturbed "
            "variation (the B06 sampler draws seagull-family shapes within "
            "+/-0.05 per coefficient of the seagull fit) while still "
            "excluding its own tail rather than reaching for the maximum. "
            "\n"
            "Space: standardised CST coefficient space (geometry.standardize, "
            "using B08's standardization.npz artifact -- the training-split "
            "statistics, applied identically to the reference signature and "
            "to every dataset row's own coefficients). Distance is Euclidean "
            "in that 20-column space (order-9 upper then lower coefficients "
            "concatenated). "
            "\n"
            "Reference signature: seeds/seagull.dat, fit directly at the "
            "committed CST order (9), via prior.fit_avian_reference. Single "
            "station -- see slot spanwise_resolution_avian_section. "
            "\n"
            "Region extent = the 90th percentile of the avian family's "
            "(family=='seagull', n=195) own distances to the reference, "
            "computed over all 982 dataset rows' distances via "
            "prior.derive_region_extent, restricted to the avian subset "
            "before taking the percentile (not the whole population's "
            "distances). "
            "\n"
            "Rounding applied to the stored value: NONE. Stored and recorded "
            "here exactly as np.percentile returned it: 1.3003770283589529. "
            "This is deliberate, the "
            "value the model reads is the value the paper discloses. "
            "\n"
            "Separation at this extent: 175 of 195 avian rows fall inside "
            "it (by construction of a 90th-percentile threshold on the "
            "avian subset's own distances); 0 of 787 non-avian rows fall "
            "inside it -- the four non-avian families' distances to the "
            "seagull reference are all well outside this radius (family "
            "medians: e387 4.980, s1223 8.054, sd7003 5.723, sg6043 4.351, "
            "vs. avian median 0.983 and this 90th-percentile extent of "
            "1.300), so the separation is clean on this data, not a near "
            "miss. "
            "\n"
            "Falsification check PASS (run_b09_avian_reference.py, deleted "
            "after passing): the seagull SEED row's own distance to the "
            "reference is exactly 0.0 and is the minimum over all 982 rows "
            "(expected -- it is the same fit computed twice); the avian "
            "family's median distance (0.983) is strictly below every other "
            "family's median (4.351 to 8.054); the stored extent equals a "
            "fresh recomputation of the same percentile exactly "
            "(1.3003770283589529 == 1.3003770283589529, bit for bit); "
            "recomputing the percentile from the RAW, unstandardised "
            "signature and coefficients gives a materially different value "
            "(0.14401303280911013), confirming the standardised- and "
            "raw-space computations are not being silently confused."
        ),
    ),

    "checkpoint_selection_metric": ParamSlot(
        key="checkpoint_selection_metric",
        treatment=Treatment.STATED_VALUE,
        note="Excluded terms named; weight coupling stated",
        set_at="B15, the build",
        value={
            "definition": (
                "unweighted sum of reconstruction_term (real-geometry reconstruction, "
                "decoded at the encoder's posterior mean, no sampling noise) and "
                "target_consistency_term (ensemble-mean-predicted normalised label of a "
                "freshly generated shape, at the row's own stored target and flag), "
                "evaluated on the validation split using ONE fixed set of latent codes "
                "drawn once before training and reused unchanged every epoch, so an "
                "epoch-to-epoch change in the metric reflects the model and not a "
                "different random draw"
            ),
            "excluded_terms": ["divergence", "safeguard", "spread_penalty", "avian_prior"],
            "weight_coupling": "none",
        },
        derivation=(
            "CORRECTION applied (the committed specification; carried from the build plan row Q58): the "
            "prior build scaled this metric by one of the objective weights, so the "
            "selection criterion moved whenever that weight moved. This build's metric "
            "(model.selection_metric_on / the per-epoch computation inside "
            "model.train_cvae) reads the two unweighted term values directly and sums "
            "them; it is never multiplied by divergence_weight, safeguard_weight, "
            "target_weight, spread_weight or avian_weight -- removing the coupling "
            "rather than disclosing it, "
            "\n"
            "Excluded terms and the reason: divergence, safeguard, spread and avian are "
            "all weighted regularisers or auxiliary signals, not statements about whether "
            "the reconstruction is faithful or the requested target is met; the committed specification requires "
            "any excluded term be named with a reason, done here. "
            "\n"
            "Verified as actually binding, not merely defined: B15's falsification check "
            "(run_b15_train_and_select_checkpoint.py, deleted after passing) trained three "
            "seeds at the real architecture (see latent_dimension_network_width) with "
            "TEST-ONLY objective weights (divergence 1.0, safeguard 1.0, target 1.0, "
            "spread 0.1, avian 1.0 -- not committed anywhere, B17's job after B16's real "
            "sweep) over 150 epochs each. Reloading each seed's saved best-checkpoint "
            "state and recomputing the metric on the same fixed validation codes matched "
            "the recorded best value exactly on every seed (bit-for-bit): seed "
            "1883400585 best=1.118158 (epoch 146 of 149), seed 418495999 best=1.124069 "
            "(epoch 138 of 149), seed 463744830 best=1.115153 (epoch 122 of 149); every "
            "reloaded-best value was <= the corresponding reloaded-final-epoch value "
            "(final metrics 1.149136 / 1.154572 / 1.133801); the best epoch equalled the "
            "final epoch on 0 of the 3 seeds (limit: no more than 1), so selection is "
            "actually binding on this mechanism, not defaulting to the last epoch. These "
            "numbers are mechanism evidence only, at test-only weights -- not a "
            "reportable trained model."
        ),
    ),

    "latent_dimension_network_width": ParamSlot(
        key="latent_dimension_network_width",
        treatment=Treatment.STATED_VALUE,
        note="Collapse diagnostics reported",
        set_at="B15, the build",
        value={
            "latent_dim": 8,
            "hidden_width": 64,
            "depth": 2,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "epochs": 150,
            "batch_size": 64,
            "warmup_epochs": 20,
            "liveness_threshold": 0.01,
        },
        derivation=(
            "No literature source hands down specific numbers (the committed specification requires the schedule "
            "reported and collapse diagnosed by per-dimension divergence and an active "
            "unit count, not any particular width). Chosen as a conventional, stated-value "
            "encoder/decoder MLP sized to the dataset (785 training rows, 20-dim "
            "standardised-CST geometry, 22-dim conditioning), disclosed as run procedure. "
            "This architecture is real and committed here; it is what B16's sweep and "
            "B17's final selection reuse unchanged -- only the five objective weights "
            "remain open past this step (reconstruction_divergence_weighting, "
            "prior_term_weight, target_and_spread_weights, safeguard_weight all still "
            "PENDING; see those slots). Divergence weight warmup: linear ramp of the "
            "schedule multiplier from 0 to 1 over the first 20 epochs, held at 1 "
            "thereafter (multiplies whatever divergence_weight B17 eventually commits; "
            "the schedule shape, not the weight's full value, is what is fixed here). "
            "\n"
            "Collapse diagnostics, from B15's falsification-check run at TEST-ONLY "
            "weights (see checkpoint_selection_metric's derivation for why these numbers "
            "are mechanism evidence, not a reportable result): per-dimension KL divergence "
            "on the validation split at each seed's own best checkpoint, live-dimension "
            "count against a stated liveness threshold of 0.01 nats (a disclosed, round "
            "figure, not itself drawn from a source): seed 1883400585, 5 of 8 dimensions "
            "live; seed 418495999, 3 of 8; seed 463744830, 3 of 8. Recorded as evidence the "
            "diagnostic itself runs and produces a real per-dimension breakdown, not as the "
            "paper's reportable collapse figure -- that reading is only meaningful once "
            "B17's real weights are trained."
        ),
    ),

    "surrogate_members_training_schedule": ParamSlot(
        key="surrogate_members_training_schedule",
        treatment=Treatment.STATED_VALUE,
        note="Disagreement not claimed as calibrated error",
        set_at="B13, the build",
        value={
            "member_count": 5,
            "hidden_width": 32,
            "depth": 2,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "epochs": 200,
            "batch_size": 64,
        },
        derivation=(
            "No literature source hands down specific numbers for these (the committed specification require "
            "the surrogate-in-place-of-solver choice and the disagreement framing be stated, "
            "not any particular architecture). Chosen as a conventional, stated-value MLP "
            "sized to the dataset: 785 training rows, 20-dimensional standardised-CST input, "
            "a small width/depth to avoid overfitting a set this size, disclosed as run "
            "procedure rather than justified against a source that does not speak to it. "
            "5 members, reseeded independently via dataset.rng_for(13, member_index) -> one "
            "integer draw seeding torch (model.seed_int), so every member's initialisation "
            "and batch order is reconstructible from the one recorded base seed and offset "
            "rule, same as every other randomness source in this build. "
            "\n"
            "Predicts the normalised label from geometry ALONE (the 20-column standardised "
            "CST coefficient vector) -- not from the full conditioning array, which would "
            "let the surrogate see the target it is meant to predict. "
            "\n"
            "CORRECTION applied: the held-out-error baseline is the TRAINING "
            "split's own mean label, not the full dataset's -- computing the trivial "
            "baseline over rows the ensemble will later be judged against (validation rows "
            "included) would leak validation-set information into the baseline itself. "
            "\n"
            "Run (run_b13_surrogate_ensemble.py, deleted after passing; reusable logic kept "
            "in model.py's train_surrogate_member / build_surrogate_ensemble): trained on "
            "the 785-row training split (B12's split.npz), evaluated on the 197-row "
            "validation split, both in RAW label units (max L/D). Per-member MAE: "
            "2.9423, 3.0782, 3.2067, 3.1747, 2.8300. Ensemble mean MAE: 2.6538. Training-"
            "split-mean baseline MAE: 13.1704. Full label range: 121.6752. Per-sample "
            "spread across members on validation rows: min 0.0034, mean 0.0144, max 0.0335 "
            "(normalised-prediction units); 0 of 197 rows had exactly zero spread. "
            "\n"
            "Falsification check PASS: fraction of validation rows with exactly zero "
            "spread (0.0) is within the stated small-fraction limit; ensemble MAE (2.6538) "
            "is materially below the training-split-mean baseline (13.1704); ensemble MAE "
            "does not exceed the full label range (121.6752), which is what a normalised/"
            "raw unit confusion would produce. Frozen ensemble written to "
            "surrogate_ensemble.pt; not retrained between arms, per B13's own text."
        ),
    ),

    "solver_operating_point_settings": ParamSlot(
        key="solver_operating_point_settings",
        treatment=Treatment.STATED_VALUE,
        note="Full recurring set reported",
        set_at="B04",
        # Values and their justification supplied directly by the author
        # (not derived here, not carried from the superseded
        # build). the committed specification requires the set be reported with a recognised
        # justification; it does not hand down specific numbers. Recorded
        # verbatim below, attributed to the author, since the proxy-panel-
        # study claims it cites were not run here and are not
        # independently verified here.
        value={
            "reynolds": 300000.0,
            "mach": 0.0,
            "ncrit": 9.0,
            "alpha_start": 0.0,
            "alpha_end": 8.0,
            "alpha_step": 1.0,
            "n_panels": 220,
            "iter_limit": 400,
        },
        derivation=(
            "Author-supplied, with justification, 2026-08-06. Quoted as given: "
            "reynolds=300000 because it must sit above the transitional regime "
            "the committed specification's cited sources flag XFOIL as unreliable in (Re 68,000-159,000, "
            "Preprints.org 202605.0263), must support attached-flow cruise (the "
            "locked regime), and for seagull-class sections at plausible cruise "
            "speeds the relevant band is Re 200,000-500,000; a proxy panel study "
            "(NACA 2412, Re=1e6, N=220) showed transition-model sensitivity at "
            "the high end, Re=1e5 sits at the transitional extreme the committed specification flags as "
            "least reliable, and 300,000 is mid-band with full proxy convergence. "
            "mach=0.0, incompressible: at Re 300,000 and seagull-cruise speeds "
            "the flow sits well below M=0.1, standard across every comparable "
            "source read for the committed specification. ncrit=9.0, 'typical wind tunnel conditions' "
            "per ScienceDirect S259012302600544X (cited at the committed specification); the author "
            "states no source in the the committed specification citation set uses a different value. "
            "n_panels=220, from an author-run empirical panel-convergence study "
            "(NACA 2412, Re=500,000): N=140 gave only 8/9 converged angles, "
            "N=180 and above gave 9/9 with |dCL|<1e-3 and |dCD|<5e-5 relative to "
            "N=400; 220 is chosen with margin above the N=180 threshold, inside "
            "Kallstrom SJSU thesis Sec 3.2's cited 160-400 plateau range, and "
            "close to the 201-cosine-point precedent the committed specification cites (S259012302600544X). "
            "iter_limit=400, stated as a safety margin only (proxy shapes "
            "converge in under 50 iterations), consistent with every the committed specification-cited "
            "source. alpha_start/end/step = 0/8/1 (9 points), to cover the "
            "attached-flow cruise range for bird-inspired sections while staying "
            "below the stall onset the author expects for the B03 seed family's "
            "thickness and camber at this Reynolds number. "
            "Independently confirmed in this build, not merely asserted: "
            "running the committed command stream and these exact settings "
            "(order-7 test-only CST fit, not the committed order) against the "
            "real seeds/e387.dat file through the real XFOIL 6.99 binary in "
            "this repo gave 9 of 9 requested angles converged. Separately, "
            "sweeping the same seed to alpha=14 (outside the committed range) "
            "made XFOIL hang indefinitely with no distinguishing output before "
            "the hang, which is the concrete failure mode the committed alpha "
            "range and the timeout-kill mechanism in solver.py both exist to "
            "avoid. The proxy-shape convergence and panel-sensitivity claims "
            "above (NACA 1408/2412/4409, N=140/180/400 comparisons) are the "
            "author's own reported figures and were not independently rerun in "
            "this build."
        ),
    ),

    "per_call_timeout": ParamSlot(
        key="per_call_timeout",
        treatment=Treatment.RANGE_WITH_RULE,
        note="Committed after the solve time distribution was observed on the "
        "real, population-scale B06 output, per the committed specification. "
        "REBUILT during the build after cst_order changed 5->9 (see "
        "that slot); the order-5 timing pass is superseded outright and "
        "archived, not read, in "
        "b05_population_progress_order5_superseded.jsonl.",
        rule_text=(
            "A stated upper percentile of successful solve times, with a stated "
            "margin. Committed only after the solve-time distribution has been "
            "observed and the reclassification count the new value implies has "
            "been recorded (the committed specification; B05 logic). Rule as applied here: the 95th "
            "percentile of successful solve times, times a stated 2.0x margin -- "
            "unchanged since the build; the rule itself was fixed before any "
            "population existed, only the data it is applied to has changed, "
            "twice now (5-seed partial, order-5 population, order-9 population)."
        ),
        set_at="B05 (population-scale, the build, order-9 rebuild)",
        value=7.56,  # seconds.
        derivation=(
            "SUPERSEDED: the order-5 population-scale result (13.47s) no "
            "longer applies to anything -- population.npz itself was "
            "rebuilt at order 9 with a different perturbation width (0.05, "
            "was 0.07), so its solve-difficulty distribution is different "
            "data, not a revision of the same data. "
            "\n"
            "Method (run_b05_population_pass.py + finalize_b05_population.py), "
            "unchanged: ran solver.py on all 1005 rows of the order-9 "
            "population.npz, at the committed B04 operating point, 60s "
            "instrumentation ceiling. Resumable via "
            "b05_population_progress.jsonl (the order-5 file was archived "
            "first, not overwritten); ran to completion (1005 of 1005) in "
            "one background pass. "
            "\n"
            "Status counts (1005 total): converged 862, partially_converged "
            "129, timeout 10, failed 4, environment_fault 0. 991 of 1005 "
            "('successful') form the distribution the percentile rule is "
            "applied to. Compared to the order-5 population: fewer timeouts "
            "(10 vs 19) and fewer partial convergences (129 vs 164) -- the "
            "order-9 population is, on this evidence, somewhat easier for "
            "the solver than order-5's was, though both still show a real "
            "tail the clean 5-seed sample never could. "
            "\n"
            "Distribution (successful solves only, n=991): p50=0.8682s "
            "p75=1.3379s p90=3.3741s p95=3.7811s p99=6.9856s min=0.5230s "
            "max=11.7853s. Per family (n, p95, max): e387 n=196 p95=6.6173s "
            "max=11.7853s; s1223 n=200 p95=3.3171s max=4.1969s; sd7003 "
            "n=199 p95=3.7229s max=9.0261s; seagull n=195 p95=3.5688s "
            "max=5.3487s (avian family); sg6043 n=201 p95=3.4587s "
            "max=6.9758s. "
            "\n"
            "Committed value: p95 (3.7811s) x 2.0 margin = 7.5623s, rounded "
            "7.56s. "
            "\n"
            "Reclassification count at the committed value: 4 of 991 "
            "successful solves (0.40%) would newly read as timeouts -- 3 "
            "e387, 1 sd7003, 0 seagull (avian), 0 s1223/sg6043. Full "
            "per-candidate table is in finalize_b05_population.py's output, "
            "retained in BUILD_LOG.md. The 10 already-timed-out-at-60s rows "
            "stay timeouts at any candidate below 60s and are not part of "
            "this count. "
            "\n"
            "Fairness check against the committed specification's stated risk (same "
            "framing as the order-5 result): 0 of 195 avian-family "
            "(seagull) successful solves reclassified vs. 4 of 796 "
            "non-avian (0.50%) -- again not evidence of a family-selective "
            "effect at this rate and sample size, and again not a clearance "
            "of the real prior-on/off risk that only exists once B23 runs."
        ),
    ),

    "minimum_converged_sweep_points": ParamSlot(
        key="minimum_converged_sweep_points",
        treatment=Treatment.RANGE_WITH_RULE,
        note="No inherited value; tolerance stated first",
        rule_text=(
            "The tolerance on truncation bias is stated first. The smallest "
            "minimum point count whose mean bias and upper-percentile absolute "
            "bias both sit inside that stated tolerance is then chosen "
            "mechanically (the committed specification; B20 logic). The prior build's threshold does "
            "not carry forward; the build plan retired it and no published basis exists. "
            "\n"
            "The tolerance itself is the module-level constant "
            "B20_TRUNCATION_TOLERANCE in this file, written before the analysis was "
            "run and before any candidate count's bias was computed. This slot does "
            "not restate it; there is one copy."
        ),
        set_at="B20 (the build)",
        value={
            "minimum_converged_points": 8,
            "sweep_length": 9,
            "mean_relative_bias_tolerance": 0.010,
            "upper_percentile": 95.0,
            "upper_percentile_tolerance": 0.020,
        },
        derivation=(
            "TOLERANCE FIRST, COUNT SECOND, and the order is enforced by where the "
            "numbers live rather than asserted. B20_TRUNCATION_TOLERANCE was written "
            "into this file before analysis.py existed and before any bias was "
            "computed. run_b20_truncation_analysis.py reads it and defines none of "
            "it, and analysis.select_minimum_point_count takes the tolerance and the "
            "table as its only two inputs. The count is what the rule returned. "
            "\n"
            "Population: every FULLY converged sweep in the dataset, 862 of the 982 "
            "labelled rows, each 9 points. Per family: e387 152, s1223 187, sd7003 "
            "173, seagull 169, sg6043 181. Partially converged sweeps are excluded, "
            "because the bias at count k is defined against the whole sweep and a "
            "sweep that never had a whole sweep supplies no reference. "
            "\n"
            "Bias at candidate k, for one sweep: (max CL/CD over the first k "
            "converged points) minus (max CL/CD over the whole sweep), divided by the "
            "latter. Relative, because the labels span 50.499 to 172.174 and a fixed "
            "L/D allowance would be a different requirement at the two ends. The "
            "quantity is at most zero, since truncating a maximum can only lower it, "
            "so the mean signed and mean absolute figures agree in magnitude at every "
            "k and both are recorded. "
            "\n"
            "Bias table (k, mean signed, mean absolute, p95 absolute, worst absolute, "
            "sweeps whose peak already lies inside the first k): "
            "1, -0.368755, 0.368755, 0.716993, 0.829792, 5/862; "
            "2, -0.240253, 0.240253, 0.519406, 0.656807, 20/862; "
            "3, -0.153196, 0.153196, 0.367939, 0.508313, 54/862; "
            "4, -0.091236, 0.091236, 0.259989, 0.376689, 115/862; "
            "5, -0.046484, 0.046484, 0.174702, 0.279216, 291/862; "
            "6, -0.017436, 0.017436, 0.099722, 0.172698, 545/862; "
            "7, -0.005302, 0.005302, 0.044219, 0.121067, 725/862; "
            "8, -0.001786, 0.001786, 0.013314, 0.066083, 796/862; "
            "9, 0.000000, 0.000000, 0.000000, 0.000000, 862/862. "
            "\n"
            "SELECTED: 8. It is the smallest k satisfying both tolerances. k=7 "
            "satisfies the mean (0.005302 <= 0.010) and fails the upper percentile "
            "(0.044219 > 0.020), so the tail is what binds here and the mean alone "
            "would have returned a smaller count. k=8 satisfies both (0.001786 and "
            "0.013314). Full table in b20_truncation.json. "
            "\n"
            "What the committed count costs, stated rather than left to be inferred: "
            "at k=8 the worst single sweep in the population is still truncated by "
            "6.6 percent of its own label, which is outside both tolerances and is "
            "not claimed to be inside either. The tolerance is on the mean and the "
            "95th percentile by construction, not on the worst case, and the worst "
            "case is recorded for exactly that reason. "
            "\n"
            "Peak angle distribution, recorded alongside per B20's own logic text "
            "(alpha in degrees at which each sweep attains its max CL/CD, over the "
            "862 sweeps): 0 deg 5 (0.6%), 1 deg 15 (1.7%), 2 deg 34 (3.9%), 3 deg 61 "
            "(7.1%), 4 deg 176 (20.4%), 5 deg 254 (29.5%), 6 deg 180 (20.9%), 7 deg "
            "71 (8.2%), 8 deg 66 (7.7%). The peak sits at 4 to 6 degrees for 71 "
            "percent of sweeps and in the last two angles for 16 percent, which is "
            "why the tail bias falls away slowly and why the count lands high. "
            "\n"
            "Consequence for the dataset, computed after the count was selected and "
            "with no part in selecting it: at a minimum of 8 converged points, 975 of "
            "the 982 labelled dataset rows clear it (862 at 9 points and 113 at 8), "
            "and 7 rows at 7 points do not. "
            "\n"
            "Falsification check PASS (check_b20_truncation_analysis.py, deleted "
            "after passing). All three clauses hold on the committed table: bias at "
            "k=1 (0.368755) is materially larger than at the full sweep length "
            "(0.000000); mean absolute bias is non-increasing across the whole range; "
            "bias at the full sweep length is exactly 0.0 on both the mean and the "
            "worst case. A fourth clause was added beyond what the build plan asks, because "
            "the first three are close to structural on a correct implementation and "
            "a check that cannot fail is not evidence: the wrong-but-wired table was "
            "built deliberately, taking the reference from the truncated set instead "
            "of the whole sweep, and the same three clauses were run against it. They "
            "failed there, on clause 1, with a bias of exactly 0.000000 at every "
            "candidate. The clauses separate the two."
        ),
    ),

    "plausibility_filter_criterion": ParamSlot(
        key="plausibility_filter_criterion",
        treatment=Treatment.STATED_VALUE,
        note="Difference from the closest source stated",
        set_at="The build, at the author's direction, after standing UNRESOLVED "
        "from B01. the committed specification concerns a POST-convergence plausibility check on "
        "solver output, distinct from the committed specification's PRE-solver geometric filter that "
        "B03 builds, and no step's Specification entries list in the "
        "reviewed plan (B01 through B26) cites the committed specification. It is set at the last "
        "step that can precede the pre-registration rather than left for a "
        "step that does not exist. Implemented in "
        "analysis.physical_plausibility and applied inside "
        "evaluate.evaluate_coefficients. Recorded as a departure from the "
        "plan's own step assignment, since the build plan assigns it to nothing.",
        value={
            "criterion": "per converged point: CD strictly positive, and both "
            "transition locations within [0, 1] chord inclusive",
            "applied_to": "every converged point of every sweep, before the label's "
            "maximum is taken",
            "affected_points": 0,
            "points_examined": 8711,
            "affected_shapes": 0,
            "shapes_examined": 982,
        },
        derivation=(
            "CRITERION. On every point XFOIL reported as converged, require drag "
            "strictly positive and both transition locations inside [0, 1] chord. A "
            "point failing either is dropped before the label is computed, so it "
            "cannot be selected by the label's own maximum. Exactly 0.0 and exactly "
            "1.0 are both admitted for transition, because 1.0 is how the polar dump "
            "records no transition with laminar flow to the trailing edge, which is "
            "an ordinary result at this Reynolds number and not a fault. "
            "\n"
            "Why the test is on the points and not on the finished label: both "
            "failures land INSIDE the label rather than beside it. Drag at or below "
            "zero from a viscous solve is not a small drag, it is a non-physical "
            "result, and it produces an unbounded or negative lift-to-drag ratio that "
            "a maximum would preferentially select. A filter applied after the "
            "maximum would be filtering the consequence. "
            "\n"
            "AFFECTED RATE, which the committed specification requires reported: ZERO. Measured over all "
            "8711 converged points across the 982 labelled dataset rows, 0 points "
            "fail either clause and 0 shapes are affected. Independently confirmed "
            "on the live evaluation path rather than on stored data alone: B19's "
            "gate one was re-run after this filter was wired into "
            "evaluate.evaluate_coefficients, and all 25 gate rows still reproduced "
            "their stored labels at exactly 0.000000 relative difference, so the "
            "filter moved no label. "
            "\n"
            "The filter is therefore INERT on this build and is reported as inert, "
            "the same discipline the safeguard term is reported under. Its criterion "
            "is NOT loosened or tightened to make it bind. Choosing a threshold so "
            "that a term starts binding is calibration against an outcome, which is "
            "the specific failure B18 exists to prevent. "
            "\n"
            "THE DIFFERENCE FROM THE CLOSEST SOURCE, which is exactly what the committed specification "
            "requires when the closer criterion is not used. the committed specification resolves the "
            "closest source to NeuralFoil (arXiv:2503.16323 section E), which "
            "filters on solver internals: negative momentum thickness, a shape "
            "factor below its physical floor, and non-physical edge velocities, at a "
            "reported 0.03 percent affected rate. This build does not test any of "
            "those three. XFOIL's accumulated polar dump carries alpha, CL, CD, CDp, "
            "CM and the two transition locations, and no boundary layer internal. "
            "Reaching them needs a different XFOIL output mode, a second parser, and "
            "a re-run of the dataset. The criterion used here is therefore WEAKER "
            "than the closest source's, and it is stated as weaker rather than "
            "presented as equivalent. It cannot catch a converged solution whose "
            "boundary layer state is unphysical while its drag and transition "
            "locations both look ordinary, and the committed specification names that failure by name. "
            "\n"
            "WHAT WAS NOT RETAINED, and why. The superseded build filtered on a band "
            "on the output ratio. That value is void under this rebuild, and the committed specification "
            "states in its own words that a band on the output ratio does not catch "
            "a converged solution with unphysical boundary layer state. Retaining it "
            "would have carried an inherited number and bought nothing the criterion "
            "above does not, so it is not carried."
        ),
    ),

    "requested_target_band": ParamSlot(
        key="requested_target_band",
        treatment=Treatment.STATED_VALUE,
        note="Reported normalised",
        set_at="The build, at the author's direction, after standing UNRESOLVED "
        "from B01. No step in the reviewed plan names where the band is "
        "chosen. B21's pilot and B23's full run both consume 'every "
        "requested target' and B22's pre-registration must disclose it, so "
        "it is set at the last step that can precede the pre-registration "
        "rather than left for a step that does not exist. Recorded as a "
        "departure from the build plan's own step assignment, since the build plan "
        "assigns it to nothing.",
        value={
            "normalised_low": 0.08761877789267145,
            "normalised_high": 0.5522084150555188,
            "raw_low": 61.15975877192982,
            "raw_high": 117.68878896594654,
            "n_targets": 11,
            "spacing": "evenly spaced, both endpoints inclusive",
            "rule": "the 5th to 95th percentile of the TRAINING SPLIT's own "
            "normalised labels",
        },
        derivation=(
            "DERIVED FROM THE TRAINING DATA, NOT FROM ANY EVALUATION OUTCOME, and "
            "the distinction is the point of the choice. The band is the 5th to 95th "
            "percentile of the training split's own normalised labels, over the same "
            "785 rows B08's normalisation was derived on. Stored unrounded, so the "
            "value the generator reads is the value the paper discloses. Reported "
            "normalised per the committed specification, with the raw equivalents alongside and not in place "
            "of them. "
            "\n"
            "WHY THIS RULE. Requesting a target the training set has no examples of "
            "is extrapolation, and whether it is extrapolation is a property of the "
            "dataset, knowable before anything is generated or solved. The training "
            "label distribution is strongly lopsided and the tail is nearly empty: "
            "p50 sits at 0.3314 normalised, p95 at 0.5522, p99 at 0.5787, and p100 "
            "at 1.0000. Only 2 of the 982 labelled rows lie above 0.70 normalised. "
            "The top thirty percent of the normalised scale contains two shapes. A "
            "band running to 1.0 would spend a third of its requested range on a "
            "region represented by two training examples. "
            "\n"
            "The 5 percent trim at each end is a round figure and is this study's "
            "own. No source supplies one. It is symmetric, and it is applied to the "
            "training split rather than the full dataset for the same reason B08's "
            "row set is the training split: a band set from rows the model is later "
            "validated on would put validation information into the conditioning "
            "range. "
            "\n"
            "CONSISTENCY WITH B21's PILOT, which is corroboration and explicitly NOT "
            "the derivation. The first pilot ran on an 11-point grid over the full "
            "[0, 1] range and admitted 10 of 10 pairs at every target from 0.10 to "
            "0.60, 5 of 10 at 0.70, 1 of 10 at 0.80, and 0 of 10 at both 0.90 and "
            "1.00. This band falls entirely inside the region where that pilot "
            "admitted nearly everything. The two agree. The band was not selected to "
            "make them agree: the percentile rule reads only the training labels, "
            "and it would have returned this same band had the pilot never run. The "
            "agreement is worth stating because it identifies the cause of the "
            "pilot's dead region as extrapolation beyond the training support rather "
            "than as an unexplained solver behaviour. "
            "\n"
            "DISCLOSURE THAT TRAVELS WITH THIS BAND, and it is not a footnote. This "
            "study never requests a target above 117.689 raw max(CL/CD), against a "
            "dataset maximum of 172.174 and a seed-population maximum in the same "
            "place. No result here speaks to behaviour at the top of the efficiency "
            "range, and no reader should be able to take the study as having probed "
            "it. The single extreme converged result at 172.174 is retained in the "
            "training data per the committed specification, with the solver mechanism named as the caveat "
            "and no outlier test applied, and it sits outside the requested band. "
            "\n"
            "11 targets, evenly spaced with both endpoints inclusive. The count "
            "matches the grid B16, B18 and B21 already generate on, so this build "
            "carries one target count rather than two. It is also the cluster count "
            "for the committed specification's purposes, and 11 sits inside Cameron, Gelbach and Miller's "
            "five to thirty few-clusters band, which is why the cluster unit and "
            "bootstrap method slot commits a bootstrap-t refinement rather than a "
            "bare percentile cluster bootstrap. "
            "\n"
            "CONSEQUENCE, discharged rather than noted. B21's first pilot measured a "
            "pair yield of 0.627273 on the full [0, 1] grid and derived a launch "
            "target of 160 pairs from it. That yield is not the yield of this band. "
            "B21 was re-run on this committed band in the same build pass and the "
            "launch target re-derived from the new measurement; see slot "
            "samples_per_target. The 160-pair figure is superseded outright and is "
            "not carried."
        ),
    ),

    "samples_per_target": ParamSlot(
        key="samples_per_target",
        treatment=Treatment.RANGE_WITH_RULE,
        note="Derived from the floor by yield inflation",
        rule_text=(
            "Derived from the analysed-pair floor by yield inflation, using the "
            "pair yield measured in a pilot run of this pipeline rather than a "
            "published rate (the committed specification; B21 logic)."
        ),
        set_at="B21 (the build), re-run on the committed requested target band "
        "after that band was set later in the same build pass. Both the "
        "launch target in pairs and the per-target sample count are "
        "committed.",
        value={
            "launch_target_pairs": 101,
            "n_requested_targets": 11,
            "samples_per_target": 10,
            "launched_pairs_at_that_grid": 110,
            "floor_analysed_pairs": 100,
            "measured_pair_yield": 0.9909090909090909,
            "pair_yield_standard_error": 0.009041851079766345,
            "inflation_factor": 1.0091743119266054,
        },
        derivation=(
            "DERIVED, NOT CHOSEN. launch_target_pairs = ceil(floor / measured pair "
            "yield) = ceil(100 / 0.990909) = 101 pairs. The floor was written into "
            "this file before any pilot ran, and the yield was measured by the "
            "pilot. Neither was adjusted after the other was known. "
            "\n"
            "samples_per_target = 10, from ceil(101 / 11) at the committed band's 11 "
            "targets, which launches 110 pairs. That is 9 pairs above the derived "
            "101 and it is the smallest whole number of samples per target that "
            "reaches the launch target, since 9 per target would launch only 99. The "
            "excess is a consequence of an integer grid and is stated rather than "
            "trimmed by launching an uneven number of samples at different targets, "
            "which would unbalance the clusters the interval is resampled over. "
            "\n"
            "MEASURED PAIR YIELD 0.990909, at the pair level, from 109 of 110 pilot "
            "pairs in which BOTH members cleared admission. It is not derived from "
            "the single-shape admission rate, which was 0.995455 (219 of 220 shapes) "
            "on the same pilot. Standard error 0.0090 on 110 pairs. "
            "\n"
            "Recorded honestly about what the check could show at this yield: the two "
            "rates differ by 0.004545 here, and both round to the same 101-pair "
            "launch target, so the pair-level measurement did not change the answer "
            "on this band. It is still measured at the pair level, because whether "
            "the two levels agree is a property of the data and not something the "
            "method may assume. On the FIRST pilot they did not agree, and the "
            "difference there was 20 pairs. "
            "\n"
            "Pilot: the committed model (committed_model.pt, B18), the 11 targets of "
            "the committed band, 10 samples each, one latent code per (target, "
            "sample) passed to both arms, generation seed 1961077683 from "
            "dataset.rng_for(21, 0) via model.seed_int. Admission per B20 and the committed specification: a "
            "label present AND at least 8 of 9 angles converged and surviving the "
            "physical plausibility filter. Full per-shape records in "
            "b21_paired_yield.json. "
            "\n"
            "SUPERSEDED OUTRIGHT, recorded and not carried. An earlier pilot in the "
            "same build pass ran on the [0, 1] diversity grid, before "
            "requested_target_band was committed. It measured a pair yield of "
            "0.627273 and derived a launch target of 160 pairs. That yield belongs to "
            "that grid. It was a mixture of a near-perfect regime below 0.60 "
            "normalised and a near-total-failure regime at 0.90 and above, where the "
            "prior-on arm timed out on 10 of 10 shapes at each of the top two "
            "targets. Committing the band removed the extrapolation region that "
            "produced those failures, and the pilot was re-run on the band rather "
            "than the old figure being reused. The 0.627273 and the 160 are void and "
            "appear nowhere outside this note and BUILD_LOG.md. "
            "\n"
            "DIFFERENTIAL ATTRITION BETWEEN ARMS (M19, the committed specification), on the committed band: "
            "prior-on admitted 110 of 110 shapes, prior-off 109 of 110. Exactly one "
            "pair was lost, and it was lost on the prior-OFF member, a partially "
            "converged shape at 7 usable points against a minimum of 8. Zero "
            "timeouts, zero plausibility rejections and zero environment faults in "
            "either arm. Zero points were dropped by the committed specification's physical plausibility "
            "filter in either arm. "
            "\n"
            "This bears directly on a risk carried since B05 and it revises what the "
            "first pilot appeared to show. On the [0, 1] grid the arms looked "
            "strongly asymmetric, 71 of 110 against 87 of 110, with 34 prior-on "
            "timeouts against 20. That asymmetry was concentrated entirely at "
            "targets outside the training support, and it does not survive the move "
            "to the committed band. The reading carried forward is that the apparent "
            "arm effect on attrition was an extrapolation effect, not a prior effect. "
            "It is NOT a clearance. The pilot is 110 pairs, one lost pair cannot "
            "distinguish equal rates from slightly unequal ones, and M19's own "
            "guidance caution is that equal rates do not establish the absence of "
            "bias any more than unequal rates establish its presence. The real test "
            "is M19 computed on B23's full run. "
            "\n"
            "Falsification check PASS (check_b21_paired_yield.py, re-run against this "
            "pilot and deleted after passing; a superseded run's passing check is not "
            "evidence about this one). Both clauses, from an independent plain "
            "recount over the stored per-shape records: the recorded pair yield "
            "equals the direct count of both-admitted pairs over pairs launched, "
            "exactly (0.9909090909); and the recorded pair yield is not above the "
            "single-shape admission rate (0.9909090909 against 0.9954545455). Noted "
            "with the result: at this yield clause 2 has little room to bite and "
            "passing it is weak evidence on its own, which is a property of a "
            "near-perfect pilot and not of the check. Clause 1 is the load-bearing "
            "one here and it is exact."
        ),
    ),

    "analysed_pair_floor": ParamSlot(
        key="analysed_pair_floor",
        treatment=Treatment.STATED_VALUE,
        note="Expressed in analysed pairs",
        set_at="B21 (the build). Written into this file BEFORE the pilot was run "
        "and before any pair yield was measured. The floor is a stated "
        "value and must not move with the yield; only the launch target "
        "derived from it may.",
        value=100,
        derivation=(
            "100 ANALYSED PAIRS. Expressed in analysed pairs, per the committed specification, not in "
            "launched runs and not in shapes. "
            "\n"
            "JUSTIFICATION TYPE, named as the committed specification requires: PRECISION BASED. the committed specification names "
            "three types, power based, precision based and convention based, and "
            "records the precision based form as the methodologically recommended one "
            "for simulation work, with a worked formula in Morris, White and Crowther "
            "2019 doi:10.1002/sim.8086 section 5.2. This floor uses that formula. "
            "\n"
            "The formula is n = (SD / target Monte Carlo standard error) squared, for "
            "a mean estimate. It needs an SD for the paired difference distribution, "
            "and no such estimate exists before evaluation. Inventing one is not "
            "available. So the target is stated RELATIVE to the SD instead of in "
            "absolute units: require the Monte Carlo standard error of the primary "
            "location statistic to be at most a stated fraction f of the paired "
            "difference distribution's own standard deviation. The SD then cancels "
            "and n = 1 / f squared, which needs no estimate of anything. "
            "\n"
            "f = 0.10, so n = 100. The 0.10 is a round figure and is this study's own. "
            "R9 not-found 1 records that no numeric floor for analysed pairs exists in "
            "generative design, so there is nothing to adopt and every constant here "
            "would have been stated whichever route was taken. What f buys is legible: "
            "at f = 0.10 the resampling noise on the reported location statistic is a "
            "tenth of the spread of the thing being located. "
            "\n"
            "LIMITATION THAT TRAVELS WITH THIS FLOOR, stated rather than assumed away. "
            "The formula above is the independent-observations one. This design "
            "clusters by requested target, and the build plan row Q119 already records "
            "the cluster count as sitting inside Cameron, Gelbach and Miller's five to "
            "thirty few-clusters band. Whenever the between-cluster component is "
            "non-zero, 100 analysed pairs buy LESS precision than the formula implies, "
            "and the shortfall grows with that component. The floor is therefore a "
            "floor and not a guarantee of the stated precision. What the paper reports "
            "is the cluster bootstrap interval at B24, which measures the realised "
            "precision directly, so a shortfall is visible in the reported interval "
            "rather than hidden inside an assumption. No correction factor is applied "
            "here, because applying one would need the intra-cluster correlation, "
            "which is the same unknown quantity the SD was, and guessing it would put "
            "an invented number inside a pre-registered floor. "
            "\n"
            "The floor is a floor. the committed specification govern extension above it, and B23's "
            "own completion condition names the extension procedure that applies if it "
            "is not reached."
        ),
    ),

    "cluster_unit_bootstrap_method": ParamSlot(
        key="cluster_unit_bootstrap_method",
        treatment=Treatment.STATED_VALUE,
        note="Small cluster case addressed",
        set_at="The build, ahead of B22, and NOT at B24 as this file previously "
        "recorded. the committed specification's own treatment text says the method is 'chosen "
        "before evaluation', and B24 runs after B23. Recording it as B24 "
        "would have placed the choice after the evaluation it governs, "
        "which is the thing the committed specification forbids. The step assignment is corrected "
        "here and the correction is recorded in KNOWN_DISCREPANCIES rather "
        "than made silently. B24 still IMPLEMENTS the estimator; it no "
        "longer chooses it.",
        value={
            "cluster_unit": "the requested target",
            "n_clusters": 11,
            "method": "wild cluster bootstrap-t with Rademacher weights",
            "also_reported": "the unrefined percentile cluster bootstrap",
            "distinct_weight_vectors": 2048,
        },
        derivation=(
            "CLUSTER UNIT: the requested target. Every pair generated at one "
            "requested target shares that target, and the paired difference at that "
            "target is not independent of its neighbours in the same cluster. The "
            "committed requested target band has 11 targets, so there are 11 "
            "clusters. "
            "\n"
            "METHOD: the wild cluster bootstrap-t with Rademacher weights, which is "
            "the refinement the foundational source names. Cameron, Gelbach and "
            "Miller 2008 (doi:10.1162/rest.90.3.414) define few clusters as five to "
            "thirty, report that standard asymptotic cluster robust tests can over "
            "reject considerably in that range, and report rejection rates of about "
            "10 percent against a nominal 5 percent reducible to nominal by exactly "
            "this refinement. 11 clusters sits inside that band, so the refinement is "
            "required rather than optional here. "
            "\n"
            "the build plan row Q119 records the superseded build's estimator as a "
            "percentile cluster bootstrap with NO refinement, which is the specific "
            "construction the committed specification names. That estimator is still computed and reported "
            "alongside, because the difference between a refined and an unrefined "
            "interval on this data is itself informative and the committed specification requires a "
            "disagreement between two measures to be presented rather than resolved "
            "by picking one. The refined interval is the one the primary claim rests "
            "on. "
            "\n"
            "LIMITATION, stated rather than discovered later. With 11 clusters, "
            "Rademacher weights admit 2 to the 11th = 2048 distinct weight vectors "
            "in total. The bootstrap distribution is therefore supported on at most "
            "2048 points however many resamples are drawn, which bounds the "
            "achievable resolution of a p-value and of the interval endpoints "
            "regardless of the resample count committed at slot resample_count. This "
            "is a property of the cluster count and the weight distribution, not of "
            "the implementation, and it is reported with the interval. "
            "\n"
            "The whole-cluster resampling requirement the committed specification also imposes is what B24's "
            "own falsification check tests, and that check is written against the "
            "specific failure of drawing clusters and then resampling pairs inside "
            "them, which produces an interval of the right shape that is too narrow."
        ),
    ),

    "primary_statistic": ParamSlot(
        key="primary_statistic",
        treatment=Treatment.STATED_VALUE,
        note="Committed before the distribution is seen",
        set_at="B22 (the committed specification requires commitment before the difference "
        "distribution is known; B22 is where the pre-registration "
        "is issued and the committed specification is among its listed specification "
        "entries; B24 builds the analysis machinery but the committed specification is not "
        "in B24's own specification-entries list). Committed in "
        "the build, ahead of B22, so the pre-registration transcribes "
        "it rather than deciding it. No evaluation outcome has been "
        "analysed at the point of commitment.",
        value={
            "statistic": "the arithmetic mean of the M01 paired differences",
            "quantity": "for each matched pair, the prior-off arm's absolute target "
            "satisfaction error minus the prior-on arm's, both in "
            "normalised units, oriented so a positive value favours the "
            "prior",
            "secondary": ["M05 paired median difference", "M07 paired win fraction"],
            "sensitivity_only": ["M06 trimmed mean difference"],
        },
        derivation=(
            "THE MEAN of the M01 paired differences. Exactly one outcome is "
            "designated primary for inference, per the committed specification, and every other statistic is "
            "labelled secondary or post hoc against it, per the committed specification. "
            "\n"
            "WHY THE MEAN, and the reason is internal consistency rather than "
            "preference. The analysed pair floor is precision based and was computed "
            "from Morris, White and Crowther 2019 section 5.2's formula, n = (SD / "
            "target Monte Carlo standard error) squared. That is a formula for the "
            "Monte Carlo standard error of a MEAN. Committing a median or a trimmed "
            "mean as primary would leave the floor justifying the precision of a "
            "statistic the paper does not report, and the pre-registration would "
            "then contain a sample size argument that does not apply to its own "
            "primary outcome. "
            "\n"
            "The other two families the committed specification names are retained and labelled. the build plan "
            "already places the median at M05 as a convention-tier secondary and the "
            "trimmed mean at M06 as a sensitivity statement, and the committed specification forbids "
            "promoting M06 to the headline. M04's distribution shape figures exist "
            "precisely so a divergence between the mean and the robust measures can "
            "be explained rather than adjudicated, which is the committed specification's requirement. "
            "\n"
            "WHAT the committed specification REQUIRES STATED, and it is stated: no rule, formula or "
            "threshold exists for choosing between a mean and a robust location "
            "statistic from information available before data collection. R10 "
            "question 3 is a clean not-found on exactly that. The choice above is "
            "therefore made on a stated internal-consistency ground and not on a "
            "published decision rule, and it is not presented as following practice. "
            "\n"
            "the committed specification governs what happens if the committed primary turns out to fit the "
            "realised distribution poorly. The deviation is reported with the "
            "original plan disclosed alongside it, and is never silently "
            "substituted. That is a reporting rule, not an escape hatch: the mean "
            "stays the primary and a robust measure does not quietly take its place."
        ),
    ),

    "interval_level": ParamSlot(
        key="interval_level",
        treatment=Treatment.STATED_VALUE,
        note="",
        set_at="B24 (the committed specification's own text concerns interpreting an interval that "
        "spans zero, not the confidence level itself; the build plan's "
        "table cites the committed specification as this row's governing entry regardless. "
        "Recorded as given; the apparent mismatch is noted, not "
        "resolved.) Committed during the build, ahead of B22, so the "
        "pre-registration transcribes the level rather than leaving "
        "it to a step that runs after the evaluation it describes.",
        value=0.95,
        derivation=(
            "95 percent. A conventional level, committed and disclosed AS "
            "conventional rather than presented as derived. No source in the corpus "
            "states a confidence level for this kind of interval, and the committed specification's own text "
            "concerns how an interval spanning zero is READ rather than how wide it "
            "is. There is nothing to derive from, and inventing a non-standard level "
            "would cost the reader a familiar reference point and buy nothing. "
            "\n"
            "It sets the endpoints of M02, the cluster-resampled interval on the "
            "primary, and of the same estimator's application to M05 and M06. One "
            "level throughout; the paper does not report a 95 percent interval on "
            "one measure and a different level on another. "
            "\n"
            "THE READING RULE TRAVELS WITH THE LEVEL, per the committed specification, and it is not "
            "optional. An interval spanning zero is reported as inconclusive about "
            "the SIGN of the effect and never as evidence of no effect. the literature scan "
            "records a 2026 survey finding that over half of published articles "
            "misinterpret a crossing interval as no difference, so the rule is "
            "stated in the paper rather than assumed of the reader. If absence of an "
            "effect were the claim being made, the correct tools would be "
            "equivalence testing or a defined region of practical equivalence, and "
            "THIS STUDY DEFINES NEITHER, so it cannot make that claim in either "
            "direction. "
            "\n"
            "Also recorded, per the committed specification's own not-found: no numeric threshold exists for "
            "how narrow a crossing interval must be before it is informative rather "
            "than inconclusive (R10 not-found 3). The paper does not invent one."
        ),
    ),

    "resample_count": ParamSlot(
        key="resample_count",
        treatment=Treatment.AUTHORS_CHOICE,
        note="Target named",
        disclosure_obligation=(
            "Report the bootstrap resample count together with the target it "
            "was chosen for (standard error, confidence interval, or a "
            "hypothesis test near a threshold), since the appropriate count "
            "differs by an order of magnitude or more across those targets."
        ),
        set_at="B24 implements it. Committed during the build, ahead of B22, so the "
        "pre-registration transcribes the count rather than leaving a "
        "resampling parameter to be chosen after the data exists.",
        value=9999,
        derivation=(
            "9999 RESAMPLES. "
            "\n"
            "THE TARGET IT WAS CHOSEN FOR, which is the whole of the committed specification's disclosure "
            "requirement: a CONFIDENCE INTERVAL, together with the Monte Carlo error "
            "of that interval's ENDPOINTS, which the committed specification requires reported as a quantity "
            "distinct from the Monte Carlo error of the point estimate. It was not "
            "chosen for a standard error and it was not chosen for a hypothesis test "
            "sitting near a threshold. the committed specification records that the appropriate count differs "
            "by an order of magnitude or more across those three targets, with "
            "reported figures spanning 200 to 100,000, which is exactly why the "
            "target has to be named alongside the number. "
            "\n"
            "Why a count at this end of that range. Interval endpoints are quantiles "
            "of the bootstrap distribution, and a quantile in the tail needs far more "
            "resamples to stabilise than a standard error computed from the whole "
            "distribution does. Hesterberg (arXiv:1411.5279) reports Efron and "
            "Tibshirani's 1993 figures once removed and recommends more than them. "
            "the committed specification's endpoint Monte Carlo error is then computed and reported, so the "
            "adequacy of this count is not asserted here but measured and shown; if "
            "the endpoints prove unstable at 9999 that is visible in the reported "
            "figure rather than hidden. "
            "\n"
            "Why 9999 and not 10000. At the committed 95 percent level, (B + 1) times "
            "alpha over 2 is 10000 times 0.025 = 250 exactly, so the percentile "
            "indices are integers and no interpolation convention has to be chosen or "
            "disclosed. An even count would force one. "
            "\n"
            "Author's choice under a disclosure requirement, per the committed specification's own treatment, "
            "and no source endorses a count for a paired solver-in-the-loop bootstrap "
            "specifically (R10 not-found 5). "
            "\n"
            "A separate bound that this count cannot lift, recorded so the two are "
            "not confused: with 11 clusters, the wild cluster bootstrap's Rademacher "
            "weights admit at most 2048 distinct weight vectors, so the bootstrap "
            "distribution is supported on at most 2048 points no matter how many "
            "resamples are drawn. See slot cluster_unit_bootstrap_method. Raising the "
            "resample count above 9999 would not relax that."
        ),
    ),

    "seeds": ParamSlot(
        key="seeds",
        treatment=Treatment.STATED_VALUE,
        note="One base seed, documented derivation rule",
        set_at="B01 nominally, per the build plan's own schedule, but left PENDING "
        "there since nothing needed a concrete draw yet Set during the build, at B06, "
        "the first step that actually draws (the bounded sampler). "
        "Not backdated into B01's own entry.",
        value=20260806,
        derivation=(
            "BASE_SEED = 20260806 (dataset.py), the calendar date of the "
            "day this was set, an arbitrary but reproducible choice -- "
            "not derived from any data. "
            "\n"
            "Offset rule (dataset.rng_for): every RNG stream anywhere in this "
            "build is np.random.default_rng(BASE_SEED + 1000 * "
            "build_step_number + substream), where build_step_number is the "
            "plan's own B-number (e.g. 6 for B06) and substream is a small "
            "integer distinguishing multiple independent streams within one "
            "step (e.g. B06 uses substream 0 for the width-selection search "
            "and substream 1 for building the committed population, so "
            "re-running the search does not perturb the population draw's "
            "stream or vice versa). Any stream used anywhere in the pipeline "
            "is reconstructible from this one recorded integer plus the "
            "step/substream it belongs to."
        ),
    ),

    "diversity_definition": ParamSlot(
        key="diversity_definition",
        treatment=Treatment.STATED_VALUE,
        note="One definition, averaged across the conditioning range",
        set_at="B16 (the build), consumed unchanged at B25. The plan assigns "
        "this to B25, but B16's own logic text requires the sweep to "
        "record 'the diversity statistic defined at B25' and B25 sits far "
        "downstream of B16. the committed specification forbid two definitions sharing "
        "the word, so it is fixed once at the first step that consumes "
        "it. Recorded as a departure from the build plan's step assignment. "
        "(The plan's schedule also fixes a 'tracking population' of "
        "matched pairs at B25; that item carries no T-number in the build plan's "
        "table and is not made a separate slot here.)",
        value=DIVERSITY_DEFINITION,
        derivation=(
            "See the module-level constant DIVERSITY_DEFINITION in this file for "
            "the definition in full, which is the one copy. Fixed during the build "
            "before B16's sweep was run. The sample count (20), the target count "
            "(11) and the statistic (mean pairwise Euclidean distance in "
            "standardised CST coefficient space) are stated values under the "
            "committed specification, disclosed as this study's own. PcDGAN's closed-form "
            "determinantal-point-process score (arXiv:2106.03620 §3.5 Eq. 14) is "
            "the nearest published construction; it is not adopted, because it "
            "brings a kernel and a subset-size choice this build has no basis for "
            "setting, and mean pairwise distance is the straightforward spread "
            "statistic in the space the prior is already measured in."
        ),
    ),

}


# ---------------------------------------------------------------------------
# Known discrepancies between this record and other project documents.
# Recorded, not resolved, because resolving them is not this step's job.
# ---------------------------------------------------------------------------

KNOWN_DISCREPANCIES = (
    "the committed specification's prose ('Which remain the author's to choose') names "
    "nine slots as the author's choice: CST order, per-seed count, "
    "standardisation row set, region extent percentile, every gate threshold, "
    "the consistency gate tolerance, the interval level, the resample count, "
    "and the requested target band. the build plan's own section 14 table gives "
    "'Interval level' and 'Requested target band' the treatment STATED VALUE, "
    "not author's choice, and 'gate thresholds' and 'consistency gate "
    "tolerance' are not rows in that table at all. This file follows Task "
    "IV's table for the `treatment` field, per B01's own instruction to read "
    "treatment from that table, and records the mismatch here instead of "
    "silently picking a side.",

    "'Per-seed count' (B06, author's choice, disclosed) and 'flag-clear "
    "fraction' (B10, range with rule) and 'tracking population' (B25, stated "
    "value) appear in the build plan's open-parameter schedule but are not rows in "
    "the committed specification. They are not represented as slots in this file, "
    "since B01's instruction is to record slots from the the build plan table.",

    "The safeguard's curvature and thickness bounds (B14, the build) are a "
    "further such case: B14's own open-parameters text names them as stated "
    "values re-derived from the training population, but the committed specification's "
    "table has no row for them distinct from 'Safeguard weight' (the committed specification, which "
    "governs the WEIGHT, not the bounds the safeguard's hinge terms are "
    "measured against). Not represented as a params.py slot for the same "
    "reason as the three above; the bounds, their derivation, and the values "
    "they produced (thickness_lower_bound=0.055425, curvature_upper_bound="
    "0.000116, both from the 785-row training population at the committed "
    "B03 interior grid and the B03 margin_fraction=0.20) are recorded in "
    "BUILD_LOG.md's B14 entry and reproducible via model.derive_safeguard_bounds.",

    "B18's gate thresholds (the build) are a fifth such case. the build plan's own "
    "open-parameter schedule gives 'Gate thresholds' the treatment author's "
    "choice, committed before the gate runs, but the committed specification's table has "
    "no row for them at all. They are therefore recorded as the module-level "
    "constant B18_GATE_THRESHOLDS in this file rather than as a params slot, "
    "following the precedent already set for the per-seed count, the flag-clear "
    "fraction, the tracking population and the safeguard bounds. They are fixed "
    "in that constant before the gate is run and before the zero-weight control "
    "that must fail against them is trained.",

    "B19's consistency gate tolerance and row count (the build) are a sixth such "
    "case. the build plan's open-parameter schedule gives 'Consistency gate tolerance' the "
    "treatment author's choice, disclosed with the absence of a published tolerance "
    "stated, but the committed specification's table has no row for it. It is recorded as the "
    "module-level constant B19_CONSISTENCY_GATE in this file, on the same precedent "
    "as B18_GATE_THRESHOLDS, and is fixed there before either gate was run. The "
    "evaluation path's decode resolution (160 points per surface) is a further "
    "stated value with no the build plan row, and is recorded inside the same constant "
    "rather than being left implicit in evaluate.py.",

    "Three step assignments settled during the build, at the author's direction, and "
    "recorded rather than made silently. (a) 'Requested target band' is set at no "
    "step in the build plan's plan at all, though B21 and B23 both consume it and B22 "
    "must disclose it; it is committed during the build, before B22. (b) 'Plausibility "
    "filter criterion' is cited by no step's Specification entries list "
    "anywhere in B01 through B26; it is likewise committed during the build, before "
    "B22. (c) 'Cluster unit and bootstrap method' was recorded here as set at B24, "
    "which is WRONG on the committed specification's own terms: the committed specification's treatment text says the method is "
    "chosen before evaluation, and B24 runs after B23. The choice is moved ahead "
    "of B22 and B24 now implements rather than chooses it. In the same build pass and "
    "for the same reason, 'Primary statistic', 'Interval level' and 'Resample "
    "count' were all committed ahead of B22, so the pre-registration transcribes "
    "them instead of deferring a decision past the run it governs.",

    "Diversity definition, step assignment (the build). the committed specification has a "
    "row for it and the build plan's schedule assigns it to B25, but B16's own logic "
    "text requires the sweep to record the diversity statistic and B25's "
    "prerequisite (B23) is far downstream of B16. The definition is fixed once at "
    "B16 and consumed unchanged at B25, because the committed specification forbid two "
    "definitions sharing the word. The slot records B16 as where it was set.",

    "Divergence weight, step assignment and treatment (the build). the build plan's row "
    "'Reconstruction and divergence weighting' stood set_at=UNRESOLVED from B01 "
    "through the build, because no step in the reviewed plan states where the "
    "divergence weight's full value is chosen. It was first carried into B16's "
    "sweep on B17's 'and by the same procedure' authority, and then withdrawn from "
    "the sweep during the build and set as a STATED VALUE of 1.0 on the locked "
    "model family, because applying a selection rule to it returned a value that "
    "contradicts that locked element. the build plan's table gives the row RANGE WITH "
    "RULE, so the committed treatment now differs from the table's. Recorded, not "
    "resolved. The reasoning is in WEIGHT_SELECTION_RULE['amendment_record'] and "
    "in the slot's own derivation.",

    "Safeguard weight, treatment (the build). Previously flagged above as a "
    "mismatch between B14's and B17's procedural text (which group it with the "
    "swept weights) and the build plan's table (STATED VALUE). It is set as a stated "
    "value, which resolves the mismatch in the table's favour. The ground is "
    "measurement rather than preference: the safeguard term is identically zero on "
    "this build, so no sweep can identify its weight. See the slot's derivation.",

    "Pre-registration as issued against this record (the build, B22). THREE RULE "
    "TEXTS CORRECTED, in this record's favour, at issue rather than left standing, "
    "because filling the value and keeping the text would have put a false "
    "derivation in the issued document. (a) The CST order's rule read 'from a "
    "named discretisation level'. Slot cst_order records the committed specification's FIRST justification "
    "type, convergence of a target force prediction as the order increases, after "
    "the fitting-error-convergence result was superseded during the build. No Kulfan "
    "discretisation level was used. (b) The safeguard weight's rule read 'same "
    "procedure' and 'this weight changes the objective'. Slot safeguard_weight "
    "records a stated value and not a swept selection, and a term that is "
    "identically zero on this build, so the weight provably does not change the "
    "objective. (c) Admission rule 4 read that the efficiency value 'lies inside "
    "the plausibility band'. No such band exists. Slot "
    "plausibility_filter_criterion records a per-point test on drag and transition "
    "location, applied before the label's maximum is taken, and declines the "
    "superseded build's band on the output ratio outright.",

    "Values this record carries that the issued pre-registration does not "
    "(the build, B22). B22's falsification check failed its clause that every "
    "parameter marked set has its value in the document. Seven slots were absent "
    "or partly absent. "
    "\n"
    "TWO RESOLVED, later during the build, at the author's direction, by reissuing the "
    "document IN FULL and BEFORE B23 rather than amending it in place. "
    "reconstruction_divergence_weighting now appears as two rows, reconstruction "
    "weight 1.0 and divergence weight 1.0. cluster_unit_bootstrap_method now "
    "appears under the build plan's own row name and carries the wild cluster bootstrap-t "
    "with Rademacher weights, the unrefined percentile bootstrap reported "
    "alongside it, and the 2048 weight-vector bound. Neither value moved and "
    "neither was decided from an outcome. The divergence weight was set "
    "before B16's sweep was run, and the bootstrap method ahead of B22, "
    "because the committed specification says the method is chosen before "
    "evaluation. Nothing has "
    "been evaluated in this build since B07's labelling pass, so the reissued "
    "document's account of what was available when it was written stands "
    "unchanged. The omission was a transcription gap in the draft's table and not "
    "a late decision. The divergence weight was the serious one, since a weight "
    "that changes the objective and appears in no committed list is the failure "
    "this file's own opening paragraph says the file exists to prevent. "
    "\n"
    "FIVE STILL ABSENT, recorded and not resolved. "
    "efficiency_target_definition, conditioning_composition, "
    "spanwise_resolution_avian_section, avian_section_geometry_figures and "
    "cst_fit_error_acceptance, along with B19_CONSISTENCY_GATE's gate zero and its "
    "160-point evaluation decode resolution. All five plausibly belong in Method "
    "sections 2.1 to 2.8 rather than in the pre-registration's own parameter "
    "table, which is the open question. The same reissue route stays available "
    "until B23 starts. No value in this file moved at any point during the build.",

    "Safeguard bounds, measurement-path defect (the build, FIXED). The B14 bounds "
    "had been derived through a decode-then-interpolate path and applied through a "
    "direct-basis-evaluation path, disagreeing by 1.10x on training-population "
    "curvature. B14 was re-opened at the author's direction and the derivation now "
    "runs on the enforcement path. Superseded: thickness_lower_bound 0.055425, "
    "curvature_upper_bound 0.000116. Committed: 0.055426321765484482 and "
    "0.00010535145925184528. No verdict changes, since the term is inert either "
    "way. The bounds remain outside the committed specification's table, which is the "
    "separate discrepancy recorded above.",

    "Diversity is evaluated on two different grids at two different steps "
    "(B16 and B18, then B25). One definition, DIVERSITY_DEFINITION, fixed at "
    "B16 and consumed "
    "unchanged: 20 samples per target, mean pairwise Euclidean distance in "
    "standardised CST coefficient space within each target, arithmetic mean across "
    "targets. B16's sweep and B18's gate evaluate it on the internal 11-point grid "
    "over the full normalised label range, because the requested target band was "
    "not committed until later, after both had run. B25 evaluates it on the "
    "committed "
    "band, because M11 says 'across the whole requested range' and B25's own logic "
    "text says 'at each requested target'. The figures on the two grids are NOT "
    "comparable point for point, and in particular the diversity cost of the committed "
    "avian weight recorded in BUILD_LOG.md is an internal-grid figure. Neither is "
    "recomputed on the other's grid. See B25_METRICS['diversity_grid_basis'].",

    "M06's trim fraction has no home in the build plan's own schedule (the build, B24). "
    "the build plan M06 says the trim fraction is a stated value and assigns the "
    "computation to B24. the build plan's open parameter schedule gives B24 only the "
    "interval level, the resample count and the base seed, so no step in the build plan "
    "sets it, and the committed specification has no row for it. It is set at B24 as 0.10 per "
    "tail and recorded in the module constant B24_ANALYSIS, on the same precedent as "
    "the gate thresholds, the consistency gate tolerance and the safeguard bounds. "
    "It joins the set of values this record carries that the issued pre-registration "
    "does not, which already holds five slots, gate zero and the 160-point evaluation "
    "decode resolution. It is a sensitivity-only parameter, not an outcome, an "
    "admission rule or a reporting rule, so the pre-registration's prohibition on in-place amendment is not engaged; it is "
    "listed here so the absence is visible rather than discovered.",
)


def unset_slots():
    """Every slot whose value is still PENDING."""
    return [p for p in PARAMS.values() if isinstance(p.value, Pending)]


def slots_by_treatment(treatment: Treatment):
    return [p for p in PARAMS.values() if p.treatment is treatment]


if __name__ == "__main__":
    print(f"{len(PARAMS)} slots recorded.")
    print(f"{len(unset_slots())} slots pending.")
    for t in Treatment:
        print(f"  {t.value}: {len(slots_by_treatment(t))} slots")
