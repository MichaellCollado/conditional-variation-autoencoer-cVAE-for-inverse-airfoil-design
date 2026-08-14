# An avian derived shape prior as a flag gated objective term in a conditional VAE for inverse airfoil design

## Abstract

Inverse airfoil design returns a geometry meeting a stated performance target.
Generative models are used because the map is one to many. This
study places an avian derived shape prior inside the training objective of a
conditional variational autoencoder rather than inside its conditioning vector,
and gates it with a flag so one trained model produces both arms of an
ablation. The prior is a squared hinge on the distance between
decoded geometry and a measured seagull section in standardized class shape
transformation space, with the region extent derived from the avian family's
own spread, being that section and its perturbations. The model was trained on 982 XFOIL labeled sections and evaluated
across eleven requested efficiency targets on paired latent codes. On 109
analyzed pairs the mean paired difference in normalized target satisfaction
error was 0.0114, with a 95 percent wild cluster bootstrap-t interval of
[-0.0023, 0.0251] that spans zero, so no difference was detected. A verification suite run before evaluation confirms the mechanism engaged, since the prior moved shapes
0.3627 closer to the reference in 72.3 percent of pairs while narrowing
diversity by a factor of 2.54. The study compares a prior against no prior, so
the control is not attributed to biology.

**Keywords.** inverse airfoil design, conditional variational autoencoder,
biomimetic shape prior, ablation, XFOIL

---

# 1. Introduction

## 1.1 The problem class and the biological candidate

Forward analysis maps a geometry to its performance. Inverse design reverses
that map, taking a performance target and returning a geometry that should
meet it.

The target is aerodynamic efficiency in cruise, defined as the maximum lift to
drag ratio over a stated angle of attack sweep. The envelope is attached flow
at low angle of attack, chosen because it is where a cruise section spends its
working life and where the panel method used here carries the fewest
documented objections. Those objections are specific. XFOIL cannot resolve the
three-dimensional breakdown of a laminar separation bubble, an inability
attributed to its two-dimensional viscous inviscid formulation
(Brunelli et al., 2026).

Avian wing cross sections are a measured shape family that has been under
selection pressure for efficient flight, which makes them a candidate design
prior. It does not make them a better airfoil, and this study does not claim
that it does. Throughout this paper the avian family is represented by one
seagull derived reference shape, so the term stands for that single measured
section and its perturbations rather than a survey across species.

Every performance claim in this literature is scoped, and this study scopes
its own the same way. No source read across the searches makes an unqualified
claim that a biological reference is aerodynamically superior across all
tested conditions. Biological sections outperform an Eppler 193 below 8
degrees, with the comparison changing above that (Mandadzhiev, 2017). A review
of albatross, falcon and owl sections scopes its claim per mission profile
rather than per metric (Ajayi et al., 2026). Pterosaur derived profiles are
reported as different in character rather than uniformly better (Tunca et al.,
2026).

## 1.2 Conditioning, the nearest work, and the gap

Conditioning a generative model on a requested performance value is standard
practice, and the input is almost universally a performance target. Yonekura et al.
(2024) condition a hybrid conditional VAE and WGAN-gp on lift coefficient supplied
to encoder, decoder and discriminator. Graves and Barati Farimani (2024) condition
a denoising diffusion model on lift and drag. Yilmaz and German (2020) condition on
metrics derived from stall condition and drag polars. Meng and Tao (2025) condition
on lift and drag over a 14 parameter geometry.

Conditioning specifically on efficiency is established within that practice,
so the target is not this study's novelty. No searched source claims a single
scalar efficiency quantity as a novel conditioning target in itself, and
contributions attach instead to architecture, representation or control
(Liu et al., 2024; Yang et al., 2026). Chen et al. (2022) directly
predict airfoils reaching 80.8 and 95.8 percent of the average optimal lift to
drag value without further optimization.

Two published works sit nearest. Wada et al. (2024) train a physics guided
conditional Wasserstein GAN in which physical validity is computed by XFOIL and
fed back into training, and add a shape smoothing penalty at weight one.

**Table 1.** Positioning against the nearest published work, on the axes the field
uses to separate a study from its predecessors. All three neighbors were read at
full text. Cells marked not stated record a field the source does not report.
Cells marked inferred record a reading taken from what the source shows rather
than from an explicit statement in it.

| Axis | Physics guided GAN (Wada et al., 2024) | Bio-inspired generative design (Jiang et al., 2023) | Conditional VAE-WGAN-gp (Yonekura et al., 2024) | This study |
|---|---|---|---|---|
| Shape parameterization | Raw coordinates, 248 points, dimension 496 | Wing planform outline, ordered points, a 192 by 2 matrix fitted by B-spline | Raw coordinates, 248 points, dimension 496 | Class shape transformation, order 9 per surface, 20 coefficients |
| Model family | Conditional WGAN with gradient penalty | Modified Bézier-GAN, unconditional at the network level, inferred | Hybrid conditional VAE and WGAN-gp | Conditional VAE |
| Conditioning input | Lift coefficient, continuous scalar, supplied to generator and discriminator | None at the network level, inferred. Output is steered by the species ratio in the training set | Lift coefficient, continuous scalar | One scalar, normalized efficiency target |
| Added objective term, and where it acts | XFOIL computed physical validity, plus a smoothing penalty at weight one | None shown. Only the general GAN objective is given, and biological knowledge enters by dataset curation | None of a shape kind. The term written as a prior is the latent divergence | Avian shape reference, flag gated in the objective |
| Solver and regime | XFOIL, angle of attack 5 degrees, Reynolds 3,000,000, Mach not stated | FlapSim blade element model, unsteady flapping wing. Reynolds and Mach not stated | XFOIL | XFOIL, attached flow cruise, Reynolds 300,000, Mach 0 |
| Evaluation reference | XFOIL directly, on generated shapes | FlapSim directly, on generated shapes | Solver | XFOIL directly, on every reported value |
| Comparison or ablation design | Four separately trained models for the physics comparison. Smoothing compared as two separately trained runs of one architecture | Dataset ratio study at 17, 33 and 66 percent dominant species. Every run contains all three bird types, so no null arm | Four separately trained models differing in loss and structure at once | One trained model, one flag, paired latent codes |

The last row is where this study is not a subset of either neighbor. The first
column shares the mechanism and not the prior. The second shares the biological
prior and not the mechanism. Wada et al. do compare a smoothing term on against
off within one architecture, but the two settings are separately trained runs, so
that comparison carries training trajectory alongside the term. Neither neighbor
is ranked by that table, since both do real work in problems this study does not
attempt.

The full text also shows a point of contact rather than a separation. Wada et al.
label with XFOIL, as this study does, at a Reynolds number of 3,000,000 against
300,000 here. Two of the three neighbors therefore share this study's solver, and
all three read performance from a solver run directly on generated shapes.

The gap is an avian derived shape prior applied as a flag gated term in the
training objective. The setting is a conditional VAE performing inverse airfoil
design toward efficiency targets, in the XFOIL attached flow cruise regime. The
prior's contribution is isolated by a clean prior on against prior off ablation.

That positioning rests on an absence and is stated as a not-found result
rather than a claim of priority. Ten literature scans were run over the
combination and its neighborhood. They ran primarily across 2021 to 2026, and
extended to ten years where the shorter window returned nothing. Queries crossed
four term groups covering biological reference, mechanism, model family and
domain. Four independent scans recorded the same core absence. No published
work reached by those scans places a biologically derived shape prior as a flag
gated, switchable or on-off term inside the training objective of a conditional
generative model for fixed wing inverse airfoil design. Two adjacent absences bound it.

## 1.3 Research questions, objectives and scope

Three questions follow.

1. Is the combination stated above distinct from the nearest existing work?
2. Can a conditional VAE trained on XFOIL labeled data perform inverse airfoil
   design toward a single scalar efficiency target, producing geometries that are
   geometrically valid and evaluable by the solver?
3. Does a flag gated avian shape prior term in the training objective measurably
   change the model's ability to satisfy efficiency targets, relative to an
   otherwise identical model with the prior off?

Question 3 is the study's question and the first two are preconditions for
reading its answer. If the combination is not distinct, the ablation measures
something already known. If the model does not perform inverse design at all,
the ablation compares two models that ignore their input. The objectives
follow. Build a conditional VAE for single scalar efficiency targeted inverse
airfoil design on XFOIL labeled data.

The contribution is claimed at the mechanism. Neither the efficiency
conditioning target nor the ablation method is claimed as novel, since the
same-architecture, single-term-toggled ablation is established in general
generative modeling (Creswell et al., 2017; Jindal & Wang, 2021;
Li et al., 2021; Sun et al., 2023; Zhou et al., 2021),
though none of those is an airfoil paper.

The scope of the baseline bounds everything that follows. The study tests
whether an avian prior beats no prior. It does not test whether any benefit is
specifically biological rather than what any fixed shape reference of
comparable elaboration would give.

---

# 2. Methodology

> **Figure 1.** Method schematic. Figure 1 shows the pipeline, which runs from the seed library of five defining sections, through
> dataset construction and solver labeling, through training with the flag gated
> prior term, to paired generation and solver evaluation. The figure plots no
> quantity. Its function is to show where the two arms diverge. They share one
> trained model, one set of weights, one conditioning layout and one latent code
> per pair, and separate at a single term in the training objective.
> `figures/figure_1_method_schematic.pdf`

## 2.1 Geometry representation

### 2.1.1 The avian anchor

The anchor is a seagull wing cross section from Liu et al. (2004, 2006). Its
spanwise resolution is mixed, because the source splits it that way. The shape
comes from coefficients averaged over 2y/b from 0.166 to 0.772, beyond which
the primary feathers separate. The magnitude comes from envelope equations at
the single station 2y/b = 0.4. No single-resolution reconstruction therefore
exists to be chosen.

Measured on the committed interior grid of 200 cosine-spaced points restricted to
[0.005, 0.995] chord, the reconstruction has a maximum thickness of 0.097466 chord
at x/c 0.2017, a maximum camber of 0.100276 chord at x/c 0.4409, and a peak
absolute camber second difference of 6.418739e-05, over 300 raw coordinate points.

### 2.1.2 The shared basis and the seed library

Geometry is a class shape transformation (CST) at order 9 per surface, giving 20
coefficients, with standard exponents N1 = 0.5 and N2 = 1.0 and a trailing
edge thickness parameter of zero. The transformation writes a section as a class
function times a shape function, so a geometry becomes a short coefficient
vector rather than a list of points. No field-wide standard order was identified, so the
order was set by convergence of a target force prediction, one of three
justification types that recur (Anusonti-Inthra, 2024; Kulfan, 2008;
Sharpe & Hansman, 2025). All five seeds were fitted and solved at each
order from 3 to 14, under the rule that the worst-case relative change in
maximum lift to drag must stay below 2 percent and stay there. Order 9 is what
that rule returns. Order 8 misses by 1.63 percentage points on one step, so
neighboring orders would also be defensible.

Five sections define the design space, being the seagull anchor plus S1223,
E387, SD7003 and SG6043. The published analogue of the avian section is among
them, since Liu et al. (2006) report that the seagull and merganser airfoils
resemble the S1223 after both are brought to a common maximum camber and
thickness. Avian-like geometry therefore enters the training distribution of
both arms twice.

The trailing edge is closed structurally, since the decode adds no linear trailing
edge term and then sets both surfaces' trailing edge ordinates to their shared
midpoint. XFOIL will otherwise forcibly collapse an unclosed trailing edge,
changing the shape the solver evaluates from the shape the model emitted.

Geometry and labels are standardized using the training split alone, being 785
of 982 rows, because constants computed over rows the model later validates on
would carry validation-set information into training-time scaling. The
literature does not settle this, and the one located source addressing it
directly applies its statistic across both splits (Zhang et al., 2019).

## 2.2 Dataset construction and conditioning

### 2.2.1 Sampling, filtering and labeling

The design space is populated by perturbing the five seeds in coefficient
space. Each accepted coefficient satisfies perturbed equals seed plus delta,
with delta drawn independently and identically from a uniform distribution on
[-0.05, +0.05] across all 20 coefficients.

A geometric validity filter runs before any shape reaches the solver,
rejecting a decoded shape if the surfaces cross on the interior grid, if
maximum thickness falls outside 0.068024 to 0.145590 chord, or if the peak
absolute camber second difference exceeds 0.0003794664188666495. Those bounds
come from the five seeds' own measured range with a margin fraction of 0.20,
fixed before any shape was sampled, and one draw was rejected during the
build.

XFOIL 6.99 is run headless through a Python subprocess at the operating point
in Table A1. The transition amplification factor of 9 corresponds to the
average wind tunnel entry in the solver's documented situational table
(Berger et al., 2026; Drela, 2001;
Yang et al., 2025), and a paneling study found no
accuracy gain beyond a threshold (Kallstrom, n.d.). Liu et al. (2004) state a
Reynolds range for birds of 4x10^4 to 7x10^5 and expect laminar separation
above roughly 10 degrees, so the committed Reynolds 300,000 sits inside that
range and the sweep stops at 8 degrees. This study ran no Reynolds convergence
probe and asserts no such comparison.

### 2.2.2 Attrition, size and the split

Timeouts are reported separately from true convergence failures throughout, and the
separation is this study's own choice, since no peer-reviewed or preprint source
found in the searches reports wall clock timeouts as a distinct category. A timeout
means the shape is slow and not that it is unsolvable.

**Table 2.** Dataset attrition and per-family composition. Attempted reconciles
exactly against kept plus every discard category, at 1005 = 982 + 23 + 0 + 0.

| Category | Count | Family | Attempted | Kept | Timeouts |
|---|---|---|---|---|---|
| Attempted | 1005 | E387 | 201 | 188 | 13 |
| Kept and labeled | 982 | S1223 | 201 | 200 | 1 |
| Discarded, timeouts | 23, rate 0.0229 | SD7003 | 201 | 198 | 3 |
| Discarded, convergence failures | 0 | Seagull, avian | 201 | 195 | 6 |
| Discarded, environment faults | 0 | SG6043 | 201 | 201 | 0 |
| Discarded, plausibility re-check | 0 | **Total** | **1005** | **982** | **23** |

Table 2 gives the attrition. Not one shape failed to converge for an
aerodynamic reason.

The dataset holds 982 labeled shapes over a label range of 50.49872773536895
to 172.17391304347828. The realized flag-clear fraction is 0.095723, being 94
rows, selected by a rule fixed beforehand as the smallest candidate leaving
every well-populated label decile at least 5 flag-clear rows. The validation
fraction is 0.20, disclosed as a conventional round figure, stratified by family
under one recorded seed, giving 197 validation and 785 training rows, and one
split serves both models.

### 2.2.3 The conditioning vector

**Table 3.** Conditioning vector layout. The block's width follows from the
representation, since at order 9 the signature is 20 standardized coefficients
matching the geometry the model emits.

| Column | Contents |
|---|---|
| 0 | The normalized target, being the row's own normalized label |
| 1 to 20 | The standardized avian signature block where the flag is set, or a zero block where it is clear |
| 21 | The flag, 1.0 when set and 0.0 when clear |

Table 3 gives the conditioning layout. Reynolds number and angle of attack are
not conditioning inputs, since both
are fixed across the training data. The user supplies one scalar target
efficiency in normalized units.

The signature block is constant within its class, so every flag-set row
carries the same fixed avian signature and every flag-clear row a zero block
of 20 columns alongside the separate flag column. It therefore carries no
conditional mutual information with the output, and this follows from
published results rather than from any measurement made here.

## 2.3 The avian prior and the training objective

### 2.3.1 Why the prior lives in the objective

A conditioning-vector mechanism does not hold for a block constant within its
class. The signature block is identical across every flag-set row and the flag
is drawn independently of the target and of the geometry, so the entire
conditioning array reproduces exactly from the target and the flag alone.

Information about the avian reference could enter by two routes. Altering the
training distribution would confound the ablation directly, since the arms would
then differ in what they were trained on as well as in the prior. A term in the
training loss can instead read the flag and act only where it is set. The objective
was chosen.

### 2.3.2 The committed mechanism

The prior is a flag gated squared hinge on the distance between decoded
geometry and the fixed avian reference, in standardized coefficient space. For
each generated shape the Euclidean distance is taken between its 20
standardized coefficients and the reference's.

**Table 4.** The committed prior mechanism, and the populations the region
separates. Separation counts are over all 982 labeled dataset rows.

| Committed element | Value |
|---|---|
| Prior term weight | 5.862918756788936 |
| Region extent | 1.3003770283589529 |
| Space | Standardized CST coefficient space, 20 columns, upper then lower |
| Distance | Euclidean |
| Penalty form | Squared hinge, zero inside the extent |
| Normalization | Divided by the count of flagged rows in the batch |
| Avian family rows inside the extent | 175 of 195 |
| Non-avian family rows inside the extent | 0 of 787 |

Table 4 records the committed values. Both sides are standardized with the
same frozen artifact, since distance and extent must live in one space for the
extent to mean anything.

The extent is the 90th percentile of the avian family's own distances to the
reference, applied to that subset alone so it describes the family's spread
rather than the dataset's. The avian count in Table 4 follows from the
construction. The record of how that percentile was set sits in Appendix A.1.

The hinge belongs to an established family documented under free bits, delta-
VAE and divergence thresholding, reported at second hand (Felhi et al., 2020; Zhu et al., 2020). Every located instance operates on a latent
divergence, and none on a distance in a coefficient space to a fixed reference
shape.

### 2.3.3 The objective

Each training batch runs the model twice. Pass one reconstructs real geometry at a
dataset row's own stored target and flag. Pass two generates freely, from fresh
latent codes decoded at the same stored targets and flags. The total objective is
the sum of the two passes' terms.

**Table 5.** Objective terms, committed weights, and the pass each acts in. Every
weight that changes the objective appears here. Reconstruction carries an implicit
weight of one and is the scale every other weight is expressed against.

| Term | Weight | Where |
|---|---|---|
| Reconstruction | 1.0, implicit | Pass one |
| Latent divergence | 1.0, times the warmup schedule multiplier | Pass one |
| Safeguard | 1.0 | Both passes |
| Target consistency | 99.09702261385384 | Pass two |
| Ensemble spread penalty | 54.307616878345584 | Pass two |
| Avian prior, flag gated | 5.862918756788936 | Pass two |

Table 5 lists every term and weight. The divergence warmup is a linear ramp of the schedule multiplier from zero to one
over the first 20 epochs, held at one thereafter. The ensemble spread penalty is
a penalty on disagreement across the frozen surrogate ensemble, which
discourages shapes the ensemble cannot agree about. The target consistency term
and the geometric safeguard are described later in this subsection.

The prior acts on pass two only. Applied to reconstruction it would fight the
data on 90.4 percent of rows, since the realized flag-set fraction is
0.904277, and on each of those rows the two instructions are directly opposed.

The target consistency term ties output to the requested target, because a
conditional VAE can otherwise reconstruct well from the latent code alone and
treat conditioning as surplus. At generation time the code is drawn at random,
so a decoder that learned to lean on it produces shapes unrelated to what was
asked for. The term needs a gradient with respect to the generated
coefficients and the solver cannot supply one, so a frozen surrogate ensemble
stands in its place. The published alternatives are named rather than passed
over.

The surrogate has five members at hidden width 32 and depth 2, trained with
Adam at learning rate 0.001 for 200 epochs at batch size 64, each predicting
the normalized label from the 20 standardized coefficients alone and each
reseeded independently. It is trained once, frozen, and not retrained between
arms.

An arm-identical geometric safeguard penalises generated shapes below a
thickness floor or above a curvature ceiling, at the values in Table A2, and
takes no conditioning input at all. Its weight is 1.0 and is reported together
with the measurement that the term does not bind, since it evaluates to
exactly 0.0 at initialization and zero of 785 training rows violate either
bound.

Only the avian term reads the flag, and this is structural, since none of the
other six term functions accepts a flag argument. Checkpoint selection uses
the unweighted sum of reconstruction and target satisfaction on the validation
split, over one fixed set of latent codes reused at every epoch, and is not
multiplied by any objective weight.

## 2.4 Weight selection and mechanism verification

### 2.4.1 The weight

The prior term weight is 5.862918756788936, and it was fixed before any solver
result was seen, since choosing it from the outcome it produces would be
p-hacking. The selection rule was written before the sweep existed, amended
once before any sweep had run with the amendment and its cause recorded, then
applied to the sweep table alone.

Each swept weight has its own ladder, being the ordered set of values swept for
it. For each swept weight, over that ladder, three dimensionless quantities are
formed, each normalized to the ladder's observed range. Reconstruction cost
is how far a candidate's validation reconstruction sits above the ladder's
best. Diversity cost is how far its generative diversity sits below the
ladder's best. Effect gain is how far the mean distance to the avian reference
on the flag-set arm sits below the ladder's worst. The score is the effect gain
less the two costs, and the selected value is the highest score.

**Table 6.** Prior weight sensitivity sweep. Diversity figures here are computed on
the sweep's internal grid of 11 evenly spaced normalized targets from 0.0 to 1.0,
which is not the requested target band committed later. Figures on the two grids
are not comparable point for point.

| Weight | Validation reconstruction | Mean distance, prior-on | Mean distance, prior-off | Diversity, prior-on | Diversity, prior-off | Live latent dimensions | Score |
|---|---|---|---|---|---|---|---|
| 0 | 1.094948 | 3.827637 | 3.977475 | 3.638577 | 3.852678 | 4 | 0.000000 |
| 0.05862918756788936 | 1.099337 | 3.450586 | 3.714686 | 2.868309 | 3.210567 | 3 | -0.096545 |
| 0.5862918756788936 | 1.133591 | 2.215275 | 2.742969 | 2.007895 | 2.605049 | 3 | 0.098563 |
| **5.862918756788936** | **1.173257** | **1.170410** | **1.881586** | **1.027770** | **1.897927** | **5** | **0.184391** |
| 58.62918756788936 | 3.217894 | 1.286437 | 2.536815 | 0.401188 | 0.907988 | 8 | -0.291845 |
| 586.2918756788936 | 9.648997 | 1.269190 | 3.405768 | 0.425645 | 0.969242 | 8 | -1.029619 |

> **Figure 2.** Prior weight sensitivity. Validation reconstruction and the
> diversity statistic against candidate weight, on paired axes, with the committed
> weight marked. The figure makes visible the trade the committed value represents.
> `figures/figure_2_prior_weight_sensitivity.pdf`

Table 6 and Figure 2 give the sweep. The selected candidate is a single maximum
with no tie, at an effect gain of
1.0000, a reconstruction cost of 0.0092 and a diversity cost of 0.8065. The
committed weight buys the full available reduction in mean distance for almost
no reconstruction cost and pays for it in generative diversity, from 3.638577
down to 1.027770. The rule charged that cost in full and the weight still won.
The prior saturates at parity, since both weights above the committed one give
a worse mean distance while reconstruction degrades by roughly a factor of
three and then eight. Two costs of the design are stated rather than hidden.

### 2.4.2 The mechanism gate

A verification suite runs between training and evaluation, in three groups.
Arm blindness confirms that every term except the prior is blind to which arm
is running. Gate response confirms the prior term switches as committed.
Effect confirms the prior has measurable consequence for the shapes the model
produces. No ablation number is trusted until the suite passes, it calls no
solver, and if it fails the evaluation does not proceed.

**Table 7.** Mechanism gate results for the committed model and the zero-weight
control. Groups one and two evaluate terms on one fixed batch of real validation
geometry with the flag column flipped and every other input held identical. Group
three is measured on paired generation over 11 targets at 20 samples each, with no
solver called.

| Group | Test | Committed model | Control | Threshold | Control required |
|---|---|---|---|---|---|
| One | Six non-prior terms, flags flipped | Bit-identical on all six | Bit-identical on all six | Tolerance exactly 0.0 | Pass |
| Two | Prior term, flags clear then set | 0.0, then 17.904039819460206 | 0.0, then 17.904039819460206 | Zero, then positive | Pass |
| Three, M12 | Mean distance, prior-on arm | 1.2596124323279743 | 4.799246120278793 | Inside extent 1.3004 | Fail |
| Three, M12 | Mean distance, prior-off arm | 1.6223132464898453 | 4.672799691743103 | Outside extent | Fail |
| Three, M12 | Separation, off less on | 0.36270081416187105 | -0.1264464285356901 | At least 0.1300377028358953 | Fail |
| Three, M13 | Direction consistency | 0.7227272727272728 | 0.39545454545454545 | At least 0.60 | Fail |
| Three, M14 | Arm effect against sampling noise | 0.6671631366396271 | 0.15182644118730707 | At least 0.25 | Fail |

Table 7 gives the verdicts. Groups one and two use real validation geometry
rather than generated shapes,
because on generated geometry the flags-set test would fail precisely when the
prior worked perfectly and pulled every flagged generation inside the region. A
well-behaved gate would then reject a well-behaved model.

The three effect quantities are defined by this study. M12 is the mean
distance to the avian reference per arm.

The zero-weight control was trained at prior weight zero with every other
committed weight unchanged, at the same seed, and was required to pass groups
one and two and to fail every test in group three. The committed model passed
all ten tests, with its checkpoint at epoch 146 and six live latent dimensions.
The control passed seven and failed exactly the three required. None of the
three margins is a near miss in either direction.

Target tracking is not gated in this build. An earlier design gated it, and the
reasoning still holds. What did not hold was the threshold, which had no stated
derivation and could not acquire one, since no located source reports this
construction with a threshold at all. Tracking is instead reported in section 3.5
against a computed condition-blind baseline, meaning a generator run with the
requested target column shuffled so that it stands as a chance reference.

## 2.5 Experimental design and committed outcomes

The ablation is prior on against prior off with everything else identical, and
the baseline is the prior-off arm of the same network. For every requested
target and sample index one latent code is drawn and decoded twice, once with
the flag set and once with it clear, so the two members of a pair differ in
the prior and not even in sampling noise.

The requested target band runs from 0.08761877789267145 to 0.5522084150555188
normalized, being 61.15975877192982 to 117.68878896594654 in raw lift to drag,
across 11 evenly spaced targets. The rule that set it is the 5th to 95th
percentile of the training split's own normalized labels, since requesting
targets above that would measure extrapolation rather than target
satisfaction.

The commitment is expressed in analyzed pairs rather than launched runs, at a
floor of 100, where an analyzed pair is one in which both members satisfied
every admission rule. The launch target is that floor divided by a measured
pair yield of 0.990909 from a pilot run, giving 101 pairs, launched as 10
samples at each of 11 targets, being 110 pairs and 220 solver calls.

A record enters the analysis only if its sweep converged on at least 8 of the
9 requested angle points. A maximum taken from too few points is not a maximum
over the sweep. No published basis for such a rule was located. The
tolerance was fixed before any candidate's bias was computed, requiring the
smallest k at which the mean absolute relative bias is at most 0.010 and the
95th percentile at most 0.020, over the 862 fully converged sweeps in the
dataset. The rule turns on the tail clause, since k equal to 7 gives 0.0053
and 0.0442 while 8 gives 0.0018 and 0.0133.

Pairing is complete-case, so a pair with one surviving member is excluded
entirely. This is the standard default and it is not bias-free, since the
closest theoretical treatment finds limited evidence that dropping incomplete
pairs protects against attrition bias in general (Bai et al., 2024).

Because the prior acts only where the flag is set, a flag-clear subset
correlated with the label would mean the avian pull was applied to a label-
correlated slice of the training data. The flag draw was therefore tested
against the label before training, with no threshold set to be cleared. The
standardized mean difference is -0.0357, the two-sample distribution statistic
is 0.0653 against an asymptotic critical value of 0.1473, and all 10
well-populated deciles contain flag-clear rows.

The target tracking slope is read against a measured condition-blind baseline
rather than an assumed one, and the construction is this study's own, since no
published source constructs a deliberately unconditioned generator as a chance
reference for a conditioning metric. The requested target column is shuffled
by one permutation of the 110 launched slots and the shapes are generated
again from the original latent codes, with everything else held identical.

Two gates run before any generated shape is evaluated, and both passed. Gate
zero solves one known airfoil through the same path with coordinates read raw,
requiring converged status on all 9 angles, and returned 9 of 9 in 0.728
seconds.

The pre-registration was issued after the model was trained and before any
generated shape reached the solver. It commits one primary outcome and four
secondaries, declares trimmed means a sensitivity check that may not be
promoted to a headline, applies no multiplicity correction while reporting
every statistic computed, and makes no directional prediction anywhere. It commits that an interval
spanning zero is reported as no detected difference and that absence of an
effect will not be claimed without a pre-registered equivalence test, of which
none exists. The full record sits in Appendix A.1 and Table A1.

---

# 3. Results

Everything in this section postdates the commitments recorded in section 2.5 and
Appendix A.1. No result here was visible when those commitments were made.

## 3.1 The mechanism engaged

This subsection comes first because the primary outcome is uninterpretable
without it. A null result on target satisfaction means one thing if the prior
moved shapes and did not help, and something else if the prior never moved
them at all. The two are not distinguishable from the primary outcome alone.

Figure 3 shows the distributions. The prior moved shapes toward the reference. The prior-on arm sits at a mean
distance of 1.2596124323279743 and the prior-off arm at 1.6223132464898453, a
separation of 0.36270081416187105. The region extent, fixed in advance at
1.3003770283589529, sits between them, so the prior-on arm lands inside the
region and the prior-off arm does not. It moved them consistently rather than
occasionally, at a direction consistency of 0.7227272727272728 against a chance
value of 0.50, so roughly seven pairs in ten move in the intended direction. The
movement is large relative to the model's own noise, at an arm effect against
sampling noise of 0.6671631366396271, so the arm difference is roughly two
thirds of the model's own sampling variation. The region also separates the
families it was derived to separate, since Table 4 reports 175 of 195 avian rows
inside the extent and 0 of 787 non-avian rows.

> **Figure 3.** Distance to the avian reference, per arm. The distribution of
> distances for each arm, with the derived region extent drawn as a vertical line
> and the training population's avian and non-avian distances behind them. Three
> claims resolve in one panel, being whether the prior-on arm sits inside the
> region, whether the prior-off arm sits outside it, and whether the region is a
> real boundary in the training data.
> `figures/figure_3_distance_to_reference.pdf`

This warrants reading section 3.3 as a measurement of the prior. It warrants
nothing about target satisfaction, since these quantities measure where shapes
went in coefficient space.

## 3.2 Sample and attrition

220 solver calls were launched across both arms and 109 pairs were analyzed.

**Table 8.** Sample, evaluability and attrition per arm. Every denominator is
stated in the row that uses it, and zero counts are printed rather than omitted.
The plausibility filter is a pre-solver geometric check that rejects a shape
before a solver call is spent on it. The minimum converged point rule is an
admission rule requiring a minimum count of converged sweep points before a
record is analyzed.

| Quantity | Prior on | Prior off | Denominator |
|---|---|---|---|
| Launched | 110 | 110 | Not applicable |
| Rejected by the plausibility filter, pre-solver | 0 | 0 | Launched, 110 per arm |
| Timeouts | 0 | 0 | Launched, 110 per arm |
| Environment faults | 0 | 0 | Launched, 110 per arm |
| Convergence failures | 0 | 0 | Launched, 110 per arm |
| Produced a usable efficiency value | 110 | 110 | Launched, 110 per arm |
| Excluded by the minimum converged point rule | 1 | 0 | Records producing a value, 110 |
| **Admitted** | **109** | **110** | Launched, 110 per arm |
| Evaluability rate | 1.0 | 1.0 | Launched less timeouts less faults |
| Admission rate | 0.990909090909091 | 1.0 | Launched, 110 per arm |
| Mean converged sweep points per admitted record | 8.963302752293577 | 8.936363636363636 | Admitted, 109 and 110 |
| Range of converged points | 8 to 9 | 8 to 9 | Admitted records |
| **Pairs with both members admitted** | **109** | | Pairs launched, 110 |
| Pairs losing only the prior-on member | 1 | | Pairs launched, 110 |

Table 8 gives the counts. Every generated shape in both arms produced a usable
efficiency value.

The two levels of attrition are different numbers and the second is the one
the analysis uses. At the shape level the pipeline lost one shape of 220, a
rate of 0.0045, and at the pair level one pair of 110, a rate of 0.0091, since
a pair dies when either member does. A paper reporting only the shape-level
figure would understate the loss by a factor of two. The lost pair sits at
requested target index 9, so that target contributes 9 pairs and every other
contributes 10.

Secondary outcome S4 is the difference between the arms' admission rates, at
-0.009090909090909038, with a pooled rate of 0.9954545454545455, a standard
error of 0.00907022440178424 and a test statistic of -1.0022805046720487. The
caution travels with it and is not softened, since unequal rates do not
establish bias and equal rates do not establish its absence (Bell et al.,
2013). The missingness depends on the generated shape rather than on the arm,
so it is not missing completely at random (Morris et al., 2019). The single
lost shape settles neither reading.

## 3.3 The primary outcome

For each matched pair the error is the absolute difference between the achieved
efficiency and the requested efficiency, in normalized units. The paired difference
is the prior-off arm's error minus the prior-on arm's, so a positive value means
the prior-on shape landed closer to what was asked. Pairs are grouped into
clusters for inference. A cluster is one requested efficiency target, since the
ten pairs sharing a target are not independent of each other.

**Table 9.** Primary outcome, both interval constructions, and the Monte Carlo
error of the endpoints. The committed estimator is the wild cluster bootstrap-t
with Rademacher weights, resampling whole requested-target clusters, with the
percentile-t taken from unrestricted residuals. That construction reweights
whole clusters at random and resamples a studentized statistic, and Rademacher
weights are drawn as plus one or minus one with equal probability. The table
names it the refined estimator, and names the plain percentile cluster bootstrap
the unrefined one. Endpoint standard deviations are
measured across 20 independent bootstrap repetitions.

| Quantity | Value |
|---|---|
| Analyzed pairs, clusters | 109, 11 |
| **Mean paired difference, normalized** | **0.01141141083603053** |
| Raw equivalent | 1.3884855281009814 lift to drag |
| Cluster-robust standard error | 0.005801960076376846 |
| Standard error treating pairs as independent | 0.0029926044165854843 |
| Ratio of the two standard errors | 1.938766127665077 |
| Cluster-robust test statistic | 1.966819951501049 |
| **95 percent interval, committed refined estimator** | **[-0.0022678114, 0.0251453683]** |
| Raw equivalent | [-0.275936, 3.059567] lift to drag |
| Bootstrap t quantiles | [-2.367123744436962, 2.357689820512483] |
| Resamples drawn, degenerate replicates | 9999, 0 |
| p value, null imposed | 0.08810881088108811 |
| 95 percent interval, unrefined percentile cluster bootstrap | [0.0007016909, 0.0225552005] |
| Refined width over unrefined width | 1.2544062787001466 |
| Exact enumeration of all 2048 weight vectors | [-0.0020148269, 0.0248376485], exact p 0.0908203125 |
| Endpoint SD across repetitions, committed estimator | 0.000256018 lower, 0.000235842 upper |
| Endpoint SD across repetitions, percentile bootstrap | 0.000132307 lower, 0.000153043 upper |

Table 9 and Figure 4 give the primary outcome. The interval spans zero, and this
is reported as no detected difference. It is
inconclusive about the sign of the effect, which is a statement about what the
data determined rather than about the world. It is not reframed as a trend, a
suggestion, a direction, or a result that would have reached some threshold
with more data.

The two arms' own errors, as descriptive context, are mean absolute errors of
0.04918267335270081 and 0.06059408418873134 normalized, being 5.984311 and 7.372796
in raw lift to drag, with medians of 0.04018551890611097 and 0.05406743193750507.
Mean achieved efficiency is 0.35428437456443845 and 0.36755380556421174 normalized,
against a mean requested value of 0.3182086803744149.

Clustering matters materially, since several samples share each requested
target and the paired difference varies systematically across targets, so
ignoring it would understate the standard error by a factor of 1.94. The two
interval constructions disagree on whether zero is included, and the
disagreement is reported rather than resolved by choosing the more convenient
one. The refined interval is the one the primary claim rests on. It was committed
before evaluation for exactly the reason this disagreement illustrates.
Standard cluster-robust inference over-rejects considerably in the five to
thirty cluster range, and those rejection rates are reducible to nominal by a
wild cluster bootstrap-t refinement (Cameron et al., 2008). This design has 11
clusters. The unrefined figure is reported because the difference is visible
only if both are shown, and it is the basis of no claim here.

> **Figure 4.** Paired difference display. Each pair's difference plotted against
> that pair's requested target, with the committed location statistic and its
> interval drawn as reference lines. The construction is borrowed from outside this
> domain and the borrowing is disclosed, since no searched source describes a
> plotting convention built for paired data as distinct from two independent groups
> anywhere in this literature. The difference plot is the domain-neutral standard
> for paired measurement (Bland & Altman, 1986).
> `figures/figure_4_paired_difference.pdf`

## 3.4 Distribution shape and the secondary outcomes

No multiplicity correction is applied. Every statistic computed is reported and
none is withheld.

The shape of the paired difference distribution is reported before the
secondaries because it makes their agreement with the primary legible. Over
109 pairs the sample standard deviation is 0.031243707369071894 and the sample
skewness is 1.033657148172426, with a minimum of -0.046670307150886436 and a
maximum of 0.11221710260985815. The distribution is right-skewed and the skew
favors the prior, since the longer tail sits on the side where the prior-on
shape landed closer to its target. The mean therefore sits above the median,
which is the ordinary consequence of a positive skew rather than a finding.

The four secondaries do not use the primary's estimator, and the substitute
failed one clause of its own validation. This is reported before the results
it bears on. A median, a fraction and a slope difference have no cluster-
robust variance formula, so the secondaries keep the identical resampling unit
and studentize by a delete-one-cluster jackknife standard error instead. To
studentize is to divide the estimate by its own standard error before
resampling. The delete-one-cluster jackknife finds that standard error by
recomputing the statistic with each requested-target cluster dropped in turn,
then measuring how far the estimate moves. That
construction was validated on synthetic data at this study's own shape, being 11
clusters by 10 pairs, across 400 datasets per design.

**Table 10.** Interval estimator coverage validation for the secondaries, against a
nominal 0.95, on synthetic data at 11 clusters by 10 pairs across 400 datasets per
design.

| Design and statistic | Refined | Unrefined | Pair-level |
|---|---|---|---|
| Median, clustered | 0.9250 | 0.9325 | 0.4950 |
| Median, no clustering | 0.9325 | 0.9100 | 0.9375 |
| Slope, clustered | 0.9400 | 0.9000 | 0.4025 |
| Slope, no clustering | 0.9575 | 0.9375 | 0.9500 |

Table 10 gives the coverage. The refined interval covers at or near nominal
everywhere and is materially
wider than a pair-level interval where the clustering is real. Where the
unrefined interval undercovers, at 0.9000 for the slope in the clustered
design, the refinement fixes it at 0.9400, and the negative control has teeth,
since the pair bootstrap covers 0.4950 and 0.4025 there. The failure is on one
clause of six, for both statistics tested, and it is reported as a failure.
With no between-cluster component the refined interval runs 2.82 times wider
than the pair interval for the median and 1.60 times wider for the slope,
against a tolerance of 0.40 fixed in advance, and the mechanism is a noisy
jackknife standard error over 11 clusters. The refined intervals below are
therefore conservative, so their coverage is sound and their width is not a
tight statement of precision. Both constructions are printed and they agree on
every measure, so no reported reading turns on the choice. It is not a reason
to call the clause satisfied and it is not called satisfied.

**Table 11.** Secondary outcomes and the declared sensitivity check. Markers are
`[P]` for the single pre-registered primary, `[S]` for a pre-registered secondary,
and `[SENS]` for a declared sensitivity check that may not be promoted to a
headline. No measure outside those categories was computed, so no row is post hoc.
The convention is defined here because no published in-table convention was located, the
closest located being bold text marking the better-performing model (Regenwetter et
al., 2023), which marks something else entirely.

| Marker | Measure | Value | Null | Interval, refined | Interval, unrefined | Verdict |
|---|---|---|---|---|---|---|
| `[P]` | Mean paired difference | 0.01141141083603053 | 0.0 | [-0.0022678114, 0.0251453683] | [0.0007016909, 0.0225552005] | Spans zero |
| `[S]` | S1, median paired difference | 0.006570038434836417 | 0.0 | [-0.0018440144, 0.0154183449] | [-0.0021950316, 0.0177107097] | Spans zero |
| `[S]` | S2, paired win fraction, 68 of 109 | 0.6238532110091743 | 0.5 | [0.4414295718, 0.8111454106] | [0.4678899083, 0.7727272727] | Spans 0.5 |
| `[S]` | S3, arm difference in tracking slope | -0.02239513088555367 | 0.0 | [-0.1052752484, 0.1726590705] | [-0.0869034422, 0.0431466918] | Spans zero |
| `[S]` | S4, difference in admission rate | -0.009090909090909038 | 0.0 | See Table 8 | See Table 8 | Marker, not a clearance |
| `[SENS]` | Trimmed mean difference, 0.1 per tail | 0.007978910334693607 | 0.0 | [-0.0032397284, 0.0224139885] | [-0.0013165139, 0.0194912327] | Spans zero |

Table 11 collects every outcome. The median's raw equivalent is
0.7994106440401215 lift to drag and its
jackknife standard error is 0.004035802194701365. The win fraction carries a
cluster-robust standard error of 0.08193804664466926, a test statistic against
0.5 of 1.511547029504785, and a bootstrap-t p value of 0.16101610161016103.
The slope difference carries a jackknife standard error of 0.04304643180435917
and bootstrap t quantiles of [-4.531251330435263, 1.9253655656348052], and its
two constructions disagree in width and centring because that studentized
distribution is strongly asymmetric at 11 clusters. The trimmed mean removes
10 values from each tail of 109. Its trim fraction was set at the analysis step,
after the pre-registration was issued, and the issued document does not name it.
It is a sensitivity parameter rather than an outcome, an admission rule or a
reporting rule, so the prohibition on amendment is not engaged.

The primary and the robust measures do not disagree, and saying so is as much
an obligation as reporting a disagreement would be. Neither of the two
conditions fixed in advance holds, since every location statistic in Table 11
favors the prior and every interval is inconclusive about the sign. What
differs is magnitude, since the mean is 1.74 times the median, which is the
positive skew doing what a positive skew does.

## 3.5 Conditioning behavior

This subsection answers research question 2 and is a precondition for reading
section 3.3 rather than a result about the prior. If neither arm used its requested
target, the primary outcome would compare two models that ignore their input.

**Table 12.** Target tracking slopes and correlations against the measured
condition-blind baseline, per arm. Both axes are normalized, so a model tracking
its request perfectly has slope exactly 1. All fits are over the 109 matched pairs.
The baseline run shuffled the requested target column by one permutation of the 110
launched slots and regenerated from the original latent codes, fitting against the
original requested column, which the model never received.

| Quantity | Prior on | Prior off |
|---|---|---|
| **Target tracking slope** | **0.726516850527816** | **0.7489119814133697** |
| Tracking correlation | 0.9714639893139583 | 0.9271562828312793 |
| **Baseline slope** | **0.051234111023757155** | **0.04752656168508205** |
| Baseline correlation | 0.0654359552660988 | 0.05820821594766166 |
| Ratio of tracking slope to baseline slope | 14.2 | 15.8 |
| Baseline run, produced a label | 110 of 110 | 110 of 110 |
| Baseline run, pairs with both members admitted | 110 | 110 |
| Baseline run, slots the permutation left in place | 2 | 2 |

Table 12 and Figure 5 give the tracking result. The baseline construction is
this study's own, since no located source
constructs a deliberately unconditioned or shuffled generator as a chance
reference for a conditioning metric. The closest published anchor is a
baseline blind to design performance rather than blind to the condition
(Regenwetter et al., 2023).

Both arms perform inverse design, each responding to its requested target
roughly fourteen to sixteen times more strongly than the same model does when
the target is unrelated to the slot it is filling. Target satisfaction error
alone could not have shown this, since a model emitting a fixed distribution
centered near the middle of the band would post a respectable mean absolute
error and a slope near zero. Neither arm tracks perfectly, and the shortfall
is stated rather than rounded away, since a slope of 0.74 against an ideal of
1 means the model moves about three quarters of the distance between two
targets.

> **Figure 5.** Signed target satisfaction error against requested target, per arm.
> One series per arm, with the mean signed error marked at each target and a
> dispersion band of plus and minus one sample standard deviation across the ten
> samples at that target. The fitted trend through each series is derived rather
> than separately fitted, at a slope of the tracking slope less one, being
> -0.273483149472184 and -0.2510880185866303. The signed form is specified because a
> model that tracks its request produces a flat band around zero while a model that
> ignores it produces a descending trend, and absolute error folds an overshoot and
> an undershoot onto the same value. Plotting error against the conditioning value
> with a dispersion band is what the domain does in place of an identity-line
> scatter (Heyrani Nobari et al., 2021).
> `figures/figure_5_signed_error_per_arm.pdf`

## 3.6 Generated geometry and diversity

Figure 6 draws the generated family. Section 3.1 reports the prior's effect on
shape as a scalar in a 20-dimensional
standardized coefficient space. Whether that separation is visible in the shapes
themselves is a different question, and no number in this paper answers it.

> **Figure 6.** Generated shape family against the avian reference. A grid of
> generated sections, both arms shown, with the avian reference outline overlaid on
> every panel, each shape labeled with its own requested value and its own achieved
> error. Panel selection is a rule rather than a choice per panel, covering four
> requested targets evenly spanning the committed band at indices 0, 3, 7 and 10,
> plus every target at which the admission rule excluded a shape, which is index 9.
> Every sample at each shown target is drawn and no sample is selected. Arm color
> is consistent across every figure, with the prior-on arm in blue and the prior-off
> arm in orange, and the avian reference black and dashed. The labeled family grid
> follows published practice (Heyrani Nobari et al., 2021). The overlay on a named
> biological reference outline does not, since no source found does it.
> `figures/figure_6_shape_family_against_reference.pdf`

One diversity definition is used everywhere the word appears. At each
requested target, 20 shapes are generated from independently drawn latent
codes with the target and the flag held fixed. The within-target statistic is
the mean pairwise Euclidean distance across those 20 shapes in standardized
coefficient space, and the metric is the arithmetic mean of that statistic
across the 11 targets. Both arms share every latent code, so their difference is
the flag's doing.

**Table 13.** Generative diversity per requested target, per arm. The prior-off
figure exceeds the prior-on figure at every one of the 11 targets. The metric
carries no interval and supports no inferential claim, since it is descriptive
rather than the primary or one of the four secondaries.

| Index | Normalized target | Raw, lift to drag | Prior on | Prior off |
|---|---|---|---|---|
| 0 | 0.087619 | 61.160 | 0.733791 | 1.457369 |
| 1 | 0.134078 | 66.813 | 0.256880 | 1.166326 |
| 2 | 0.180537 | 72.466 | 0.416556 | 1.360873 |
| 3 | 0.226996 | 78.118 | 0.241667 | 1.006296 |
| 4 | 0.273455 | 83.771 | 0.559244 | 1.760471 |
| 5 | 0.319914 | 89.424 | 0.523132 | 1.523788 |
| 6 | 0.366373 | 95.077 | 0.398417 | 1.087957 |
| 7 | 0.412832 | 100.730 | 0.645968 | 1.484618 |
| 8 | 0.459290 | 106.383 | 0.840111 | 1.584819 |
| 9 | 0.505749 | 112.036 | 0.591506 | 1.277201 |
| 10 | 0.552208 | 117.689 | 0.705148 | 1.314640 |
| **Mean across the range** | | | **0.5374928585930486** | **1.365850567488148** |

Table 13 and Figure 7 give the per-target figures. The prior narrows the range
of shapes the model produces, at a difference of
-0.8283577088950994 and a prior-off figure 2.54 times the prior-on figure.
This is the realized cost of the prior and it was charged in advance, since
the weight selection rule charged diversity in full at a cost of 0.8065 of the
ladder's range and the weight still won. The size of the gap is not uniform
across targets and no reading of where it concentrates is offered.

This metric separates two states that no other reported quantity
distinguishes, since a collapsed model and a healthy one can post the same
mean distance to the avian reference. On this evidence the prior-on arm has
not collapsed, since its diversity is 0.537 rather than near zero and varies
across targets.

> **Figure 7.** Diversity across the requested range. The within-target diversity
> statistic at each of the 11 requested targets, per arm, with the across-range mean
> marked on each series. The figure exists because the mean cannot show whether a
> diversity difference is uniform across the range or concentrated somewhere in it
> (Regenwetter et al., 2023).
> `figures/figure_7_diversity_across_range.pdf`

## 3.7 Surrogate honesty

This check is reported whichever way it falls, and it did not fall the way the
check was built to catch.

**Table 14.** Surrogate against solver on admitted generated shapes, in raw maximum
lift to drag, with the held-out reference the gap is read against. The sign is
predicted less solver, so a positive value means the surrogate read a shape as more
efficient than the solver found it. The denominator is 219 admitted shapes of 220
launched, since the one excluded shape carries no admitted solver value.

| Quantity | Pooled | Prior on | Prior off |
|---|---|---|---|
| n | 219 | 109 | 110 |
| **Mean absolute difference** | **5.601578499890196** | 5.304728422879385 | 5.895729939837272 |
| **Mean signed difference** | **-4.638444721821944** | -3.8740464450913783 | -5.395893923309508 |
| Correlation | 0.9624665073921944 | 0.9787483792556818 | 0.9579898806536891 |
| Ensemble held-out mean absolute error | 2.6538 | | |
| Training-split-mean baseline | 13.1704 | | |
| Full label range, for scale | 121.6752 | | |

Table 14 gives the comparison. The gap on generated shapes is 2.11 times the
held-out reference, so the surrogate is materially less accurate on the shapes
the generator produced than on data it was fitted near.

The mean signed difference is negative, so the surrogate reads generated
shapes as less efficient than the solver finds them, which is the opposite
direction from the failure this check exists to detect. A generator exploiting
blind spots in the model it was optimized against would drive shapes into
regions where the surrogate scores them generously. The error is dominated by
a consistent offset rather than by scatter, since the mean signed difference
is 82.8 percent of the mean absolute difference.

The primary outcome therefore cannot have been inflated by the prior-on arm
exploiting the surrogate, because exploitation would require over-prediction.
It does not establish that no exploitation occurred anywhere, since a
concentration of over-prediction at the top of the range would be diluted in a
mean over 219 shapes, which is a question for Figure 8.

> **Figure 8.** Surrogate against solver. The ensemble's predicted efficiency
> against the solver's value on every admitted generated shape, with the identity
> line drawn and the held-out reference error marked as a band. The figure carries
> what the scalars cannot, being whether the disagreement is uniform across the
> efficiency range or concentrated at one end. Exploitation, had it occurred, would
> appear as a fan opening toward the top of the range.
> `figures/figure_8_surrogate_against_solver.pdf`

No extreme converged result occurred. The largest single paired difference
sits 3.23 sample standard deviations above the mean, and dropping it moves the
mean from 0.011411 to 0.010478, a change of 8.2 percent against 30.1 percent
for trimming the outer tenth of each tail. So the tail as a whole carries
about a third of the mean's location and no individual record carries an
eighth of it.

---

# 4. Discussion

## 4.1 What the result answers

Question 1, whether the combination is distinct from the nearest existing work, is
answered in section 1.2 as a not-found result with its scope and its caveat.
Nothing in the Results bears on it and none is claimed to.

Question 2 is answered unambiguously on this run. Every one of the 220 generated
shapes produced a usable efficiency value, at an evaluability rate of 1.0 in both
arms, and both arms track their requested target at a slope near 0.74 against a
measured condition-blind baseline near 0.05. Neither tracks perfectly, and that
shortfall is a property of the model rather than of the prior.

Question 3 is the study's question and the answer is that no difference was
detected. On 109 analyzed pairs the mean paired difference in normalized
target satisfaction error is 0.01141141083603053, with a 95 percent interval
of [-0.0022678114, 0.0251453683] that spans zero. That is not a finding that
the prior has no effect. No equivalence test is pre-registered, so absence is
not claimed and cannot be, and the interval did not determine the sign.

The honest account has three clauses, and dropping any one misrepresents the
result in a predictable direction. The primary outcome detected no difference.
Every location measure and the win fraction favor the prior, and none of
their intervals determines the sign. The mechanism engaged, and both arms
perform inverse design.

At the width the evidence supports, the finding is this. A flag gated avian
shape prior in the training objective of a conditional VAE produced no detected
difference in normalized target satisfaction error. It was ablated against an
otherwise identical model, on 109 matched pairs sharing latent codes. The prior
verifiably changed the geometry the model produced and narrowed its generative
diversity by a factor of about 2.5.

## 4.2 Why the measures differ, and why they do not disagree

The mean and the median are not two estimates of one quantity. The mean asks
how much closer the average pair landed, which is what a designer running the
pipeline many times would care about. The median asks where the bulk of pairs
sits, which is what a designer running it once would care about. Neither is
correct in general.

The condition the pre-registration wrote a rule for did not arise, since
neither of the two fixed tests for a reportable disagreement holds. Stating
that plainly is an obligation of the same kind as reporting a disagreement
would have been, because otherwise it is not visible whether the measures were
checked against each other at all.

What differs is magnitude. The mean is 1.74 times the median, and that gap has
one cause, being a sample skewness of 1.034. No single record produces it,
since dropping the largest pair moves the mean by 8.2 percent against 30.1
percent for the tail as a whole.

One property of the measured quantity is worth stating even though no single
excursion drives this result. Target satisfaction error is an absolute
difference, so it is blind to sign.

## 4.3 Comparative position

No benchmark was run. No published method was reimplemented, none was run on
this study's data, and no shared task, dataset, solver configuration or
evaluation protocol connects this study to any work compared below.

Table 1 gives the positioning. The last row is where this study sits furthest
from its neighbors, since none of the three runs a single trained model with
one term switched. The evaluation reference row is not a separation. Reading
the three at full text confirms that each takes its performance numbers from a
solver run directly on generated shapes, as this study does. The solver against
estimator contrast holds instead against the closest comparable study outside
that table, which computes its headline conditioning metric against a trained
estimator and confirms the trends qualitatively (Heyrani Nobari et al., 2021).

The closest published quantity to this study's target satisfaction error is
PcDGAN's label error, at 0.0284 for its proposed model and 0.119 for its CcGAN
baseline (Heyrani Nobari et al., 2021). Both of this study's arms sit between
those figures, at 0.04918 and 0.06059, and that placement is the whole of what
the comparison shows. Seven differences prevent it from showing more. The
evaluation reference differs, since PcDGAN's error is computed against a
trained estimator and this study's against the solver. The conditioning range
differs, at 100 conditions spanning 0.05 to 0.95 against 11 targets spanning
0.088 to 0.552. The sample count differs by two orders of magnitude. The
replication differs, since PcDGAN averages across 10 training runs and this
study reports one.

Two solver-side comparisons are worth stating at their true scope. This
study's generated shapes produced a usable efficiency value at a rate of 1.0
on 220 shapes, against published figures of 94.1 percent validity on 256
generated designs (Zamrai & Mohd Yusof, 2025) and 56 percent convergence in a
large generation campaign over a deliberately broad input space (Sharpe &
Hansman, 2025), while the nearest published surrogate fidelity gap is around 9
percent (Fazliani et al., 2026).

Four trade-offs follow, each with a measured cost and a measured benefit. A
cleaner ablation buys attribution and costs effect size, since a study
comparing separately trained architectures measures a difference that includes
loss composition, network structure and training trajectory together. A locked
solver buys reproducibility and costs fidelity. A biological prior buys
geometric control and costs generative diversity, at a prior-off diversity
2.54 times the prior-on figure.

## 4.4 Engineering, ethical and societal implications

The prior degraded one measurable thing. Generative diversity fell from
1.365850567488148 to 0.5374928585930486, consistently at every requested
target. Nothing else degraded, since evaluability was perfect in both arms, the
tracking slope difference has an interval spanning zero, and the converged point
count and surrogate gap both run marginally in the prior-on arm's favor. The
narrowing is not collapse, since the prior-on figure is 0.537 rather than near
zero and varies across targets.

If the objective is target satisfaction alone, a designer should not add this
prior, since it carries a measured cost of a 2.5-fold reduction in diversity
and no measured benefit on the objective. That is not the same as concluding
the prior is useless, since the interval did not determine the sign, so a
designer choosing not to adopt is choosing rationally under uncertainty rather
than acting on a demonstrated absence of benefit.

Three guards protect the pipeline from producing implausible shapes, and none
fired on the evaluated run. A guard that never fires is a guard that was never
tested, so zero rejections establishes that no evaluated shape violated any
stated criterion without establishing that the criteria would catch what they
were written to catch. The largest safety statement in this paper is a
negative one. Its documented limits apply to every number here, and nothing in
this paper certifies any generated shape as fit for use.

The efficiency target is itself a sustainability quantity, since maximum lift
to drag in cruise sets fuel burn per unit distance for a section spending its
working life in cruise. This study did not improve it and does not claim to.
The pipeline cost roughly 2,700 solver calls plus 24 training runs. Inference is
a forward pass at negligible cost, so a trained generative model amortises its
build cost across every subsequent request (Chen et al., 2022). This study
issued 110 requests and did not reach break-even, because it was built to
measure a mechanism rather than to produce designs.

No source in any of the ten literature scans addresses ethical or societal
implications for generative inverse airfoil design, so this part stands on
general grounds and cites nothing. The design space is entirely determined by
five sections and everything outside their perturbed neighborhood is
unreachable, which is a decision about whose design problems the model can
serve rather than a defect.

Fairness as the machine learning literature defines it does not map onto this
study, and saying so is more honest than constructing an analogy, since there
are no human subjects, no protected attributes and no allocation of a resource
among people. What does map is uneven service across the design space.
Tracking is imperfect, error is not uniform across the band, and training data
is unevenly distributed across the five families. A designer whose problem
sits near a well-represented family is therefore better served than one whose
problem sits near a poorly represented one.

The interface presents 22 conditioning inputs and exposes one real control, so
a user handed this vector without the disclosure in section 2.2.3 would
reasonably believe they had twenty-two levers when they have one continuous
lever and one binary one. Disclosure is the mitigation rather than the fix.

The most likely harm from this class of method is not misuse by a bad actor.
It is a plausible-looking artifact being trusted further than its evidence
supports, since this pipeline produces sections that look like airfoils, carry
a solver-computed efficiency value, and can be generated in seconds. The
specific failure mode is that a generated section carries a number, since a
shape with an attached efficiency of 118 reads as more validated than a sketch
and is not. That number is a two-dimensional panel-method prediction at one
operating point, from a solver documented as unable to resolve three-
dimensional laminar separation bubble breakdown and as tending to over-predict
maximum lift. Three mitigations are already in the design rather than proposed
for future work. Every reported value comes from the solver and none from the
surrogate.

## 4.5 Significance and contribution

The contribution is claimed at the mechanism and nowhere else. Conditioning a
generative model on an efficiency target is established practice, and the
single-term-toggled ablation is established in general generative modeling.
What no located work does is place a biologically derived shape prior as a flag
gated term inside the training objective of a conditional generative model for
fixed wing inverse airfoil design, rather than inside its conditioning vector.
Three design choices accompany that placement. One trained model produces both
arms, so the comparison carries no difference in loss composition, network
structure or training trajectory. Paired latent codes remove sampling variation
from the comparison. Every reported value comes from the solver rather than from
the surrogate the training objective used, so the evaluation reference is
independent of the quantity trained against.

The significance is what those choices let the result mean. A study comparing
separately trained architectures measures a difference containing everything
that differs between them, and cannot attribute it to one term. This design can,
which is why a null here carries information. The mechanism verifiably engaged
before the null was read, so no detected difference means the term acted and did
not help on this measure. It does not mean the term may never have acted at all.
For a designer the actionable content is a trade at measured size. The prior
buys geometric control toward a chosen reference and costs a factor of about 2.5
in generative diversity, with no measured benefit on target satisfaction and an
interval that did not determine the sign. For method work the contribution is
the isolation design itself, which transfers to any objective term a study wants
to attribute rather than only to add.

## 4.6 Limitations

Seventeen limitations bound this study. The first two bound what the result can be
attributed to at all, the next two bound the evidentiary base every number rests
on, and the rest bound specific quantities or parts of the inference machinery.

**Single baseline.** The study compares prior against no prior rather than
biology against a matched alternative. Nothing in the Results distinguishes an
avian section from a human-designed section, a NACA section, or an arbitrary
but fixed point in the same coefficient space. A prior pointing at any of those
would be expected to move shapes toward it.

**The anchor is also a seed, and so is its published analogue.** Avian
geometry is present in the training distribution of both arms, since 195 of
the 982 rows are perturbations of the anchor and a second seed is its
published analogue. What the ablation measures is the marginal effect of
adding an objective term pointing at a shape family already present in the
data both arms saw.

**No physical validation, and the solver's documented limits apply to every
number.** No generated section was manufactured or tested, so every efficiency
value, error and tracking slope is a two-dimensional panel-method prediction.
The solver cannot resolve the three-dimensional breakdown of a laminar
separation bubble (Brunelli et al., 2026), tends to over-
predict maximum lift in high-lift low-Reynolds conditions (Pascoa, 2016), and
shows poor reliability in transitional regimes at Reynolds 68,000 to 159,000
(Demie et al., 2026).

**One trained model, one training seed, no replication.** The committed model
was trained once at one recorded seed, so nothing measures how much of the
reported outcome would survive a different training draw. This bounds the
result differently from the sample size, since the 109 analyzed pairs bound
the sampling variation of the evaluation and the interval quantifies it, while
the single training seed bounds the variation of the model and nothing
quantifies it.

**Every reported effect is conditional on the committed prior weight.** Table 6
shows both benefit and cost varying across four decades of weight. Table
6 shows both benefit and cost varying across four decades of weight. The
geometric effect saturates at the committed value and worsens above it, so the
committed weight is not an arbitrary point on a monotone curve.

**The measured cost in generative diversity.** The prior narrowed diversity by a
factor of 2.54 at every requested target, so a designer generating a portfolio
receives roughly two fifths as much spread per unit of compute. It is descriptive
with no interval, it is not collapse, and it is conditional on the committed
weight.

**The primary statistic sits on a skewed distribution.** The mean was
committed before the distribution was known and the distribution turned out
right-skewed at 1.0337. The measure is reported anyway, with the median beside
it and the mechanism of the 1.74-fold gap stated, since changing the primary
after seeing the distribution is precisely the failure a pre-registration
exists to prevent.

**Complete-case pairing is not bias-free.** Dropping incomplete pairs does not
recover the average treatment effect when attrition depends on the matching
variables, and the closest treatment finds limited evidence that it protects
against attrition bias in general (Bai et al., 2024). This study's pair-level
attrition rate is 0.91 percent and the direction of any resulting bias is
unmeasured.

**Optional stopping above the floor is bounded but not removed.** The
committed floor prevents stopping as soon as a result looks favorable. It
does not prevent continuing past the floor until one becomes favorable, and
this study pre-specifies no graded increment schedule with an adjusted
criterion. Three things bound the residual risk. The sweep was not extended,
so no extension decision was taken at all.

**The secondaries' interval estimator failed one clause of its own
validation.** The failure is reported in section 3.4 and repeated here because
a failed clause reported in Results but absent from Limitations would read as
having been quietly retired. The refined intervals are conservative, so their
coverage is sound and their width is not a tight statement of precision.

**Latent capacity is concentrated in fewer dimensions than the architecture
provides.** Six of eight dimensions clear the liveness threshold of 0.01,
while two carry 87.1 percent of the total divergence and three carry 98.2
percent, as Figure A2 shows. The effective capacity is therefore smaller than
the nominal eight, which bounds the diversity the architecture can express
independently of anything the prior does.

**Timeouts fall unevenly across families, and the cause is a resource limit.** All
23 dataset losses were wall-clock timeouts and none was a convergence failure, so
the family best represented is the one the solver solved fastest and nothing
aerodynamic distinguishes them. The consequence is a mild imbalance rather than a
distortion, at 188 against 201 across the extremes.

**The timeout separation has no precedent in the sources located and changes
every denominator it touches.** No peer-reviewed or preprint source found
reports timeouts as a category distinct from aerodynamic non-convergence. The evaluability
denominator excludes timeouts, so it is not the launched count in general,
though on this run both are 110.

**The avian section's spanwise resolution is mixed.** The shape comes from
span-averaged coefficients and the magnitude from envelope equations at a
single station, because the source splits them that way (Liu et al., 2004,
2006). The reference is therefore not a section of any real wing at any real
spanwise position, so any claim that the prior points at the seagull section
is loose.

**The surrogate is a gradient source only.** Its gap on generated shapes is
2.11 times its held-out reference error, which bounds the quality of the
gradient the generator was trained against and bounds no reported outcome.
Ensemble disagreement is used as a gating signal and is not claimed to
estimate solver error, and no correlation between disagreement and true error
is reported because none is computed and none exists to adopt.

**The plausibility band is pre-registered and it never fired.** Zero converged
points were dropped in either arm, the pre-solver filter rejected zero of 220
shapes, and the geometric safeguard was identically zero throughout training.
A guard that never fires is a guard that was never tested.

**The hinge form carries a published optimization caution.** The form is
established under free bits, delta-VAE and divergence thresholding, reported
at second hand (Felhi et al., 2020; Zhu et al., 2020), and the
reported caution is that the hinge is non-smooth and creates its own
optimization difficulties. This study uses the hinge and does not claim the
criticism fails to apply, since nothing here measures whether the non-
smoothness affected training and no diagnostic in this build would have
detected it.

## 4.7 Future work

Four items follow. Each closes a question this study raises and cannot answer from
inside its own design. Items that would merely extend the present sample are
absent, for the reason the optional stopping limitation gives.

**A matched non-biological baseline.** The design is a third arm carrying the
same architecture, seed library, objective and weight selection procedure,
with the reference in the prior term replaced by a fixed non-biological
section of comparable geometric elaboration, run against the same requested
band with the same paired construction. This study measured that a prior
pointing at a named section moves shapes toward that section and produces no
detected change in target satisfaction. It could not distinguish whether
either follows from the reference being avian or from its being a fixed
reference at all. A matched arm answers that directly, and it would partially
address the overlap limitation, since avian geometry would still be present in
the training data but present equally in both arms. One design question is
part of the work rather than settled before it, since what makes a non-
biological reference matched needs a definition.

**Higher fidelity verification where the panel method is weakest.** Take a
subset of the shapes this study already generated and re-evaluate them under a
transition-sensitive RANS method or in a tunnel, concentrating on the
conditions where the panel method's documented limits bite hardest. The subset
matters more than a uniform sample, because a shape whose efficiency depends
on an extended laminar run is exactly the shape the solver is least able to
adjudicate and also the shape most likely to post a high efficiency value.

**A spanwise family rather than one composite section.** Replace the single
reference with a family of sections reconstructed across the source's stated
valid spanwise range, so the prior points at a region defined over that family
rather than at the neighborhood of one composite point. The region extent in
this study is derived from the perturbed avian family's own spread around one
reference, while a spanwise family would supply a spread that is
aerodynamically meaningful rather than sampler-generated.

**A robust primary outcome committed in advance.** No source located gives a
decision rule, formula or threshold for choosing between a mean and a robust
location statistic from information available before data collection. That
reads as a gap in the methodological literature for paired computational
experiments generally.
The work would be a study that commits a robust location statistic as its
primary outcome before its own difference distribution is known, and states
the grounds on which it did so, with the contribution being the grounds rather
than the result.

---

# 5. Conclusion

No published work was found placing a biologically derived shape prior as a flag
gated objective term in a conditional generative model for fixed wing inverse
airfoil design. That absence rests on ten literature scans and is strong evidence
rather than proof.

A conditional VAE was trained on 982 solver-labeled sections in a shared order-9
class shape transformation basis, with a squared hinge penalty pulling freely
generated geometry toward a measured seagull reference, gated by a flag so that
one trained model produces both arms. The arms were compared on paired latent
codes across eleven requested efficiency targets, with every reported value from
the solver.

On 109 analyzed pairs the prior produced no detected difference in normalized
target satisfaction error. The mean paired difference is 0.011411 with a 95
percent interval of [-0.0022678114, 0.0251453683], which spans zero, and every
robust location measure points the same way without any interval determining the
sign. The mechanism verifiably engaged, since the prior moved generated shapes
0.3627 closer to the reference in 72.3 percent of paired comparisons and moved
the prior-on arm inside a region whose extent was fixed before training. Both
arms performed inverse design at roughly fifteen times the measured
condition-blind baseline. Generative diversity fell by a factor of 2.54.

That does not say the prior helps, and it does not say the prior does not help,
since no equivalence test was pre-registered. It does not attribute the measured
geometric control to the reference being biological, because the study compares a
prior against no prior, and because the avian anchor and its published analogue
both sit in the training data of both arms. The control that separates them is a third arm carrying a matched
non-biological reference. No study located runs one.

---

# Code availability

The code supporting this study is available at the provided Github repository:
https://github.com/MichaellCollado/conditional-variation-autoencoder-cVAE-for-inverse-airfoil-design.

The repository holds the pipeline that produced every number reported here. It
carries the geometry and dataset construction code, the interface that calls the
solver, the training code including the flag gated objective term, the surrogate
ensemble, the weight sensitivity sweep, the verification suite, the evaluation
and interval estimation code, and the figure driver. It also carries the five
seed coordinate files under `seeds/`, the parameter file recording the
automated checks and thresholds with their committed values, and the
pre-registration document summarized in Appendix A.1.

The repository does not carry the solver. XFOIL is third-party software under
the GNU General Public License version 2 or later, and is obtained separately at
the version recorded in Table A3. No pretrained model is included, since both
models are trained from the labeled dataset by the code in the repository.

---

# Appendix

## A.1 Pre-registration record

The pre-registration was issued after the model was trained and before any
generated shape was passed to the solver. Training artifacts were available when
it was written, including the training and validation curves, the checkpoint
selection metric, the latent dimension diagnostics, the surrogate's held-out
error, the weight sensitivity sweep and the mechanism gate. No evaluation result
was available and none had been produced. The prior weight was selected from the
sweep, which is a training-stage artifact, so the weight was set with the training
evidence visible and the evaluation evidence not yet in existence.

One primary outcome is committed, being the mean paired difference in
normalized target satisfaction error with a 95 percent interval from a wild
cluster bootstrap-t resampling whole requested-target clusters at 9999
resamples. The mean was chosen on convention, since no published rule was
located for choosing between a mean and a robust location statistic before the
shape of the distribution is known. A vulnerability was declared in advance, being
that the mean is sensitive to a small number of large excursions and that a
converged solution carrying an unusually long laminar run can return an
extreme but internally consistent efficiency value. A rule selecting the
statistic from the realized skewness was considered and rejected, because it
would make the primary outcome depend on the analyzed data.

Four secondaries are committed, each answering a question the primary cannot.
The median paired difference reports where the bulk of pairs sits rather than
how much closer the average pair landed. The paired win fraction reports how
often the prior helps, independent of how much. The difference between the
arms' tracking slopes reports whether the prior changed conditioning behavior
rather than only error. The difference between the arms' admission rates
reports whether the prior changed which shapes survived to be measured.
Trimmed means are declared a sensitivity check. The prior mechanism metrics
are gate quantities measured before evaluation and are not evaluation
outcomes.

No directional prediction is made. The study does not predict that the prior will
improve target satisfaction, does not predict that it will degrade it, and does not
predict that any effect will be larger or smaller in any part of the requested
range. The primary and every secondary are two-sided, and no subgroup analysis is
pre-registered.

The region extent was set as the 90th percentile of the avian family's own
distances to the reference. That percentile is a hand-set value rather than a
derived one. No published derivation procedure for such a region was found, and
every located variant of the thresholded-penalty family reports its threshold as
a hand-set hyperparameter. The value is disclosed as a choice on the same terms
as the seed library and the validation fraction in Table A1. It was not searched
to produce a particular separation between the avian and non-avian populations,
and the separation counts in Table 4 follow from the construction rather than
motivating it.

The reporting rules commit three things. An interval spanning zero is
reported as no detected difference. Absence of an effect is not claimed without
a pre-registered equivalence test. A disagreement between the mean and the
median is reported as a finding with equal prominence rather than resolved by
selecting a winner. The admission rules commit the minimum converged point
count, the per-point physical plausibility criterion, complete-case pairing,
and that the admission field is read explicitly with no default value. The
sweep size rule commits a floor of 100 analyzed pairs and requires the floor
result reported beside any extended result. No extension occurred. The
document forbids in-place amendment, so one departure is recorded here rather
than corrected. The diversity metric's committed grid reads as 11 evenly
spaced normalized targets from 0.0 to 1.0 in the value column while the same
row's rule column reads averaged across the requested range, and it was
computed on the requested band, following the rule column.

## A.2 Committed parameters

**Table A1.** Every fixed parameter value committed before evaluation, with the
rule by which each was set. Values are given at the precision the pipeline reads.

| Parameter | Value | Rule by which it was set |
|---|---|---|
| CST order per surface | 9, giving 20 coefficients | Convergence of the target force prediction under 2 percent, orders 3 to 13 |
| Seed library | Seagull anchor, S1223, E387, SD7003, SG6043 | Author's choice, disclosed as such |
| Perturbation width | Uniform on [-0.05, +0.05] per coefficient | Smallest searched width spanning the geometric target range within 5 percent of each edge |
| Plausibility thickness band | 0.068024 to 0.145590 chord | Seed range times a margin fraction of 0.20 |
| Plausibility camber curvature ceiling | 0.0003794664188666495 | Largest seed value times 1.20 |
| Reynolds number, Mach number | 300,000, 0 | Stated operating point, inside the avian range reported by Liu et al. (2004) |
| Transition amplification factor | 9 | Average wind tunnel entry in the solver's documented table |
| Angle sweep | 0 to 8 degrees inclusive, 1 degree step, 9 points | Attached flow envelope, below expected leading edge separation |
| Panel count, iteration limit | 220, 400 | Stated values |
| Per-call timeout | 7.56 s | 95th percentile of measured successful solve times times a margin of 2.0 |
| Label range | 50.49872773536895 to 172.17391304347828 | Realized on the committed population |
| Validation fraction | 0.20, stratified by family | Conventional round figure, disclosed as such |
| Standardization row set | Training split alone, 785 of 982 rows | Excludes validation distributional information from training-time scaling |
| Flag-clear fraction, requested and realized | 0.10 requested, 0.095723 realized | Smallest candidate leaving every well-populated label decile at least 5 flag-clear rows |
| Prior term weight | 5.862918756788936 | Highest score on the ladder, ties to the smaller weight |
| Region extent | 1.3003770283589529 | 90th percentile of the avian family's own distances to the reference |
| Target consistency weight | 99.09702261385384 | Same selection rule, own ladder |
| Ensemble spread weight | 54.307616878345584 | Same selection rule, own ladder |
| Divergence weight, warmup | 1.0, linear ramp over 20 epochs | Exact evidence lower bound on the locked model family |
| Safeguard weight | 1.0 | Stated value. Parity point undefined, since the term is exactly zero at initialization |
| Surrogate ensemble | 5 members, width 32, depth 2, Adam at 0.001, 200 epochs, batch 64 | Stated configuration, trained once and frozen |
| Checkpoint selection metric | Unweighted sum of reconstruction and target satisfaction on validation | Excludes every weighted regulariser, including the prior |
| Selected checkpoint | Epoch 146, six live latent dimensions | Committed metric |
| Requested target band | 0.08761877789267145 to 0.5522084150555188, 11 targets | 5th to 95th percentile of the training split's normalized labels |
| Sweep floor, launch target | 100 analyzed pairs, 101 pairs launched as 110 | Floor divided by a measured pair yield of 0.990909 |
| Minimum converged point count | 8 of 9 | Smallest k at mean absolute relative bias at most 0.010 and 95th percentile at most 0.020 |
| Latent dimension liveness threshold | 0.01 per-dimension divergence | Stated value |

Parameters omitted from Table A1 because the body states them in full are the
class function exponents, the interior measurement grid, the conditioning vector
layout, the diversity definition, the interval level and resample count, and the
committed training length.

## A.3 Thresholds, guards and environment

**Table A2.** Thresholds and guards that constrain the pipeline but appear in no
reported outcome. None of the three guards fired on the evaluated run.

| Item | Value or criterion | Where it acts |
|---|---|---|
| Geometric safeguard, thickness floor | 0.055426321765484482 chord | Training objective, both passes |
| Geometric safeguard, curvature ceiling | 0.00010535145925184528 | Training objective, both passes |
| Pre-solver plausibility filter | No surface crossing, thickness inside the stated band, camber curvature below the stated ceiling | Before any solver call |
| Post-convergence plausibility band | Drag strictly positive, both transition locations inside [0, 1] chord | On converged output, before the maximum |
| Gate zero, solver responsiveness | Converged status on all 9 angles for one known airfoil | Before evaluation |
| Gate one, pipeline consistency | Relative difference at or below 0.01 on 25 dataset rows of known label | Before evaluation |
| Mechanism gate, direction consistency | At least 0.60, against a chance value of 0.50 | Between training and evaluation |
| Mechanism gate, mean distance separation | At least 0.1300377028358953, being 0.10 of the region extent | Between training and evaluation |
| Mechanism gate, arm effect against noise | At least 0.25 | Between training and evaluation |
| Arm blindness tolerance | Exactly 0.0 | Between training and evaluation |
| Truncation tolerance, mean and tail | 0.010 and 0.020 absolute relative bias | Admission rule derivation |
| Secondaries' width agreement tolerance | 0.40 | Estimator validation, failed and reported |

The automated checks and thresholds are recorded in the build's parameter
file with their committed values and the step each is applied at. The register is
not reproduced here and it contains no result. Table A1 gives the committed
parameter values and Table A2 the thresholds and guards that constrain the
pipeline without appearing in a reported outcome. The register also holds
operational guards, such as subprocess timeouts and loader tolerances, that bear
on no reported number. The file itself is in the repository named under Code
availability.

**Table A3.** Environment and measured run cost. Every training run, every solver
call and every analysis ran on the CPU of one machine, with no cluster, no
accelerator and no second machine.

| Item | Value |
|---|---|
| Solver | XFOIL 6.99, GNU General Public License version 2 or later, Copyright (C) 2000 Mark Drela, Win32 executable called as a subprocess |
| CPU, memory | 11th Gen Intel Core i5-11400H, 6 physical cores, 12 logical, 2.70 GHz base, 15.74 GB |
| Operating system | Microsoft Windows 11 Home Single Language, 10.0.26200, build 26200 |
| GPU used | None. The installed torch is the CPU build and CUDA is unavailable |
| Python, numpy, torch | 3.14.6, 2.5.1, 2.13.0+cpu |
| matplotlib | 3.11.1, used by the figure driver alone and by no pipeline module |
| Seed coordinate files | Five files under `seeds/`, at 61, 81, 61, 299 and 81 coordinate points. No license statement accompanies any of them and none is asserted |
| Pretrained models used | None. Both models are trained here from the labeled dataset |
| Dataset labeling pass | 1005 calls, 1548.3 s total, 1.541 s mean, 7.597 s maximum |
| Paired generation and evaluation | 220 calls, 265.3 s total, 1.206 s mean, 6.912 s maximum |
| Condition-blind baseline | 220 calls, 252.0 s total, 1.146 s mean, 5.861 s maximum |
| Weight sensitivity sweep | 21 distinct training runs in 309 s |
| Committed model and control | 2 training runs, 150 epochs each |

Every derived dataset in this build is produced by this build's own code from
the five seed files, and every source of randomness derives from one
documented base seed by a stated offset rule. Table A3 records the environment
the run was made in. That table alone does not let a third party repeat the run,
since it fixes the conditions rather than supplying the procedure. The code that
supplies the procedure is in the repository named under Code availability, and a
reader wanting to reproduce a reported number should start there and read the
table beside it.

> **Figure A1.** Training curves. Per-epoch validation objective components for
> the committed training run, with the selected checkpoint marked at epoch 146.
> `figures/figure_A1_training_curves.pdf`

> **Figure A2.** Latent dimension usage. Per-dimension divergence on the
> validation split, sorted, with the liveness threshold of 0.01 drawn. Six of eight
> dimensions clear the threshold, while the first two carry 87.1 percent of the
> total divergence and the first three carry 98.2 percent. The figure shows how
> steeply capacity concentrates, which a count of live dimensions conceals. Values
> are computed at the final epoch's validation-split posterior rather than at the
> selected checkpoint's.
> `figures/figure_A2_latent_dimension_usage.pdf`

---

# References

Ajayi, A. S., Osimene, E. E. G., Hanrahan, B. C., Segun, S. E., David, B. A., Isaac, A. E., & Ehigocho, M. P. (2026). Bio-inspired wing designs for UAVs and low-speed aircraft. Journal of Applied Sciences and Applications in Engineering, 2(1), 1–11.

Anusonti-Inthra, P. (2024). Airfoil parameterization using an orthogonal class shape transformation. In AIAA SciTech 2024 Forum (AIAA 2024-2140). https://doi.org/10.2514/6.2024-2140

Bai, Y., Hsieh, M. H., Liu, J., & Tabord-Meehan, M. (2024). Revisiting the analysis of matched-pair and stratified experiments in the presence of attrition. Journal of Applied Econometrics, 39(2), 256–268. https://doi.org/10.1002/jae.3025

Bell, M. L., Kenward, M. G., Fairclough, D. L., & Horton, N. J. (2013). Differential dropout and bias in randomised controlled trials: When it matters and when it may not. BMJ, 346, e8668. https://doi.org/10.1136/bmj.e8668

Berger, M., Pellegrini, S., Senfter, T., & Pillei, M. (2026). AI-driven prediction of aerodynamic coefficients using VAE-GAN and MLP models for 2D airfoils. Results in Engineering, 29, 109504. https://doi.org/10.1016/j.rineng.2026.109504

Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. The Lancet, 327(8476), 307–310. PMID: 2868172.

Brunelli, C., Avirovic, M., Janssens, B., Marinus, B. G., Hillewaert, K., & Runacres, M. (2026). Prediction of laminar separation bubble on airfoils at low Reynolds number. Flow, Turbulence and Combustion, 116, Article 27. https://doi.org/10.1007/s10494-025-00727-7

Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. The Review of Economics and Statistics, 90(3), 414–427. https://doi.org/10.1162/rest.90.3.414

Chen, Q., Wang, J., Pope, P., Chen, W., & Fuge, M. (2022). Inverse design of two-dimensional airfoils using conditional generative models and surrogate log-likelihoods. Journal of Mechanical Design, 144(2), 021712. https://doi.org/10.1115/1.4052846

Chen, W., & Ahmed, F. (2021). PaDGAN: Learning to generate high-quality novel designs. Journal of Mechanical Design, 143(3), 031703. https://doi.org/10.1115/1.4048626

Creswell, A., Bharath, A., & Sengupta, B. (2017). Conditional autoencoders with adversarial information factorization (arXiv:1711.05175). arXiv. https://doi.org/10.48550/arXiv.1711.05175

Demie, A. B., Ancha, V. R., & Kahsay, M. B. (2026). Multi-model assessment and experimental validation of a custom high-camber airfoil for wind-lens technology application. Preprints. https://doi.org/10.20944/preprints202605.0263.v1

Drela, M. (2001). XFOIL documentation. Massachusetts Institute of Technology. https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt

Fazliani, S., Chawla, K., Guo, J., Shen, Y., Ihme, M., & Udell, M. (2026). ShapeBench: A scalable benchmark and diagnostic suite for standardized evaluation in aerodynamic shape optimization (arXiv:2605.20763). arXiv. https://doi.org/10.48550/arXiv.2605.20763

Felhi, G., Roux, J. L., & Seddah, D. (2020). Disentangling semantics in language through VAEs and a certain architectural choice (arXiv:2012.13031). arXiv. https://doi.org/10.48550/arXiv.2012.13031

Graves, R., & Barati Farimani, A. (2024). Airfoil diffusion: Denoising diffusion model for conditional airfoil generation (arXiv:2408.15898). arXiv. https://doi.org/10.48550/arXiv.2408.15898

Heyrani Nobari, A., Chen, W., & Ahmed, F. (2021). PcDGAN: A continuous conditional diverse generative adversarial network for inverse design. In Proceedings of KDD '21. https://doi.org/10.1145/3447548.3467414

Jiang, Z., Ma, Y., & Xiong, Y. (2023). Bio-inspired generative design for engineering products: A case study for flapping wing shape exploration. Advanced Engineering Informatics, 58, 102240. https://doi.org/10.1016/j.aei.2023.102240

Jindal, S., & Wang, X. E. (2021). CUDA-GHR: Controllable unsupervised domain adaptation for gaze and head redirection. In 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (pp. 467–477). (arVix:2106.10852). arXiv. https://doi.org/10.48550/arXiv.2106.10852

Kallstrom, K. (2022). Exploring Airfoil Table Generation using XFOIL and OVERFLOW. https://ntrs.nasa.gov/api/citations/20220000003/downloads/1538_Kallstrom_011822.pdf

Kulfan, B. M. (2008). Universal parametric geometry representation method. Journal of Aircraft, 45(1), 142–158. https://doi.org/10.2514/1.29958

Li, R., Liu, S., Wang, G., Liu, G., & Zeng, B. (2021). JigsawGAN: Auxiliary learning for solving jigsaw puzzles with generative adversarial networks. IEEE Transactions on Image Processing, 31, 513–524. (arXiv.2101.07555). arXiv. https://doi.org/10.48550/arXiv.2101.07555

Liu, J., Wu, J., Xie, H., Zhang, G., Wang, J., Liu, W., Ouyang, W., Jiang, J., Liu, X., Tang, S., & Zhang, M. (2024). AFBench: A large-scale benchmark for airfoil design (arXiv:2406.18846). arXiv. https://doi.org/10.48550/arXiv.2406.18846

Liu, T., Kuykendoll, K., Rhew, R., & Jones, S. (2004). Avian wings (AIAA 2004-2186). 24th AIAA Aerodynamic Measurement Technology and Ground Testing Conference. https://doi.org/10.2514/1.16224

Liu, T., Kuykendoll, K., Rhew, R., & Jones, S. (2006). Avian wing geometry and kinematics. AIAA Journal, 44(5), 954–963. https://doi.org/10.2514/1.16224

Liu, X., & He, W. (2018). Airfoil optimization design based on the pivot element weighting iterative method. Algorithms, 11(10), 163. https://doi.org/10.3390/a11100163

Mandadzhiev, B. A. (2017). Design and aerodynamic analysis of an airfoil with a bioinspired leading edge device for stall mitigation at low Reynolds number operation [Conference paper]. https://api.semanticscholar.org/CorpusID:55465526 

Meng, X., & Tao, J. (2025). An airfoil inverse design method based on target testing conditional generative adversarial network. Acta Aeronautica et Astronautica Sinica, 46(10). https://doi.org/10.7527/S1000-6893.2024.31182

Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. Statistics in Medicine, 38(11), 2074–2102. https://doi.org/10.1002/sim.8086

Pascoa, J. (2016). XFOIL vs CFD performance predictions for high lift low Reynolds number airfoils. Aerospace Science and Technology. https://doi.org/10.1016/J.AST.2016.02.031

Regenwetter, L., Srivastava, A., Gutfreund, D., & Ahmed, F. (2023). Beyond statistical similarity: Rethinking metrics for deep generative models in engineering design. Computer-Aided Design, 165, 103609. https://doi.org/10.1016/j.cad.2023.103609

Sharpe, P., & Hansman, R. J. (2025). NeuralFoil: An airfoil aerodynamics analysis tool using physics-informed machine learning (arXiv:2503.16323). arXiv. https://doi.org/10.48550/arXiv.2503.16323

Sun, H., Huang, Y., Han, L., Fu, C., Liu, H., & Long, X. (2023). MTS-DVGAN: Anomaly detection in cyber-physical systems using a dual variational generative adversarial network (arXiv:2311.02378). arXiv. https://doi.org/10.48550/arXiv.2311.02378

Tunca, S. G., Özgür, M. A., & Koşar, O. (2026). Experimental investigation of the aerodynamic performance of Rhamphorhynchus muensteri-inspired airfoil profiles at low Reynolds numbers. Journal of Engineering Research, 14(1), 398–414. https://doi.org/10.1016/j.jer.2025.08.005

Wada, K., Yonekura, K., & Suzuki, K. (2024). Physics-guided training of GAN to improve accuracy in airfoil design synthesis. Computer Methods in Applied Mechanics and Engineering, 421, 116746. https://doi.org/10.1016/j.cma.2024.116746

Yang, M., Wang, Y., & Jiang, P. (2025). Research on aerodynamic performance prediction of airfoils based on a fusion algorithm of Transformer and GAN (arXiv:2506.06979). arXiv. https://doi.org/10.48550/arXiv.2506.06979

Yang, Z., Tang, M., Du, P., & Zou, Q. (2026). AirfoilGen: A valid-by-construction and performance-aware latent diffusion model for airfoil generation (arXiv:2605.20303). arXiv. https://doi.org/10.48550/arXiv.2605.20303

Yilmaz, E., & German, B. (2020). Conditional generative adversarial network framework for airfoil inverse design. In AIAA Aviation Forum (AIAA 2020-3185). https://doi.org/10.2514/6.2020-3185

Yonekura, K., Tomori, Y., & Suzuki, K. (2024). Airfoil shape generation and feature extraction using the conditional VAE-WGAN-gp. AI, 5(4), 2092–2103. https://doi.org/10.3390/ai5040102

Zamrai, M. A. H., & Mohd Yusof, K. (2025). Generative inverse design: From single-point optimization to a diverse design portfolio via conditional variational autoencoders (arXiv:2510.05160). arXiv. https://doi.org/10.48550/arXiv.2510.05160

Zhang, S., Ye, F., Wang, B., & Habetler, T. G. (2019). Semi-supervised learning of bearing anomaly detection via deep variational autoencoders (arXiv:1912.01096). arXiv. https://doi.org/10.48550/arXiv.1912.01096

Zhou, H., Ma, R., Zhang, L., Gao, L., Mahdavi-Amiri, A., & Zhang, H. (2021). SAC-GAN: Structure-aware image composition. IEEE Transactions on Visualization and Computer Graphics, 30, 3151–3165. [DOI pending. The arXiv preprint is 2112.06596.]

Zhu, Q., Bi, W., Liu, X., Ma, X., Li, X., & Wu, D. O. (2020). A batch normalized inference network keeps the KL vanishing away (arXiv:2004.12585). arXiv. https://doi.org/10.48550/arXiv.2004.12585

