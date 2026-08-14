> **This is the pre-registration as issued, reproduced unchanged.**
>
> It is the document Appendix A.1 of `PAPER.md` summarises, and it is included
> because Code availability says the repository carries it. It was issued after
> the model was trained and before any generated shape reached the solver.
>
> Its own numbering, 2.9.1 to 2.9.12, is the numbering it carried when issued.
> `PAPER.md` has no section 2.9; the article summarises this document at
> Appendix A.1 instead. The numbers below are left as they were issued because
> the document forbids in-place amendment, and renumbering it here would be an
> amendment. Where the code refers to a commitment made here, it names the
> pre-registration rather than a section number.
>
> One departure from this document is recorded in Appendix A.1 of `PAPER.md`,
> concerning the diversity metric's grid. It is recorded there rather than
> corrected here, for the same reason.

---

# 2.9 Pre-registration

This subsection was written after the pipeline was rebuilt and trained and
before any evaluation run.

**On placement.** This sits at the end of Method, immediately before Results.
Placement carries the ordering signal structurally, so a reader reaching Results
has already passed every commitment. It also keeps the paper inside the standard
article structure, which a numbered section of its own would not have done.
Section 2.9.1 still states the timing explicitly, because placement makes the
claim credible rather than unnecessary.

Nothing follows this subsection inside Method. Everything committed here was
fixed before any number in Results existed.

No value is written here that has not been set, and no value is inferred.

---

## 2.9.1 Scope and timing

This pre-registration was issued after the model was trained and before any
generated shape was passed to the solver for evaluation. Training results were
available when it was written. These include the training and validation curves,
the checkpoint selection metric, the latent dimension diagnostics, the surrogate
ensemble's held-out error, the weight sensitivity sweep, and the prior mechanism
gate. No evaluation result was available, and none had been produced.

The distinction matters because the prior weight was selected from the sweep, and
the sweep is a training-stage artifact. The weight was therefore set with the
training evidence visible and the evaluation evidence not yet in existence. That
ordering is the point of issuing this document at this moment rather than later.

Everything below is committed. Section 2.9.12 states the only route by which any
of it changes.

---

## 2.9.2 Primary outcome

There is one primary outcome.

**The mean paired difference in normalised target satisfaction error, with a
confidence interval produced by resampling whole requested-target clusters.**

For each matched pair, the error is the absolute difference between the achieved
efficiency and the requested efficiency, expressed in normalised units. The
paired difference is the prior-off arm's error minus the prior-on arm's error, so
a positive value indicates that the prior-on shape landed closer to what was
asked. The primary outcome is the mean of that series across all admitted pairs.

The interval level is 95 percent. The resample count is 9999, reported with the
target it was chosen for.

**The rule by which the statistic was chosen.** No published rule exists for
choosing between a mean and a robust location statistic before the shape of the
distribution is known. This is a documented absence rather than an oversight in
this study's search. The mean was chosen on convention, since it is the first
member of the standard family for paired comparisons, and because any other basis
available at this moment would depend on information about the realised
distribution that does not yet exist.

**A vulnerability declared in advance.** The mean is sensitive to a small number
of large excursions. In this pipeline there is a known mechanism that can produce
them. The solver is documented to over-predict maximum lift and to be least
reliable in transitional regimes, and it cannot resolve the three-dimensional
breakdown of a laminar separation bubble. A converged solution carrying an
unusually long laminar run can therefore return an extreme but internally
consistent efficiency value. If such a value appears in one arm and not its
paired counterpart, it enters the primary outcome at full weight.

This is stated now, before any evaluation, so that it cannot later be presented
as a discovery. The consequence is committed at section 2.9.6.

**Why a conditional rule was rejected.** A rule selecting the statistic from the
realised skewness was considered and is not adopted. It would make the primary
outcome depend on the analysed data, which is the property a pre-registered
primary exists to avoid. The alternative taken is to fix the mean and to report
the robust measures alongside it always, which is section 2.9.3.

---

## 2.9.3 Secondary outcomes

**On the count.** No source found in this study's search designates a number of
secondary outcomes, and no source in the generative design literature uses the
primary and secondary vocabulary at all. What the returns support is a rule
rather than a count. Each pre-registered secondary answers a question the primary
cannot, and each is named here before evaluation. Four qualify.

**S1. The median paired difference**, with an interval from the same
cluster-resampling estimator as the primary.
It answers a question the mean cannot. The mean reports how much closer the
average pair landed. The median reports where the bulk of pairs sit. Where the
two disagree, the disagreement is itself the finding, and section 2.9.6 fixes how
it is reported.

**S2. The paired win fraction**, with a test statistic that accounts for the
clustering by requested target.
It answers how often the prior helps, which is independent of how much. A small
mean improvement produced by consistent small gains and one produced by a few
large gains are different results, and no location statistic separates them.

**S3. The difference between the two arms' target tracking slopes**, fitted over
matched pairs, with a cluster-resampled interval.
It answers whether the prior changed the model's conditioning behaviour rather
than only its error. A model can score well on target satisfaction while
substantially ignoring its request, and the primary outcome cannot detect that.

**S4. The difference between the two arms' admission rates.**
It answers whether the prior changed which shapes survived to be measured. If it
did, the paired comparison is drawn from two differently filtered populations,
and that would qualify the primary outcome rather than support it.

**What is not a secondary outcome.** Trimmed means are declared here as a
sensitivity check and not as an outcome. They are reported and they are not
promoted. The prior mechanism metrics, being distance to the reference, direction
consistency, and arm effect against sampling noise, are gate quantities measured
before evaluation and are not evaluation outcomes. Every other quantity in the
metrics specification is descriptive and is labelled as such in the results
table.

**Multiplicity position.** No correction is applied. Every statistic computed is
reported, none is withheld, and every measure that is not the primary or one of
the four named secondaries is marked post hoc in the table where it appears. The
field is split three ways on this question and no default exists, so the position
is declared rather than assumed. The marker convention is defined in the caption
of the table that uses it, because no published in-table convention for this
exists.

---

## 2.9.4 No directional prediction

**This study makes no directional prediction.**

It does not predict that the prior will improve target satisfaction. It does not
predict that the prior will degrade it. It does not predict that any effect will
be larger or smaller in any part of the requested range, at any efficiency level,
or for any subset of targets.

The primary outcome is two-sided. Every secondary outcome is two-sided. No
subgroup analysis is pre-registered, and any subgroup analysis that appears in
the paper is marked post hoc.

This is a change from the study's earlier commitments and it is stated as a
commitment rather than as a correction, because the earlier version is omitted
under section 2.9.10. The reason no prediction is made is that this study found
no published basis for one. No source places a biologically derived shape prior
as a gated objective term in generative inverse airfoil design, so there is no
literature from which to derive a directional expectation, and an intuition is
not a basis.

---

## 2.9.5 Admission rules

A generated shape enters the analysis only if all of the following hold.

1. It passes the geometric plausibility filter before reaching the solver.
2. The solver returns a converged result rather than a timeout, a convergence
   failure, or an environment fault.
3. The sweep converged on at least 8 of the 9 angle points of the requested
   sweep.
4. Every converged point contributing to the efficiency value satisfies the
   physical plausibility criterion, being drag strictly positive and both
   transition locations inside [0, 1] chord inclusive. The criterion is applied
   to converged results before the maximum is taken.

A pair enters the analysis only if both of its members satisfy all four
conditions. Pairing is complete-case. A pair with one surviving member is
excluded entirely and its surviving member is not used anywhere in the paired
analysis.

**Three commitments about the rules themselves.**

The admission field is read explicitly, with no default value. A record missing
the field is excluded and counted, not silently admitted.

The count of records excluded by rule 3 is reported with its denominator stated,
and it is reported even if it is zero. The denominator is the records that
produced an efficiency value, not the records launched.

Complete-case pairing is the standard default for paired designs and it is not
bias-free. The closest theoretical treatment finds limited evidence that dropping
incomplete pairs protects against attrition bias. This is stated as a limitation
and is not presented as a neutral choice.

**Missingness classification.** Missingness here depends on the generated shape
itself rather than on the arm it was generated in, since the solver's ability to
converge is a property of the geometry. It is therefore not missing completely at
random. This study does not assume it is ignorable. Secondary outcome S4 tests
whether the arms lost shapes at different rates, and the result is reported as a
marker in either direction. Unequal rates would not establish bias and equal
rates would not establish its absence, and neither reading is claimed.

---

## 2.9.6 Reporting rules

**An interval spanning zero is reported as no detected difference.** It is not
reported as evidence of no effect, and it is not reframed as a trend, a
suggestion, or a direction. The distinction is that an interval containing zero
is inconclusive about the sign of the effect, which is a statement about what the
data determined and not a statement about the world.

**If absence of an effect is to be claimed, an equivalence test is required.**
This study does not pre-register one, so it will not claim absence. It will
report that the interval did not determine the sign.

**Disagreement between the primary and the secondaries is reported as a
finding.** If the mean and the median point in different directions, or if one
interval excludes zero and another does not, both are reported with equal
prominence and the mechanism producing the divergence is explained. The
disagreement is not resolved by selecting whichever measure is more convenient,
and the primary is not quietly replaced.

**If the primary outcome fits the realised distribution poorly, that is
disclosed and the primary is still reported.** Section 2.9.2 declares the mean's
vulnerability in advance. If it materialises, the paper reports the mean as the
primary, reports the median beside it, states the mechanism, and does not
substitute one for the other. A deviation is not automatically a problem and an
unacknowledged deviation invalidates the inference. The commitment is to
acknowledge, not to avoid.

**Every reported statistic comes from one analysis module.** No number in the
paper is taken from a diagnostic printed during a generation run. Where any
in-run diagnostic exists at all, it is not reportable and is not reported.

**Units.** Target satisfaction error is reported in normalised units throughout.
Raw units may accompany it and never replace it.

**Rounding.** Decimal precision tracks each quantity's natural scale rather than
a single fixed rule. Full precision is retained in the recorded artifacts.

**The Monte Carlo error of the interval endpoints is reported separately from the
Monte Carlo error of the point estimate.** No claim is made about the stability of
any decimal place that this figure does not support.

---

## 2.9.7 Fixed parameter values

Every value below is fixed at the moment this document is issued. Each carries
the rule by which it was set. A value that appears here and nowhere in the
recorded parameter file, or in the parameter file and not here, is an error in
this document.

### Representation and dataset

| Parameter | Value | Rule by which it was set |
|---|---|---|
| CST order per surface | 9 | Author's choice, from a named justification type, being convergence of a target force prediction as the order increases, with the justification type stated |
| Coefficient count | 10 per surface, 20 in total | Follows from the order |
| Trailing edge thickness | Zero | Standard CST practice, stated and closed |
| Plausibility bounds, thickness and curvature | Maximum thickness inside [0.0680239706227634, 0.14558990124579074] chord. Peak absolute camber second difference at most 0.0003794664188666495. Margin fraction 0.20 | Derived from the seed population's own measured range with a stated margin, before any shape was sampled |
| Perturbation bound | 0.05, a per-coefficient additive interval on that seed's own coefficient, drawn uniform | Smallest per-coefficient interval whose accepted population spans a stated geometric range. Stated on the sampler, not enforced by rejection alone |
| Shapes per seed | 200 | Author's choice, disclosed |
| Validation fraction | 0.20 | Stated value. One split, used by both the surrogate and the generative model |
| Standardisation row set | The training split alone, 785 of 982 rows | Author's choice, disclosed, with the leakage claim scoped to it |

### Solver

| Parameter | Value | Rule by which it was set |
|---|---|---|
| Reynolds number | 300000 | Stated value, placed against the published avian range |
| Mach number | 0 | Stated value |
| Transition amplification factor | 9 | Stated value |
| Angle range and step | 0 to 8 degrees inclusive, 1 degree step, 9 requested points | Stated value |
| Panel count | 220 | Stated value |
| Iteration limit | 400 | Stated value |
| Per-call timeout | 7.56 seconds | A stated upper percentile of measured successful solve times, with a stated margin. Committed only after the count of successful solves it would reclassify was recorded, per family |

### Prior and objective

| Parameter | Value | Rule by which it was set |
|---|---|---|
| Avian reference | Seagull section, locked | Locked element |
| Region extent percentile | 90 | Author's choice, disclosed, with the absence of any published derivation procedure stated |
| Region extent | 1.3003770283589529 | The stated percentile of the avian family's own distances in standardised space. Stored and disclosed at full precision, unrounded |
| Null representation | A zero block of 20 columns, all 0.0, with a separate flag column | Stated value. A zero block with a separate indicator flag, chosen because it is the straightforward option and removes the need for a gating layer |
| Flag-clear fraction | 0.10 requested, 0.095723 realised | Large enough to support the independence test, fixed before training |
| Reconstruction weight | 1.0 | Implicit in the total objective, which adds reconstruction unweighted and scales every other term against it. There is nothing to select. Recorded because a weight that changes the objective and appears in no committed list is the failure this table exists to prevent |
| Divergence weight | 1.0 | Stated value on the locked model family, being the exact evidence lower bound. Not selected by the sweep procedure, because a selection rule applied to its own ladder returned zero, which is a conditional autoencoder and contradicts a locked element. The ladder is still run and reported as sensitivity evidence, and the committed value does not move with it |
| Prior term weight | 5.862918756788936 | Selected by a rule fixed before the sweep existed, applied to the sweep table alone, with no evaluation result seen |
| Target consistency weight | 99.09702261385384 | Same procedure |
| Spread penalty weight | 54.307616878345584 | Same procedure |
| Safeguard weight | 1.0 | Stated value. The safeguard term is identically zero on this build, so no sweep can identify the weight. Recorded here because an earlier version of this study carried it in no committed list |
| Safeguard curvature and thickness bounds | Thickness at least 0.055426321765484482 chord. Curvature at most 0.00010535145925184528 | Re-derived from the training population's own measured range |

### Model and training

| Parameter | Value | Rule by which it was set |
|---|---|---|
| Latent dimension | 8 | Stated value, reported as run procedure, with the usage diagnostic reported |
| Hidden width and depth | Width 64, depth 2 | Stated value, reported as run procedure |
| Optimiser settings, batch size, epoch count | Adam at learning rate 0.001, batch size 64, 150 epochs | Stated value, reported as run procedure |
| Divergence warmup schedule | Linear ramp of the schedule multiplier from 0 to 1 over the first 20 epochs, held at 1 after | Stated value, reported |
| Checkpoint selection metric | Unweighted reconstruction plus target satisfaction, on the validation split | Stated value. Defined on unweighted components so the criterion does not move when a weight moves. The divergence, safeguard, spread and avian terms are excluded |
| Surrogate member count, width, depth, schedule | 5 members, width 32, depth 2, Adam at learning rate 0.001, batch size 64, 200 epochs | Stated value, reported as run procedure |

### Evaluation and analysis

| Parameter | Value | Rule by which it was set |
|---|---|---|
| Requested target band | 0.08761877789267145 to 0.5522084150555188 normalised, at 11 evenly spaced targets with both endpoints included. Raw equivalent 61.15975877192982 to 117.68878896594654 | Stated value, in normalised units, reported |
| Samples per target | 10 | Derived from the floor and the measured pair yield |
| Minimum converged sweep points | 8 of the 9 requested | The smallest count whose mean and upper-percentile truncation bias both sit inside a tolerance stated before the analysis ran |
| Truncation tolerance | Mean absolute relative bias at most 0.010, and 95th percentile absolute relative bias at most 0.020 | Stated before the truncation analysis was run |
| Plausibility band on converged results | Per converged point, drag strictly positive and both transition locations inside [0, 1] chord inclusive | Stated value. Pre-registered here, which an earlier version of this study did not do |
| Consistency gate tolerance | 0.01 relative, required on every one of 25 gate rows | Author's choice, disclosed, with the absence of any published tolerance stated |
| Prior mechanism gate thresholds | Arm blindness bit-identical, tolerance exactly 0.0. Gate response exactly 0.0 with flags clear and strictly above 0.0 with flags set. Direction consistency at least 0.60. Mean distance separation at least 0.10 of the region extent. Arm effect against sampling noise at least 0.25 | Author's choice, each recorded before the gate was run |
| Interval level | 95 percent | Stated value |
| Bootstrap resample count | 9999, chosen for a confidence interval and for the Monte Carlo error of that interval's endpoints | Author's choice, reported with the target it was chosen for |
| Cluster unit and bootstrap method | The requested target, giving 11 clusters. The wild cluster bootstrap-t with Rademacher weights, with the unrefined percentile cluster bootstrap reported alongside it. With 11 clusters the Rademacher weights admit at most 2048 distinct weight vectors, so the bootstrap distribution is supported on at most 2048 points however many resamples are drawn, and that bound is reported with the interval | Stated value, chosen before evaluation. Several samples share each target, so the pairs are not independent. At this cluster count the refinement is required rather than optional, since standard cluster robust tests over-reject in the five to thirty cluster range |
| Base seed and derivation rule | 20260806. Every stream is the base seed plus 1000 times the build step number plus a substream index | One base seed. Every downstream generator is derived from it by a documented offset rule, and every distinct source of randomness is recorded |
| Diversity sample count and statistic | 20 samples at each of 11 evenly spaced normalised targets from 0.0 to 1.0 inclusive, which is the full conditioning range and not the requested target band. The statistic is the mean pairwise Euclidean distance in standardised CST coefficient space | Stated value. One definition, averaged across the requested range |

---

## 2.9.8 Sweep size

**The floor is stated in expected yield, not in launched runs.**

The committed minimum is 100 **analysed pairs**. An analysed pair is one in which
both members satisfied every admission rule in section 2.9.5.

The launch target is derived rather than chosen. It is the floor divided by the
measured pair yield, which is the fraction of launched pairs in which both
members cleared admission. The yield was measured on a pilot run of this
pipeline, at 0.990909, being 109 of 110 pilot pairs in which both members cleared
admission, and is not adopted from published convergence rates, which are
pipeline specific and vary widely across solvers and regimes.

The resulting launch target is 101 pairs. At the 11 requested targets this is
launched as 10 samples per target, which is 110 pairs, giving 220 solver calls
across both arms.

**Why yield and not launched runs.** The launched count is not the inferentially
meaningful figure. A paired design loses a pair when either member fails, so the
pair-level yield is lower than the single-shape admission rate whenever failures
are not perfectly correlated. A floor stated in launched runs would silently
become a smaller floor in analysed pairs. Both counts are reported, and the floor
is expressed in the one the analysis actually uses.

**Extension above the floor is permitted.** It is not required, and it is not
planned.

**Every extension is reported with three things.**

1. **Its trigger.** The condition that prompted it, stated as it was stated at
   the time.
2. **The result at the decision point.** Every quantity that was visible when the
   decision was taken, including the primary outcome and its interval at that
   moment.
3. **The resulting size.** The new launch target and the analysed pair count that
   followed.

An extension whose trigger, interim result, and resulting size are not all three
reported is a protocol violation and is reported as one.

**The floor result is reported alongside the final result.** If the sweep is
extended at all, the paper reports the primary outcome and its interval computed
on the first 100 analysed pairs, beside the primary outcome and its interval
computed on the full extended set. Both appear in the same table. A reader can
therefore see what the committed sample alone determined, without relying on the
extension being disinterested.

---

## 2.9.9 Optional stopping

**The floor bounds the risk from optional stopping. It does not remove it.**

A committed minimum prevents the specific failure of stopping as soon as a result
looks favourable, because no result below the floor is reportable. It does not
prevent the more subtle failure of continuing past the floor until a result
becomes favourable, and then stopping.

The published treatment of this is direct. Adding sample in pre-specified graded
increments with a correspondingly adjusted criterion at each look avoids
inflation. A naive augmentation with an unchanged criterion can substantially
raise the error rate. This study pre-specifies no graded increment schedule and
applies no adjusted criterion, so the residual risk is not eliminated.

**What bounds it here.**

The floor result is always reported beside the final result, per section 2.9.8,
so any divergence between them is visible rather than hidden.

Every extension's trigger and interim state are reported, so a reader can see
what was known at each decision point.

No directional prediction exists, per section 2.9.4, so there is no favourable
direction that an extension could be steered toward without that steering being
visible in the reported triggers.

**What remains.** This is a limitation and it is stated as one in the paper's
limitations section, not only here. A reader who believes the extension decisions
were outcome-influenced has the floor result available to check against, which is
the strongest protection this design offers and is weaker than a pre-specified
increment schedule with an adjusted criterion would be.

One asymmetry is worth stating. Stopping at the floor, rather than extending,
cannot inflate the error rate on its own. The risk lies entirely in the extension
direction.

---

## 2.9.10 Treatment of the superseded build

**An earlier version of this pipeline was built, trained, evaluated, and frozen
under a previous analysis commitment. That version is omitted from this paper in
full.**

Omission here means the following.

No result from that version appears in this paper, in any section, including the
appendix.

No count, interval, threshold, or diagnostic value from that version is carried
forward, and none informs any commitment in this document.

No narrative of that version's development appears. The paper does not describe
what was tried, what was measured, what was changed, or in what order.

No claim in this paper is supported by evidence produced under that version. Any
design choice that previously rested on such evidence is either supported
independently, per section 2.9.11, or is not made.

**Why omission rather than disclosure.** A superseded build's numbers are not
evidence, because the commitments under which they were produced are void. Partial
disclosure would place void figures beside live ones in the same document, where a
reader could not reliably tell which was which. Full omission is the treatment
that keeps every number in the paper answerable to this pre-registration.

**What omission does not mean.** It does not mean the earlier work is denied. If
asked, the authors state that a prior build existed and was set aside. It means
the paper's evidentiary base contains one build.

---

## 2.9.11 Independent grounding for retained design choices

Several design choices were originally adopted for reasons that lived in the
omitted build. Each is retained only because independent grounding now supports
it. Where no such grounding was found, the choice is stated as this study's own
rather than as following practice.

**The prior acts through the training objective rather than the conditioning
vector.** Originally justified by this study's own measurement that a
conditioning-vector mechanism was inert. Now grounded independently. A
conditioning component that is constant within its class carries no conditional
mutual information with the output, and classifier free guidance presupposes that
the dropped condition is informative, so that dropping it changes the generative
distribution. Where it carries none, the conditional and unconditional models
coincide. The mechanism change follows from published results rather than from an
omitted measurement.

**The hinge form of the prior penalty.** Originally justified by internal
reasoning. Now grounded in the established family of thresholded penalties that
are inactive inside a region and active outside it, with the published caution
that the hinge form is non-smooth and introduces its own optimisation
difficulties. That caution is carried into the limitations.

**Timeouts are separated from convergence failures.** Originally an internal
convention. No published precedent was found for the separation, so it is now
stated as this study's own choice, and its effect on every reported denominator
is shown rather than assumed to be neutral.

**The extreme converged result, if one occurs, is retained.** Originally an
internal decision. Now grounded in published practice, where such results are
retained with a stated mechanistic caveat and no statistical outlier test is
applied to a converged aerodynamic result.

**The consistency gate on the evaluation path.** Originally an internal check. No
published precedent was found for the procedure and none for its tolerance, so
both are stated as this study's own.

**The prior mechanism gate's thresholds.** Originally set by tightening until a
zero-weight control failed, which is calibration against an outcome. Now every
threshold is recorded before the gate runs, and the zero-weight control is
required to fail against thresholds it did not set.

**The seed library and the avian reference.** Locked inputs. The avian section
also appears among the seeds, so avian geometry is present in the training
distribution of both arms. The ablation therefore measures the marginal effect of
the objective term given that presence, and not the effect of avian information
as such. This is argued as a limitation and is not repaired, because both the
seed set and the reference are fixed. The limitation is wider than the seagull
alone, since the published analogue of the avian section is also among the seeds.

---

## 2.9.12 Amendment

**There is no in-place amendment to this pre-registration.**

No value in section 2.9.7 is changed after issue. No outcome in sections 2.9.2 or
2.9.3 is redefined, reordered, or reweighted. No admission rule in section 2.9.5
is loosened or tightened. No reporting rule in section 2.9.6 is relaxed.

**A commitment is void only by the following route.** The pipeline is re-run from
the start, and a new pre-registration is issued before the new evaluation begins.
The re-run is complete rather than partial, and the new document supersedes this
one entirely rather than editing it.

**A deviation discovered during analysis is reported as a deviation.** It is not
corrected in this document. The paper states what was committed, what was done,
and why they differ. An acknowledged deviation can increase the validity of a
result when it responds to a violated assumption. An unacknowledged one
invalidates the inference regardless of its merit.
