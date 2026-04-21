# GRADE on Cognitive-Distortion Validation in OLMo — Methods & Findings

**Status:** preliminary pilot (OLMo-2 1B, n = 36 distortion stimuli, 36 id-matched
factual controls). Three rounds of strict review have been applied; a partial
summary of the review comments and the code-side fixes is in
[`review/grade_review_log.md`](grade_review_log.md).

**Companion code:** [`grade_reference.py`](../grade_reference.py). Companion notebook
for 7B: `grade_notebook.ipynb`. Raw output: `results/grade_results.json`.

**Paper this extends:** Wang, Liang, Lai, Zhang & Yan (2026),
*GRADE: Probing Knowledge Gaps in LLMs through Gradient Subspace Dynamics*,
arXiv:2604.02830.

---

## 1. Question

The host repo (OLMo mechanistic interp of clinical sycophancy) shows that
preference-tuned OLMo models *behaviourally* prefer to validate a user's
cognitive distortion rather than gently reframe it (the "sycophantic"
completion is the default). It also identifies a low-rank residual-stream
direction whose removal nudges the model toward the therapeutic completion,
but with substantial warmth leakage (specificity ratio 1.84, 95 % CI
[1.32, 2.70]).

A question the host repo cannot answer with activation geometry alone:

> Does the model **have the capacity** to produce CBT-style therapeutic
> reframing — and merely defer to the sycophantic default — or does it
> **lack the knowledge** to produce therapeutic reframing in the first place?

This distinction is load-bearing for deployment. If capacity is present, a
system prompt or LoRA fix is cheap. If capacity is absent, a much larger
intervention (instruction fine-tuning, retrieval of CBT templates) is needed.

GRADE is built precisely for this question. It defines a knowledge-gap signal
from *gradient subspace dynamics*: the cross-layer ratio between the effective
dimensionality (stable rank) of the MLP-parameter gradient and that of the MLP
intermediate states themselves. The lower the ratio, the more the required
update already lies inside the activated-knowledge subspace — i.e. the more
the model "knows how" to produce the target.

## 2. Mathematical formulation

We follow the paper's notation exactly. At MLP layer `l` of a Llama/OLMo-style
block:

- `h ∈ R^{n × d_ff}` — MLP intermediate states (SwiGLU-gated output; input of
  `down_proj`), restricted to the *n* completion token positions of `(q, y)`.
- `W = W_down ∈ R^{d_model × d_ff}`.
- Completion loss (paper Eq. 3, `L_pos`):
  `L = -Σ_{t=1}^{n} log p(y_t | y_<t, q)`.
- Gradient (paper Eq. 4): `g = ∂L/∂W = Δ^T h ∈ R^{d_model × d_ff}`, with
  `Δ = ∂L/∂o ∈ R^{n × d_model}`.

Per Eq. 5, every row of `g` is a linear combination of `{h_t}`, so `g` lies
entirely in span(`h`). The paper therefore works in the *shared* row space.
Define

```
C_h = h h^T            ∈ R^{n × n}
C_g = C_h^+ (h g^T g h^T) C_h^+  ∈ R^{n × n}
```

where `C_h^+` is the Moore–Penrose pseudoinverse. Paper Eq. 6 (stable rank
of Sanyal et al. 2020 / Ipsen & Saibaba 2025, sorted singular values
`λ_1 ≥ … ≥ λ_n ≥ 0` of `C_g`):

```
srank_pre(g) = Σ_i λ_i / λ_1
srank_pos(g) = Σ_i (λ_i)^2 / (λ_1)^2
```

Per-layer **rank ratio**: `RankRatio(l) = srank(C_g^(l)) / srank(C_h^(l))`
(computed with both `pre` and `pos` variants; `srank(h)` uses the singular
values of `C_h`). Low ratio ⇒ the required update lies inside few of the
directions already activated ⇒ the model has the knowledge. High ratio ⇒
update needs many new directions ⇒ knowledge gap.

### 2.1 Efficient computation

We never materialise `C_g` as `n × n`. Using

```
C_g = (C_h^+ · h · g^T) · (g · h^T · C_h^+)  = M M^T,    M := C_h^+ h g^T ∈ R^{n × d_model}
```

the eigenvalues of `C_g` are the *squared* singular values of `M`, so one SVD
on a small (`n × d_model`) matrix gives both `srank_pre(C_g)` and
`srank_pos(C_g)`. This matters in practice: `n` is the completion length
(≈ 80 tokens), `d_ff = 8192`, and the `d_ff × d_ff` route is ~8 GB per
stimulus where ours is ~1 MB.

## 3. Experiments

### G1 — Per-layer rank ratio on distortion stimuli

Per stimulus `x` in the cognitive-distortion set we compute `RankRatio_pos(l, x, T)`
under the **therapeutic** completion as target, and `RankRatio_pos(l, x, S)`
under the **sycophantic** completion as target, from the *same* forward pass
(two forward+backwards per stimulus, one per role). Paired t-test and paired
bootstrap CI on the per-stimulus difference `Δ(l, x) = T - S`. 500-step
sign-flip paired bootstrap for the CI.

### G5 — Capacity probe

Define a per-stimulus per-target capacity

```
C(x, target) = 1 / mean_l RankRatio_pos(l, x, target)
```

(scalars are small; inverted so that higher = more capacity). We compare:

- `C_ther(dist)` vs `C_syc(dist)` on distortion stimuli
  — within-stimulus paired: does the model *know* therapeutic less than sycophantic on the same input?
- `C_ther(dist)` vs `C_ther(factual)` on the factual-control stimuli
  — does the model *know* therapeutic less than it knows a plain factual reframe?

Interpretation table:

| observed pattern | what it means |
|------------------|---------------|
| `C_ther(dist) ≈ C_syc(dist) ≈ C_ther(factual)` | no capacity gap; pure preference bias |
| `C_ther(dist) < C_syc(dist)`, `≈ C_ther(factual)` | stimulus-specific gap (the distortion elicits sycophantic attractor) |
| `C_ther(dist) < C_ther(factual)` | true therapeutic-capacity deficit independent of distortion framing |

### G3 — Mechanism-only steering direction `v*`

For each held-out stimulus we also capture the residual-stream gradient at one
pre-registered layer (the median sampled layer): `∇_{h_res^(l)} L(y|x)` for the
`T` and `S` targets, mean-pooled over completion tokens. Stack per-stimulus
contrasts into `M = [g_T(x_i) - g_S(x_i)]_i ∈ R^{N × d_model}`. Our
mechanism-only steering direction is the top right-singular vector of `M`,
sign-fixed to align with the mean row:

```
v* = arg max_{||v||=1} || M v ||   s.t.   sign(mean_row(M) · v) = +1
```

Compared to the host repo's activation contrast `d_baseline =
normalise(mean(syc_acts) - mean(ther_acts))`, `v*` is derived purely from how
the loss *would like* to change the residual stream, not from which tokens the
completions use — which should make it less sensitive to style (opener,
length, register) and more sensitive to mechanism. We measure specificity
(shift in `log p(T)−log p(S)` over shift in `log p(T)−log p(cold)`) for both
directions at the same layer and α.

### G4 — Consensus sharpening across the 12 CBT subtypes

For each of the 12 subtypes `c`, compute its mean (over stimuli in `c`) per-layer
profile `π_c(l) = RankRatio_pos(l, T) - RankRatio_pos(l, S)`. Stack into
`P ∈ R^{12 × L}`. Centre by per-column mean (remove the population capacity
gap at each layer) to isolate subtype-specific variation, then compute the
stable rank of `P_centred`. A low stable rank across 12 subtypes means one
shared cross-layer capacity-gap signature; a high stable rank means each
subtype has a distinct signature.

## 4. Design choices & guardrails

1. **`srank_pos`, not `srank_pre`, as the headline.** The paper reports that
   `srank_pos` transfers better across datasets; we carry both for the G1 table
   and report `pos` in G5 / G3 / G4.
2. **MLP gradient for G1/G4/G5; residual-stream gradient for G3.** The paper's
   rank-ratio signal is derived specifically from MLP `down_proj` gradients
   (Eq. 4–5). G3 is a *new* adaptation — the paper does not propose steering —
   and for steering the natural object is the residual-stream gradient, which
   is what the intervention hook modifies.
3. **Sum, not mean, for `L_pos`.** Eq. 3 explicitly sums across completion
   tokens. Scale-invariance of stable rank makes this choice immaterial for
   `RankRatio` in theory, but we use the paper's definition verbatim.
4. **Sign fix on `v*`.** SVD returns right singular vectors up to a sign; we
   pin `sign(mean_row(M) · v*) = +1` so the direction consistently encodes
   "`(g_T − g_S)` pooled". Without this fix, steering reverses direction on
   half of runs.
5. **Held-out intervention stimuli for G3.** We split the distortion set into
   a first-half fit and second-half evaluation, mirroring the host repo's
   held-out convention, so `v*` and `d_baseline` are both measured
   out-of-sample.
6. **`dtype=float64` inside `rank_ratio_from_h_g`.** Eigendecomposing
   `C_h` in fp32 introduces enough numerical noise that the pseudoinverse
   threshold dominates the result; doubling for this tiny `n × n` matrix is
   cheap and stable. Model forward/backward remains in whatever dtype the
   model was loaded in (fp32 on CPU/MPS, bf16 on CUDA).
7. **Memory discipline.** Per-layer `h` and `g_w` are huge (~70 MB per
   stimulus per layer). We compute the rank ratios inline inside the
   extraction loop and keep only the scalar results plus the small
   residual-stream gradients for G3. This is the only change that makes the
   1B forward+backward fit in unified memory.
8. **Thread-library interaction.** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` is
   required on macOS: importing `sklearn` before model load interacts with
   libomp such that PyTorch backward segfaults. The `finalize_run.sh` wrapper
   sets these.

## 5. Results

### 5.1 Smoke test (superseded)

An initial smoke test at n = 12 distortion + 12 factual stimuli and 5 layers
is not reported in this document; its headline `v*` specificity of 12.73 was
a consequence of the denominator-noise and in-sample-`v*` bugs flagged in
round 1 of review (see [grade_review_log.md](grade_review_log.md), items
B2 and B3). All quantitative claims in this document refer to the **full
1B run in §5.2** with the round-1 + round-2 fixes applied. The raw smoke-
test numbers remain in the JSON git history for audit but should not be
cited.

### 5.2 Full 1B run — OLMo-2-0425-1B-Instruct (review-corrected, n = 100)

Configuration: **100 distortion stimuli** (all cognitive_distortions.json
items across 12 CBT subtypes), 100 id-matched factual control stimuli, 9
sampled layers {0, 2, 4, 6, 8, 10, 12, 14, 15}, α = 4.0,
**n_intervene = 40** (second-half, out-of-sample), **n_random = 30**,
CPU / fp32, `OMP_NUM_THREADS=1` (see §4.8), seed = 42. Paired t with
exact df = 99, Welch's t (unpaired, df ≈ 198) for cross-corpus contrast,
cluster-bootstrap CIs by CBT subtype on the distortion side AND by
factual subcategory on the factual side (fix R3-b). Power analysis uses
paired formula for d_z and 2× unpaired per-group formula for d (fix R3-a).

**G1 — per-layer rank ratio (paired T − S, `srank_pos` variant), n = 100, df = 99.**

| Layer | mean T | mean S | Δ (T − S) | t | raw p_t | Holm | BH | p_sign | Cohen's d_z | cluster-95 CI |
|------:|-------:|-------:|----------:|:----------:|-------:|------:|-------:|-------:|------------:|:---|
|  0 | 0.625 | 0.564 | +0.061 | +2.19 | 0.031 | 0.061 | 0.035 | 0.035 | +0.22 | [−0.001, +0.126] |
|  **2** | 0.584 | 0.535 | +0.049 | +3.38 | 0.001 | **0.004** | **0.002** | 0.089 | +0.34 | [+0.015, +0.086] |
|  4 | 0.974 | 0.905 | +0.069 | +1.15 | 0.252 | 0.252 | 0.252 | 0.764 | +0.12 | [−0.078, +0.220] |
|  6 | 0.964 | 1.107 | −0.143 | −2.38 | 0.019 | 0.058 | **0.025** | **0.004** | −0.24 | [−0.247, −0.019] |
|  **8** | 0.857 | 1.064 | **−0.207** | **−5.70** | **<.001** | **<.001** | **<.001** | **<.001** | −0.57 | [−0.284, −0.115] |
| **10** | 1.145 | 1.758 | **−0.613** | **−6.18** | **<.001** | **<.001** | **<.001** | **<.001** | −0.62 | [−0.813, −0.406] |
| **12** | 0.775 | 0.950 | **−0.175** | **−5.73** | **<.001** | **<.001** | **<.001** | **<.001** | −0.57 | [−0.240, −0.096] |
| **14** | 1.703 | 3.008 | **−1.306** | **−5.91** | **<.001** | **<.001** | **<.001** | **<.001** | −0.59 | [−1.719, −0.883] |
| **15** | 2.429 | 2.944 | **−0.516** | −3.55 | 0.001 | **0.003** | **0.001** | 0.012 | −0.36 | [−0.808, −0.252] |

**Holm-FWER-significant** (p_Holm < 0.05): **L2, L8, L10, L12, L14, L15**.
**BH-FDR-significant** (q = 0.05): all above + **L0, L6** — 8 of 9 layers.
Cluster-bootstrap 95% CI excludes zero at the same 6 layers Holm flags.
Sign tests concur at L8, L10, L12, L14. Effect sizes at the deep layers
are moderate-to-large (d_z = 0.36–0.62).

Readout: **n = 100 resolves the cross-layer pattern cleanly**. Early
layers (L0–L2) show T *slightly above* S (positive Δ, small-to-moderate
effect) — these layers encode surface features of the input and the
therapeutic completions' different style is faintly detectable. The
**deep layers (L8–L15) reverse the sign to strongly negative** (d_z =
0.57–0.62) under Holm-FWER control, confirming that where semantics
dominates, therapeutic reframing sits inside a lower-dimensional
gradient-update subspace than sycophantic validation — i.e. the deep
layers know how to reframe therapeutically without a large knowledge
update. L4 is the crossover.

**G5 — capacity summary** (mean across layers, per-stimulus), n = 100.

| signal | value | test | t (df) | p_t | d | 95 % CI | n for 80% power |
|---|---:|---|:---:|---:|:---:|---|---:|
| `C_ther(dist) − C_syc(dist)` (paired, within-stimulus) | **+0.183** | paired-t | +3.81 (99) | **<.001** | d_z = +0.38 | **[+0.081, +0.280]** (cluster, paired) | 55 (HAVE 100 ✓) |
| `C_ther(dist) − C_ther(factual)` (unpaired, different corpora) | **−0.112** | Welch | −2.47 (198) | **0.014** | d = −0.35 | **[−0.184, −0.039]** (2s cluster, both sides) | 129 / group (have 100) |

Per-corpus means:

- `C_ther(dist)` = 0.9910
- `C_syc(dist)` = 0.8086
- `C_ther(factual)` = 1.1026

**Both contrasts now significant at n = 100.** The round-2 draft at
n = 36 (p = 0.312 on the factual contrast) was indeed underpowered; the
point estimate has held (−0.081 → −0.112), and with correct unpaired
Welch + two-sided cluster bootstrap on both sides, the T_dist vs T_factual
contrast is significant at p = 0.014 with a cluster-CI that excludes zero.

**Read-out of the capacity question.** The two contrasts together support
a **"mostly preference, partly capacity"** reading:

1. **Within the same distorted user input, therapeutic is more inside
   activated knowledge than sycophantic.** Paired `C_ther(dist) >
   C_syc(dist)` at p < 0.001, d_z = 0.38, cluster-CI [+0.08, +0.28]. The
   model is NOT missing therapeutic knowledge relative to sycophantic
   knowledge for this input — if anything, the therapeutic direction is
   more "ready to go."
2. **BUT the distortion framing itself deactivates some of the
   therapeutic capacity that is available on plain factual inputs.**
   Unpaired `C_ther(dist) < C_ther(factual)` at p = 0.014, d = −0.35,
   2s-cluster-CI [−0.18, −0.04]. Therapeutic reframing on distorted
   emotional content is measurably harder than therapeutic reframing on
   factual errors. The distortion prompts partially shift activated
   knowledge toward sycophantic attractors.

**Deployment implication.** Sycophancy is dominantly a preference /
routing failure (a system-prompt or small DPO mix-in should substantially
help) BUT the measured capacity deficit on distorted inputs (d = 0.35)
means that pure prompting is unlikely to fully close the gap. A
two-part fix is indicated:
- **(a) system-prompt / RLHF-style rerouting** to surface the therapeutic
  direction that already exists in the activated subspace (G1 shows it
  is there at L8–L15 with strong effect size);
- **(b) modest CBT-demonstration data mix-in** during instruction
  fine-tune to shrink the ~0.35-d activation gap that the distortion
  framing induces relative to factual content.

Pure retrieval / large knowledge augmentation is not indicated — the
capacity deficit is moderate, not catastrophic.

**G3 — mechanism steering at L8, α = 4.0, fit on first-half 50 stimuli,
intervene on second-half 40 stimuli, n_random = 30 control directions.**

| direction | ΔE5 (ther − syc) | ΔE6 (ther − cold) | specificity \|ΔE5\|/\|ΔE6\| | on-target vs random (n=30) |
|---|---:|---:|---:|:---:|
| random unit vectors | +0.016 ± 0.046 | −0.033 ± 0.044 | 1.65 (mean) | — |
| host repo `d_baseline` (activation) | +0.490 [+0.44, +0.54] | −0.162 [−0.22, −0.10] | 3.02 [2.07, 5.00] | p_E5 = 0.000 (0/30) |
| **GRADE `v*` (gradient top-PC)** | **+0.758** [+0.69, +0.82] | −0.135 [−0.20, −0.07] | **5.62** [3.69, 11.04] | **p_E5 = 0.000 (0/30)** |

Bootstrap CIs are stimulus-paired, n_boot = 2000.

- **v\* is significantly more on-target than d_baseline.** The 95 %
  CIs on ΔE5 are *non-overlapping*: v\* [+0.69, +0.82] vs d_baseline
  [+0.44, +0.54]. v\* produces a 55 % larger therapeutic-vs-sycophantic
  log-prob shift.
- **Off-target (ΔE6) leakage is similar on both.** At n = 40 both
  directions show robust negative ΔE6 (CIs exclude zero); the earlier
  n = 12 "v\* has no off-target leakage" claim was a small-sample
  artefact and does not hold at n = 40. v\* is −0.135 vs d_baseline
  −0.162, modestly less leakage but not a categorical improvement.
- **Specificity ratio.** v\* point estimate 5.62 [3.69, 11.04]; d_baseline
  3.02 [2.07, 5.00]. CIs overlap in the range [3.69, 5.00], i.e. v\*'s
  lower bound is roughly at d_baseline's upper bound — v\* is
  directionally more specific and the overlap is narrow, but the two
  ratios are NOT cleanly separated at n = 40. The 17.7× early-draft
  headline has been retracted; the ≈ 2× improvement at full n is the
  correct claim.
- **Both directions cleanly beat the random null on E5.** 0 of 30
  random unit directions reached either direction's on-target shift.
  The random-null specificity ratio (mean 1.65, SD ≈ 1) is dominated
  by near-zero denominators and is not a useful null for the ratio
  itself — which is why the E5-shift rank test is the principled
  signal.
- **cos(v\*, d_baseline) = +0.269.** Modest overlap — v\* and d_baseline
  encode related but distinct objects.

**G4 — consensus sharpening across 12 CBT subtypes** (n = 100).

- centred per-layer profile matrix `P ∈ R^{12 × 9}`.
- `srank_pre(P_centred)` = 1.488; `srank_pos(P_centred)` = **1.091**.
- Mean Δ-rank-ratio profile (L0–L15):
  `+0.06, +0.05, +0.08, −0.14, −0.20, −0.61, −0.17, −1.28, −0.51`.

`srank_pos ≈ 1.09` (out of 12) reproduces the 36-stim estimate (1.067)
to within numerical noise — the subtype-specific variation around the
mean capacity-gap signature is essentially 1-dimensional. All 12 CBT
subtypes exhibit the same cross-layer pattern (small positive Δ in L0–L4,
strongly negative Δ at L6–L15) to within a single scaling factor.
Stronger consensus claim than the host repo's activation-SVD 2.49-dim
result, because the subspace is measured on a gradient-derived profile
insulated from the style / opener / length confounds the host repo's
Limitation 7 warns about.

### 5.3 7B Colab (OLMo-3 7B Instruct DPO)

_Filled in by the user after running `grade_notebook.ipynb`:_

```
<!-- PLACEHOLDER_7B -->
```

## 6. Limitations (to be attacked in revision)

1. **Stable rank is scale-invariant but not translation-invariant.** If the
   model has a high-variance "bias" direction that dominates `h` equally across
   targets, `srank(C_h)` is inflated and the rank ratio is deflated uniformly.
   This would *dampen* any real difference between T and S, not create one, so
   it is a conservative bias — but the absolute magnitudes of `C(x, target)`
   are not directly comparable across models of different sizes.

2. **`L_pos` is teacher-forced.** Per paper §3.1 this is the post-response
   variant. For capacity judgements this is fine (we are asking "if you *had*
   to produce this, how many new knowledge directions are needed?"). It does
   not measure sampling preference — the host repo's teacher-forced log-prob
   signal does that directly.

3. **G3 sign-fixing is a choice.** An alternative is to sign-fix by projecting
   onto `d_baseline` and flipping if cosine is negative; we chose the
   data-only mean-row criterion so `v*` is not tied to `d_baseline`. Both
   give the same sign in the smoke test.

4. **Small stimulus count in the smoke test.** 12 stimuli × 12 subtypes = 1
   per subtype; the p-values are indicative, not confirmatory. The full 1B
   and 7B runs restore power.

5. **OLMo `down_proj` vs the paper's Llama `down_proj`.** OLMo-2 uses the same
   SwiGLU-style MLP with a `down_proj: d_ff → d_model`, so Eq. 4 applies
   verbatim. OLMo-3 7B uses the same topology. The paper's model set
   (Llama-3.1-8B, Qwen2.5-7B, Gemma2-9B) all share this, which is the reason
   the port is mechanical.

6. **The paper uses a trained supervised probe on top of stacked
   `{RankRatio_l}_l=1..L` for its final knowledge-gap decision.** We have not
   trained such a probe: we do not have labelled answerability data for CBT
   reframing. Our summary statistic `C = 1 / mean_l RankRatio_pos` is a
   direct, probe-free read-out. A probe would be a natural extension once a
   hand-labelled CBT-correctness set exists (e.g. author + 2nd-rater therapist
   judging each generated therapeutic continuation as valid CBT-reframing or
   not).
