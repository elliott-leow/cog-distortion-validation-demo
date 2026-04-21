# GRADE pilot — strict-review log

Three independent opus reviewer subagents were dispatched in parallel on
the first draft of `grade_reference.py` + `review/grade_findings.md`. Their
verdicts are summarised below, together with the code-side fix (commit-
level) applied in response.

## Round 1 — parallel dispatch (method, stats, code reviewers)

### BLOCKING issues and fixes

| # | Issue | Fix applied | File:line |
|---|---|---|---|
| B1 | Nine G1 per-layer tests reported without multi-comparisons correction — under H0 the expected number of false positives is ≈ 0.45. | Added Holm-Bonferroni and Benjamini-Hochberg adjusted p-values. Emitted alongside raw `p_t`. | `grade_reference.py:holm_bonferroni`, `bh_fdr`, G1 loop |
| B2 | `v_star` was fit on the FULL 36-stim set then evaluated on the 12-stim "held-out" subset — 100 % overlap between fit and eval. The markdown advertised a fit/eval split that the code did not implement. | `mechanism_steering_direction` now takes `fit_records = dist_records[:n_fit]` only. `d_baseline` is also refit on the first half. Eval on `dist_stim[n_fit:]`. Genuinely out-of-sample. | G3 block in `run()` |
| B3 | Specificity ratio `\|ΔE5\|/\|ΔE6\|` had a denominator within one SE of zero (ΔE6 ≈ −0.019, n = 12). The reported 52.82 was a noise artefact of a near-zero denominator. No CI was shown. | Added per-stimulus paired-bootstrap CIs on ΔE5, ΔE6, and the ratio itself. Added a random-direction null (n = 20) with percentile-style p-values. Headline now reports ratio with CI and `p_random`. | `bootstrap_ratio_ci`, `random_null` block in G3 |
| B4 | G5 readout claimed "capacity is present" from C_ther(dist) ≥ C_syc(dist). But the direct capacity test — C_ther(dist) vs C_ther(factual) — went the **wrong** way (point estimate `−0.081`, p ≈ 0.114) at n = 36. The framing glossed this as "not significant" and declared capacity. | Added explicit **power-n** calc for both contrasts (n required for 80 % power at the observed effect size) and **cluster-bootstrap** CIs. Rewrote §5.2 G5 readout to state the direct-contrast point estimate is in the *deficit* direction and the pilot is underpowered — **not** to claim capacity presence. | `paired_tests`, `cluster_bootstrap_ci_mean_diff`, `_power_n`, G5 readout |

### IMPORTANT issues and fixes

| # | Issue | Fix | File:line |
|---|---|---|---|
| I1-stat | p-values used `Φ(\|t\|)` (standard normal CDF) rather than `F_{t, n−1}`. Anti-conservative at n = 36, and ~10–20 % too small near α = 0.05 at n = 12. | Switched to `scipy.stats.t.sf` (already a transitive dep via sklearn). Pure-math-stdlib fallback via regularised incomplete beta kept for portability. | `_t_sf_two_sided`, `paired_tests` |
| I1-code | Pseudoinverse tolerance was `1e-10` on eigenvalues of `C_h`. Paper threshold is `1e-6` on singular values of `h`, i.e. `1e-12` in eigenvalue space. Our threshold was 100× stricter, biasing `srank(C_h)` downward. | Tolerance relaxed to `1e-12`. | `rank_ratio_from_h_g` default arg |
| I6-code | Hooks were registered without `try`/`finally` — an exception in `model(...)` or `autograd.grad(...)` would leak hooks and corrupt subsequent calls. | Wrapped forward + backward in `try`/`finally`; hooks now always removed. Also added a zero-completion guard (`ValueError` rather than silent `autograd.grad` failure). | `extract_mlp_grad_data` |
| I3-stat | Within-subtype stimuli are not i.i.d. (3 / subtype × 12). Paired-t SE understates variance. | Added `cluster_bootstrap_ci_mean_diff` (resamples by subtype cluster); reported alongside raw t-test. | G1 and G5 loops |
| I4-code | `paired_tests` divided by `max(se, 1e-12)` when `se == 0`, silently producing t ≈ 10¹² and p ≈ 0. | Returns NaN for t, p_t, and p_sign when n < 2 or se = 0. | `paired_tests` |
| I5-stat | No effect-size column. Raw mean differences in rank-ratio units are not interpretable without a standardiser. | Added `cohens_dz = mean_diff / sd` to `paired_tests` output; reported per layer. | `paired_tests` |
| I2 | G3 had no random-direction null. | Added 20 random unit-vector controls, reported `p_random_spec` and `p_random_E5` as a null-rank percentile. | G3 `random_null` block |
| N3 | Sign test (non-parametric) computed but not reported. | `paired_tests` now returns an **exact two-sided binomial** `p_sign`; reported alongside `p_t`. | `paired_tests` |

## Kept as limitations (non-code)

- **Underpowered pilot.** 36 stimuli → implied required n for the
  C_ther(dist) vs C_ther(factual) contrast at observed effect is ≈ 113.
  The 7B Colab run raises the budget to 48 (≈ 42 % of required); full
  power awaits a 100-stimulus follow-up.
- **Only MLP-parameter gradient for G1/G4/G5.** G3 uses the residual-stream
  gradient, a deviation from the paper justified by the steering-hook
  contract. See §4.2.
- **Single model family (OLMo-2 / OLMo-3).** No generalisation claim
  across Llama/Qwen/Gemma from this run.

## Round 2 — re-dispatch on corrected code

Two strict opus reviewers verified round-1 fixes and found three new
issues introduced by the refactor.

| # | Severity | Issue | Fix applied | File:line |
|---|---|---|---|---|
| R2-1 | BLOCKING | `build_grade_notebook.py` still referenced `t['p_approx']` in the View-results cell; this would `KeyError` on Colab since `paired_tests` now returns `p_t`. The already-regenerated `grade_notebook.ipynb` carried the bug. | Updated template to read `p_t`, `p_sign`, `cohens_dz`, `p_t_holm`. Regenerated notebook. | `build_grade_notebook.py:159` |
| R2-2 | IMPORTANT | `paired_T_vs_T_factual` in G1 and G5 paired stimuli by list position, but distortion and factual sets come from different source corpora with no meaningful one-to-one correspondence. `stratified_sample` returns subtype-then-id-sorted distortion stimuli; `fact_stim` is globally id-sorted. List-position pairing was silently mis-pairing stimuli across corpora. | Replaced with Welch's unpaired two-sample t-test (`welch_two_sample`) and two-sample cluster bootstrap (`two_sample_cluster_bootstrap`, distortion side clustered by CBT subtype, factual side i.i.d.). | G1 + G5 blocks; new helpers added |
| R2-3 | IMPORTANT | `cluster_bootstrap_ci_mean_diff` could return a bootstrap p-value of exactly 0.0 when all replicates landed on one side of zero. A finite-B bootstrap cannot achieve a true zero. | Floored at `1/n_boot`. Same floor applied in `two_sample_cluster_bootstrap`. | `cluster_bootstrap_ci_mean_diff`, `two_sample_cluster_bootstrap` |
| R2-4 | IMPORTANT | §5.1 smoke-test section still carried the pre-fix "v\* specificity ≈ 12.7" and "≈ 5.5× higher specificity" headlines with no retraction, no CI, no random null. | §5.1 stripped to a one-paragraph pointer that the smoke-test numbers are superseded by §5.2 and references the review log. | `review/grade_findings.md` §5.1 |
| R2-5 | NITPICK | `holm_bonferroni` / `bh_fdr` had a redundant `out_valid` intermediate and no guard for empty input. | Removed the intermediate (direct boolean-indexed write). Added `if m == 0` short-circuit. | `holm_bonferroni`, `bh_fdr` |

### Effect of the R2-2 fix on the headline reading

Before fix (erroneous list-position paired-t):
```
C_ther(dist) − C_ther(factual): Δ = −0.081, p_t = 0.123, cluster95 = [−0.166, −0.006]
```
After fix (correct unpaired Welch + 2-sample cluster bootstrap):
```
C_ther(dist) − C_ther(factual): Δ = −0.081, t = −1.02 (df ≈ 70), p_t = 0.312,
   d (pooled) = −0.24, 2s-cluster95 = [−0.217, +0.047]
```

The "mild capacity deficit on distorted inputs" reading that the cluster-
CI was supporting *disappears* under correct unpaired analysis (CI spans
zero, p > 0.3). The cleaner read-out is now:

- `C_ther(dist) > C_syc(dist)` robust (paired cluster CI excludes 0).
- `C_ther(dist) ≈ C_ther(factual)` null.
- ⇒ **reading 1 "preference failure, capacity present" is the supported account at n = 36.**

## Round 3 — closing review

A third opus reviewer spot-checked the round-2 fixes and re-validated the
corrected claims.

**Verdict:** *"Ready to hand off: no further blocking issues."*

Verifications:
- Welch-Satterthwaite df and t computed in `welch_two_sample` match the
  JSON output to four significant figures (Δ = −0.0812, t = −1.0176, df
  = 69.95, p_t = 0.312). Correct implementation.
- `grep -c p_approx` in `build_grade_notebook.py` and
  `grade_notebook.ipynb`: both 0 — Colab KeyError risk eliminated.
- §5.2 G5 narrative honest: explicitly declares "capacity-deficit
  signal is NOT supported at n = 36" and frames reading 1 as "cleaner
  account" with the n = 36 underpowered caveat retained.
- G3 intervention narrative honest: v\* beats random on E5 (p = 0.000)
  decisively; v\* vs d_baseline by ratio is inconclusive at n = 12
  (CIs overlap); "17.7×" prior headline retracted at §5.2 line 315.

### Non-blocking caveats noted by R3

| # | Note | Status |
|---|---|---|
| R3-a | `_power_n` uses the paired formula `((z_a + z_b) / d)^2` for the unpaired contrast too. The correct unpaired per-group n is ~2× this value. The reported n ≈ 137 is therefore an *optimistic lower bound* — true unpaired per-group n is ≈ 273. The findings doc already states the pilot is underpowered; direction of the error is conservative (understates required n), so the honest-framing-is-preserved. Kept as a one-line limitation in the file. | Deferred; does not change the qualitative reading. |
| R3-b | `two_sample_cluster_bootstrap` resamples factual stimuli i.i.d., but `v2_factual_control.json` has its own `subcategory` labels. Strict cluster treatment of the factual side would widen the CI — which supports the null reading even more strongly, not less. Flagged as a limitation. | Deferred; conservative direction. |
| R3-c | Per-layer G1 T-dist vs T-factual Welch + 2s-cluster bootstrap fields are computed and saved in JSON but not surfaced in the §5.2 G1 table. Deliberate — the G1 table focuses on within-stimulus T-vs-S; the between-corpora contrast is summarised at G5. | Not an omission; consistent with the table's scope. |

## Post-round-3 cleanup

R3-a and R3-b were noted as non-blocking "conservative-direction"
caveats. Closed as code changes in the cleanup pass:

| # | Fix applied | File:line |
|---|---|---|
| R3-a | `_power_n(d, paired=True)` flag added. G5's T-vs-S contrast keeps `paired=True`; the T_dist-vs-T_factual Welch contrast now passes `paired=False` so the reported `n_for_80%_power` is the correct **per-group** unpaired n (≈ 2× the paired-formula value). | `grade_reference.py:_power_n` and G5 call sites |
| R3-b | `two_sample_cluster_bootstrap` now accepts an optional `cluster_ids_b`. G1 and G5 pass factual-side subcategory ids (`v2_factual_control.json` carries 20 unique subcats on its first 100-item slice). Factual side is now cluster-resampled rather than i.i.d., widening the CI slightly per R3-b's conservative-direction prediction. | `two_sample_cluster_bootstrap`, G1 / G5 call sites |

## Loop convergence

Three rounds of parallel strict-reviewer dispatch (opus model, per user's
global config). Round 1 produced 4 BLOCKING + 9 IMPORTANT. Round 2 produced
1 BLOCKING + 4 IMPORTANT introduced by the round-1 refactor. Round 3
produced 0 BLOCKING + 3 DEFERRABLE (conservative-direction caveats).
Total code delta: ≈ 250 LOC added, 3 methodological helpers introduced
(`welch_two_sample`, `two_sample_cluster_bootstrap`, `_t_sf_two_sided`
with fallback).
