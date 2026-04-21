# Literature Review: Mitigations for Paper Limitations

Commissioned during iteration 10 of the review loop, in response to the user
instruction to "see if the limitations can be mitigated." This document
summarises the published literature on each of the paper's numbered
limitations and ranks the mitigations by cost / impact. Items marked
Tier A are implemented (or implementable) in this repository; items in
Tier B/C are noted in the paper's Mitigations section as work the
follow-up should do.

## 1. Style confound in contrastive direction (Limitation 7 of paper.md)

Arditi et al. (NeurIPS 2024, *Refusal in Language Models Is Mediated by a
Single Direction*, arXiv:2406.11717) construct contrastive pairs with
matched instruction templates and validate transfer across unseen prompt
formats. Zou et al. (2023, *Representation Engineering*, arXiv:2310.01405)
use token-matched pairs + linear artificial tomography (LAT) to minimise
style bleed. **LEACE** (Belrose et al., NeurIPS 2023, *LEACE: Perfect
Linear Concept Erasure in Closed Form*, arXiv:2306.03819) provides a
closed-form projection that removes a linear nuisance concept (length,
opener) while minimally perturbing the rest — directly applicable as a
post-hoc nuisance-removal step before computing `d_dist`. INLP
(Ravfogel et al., ACL 2020, arXiv:2004.07667) and RLACE (Ravfogel et al.,
ICML 2022, arXiv:2201.12091) are iterative alternatives.

## 2. Probe saturation (Limitation 7)

Hewitt & Liang (EMNLP 2019, *Designing and Interpreting Probes with
Control Tasks*, arXiv:1909.03368) introduced **selectivity** — probe AUC
on real labels minus probe AUC on random labels — precisely to flag
"probe memorises, not extracts." Voita & Titov (EMNLP 2020,
*Information-Theoretic Probing with MDL*, arXiv:2003.12298) recommend
online-code / MDL probes over accuracy. Belinkov (*Computational
Linguistics* 2022, *Probing Classifiers: Promises, Shortcomings, and
Advances*, arXiv:2102.12452) is the canonical survey. Elazar et al.
(TACL 2021, *Amnesic Probing*, arXiv:2006.00995) asks the causal
question: does erasing the direction change behaviour?

## 3. Prompt-pair leakage in probe CV (addressed in the paper's M1 below)

Standard fix: `sklearn.model_selection.GroupKFold` with stimulus id as
the group. Conneau et al. (ACL 2018, *What you can cram into a single
vector*, arXiv:1805.01070) formalises the leakage concern for paired
SentEval stimuli. Voita et al. (ACL 2019, arXiv:1905.09418) and Hupkes
et al. (JAIR 2020) use grouped splits by default.

## 4. Per-subtype direction noise floor (Limitation 11)

Marks & Tegmark (2023, *The Geometry of Truth: Emergent Linear Structure
in LLM Representations*, arXiv:2310.06824) report difference-of-means
stability curves and recommend ≥ 100 pairs per direction. Tigges et al.
(2023, *Linear Representations of Sentiment in LLMs*, arXiv:2310.15154)
and Park et al. (ICML 2024, *The Linear Representation Hypothesis and
the Geometry of LLMs*, arXiv:2311.03658) both use label-shuffle
permutation nulls to establish baselines for PCA-based geometry
statistics; this is what our M2 implements. Pimentel et al. (ACL 2020,
arXiv:2004.03061) gives sample-complexity bounds for linear probes.

## 5. Random-direction null bias in negative steering (Limitation 9)

Turner et al. (2023, *Activation Addition: Steering Language Models
Without Optimization*, arXiv:2308.10248) document that steering-vector
*norm* alone perturbs log-probs because residual norm enters LayerNorm.
Panickssery et al. (ICLR 2024, *Steering Llama 2 via Contrastive
Activation Addition*, arXiv:2312.06681) introduce **norm-matched**
random-direction baselines: scale each random direction so its
projection onto the mean activation equals the true direction's
projection. Rimsky et al. (ACL 2024) further recommend "off-topic
prompts" where the target concept is absent, to isolate
norm-induced shifts from concept-specific shifts.

## 6. Sign-agnostic specificity ratio (Limitation 10)

Wu et al. (ICLR 2024, *Interpretability at Scale: Identifying Causal
Mechanisms in Alpaca*, arXiv:2305.08809) and DAS/Boundless-DAS report
**signed** intervention effects. Makelov et al. (ICLR 2024, *Is This
the Subspace You Are Looking For? An Interpretability Illusion for
Subspace Activation Patching*, arXiv:2311.17030) formalise why
sign-flipped or magnitude-only specificity metrics admit
interpretability illusions. Meng et al. (ICLR 2023, ROME,
arXiv:2202.05262) report signed efficacy and signed specificity
separately.

## 7. Held-out direction but same-dataset intervention (Limitation 12)

Arditi et al. (2024, refusal) is the template: fit on AdvBench, evaluate
on HarmBench / MaliciousInstruct / JailbreakBench. Templeton et al.
(Anthropic 2024, *Scaling Monosemanticity* and the Claude sycophancy
feature write-ups) evaluates steering on prompts drawn from
distributions disjoint from the feature-discovery set. Zou et al.
(2023, RepE) tests on TruthfulQA after fitting on curated honesty
stimuli. Sharma et al. (ICLR 2024, *Towards Understanding Sycophancy in
Language Models*, arXiv:2310.13548) provides **SycophancyEval**, a ready
held-out benchmark — though it is not clinical-sycophancy specific and
would need adaptation.

## 8. Single model family (Limitation 1)

Field standard is ≥ 2 families × 2 scales. Arditi et al. (2024, refusal)
replicates across 13 open-weight models (Llama-2, Llama-3, Qwen, Yi,
Gemma). Marks & Tegmark (2023, Geometry of Truth) tests on LLaMA-13B
and LLaMA-2-70B. Tigges et al. (2023, sentiment) replicates on Pythia
and LLaMA. Panickssery et al. (2024, CAA) on Llama-2-7B and -13B.

## 9. Preregistration without timestamp (Limitation 13)

OSF (osf.io) and AsPredicted (aspredicted.org) provide free time-stamped,
immutable registrations; the interpretability community has adopted
them (BlackBoxNLP 2023/2024). ML Reproducibility Checklist (Pineau et
al., NeurIPS 2021, arXiv:2003.12206) explicitly lists pre-registration
of analysis choices as best practice. A GPG-signed Git tag pushed to a
public repo before analysis is also accepted as a timestamp (Nosek et
al., *Trends in Cognitive Sciences* 2018).

## 10. Style vs. content disentanglement (Limitation 7 extended)

Burns et al. (ICLR 2023, *Discovering Latent Knowledge in Language
Models Without Supervision* / CCS, arXiv:2212.03827) uses contrast pairs
of (yes, no) continuations on the same prompt so any style difference
between the two is controlled by construction. Contrastive
representation learning (Radford et al., ICML 2021, CLIP,
arXiv:2103.00020; Gao et al., EMNLP 2021, SimCSE, arXiv:2104.08821)
shows matched pos/neg pairs isolate semantics from surface form. Mallen
et al. (ICLR 2024, *Eliciting Latent Knowledge from Quirky Language
Models*, arXiv:2312.01037) extends CCS-style probing to behavioural
properties close to sycophancy. Adversarial style probes (Elazar &
Goldberg, EMNLP 2018, *Adversarial Removal of Demographic Attributes*,
arXiv:1808.06640) are medium-expensive.

---

## Priority ranking

**Tier A (implemented or cheap; hours each):**
1. GroupKFold on stimulus id (addresses limitation 3). → **M1** in
   `mitigation_experiments.py`.
2. Label-shuffle null for participation ratio & pairwise cosine
   (limitation 4). → **M2**.
3. Held-out Cohen's d (limitation 14). → **M3**.
4. Signed specificity reporting (limitation 6). → **M4** in
   `paper.md §4.10` (computed from existing sweep numbers; no re-run).
5. **Hewitt & Liang 2019 control-task selectivity (limitation 2 / 10).**
   Compute probe AUC on real labels minus AUC on random labels, on
   downsampled stimulus counts so the real-label AUC falls off the
   ceiling — this is the direct published mitigation for the
   saturated-AUC problem that drives Limitation 7. ~1–2 hours on top of
   the existing probe harness. **Flagged as follow-up**; the M1 caveat
   in §4.10 explicitly recommends this as the diagnostic M1 itself
   cannot do.
6. Norm-matched random-direction null (limitation 5). → flagged for
   implementation in follow-up; ~1 hour of work on top of the existing
   sweep harness.
7. OSF timestamp of the prespecified rule (limitation 9). → 15 minutes
   to upload; not yet done because the repo is pre-publication.
8. LEACE-based nuisance removal (limitation 1). → closed-form; ~2–4
   hours of work; flagged as follow-up.

**Tier B (day or two each):**
8. Cross-dataset intervention evaluation on SycophancyEval or an LLM-
   generated clinical held-out set (limitation 7).
9. At least one additional model family (limitation 8) — e.g.,
   Llama-3.2-1B or Qwen-2.5-1.5B at matched scale.
10. CCS-style matched-continuation probing on a rewritten subset of 20–
    30 stimuli (limitation 10).

**Tier C (de-facto research projects):**
11. Full cross-family replication (≥ 4 families, ≥ 2 scales).
12. Amnesic probing + causal mediation on a proper held-out clinical
    sycophancy benchmark.
13. Style-adversarial probes with a learned style classifier.

The current implementation covers Tier A items 1–3. Items 4–7 are
cheap enough to be added before publication; items 8+ are
scope-creeping beyond what this paper commits to.
